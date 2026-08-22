"""tbg_perturb_be — TritonBench-G × BenchEvolver-style minor perturbation.

Solution-first: the LLM rewrites one `@triton.jit` kernel under ONE mutation
contract from `EVOL_OPERATORS_TBG`. Wrapper signature and the `test_xxx()`
block are byte-for-byte preserved. Variant must pass 5 gates:
  g1 format          — kernel names + wrapper signature + test_func_name unchanged
  g2 kernel-body diff — kernel-body char diff in [0.005, 0.50]
  g3 smoke           — variant compiles + ref test_xxx() runs
  g5 (no-op for now)
  g4 behavioural     — count `result_gold` dict keys that differ; 1 ≤ n_diff ≤ ceil(0.4 * N)

NL prompt reuses INVERSE_FRAMINGS_TBG (4 framings) on the VARIANT ref.

Output Expanded row:
    reference_solution      = full variant .py (modified kernel + original test block)
    expanded_prompt         = back-derived NL
    metadata.assistant_spec = implementation plan
    metadata.operator
    metadata.behavioral_n_diff / behavioral_n_keys
    metadata.kernel_body_diff_ratio
    metadata.original_reference  (audit)

Downstream uses existing scripts/teacher_triton_rollout_tbg.py + tbg verify.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..base import Expanded, Seed
from ..llm import LLMRouter
from ..registry import register_method
from ._common import (
    base_expanded,
    check_v4pro_format,
    extract_v4pro_answer,
    strip_dryrun,
    synthesize_v4pro_wrap,
)
from ._perturb_common import (
    EVOL_OPERATORS_TBG,
    GateConfigTBG,
    extract_python,
    fmt_invariants,
    run_gates_tbg,
)
from .inversecoder import (
    INVERSE_FRAMINGS_TBG,
    BACK_DERIVE_SYSTEM_TBG,
    BACK_DERIVE_USER_TBG,
    PLAN_SYSTEM_TBG,
    PLAN_USER_TBG,
    _extract_triton_interface,
)


PERTURB_TBG_BE_SYSTEM = (
    "You are a careful Triton-kernel editor. You apply EXACTLY ONE bounded "
    "edit to a verified TritonBench-G reference under a mutation contract. "
    "The edit must be a MINOR functional perturbation — most output values "
    "should still match the original.\n\n"
    "HARD CONSTRAINTS (violating any → your output is rejected):\n"
    "  - Output ONE complete file in a single ```python ... ``` fence.\n"
    "  - The wrapper function signature, kernel names, and "
    "    `def test_xxx() ... result_gold = test_xxx()` block must be "
    "    BYTE-FOR-BYTE IDENTICAL to the original. Copy them verbatim.\n"
    "  - Only the body of ONE `@triton.jit` kernel (or the wrapper "
    "    post-kernel-launch, per contract) is modified.\n"
    "  - Do NOT introduce Python data-dependent control flow inside the "
    "    kernel; stick to `tl.*`, `tl.where`, arithmetic, and program-id "
    "    constant-modulo conditions if the contract calls for them."
)

PERTURB_TBG_BE_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "Target             : {target_hint}\n"
    "What to do         : {definition}\n"
    "May touch          : {scope}\n"
    "Must NOT change    : {do_not_change}\n"
    "Old property (orig): {old_property}\n"
    "New property (you) : {new_property}\n"
    "=== End contract ===\n\n"
    "Original TritonBench-G reference (entire file, kernels + wrapper + "
    "test block):\n"
    "```python\n{ref_src}\n```\n\n"
    "Apply EXACTLY this contract. Output the full modified file in one "
    "```python ... ``` fence."
)


@register_method("tbg_perturb_be")
class TBGPerturbBEExpansion:
    """Solution-first minor-perturbation for TritonBench-G seeds.

    Per (seed, variant): pick one operator from EVOL_OPERATORS_TBG
    (hash-rotated). LLM rewrites ONE kernel under contract. 5-gate
    validation. NL back-derive on the variant ref (reuse INVERSE_FRAMINGS_TBG).
    """

    name = "tbg_perturb_be"
    MAX_PERTURB_ATTEMPTS = 2
    # Operator fallback DISABLED (mirrors KB-BE P0 fix dated 2026-06-27):
    # with ~70-90% gate failure rates the chain doubles per-variant work
    # to 4 LLM rolls + 4 gates, killing throughput. num_variants already
    # supplies operator diversity via hash rotation.
    MAX_OPERATOR_FALLBACKS = 0

    async def expand(self, seed: Seed, llm, num_variants: int = 1,
                     **kw) -> List[Expanded]:
        if seed.source_dataset != "tritonbench_g":
            return []
        ref = seed.reference_solution or ""
        if not ref.strip():
            return []
        orig_iface = _extract_triton_interface(ref)
        if orig_iface is None:
            return []

        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))
        cfg = _gate_config_from_kw(kw)
        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)
        # Primary operator per variant via hash rotation, then fallback chain.
        primaries = [EVOL_OPERATORS_TBG[(seed_hash + i) % len(EVOL_OPERATORS_TBG)]
                     for i in range(num_variants)]
        # A-mode: ×m model dimension.
        m = router.num_prompt_models

        coros = [
            self._perturb_one(seed, primary, op_slot * m + model_idx,
                              ref, orig_iface,
                              router, seed_hash, cfg)
            for op_slot, primary in enumerate(primaries)
            for model_idx in range(m)
        ]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _perturb_one(self, seed: Seed, op_primary: Dict[str, Any],
                           op_idx: int, ref: str, orig_iface,
                           router: LLMRouter, seed_hash: int,
                           cfg: GateConfigTBG) -> List[Expanded]:
        primary_idx = EVOL_OPERATORS_TBG.index(op_primary)
        chain = [
            EVOL_OPERATORS_TBG[(primary_idx + k) % len(EVOL_OPERATORS_TBG)]
            for k in range(1 + self.MAX_OPERATOR_FALLBACKS)
        ]
        err_hist: List[str] = []
        op_used = None
        variant_src = None
        traj_wrap = ""
        last_gate = None
        for chain_idx, op in enumerate(chain):
            variant_src, traj_wrap, last_gate, op_err_hist = (
                await self._evolve_kernel(seed, op, ref, orig_iface, router,
                                          cfg,
                                          seed_hash=seed_hash, variant_idx=op_idx))
            err_hist.extend(
                [f"op={op['name']} attempt={i+1}: {e}"
                 for i, e in enumerate(op_err_hist)])
            if variant_src is not None:
                op_used = op
                break
        if variant_src is None or op_used is None:
            return [base_expanded(
                seed, self.name, op_idx, seed.original_prompt,
                extra_meta={
                    "warn": "perturb_gate_failed",
                    "operator": op_primary["name"],
                    "operator_chain": [o["name"] for o in chain],
                    "perturb_attempts": len(err_hist),
                    "perturb_last_err": err_hist[-1] if err_hist else "",
                    "framing": "passthrough",
                })]

        # NL back-derive + plan on the VARIANT ref. Reuse INVERSE_FRAMINGS_TBG.
        framing = INVERSE_FRAMINGS_TBG[op_idx % len(INVERSE_FRAMINGS_TBG)]
        variant_iface = _extract_triton_interface(variant_src)
        nl = await self._back_derive(variant_iface, variant_src, op_used,
                                     framing, router, seed_hash, op_idx)
        if not nl:
            return []
        plan = await self._plan(nl, variant_src, router, seed_hash, op_idx)
        if not plan:
            return []

        row = base_expanded(
            seed, self.name, op_idx, nl,
            extra_meta={
                "framing": framing[0],
                "framing_idx": op_idx % len(INVERSE_FRAMINGS_TBG),
                "variant_idx": op_idx,
                "operator": op_used["name"],
                "operator_primary": op_primary["name"],
                "operator_chain_index": chain.index(op_used),
                "behavioral_n_diff": last_gate.n_diff if last_gate else None,
                "behavioral_n_keys": last_gate.n_keys if last_gate else None,
                "kernel_body_diff_ratio": last_gate.body_diff if last_gate else None,
                "original_reference": ref,
                "perturb_attempts": len(err_hist) + 1,
                "assistant_spec": plan,
                "triton_interface": asdict(variant_iface) if variant_iface else None,
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, op_idx),
                "traj_content": traj_wrap,
            })
        row.reference_solution = variant_src
        return [row]

    async def _evolve_kernel(
        self, seed: Seed, op: Dict[str, Any], ref: str, orig_iface,
        router: LLMRouter, cfg: GateConfigTBG,
        *, seed_hash: int = 0, variant_idx: int = 0,
    ) -> Tuple[Optional[str], str, Any, List[str]]:
        base_user = PERTURB_TBG_BE_USER.format(
            name=op["name"],
            target_hint=op["target_hint"],
            definition=op["definition"],
            scope=op["scope"],
            do_not_change=fmt_invariants(op["do_not_change"]),
            old_property=op["old_property"],
            new_property=op["new_property"],
            ref_src=ref)
        err_history: List[str] = []
        last_wrap = ""
        last_gate = None
        consecutive_zero_diff = 0
        for attempt in range(self.MAX_PERTURB_ATTEMPTS):
            user = base_user
            if err_history:
                user += (
                    "\n\nYour previous attempt failed a validation gate:\n"
                    f"```\n{err_history[-1]}\n```\n"
                    "Re-emit the FULL modified file in one ```python ... "
                    "``` fence. The mutation contract above is unchanged. "
                    "Tighten the edit. DO NOT touch the wrapper, test "
                    "block, or kernel names.")
            resp = await router.chat_prompt_full(
                PERTURB_TBG_BE_SYSTEM, user,
                seed_hash=seed_hash, variant_idx=variant_idx)
            raw = resp.content or ""
            wrap = synthesize_v4pro_wrap(
                raw, resp.reasoning_content, fallback_lang="python")
            last_wrap = wrap
            ok_fmt, fmt_reason = check_v4pro_format(raw)
            if not ok_fmt:
                err_history.append(f"format: {fmt_reason}")
                continue
            answer = extract_v4pro_answer(strip_dryrun(raw))
            variant_src = extract_python(answer)
            if not variant_src.strip():
                err_history.append("no ```python``` fence")
                continue
            is_final = attempt == self.MAX_PERTURB_ATTEMPTS - 1
            gate = await run_gates_tbg(
                ref, variant_src, cfg,
                is_final_attempt=is_final,
                orig_interface=orig_iface,
                test_var_name="result_gold")
            last_gate = gate
            if gate.ok:
                return variant_src, wrap, gate, err_history
            err_history.append(gate.reason)
            if gate.body_diff == 0.0:
                consecutive_zero_diff += 1
                if consecutive_zero_diff >= 2:
                    err_history.append(
                        "operator-mismatch short-circuit (2× zero diff)")
                    break
            else:
                consecutive_zero_diff = 0
        return None, last_wrap, last_gate, err_history

    async def _back_derive(self, interface, ref_src: str,
                           op: Dict[str, Any],
                           framing: Tuple[str, str], router: LLMRouter,
                           seed_hash: int, variant_idx: int) -> str:
        f_name, f_desc = framing
        # BACK_DERIVE_USER_TBG fields: kernel_names, primary_wrapper,
        # primary_signature, test_func_name, ref_src, summary.
        user = BACK_DERIVE_USER_TBG.format(
            kernel_names=interface.kernel_names,
            primary_wrapper=interface.primary_wrapper,
            primary_signature=interface.primary_signature,
            test_func_name=interface.test_func_name,
            ref_src=ref_src,
            summary=(
                f"This is a minor perturbation of the original Triton kernel "
                f"under operator `{op['name']}`. New property: "
                f"{op['new_property']}. The change affects only a small "
                f"fraction of output values."))
        system = BACK_DERIVE_SYSTEM_TBG.format(
            framing=f_name, framing_desc=f_desc,
            primary_wrapper=interface.primary_wrapper)
        raw = await router.chat_prompt(
            system, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()

    async def _plan(self, nl: str, ref_src: str, router: LLMRouter,
                    seed_hash: int, variant_idx: int) -> str:
        user = PLAN_USER_TBG.format(expanded_prompt=nl, ref_src=ref_src)
        raw = await router.chat_prompt(
            PLAN_SYSTEM_TBG, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()


def _gate_config_from_kw(kw: Dict[str, Any]) -> GateConfigTBG:
    p = kw.get("perturb") or {}
    return GateConfigTBG(
        diff_lo=p.get("diff_lo", 0.005),
        diff_hi=p.get("diff_hi", 0.50),
        diff_lo_final=p.get("diff_lo_final", 0.001),
        diff_hi_final=p.get("diff_hi_final", 0.70),
        behavioral_n_low=p.get("behavioral_n_low", 1),
        behavioral_ratio_hi=p.get("behavioral_ratio_hi", 0.40),
        behavioral_timeout=p.get("behavioral_timeout", 240),
        smoke_timeout=p.get("smoke_timeout", 120),
    )
