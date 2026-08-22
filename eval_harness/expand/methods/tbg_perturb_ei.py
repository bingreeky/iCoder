"""tbg_perturb_ei — TritonBench-G × Evol-Instruct-style minor perturbation.

Prompt-first: LLM rewrites the NL request to ADD ONE small additional
behaviour under a mutation contract. Teacher v4-pro then reads (evolved NL
+ original ref) and emits a new Triton kernel + wrapper. The test_xxx()
block is copied verbatim from the original. New file is gate-validated like
BE, then emitted with `reference_solution = generated .py`.

Per (seed, variant): 1 prompt-rewrite call + up to N teacher calls + gates.
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
    PLAN_SYSTEM_TBG,
    PLAN_USER_TBG,
    _extract_triton_interface,
)


PERTURB_TBG_EI_PROMPT_SYSTEM = (
    "You rewrite a Triton-kernel implementation request to add ONE small "
    "additional behaviour, governed by a mutation contract. The rewritten "
    "request must:\n"
    "  - Read like a standalone NL task description for a Triton engineer "
    "    (no operator jargon).\n"
    "  - Preserve every functional requirement of the original task.\n"
    "  - Add the operator's new property as ONE extra sentence or bullet.\n"
    "  - Name the wrapper function and any kernel names explicitly.\n"
    "  - Mention `torch.allclose(atol=1e-2, rtol=1e-2)` as the tolerance "
    "    against the reference test harness.\n\n"
    "Output ONLY the evolved NL request — plain text, no tags."
)

PERTURB_TBG_EI_PROMPT_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "What to do         : {definition}\n"
    "Old property (orig): {old_property}\n"
    "New property       : {new_property}\n"
    "=== End contract ===\n\n"
    "Public interface (preserve):\n"
    "  wrapper:       {primary_wrapper}\n"
    "  signature:     {primary_signature}\n"
    "  kernels:       {kernel_names}\n"
    "  test_func:     {test_func_name}\n\n"
    "Original reference (for your eyes only — do not embed verbatim):\n"
    "```python\n{ref_src}\n```\n\n"
    "Write the evolved NL request now."
)


PERTURB_TBG_EI_KERNEL_SYSTEM = (
    "You implement a Triton kernel + wrapper to spec. You receive an "
    "evolved NL request and an ORIGINAL TritonBench-G reference. Emit a "
    "new file that satisfies the evolved request.\n\n"
    "Hard requirements:\n"
    "  - Wrapper function name and signature: identical to the original.\n"
    "  - Kernel name(s): identical to the original.\n"
    "  - `def test_xxx()` block and `result_gold = test_xxx()` last line: "
    "    BYTE-FOR-BYTE IDENTICAL to the original (copy them verbatim).\n"
    "  - The new behaviour must be expressible inside Triton (`tl.*`).\n\n"
    "Output ONLY the full file in a single ```python ... ``` fence."
)

PERTURB_TBG_EI_KERNEL_USER = (
    "=== Mutation contract (already encoded in the prompt; here for "
    "transparency) ===\n"
    "Operator           : {name}\n"
    "Old property       : {old_property}\n"
    "New property       : {new_property}\n"
    "Must NOT change    : {do_not_change}\n"
    "=== End contract ===\n\n"
    "Evolved NL request:\n---\n{evolved_prompt}\n---\n\n"
    "Original TritonBench-G reference (use as starting point; copy "
    "test_xxx and result_gold verbatim):\n"
    "```python\n{ref_src}\n```\n\n"
    "Emit the new file in one ```python ... ``` fence."
)


@register_method("tbg_perturb_ei")
class TBGPerturbEIExpansion:
    """Prompt-first minor-perturbation for TritonBench-G seeds."""

    name = "tbg_perturb_ei"
    MAX_PERTURB_ATTEMPTS = 2
    # Operator fallback DISABLED (mirrors KB-EI P0 fix dated 2026-06-27).
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
        primaries = [EVOL_OPERATORS_TBG[(seed_hash + i) % len(EVOL_OPERATORS_TBG)]
                     for i in range(num_variants)]
        # A-mode: ×m model dimension.
        m = router.num_prompt_models
        coros = [
            self._perturb_one(seed, p, op_slot * m + model_idx,
                              ref, orig_iface,
                              router, seed_hash, cfg)
            for op_slot, p in enumerate(primaries)
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
        evolved_prompt = ""
        variant_src = None
        traj_wrap = ""
        last_gate = None
        op_used = None

        for op in chain:
            # Step 1: evolve the NL prompt (prompt-side LLM)
            evolved_prompt = await self._evolve_prompt(
                seed, op, ref, orig_iface, router, seed_hash, op_idx)
            if not evolved_prompt:
                err_hist.append(f"op={op['name']}: prompt-rewrite failed")
                continue
            # Step 2: teacher generates new kernel + wrapper (gates)
            variant_src, traj_wrap, last_gate, op_err_hist = (
                await self._evolve_kernel(seed, op, evolved_prompt, ref,
                                          orig_iface, router, cfg,
                                          seed_hash=seed_hash, variant_idx=op_idx))
            err_hist.extend(
                [f"op={op['name']} attempt={i+1}: {e}"
                 for i, e in enumerate(op_err_hist)])
            if variant_src is not None:
                op_used = op
                break

        if variant_src is None or op_used is None:
            return [base_expanded(
                seed, self.name, op_idx,
                evolved_prompt or seed.original_prompt,
                extra_meta={
                    "warn": "perturb_gate_failed",
                    "operator": op_primary["name"],
                    "operator_chain": [o["name"] for o in chain],
                    "perturb_attempts": len(err_hist),
                    "perturb_last_err": err_hist[-1] if err_hist else "",
                    "framing": "passthrough",
                })]

        # Plan from evolved prompt + variant ref.
        variant_iface = _extract_triton_interface(variant_src)
        plan = await self._plan(evolved_prompt, variant_src, router,
                                seed_hash, op_idx)
        if not plan:
            return []

        row = base_expanded(
            seed, self.name, op_idx, evolved_prompt,
            extra_meta={
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
                "framing": "evol_instruct",
                "framing_idx": op_idx,
                "variant_idx": op_idx,
            })
        row.reference_solution = variant_src
        return [row]

    async def _evolve_prompt(self, seed: Seed, op: Dict[str, Any], ref: str,
                             orig_iface, router: LLMRouter,
                             seed_hash: int, variant_idx: int) -> str:
        user = PERTURB_TBG_EI_PROMPT_USER.format(
            name=op["name"],
            definition=op["definition"],
            old_property=op["old_property"],
            new_property=op["new_property"],
            primary_wrapper=orig_iface.primary_wrapper,
            primary_signature=orig_iface.primary_signature,
            kernel_names=orig_iface.kernel_names,
            test_func_name=orig_iface.test_func_name,
            ref_src=ref)
        raw = await router.chat_prompt(
            PERTURB_TBG_EI_PROMPT_SYSTEM, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()

    async def _evolve_kernel(
        self, seed: Seed, op: Dict[str, Any], evolved_prompt: str, ref: str,
        orig_iface, router: LLMRouter, cfg: GateConfigTBG,
        *, seed_hash: int = 0, variant_idx: int = 0,
    ) -> Tuple[Optional[str], str, Any, List[str]]:
        base_user = PERTURB_TBG_EI_KERNEL_USER.format(
            name=op["name"],
            old_property=op["old_property"],
            new_property=op["new_property"],
            do_not_change=fmt_invariants(op["do_not_change"]),
            evolved_prompt=evolved_prompt,
            ref_src=ref)
        err_history: List[str] = []
        last_wrap = ""
        last_gate = None
        for attempt in range(self.MAX_PERTURB_ATTEMPTS):
            user = base_user
            if err_history:
                user += (
                    "\n\nYour previous attempt failed a validation gate:\n"
                    f"```\n{err_history[-1]}\n```\n"
                    "Re-emit the FULL file in one ```python``` fence. "
                    "Tighten the edit. Keep wrapper + kernel name + "
                    "test_xxx byte-identical to the original.")
            resp = await router.chat_prompt_full(
                PERTURB_TBG_EI_KERNEL_SYSTEM, user,
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
                test_var_name="result_gold",
                body_kind="triton_kernels")
            last_gate = gate
            if gate.ok:
                return variant_src, wrap, gate, err_history
            err_history.append(gate.reason)
        return None, last_wrap, last_gate, err_history

    async def _plan(self, evolved_prompt: str, variant_src: str,
                    router: LLMRouter, seed_hash: int,
                    variant_idx: int) -> str:
        user = PLAN_USER_TBG.format(
            expanded_prompt=evolved_prompt, ref_src=variant_src)
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
