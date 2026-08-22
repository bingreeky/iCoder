"""tbt_perturb_be — TritonBench-T × BenchEvolver-style minor perturbation.

T-split refs are PyTorch standalone functions (`def foo(...)` calling torch
ops) with discrete `def test_xxx() ... test_results = test_xxx()` harness.

We reuse the KB operator catalogue (`EVOL_OPERATORS_KB`) — operators
"add one line near the return" work on any PyTorch function. But verify is
TBG-style dict comparison (`test_results` keys), not KB-style fuzz.

Per (seed, variant): contract → LLM rewrites the function body → 5 gates
(g1 format on TBT func name + arg names, g2 func-body diff, g3 smoke for
test_results attr, g5 no-op, g4 dict-key diff count) → NL back-derive via
TBG framings.
"""

from __future__ import annotations

import asyncio
import hashlib
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
    EVOL_OPERATORS_KB,
    GateConfigTBG,
    extract_python,
    fmt_invariants,
    run_gates_tbg,
    _extract_tbt_func_interface,
)
from .inversecoder import (
    INVERSE_FRAMINGS_TBG,
    BACK_DERIVE_SYSTEM_TBG,
    PLAN_SYSTEM_TBG,
    PLAN_USER_TBG,
)


PERTURB_TBT_BE_SYSTEM = (
    "You are a careful PyTorch editor. You apply EXACTLY ONE bounded edit "
    "to a PyTorch function under a mutation contract. The edit must be a "
    "MINOR functional perturbation — most output values should still match.\n\n"
    "HARD CONSTRAINTS:\n"
    "  - Output ONE complete file in a single ```python ... ``` fence.\n"
    "  - The function NAME and SIGNATURE must stay identical.\n"
    "  - The `def test_xxx()` block and `test_results = test_xxx()` line "
    "    must be BYTE-FOR-BYTE IDENTICAL to the original (copy verbatim).\n"
    "  - Only the inside of the function body changes.\n"
    "  - No `.tolist()`, no `.item()`, no Python `for` over a tensor, no "
    "    data-dependent control flow."
)

PERTURB_TBT_BE_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "Target             : {target_hint}\n"
    "What to do         : {definition}\n"
    "May touch          : {scope}\n"
    "Must NOT change    : {do_not_change}\n"
    "Old property (orig): {old_property}\n"
    "New property (you) : {new_property}\n"
    "=== End contract ===\n\n"
    "Original TBT reference (entire file: function + test block):\n"
    "```python\n{ref_src}\n```\n\n"
    "Apply EXACTLY this contract. Output the full modified file in one "
    "```python ... ``` fence."
)


# A simplified back-derive prompt for TBT (no Triton kernel names — just
# function name + signature).
BACK_DERIVE_USER_TBT = (
    "Public interface:\n"
    "---\n"
    "function:   {func_name}\n"
    "args:       {arg_names}\n"
    "test func:  {test_func_name}\n"
    "---\n\n"
    "Reference PyTorch (for your eyes only — do NOT embed verbatim):\n"
    "```python\n{ref_src}\n```\n\n"
    "Behavioural summary (background only — paraphrase or use to "
    "structure the request, do not echo it verbatim):\n"
    "---\n{summary}\n---\n\n"
    "Now write the user's NL request — describe what the function should "
    "compute. The result must be implementable as a Triton kernel."
)


@register_method("tbt_perturb_be")
class TBTPerturbBEExpansion:
    """Solution-first minor-perturbation for TritonBench-T seeds.

    Same operators as KB (PyTorch ops on function body), TBG-style
    dict-compare verify (`test_results` keys).
    """

    name = "tbt_perturb_be"
    MAX_PERTURB_ATTEMPTS = 2
    # Operator fallback DISABLED (mirrors KB-BE P0 fix dated 2026-06-27).
    MAX_OPERATOR_FALLBACKS = 0

    async def expand(self, seed: Seed, llm, num_variants: int = 1,
                     **kw) -> List[Expanded]:
        if seed.source_dataset != "tritonbench_t":
            return []
        ref = seed.reference_solution or ""
        if not ref.strip():
            return []
        ei = seed.evaluator_info or {}
        func_src = ei.get("func_src", "")
        iface = _extract_tbt_func_interface(func_src) if func_src else None
        if iface is None:
            return []

        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))
        cfg = _gate_config_from_kw(kw)
        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)
        primaries = [EVOL_OPERATORS_KB[(seed_hash + i) % len(EVOL_OPERATORS_KB)]
                     for i in range(num_variants)]
        # A-mode: ×m model dimension.
        m = router.num_prompt_models
        coros = [
            self._perturb_one(seed, p, op_slot * m + model_idx,
                              ref, iface, router, seed_hash, cfg)
            for op_slot, p in enumerate(primaries)
            for model_idx in range(m)
        ]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _perturb_one(self, seed: Seed, op_primary: Dict[str, Any],
                           op_idx: int, ref: str, iface,
                           router: LLMRouter, seed_hash: int,
                           cfg: GateConfigTBG) -> List[Expanded]:
        primary_idx = EVOL_OPERATORS_KB.index(op_primary)
        chain = [
            EVOL_OPERATORS_KB[(primary_idx + k) % len(EVOL_OPERATORS_KB)]
            for k in range(1 + self.MAX_OPERATOR_FALLBACKS)
        ]
        err_hist: List[str] = []
        variant_src = None
        traj_wrap = ""
        last_gate = None
        op_used = None

        for op in chain:
            variant_src, traj_wrap, last_gate, op_err_hist = (
                await self._evolve_func(seed, op, ref, iface, router, cfg,
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

        # NL back-derive + plan on the VARIANT ref.
        framing = INVERSE_FRAMINGS_TBG[op_idx % len(INVERSE_FRAMINGS_TBG)]
        variant_iface = _extract_tbt_func_interface(variant_src) or iface
        nl = await self._back_derive(
            variant_iface, variant_src, seed.evaluator_info["test_func_name"],
            op_used, framing, router, seed_hash, op_idx)
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
                "behavioral_n_diff": last_gate.n_diff if last_gate else None,
                "behavioral_n_keys": last_gate.n_keys if last_gate else None,
                "func_body_diff_ratio": last_gate.body_diff if last_gate else None,
                "original_reference": ref,
                "perturb_attempts": len(err_hist) + 1,
                "assistant_spec": plan,
                "tbt_interface": variant_iface,
                # Provide a TBG-shaped interface dict so the existing
                # scripts/teacher_triton_rollout_tbg.py picks up the
                # contract fields it already reads.
                "triton_interface": {
                    "kernel_names": [],
                    "wrapper_names": [variant_iface["name"]],
                    "primary_wrapper": variant_iface["name"],
                    "primary_signature": (
                        f"{variant_iface['name']}("
                        f"{', '.join(variant_iface['arg_names'])})"
                    ),
                    "primary_arg_names": variant_iface["arg_names"],
                    "test_func_name": seed.evaluator_info.get(
                        "test_func_name", ""),
                    "has_class_wrapper": False,
                },
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, op_idx),
                "traj_content": traj_wrap,
            })
        row.reference_solution = variant_src
        return [row]

    async def _evolve_func(
        self, seed: Seed, op: Dict[str, Any], ref: str, iface,
        router: LLMRouter, cfg: GateConfigTBG,
        *, seed_hash: int = 0, variant_idx: int = 0,
    ) -> Tuple[Optional[str], str, Any, List[str]]:
        base_user = PERTURB_TBT_BE_USER.format(
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
                    "Re-emit the FULL file in one ```python``` fence. "
                    "Tighten the edit. Keep function name, signature, and "
                    "the entire `def test_xxx()` block byte-identical.")
            resp = await router.chat_prompt_full(
                PERTURB_TBT_BE_SYSTEM, user,
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

            # TBT g1 format: re-extract interface, must match name + arg names.
            variant_iface = _extract_tbt_func_interface(variant_src)
            if variant_iface is None:
                err_history.append("g1: variant has no top-level def")
                continue
            if variant_iface["name"] != iface["name"]:
                err_history.append(
                    f"g1: function name changed "
                    f"{iface['name']!r} → {variant_iface['name']!r}")
                continue
            if variant_iface["arg_names"] != iface["arg_names"]:
                err_history.append(
                    f"g1: function args changed "
                    f"{iface['arg_names']} → {variant_iface['arg_names']}")
                continue

            is_final = attempt == self.MAX_PERTURB_ATTEMPTS - 1
            gate = await run_gates_tbg(
                ref, variant_src, cfg,
                is_final_attempt=is_final,
                orig_interface=None,           # TBG-style g1 skipped (no kernel)
                test_var_name="test_results",  # TBT
                body_kind="tbt_func")
            last_gate = gate
            if gate.ok:
                return variant_src, wrap, gate, err_history
            err_history.append(gate.reason)
            if gate.body_diff == 0.0:
                consecutive_zero_diff += 1
                if consecutive_zero_diff >= 2:
                    err_history.append("zero-diff short-circuit")
                    break
            else:
                consecutive_zero_diff = 0
        return None, last_wrap, last_gate, err_history

    async def _back_derive(self, iface, ref_src: str, test_func_name: str,
                           op: Dict[str, Any],
                           framing: Tuple[str, str], router: LLMRouter,
                           seed_hash: int, variant_idx: int) -> str:
        f_name, f_desc = framing
        user = BACK_DERIVE_USER_TBT.format(
            func_name=iface["name"],
            arg_names=iface["arg_names"],
            test_func_name=test_func_name,
            ref_src=ref_src,
            summary=(
                f"This is a minor perturbation of the original PyTorch "
                f"function under operator `{op['name']}`. New property: "
                f"{op['new_property']}. The change affects only a small "
                f"fraction of output values."))
        system = BACK_DERIVE_SYSTEM_TBG.format(
            framing=f_name, framing_desc=f_desc,
            primary_wrapper=iface["name"])
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
        smoke_timeout=p.get("smoke_timeout", 90),
    )
