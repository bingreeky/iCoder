"""kb_perturb_be — KernelBench × BenchEvolver-style minor functional perturbation.

Solution-first: the LLM rewrites ``Model.forward`` under ONE mutation contract
(operator from :data:`expand.methods._perturb_common.EVOL_OPERATORS_KB`).
The variant ref must pass the 5-gate validation pipeline (format / forward-body
diff / smoke / Triton-feasibility lint / behavioural diff). Surviving variants
get a 4-framing NL back-derive (reused from InverseCoder's KB branch) and a
high-level plan; the output Expanded row has

    reference_solution   = variant PyTorch ref  (mutated)
    expanded_prompt      = NL request describing the variant behaviour
    metadata.assistant_spec = implementation plan (for the teacher)
    metadata.operator             = operator name
    metadata.behavioral_diff_ratio
    metadata.forward_body_diff_ratio
    metadata.original_reference   = original ref (audit)

Downstream is unchanged: ``scripts/teacher_triton_rollout.py`` reads
``reference_solution`` from JSONL and uses it as the correctness oracle —
feeding it the variant ref makes the teacher target the perturbed behaviour
without any code change.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
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
    GateConfig,
    extract_python,
    fmt_invariants,
    operator_for_seed,
    run_gates,
)
from .inversecoder import (
    INVERSE_FRAMINGS,
    BACK_DERIVE_SYSTEM_KB,
    BACK_DERIVE_USER_KB,
    PLAN_SYSTEM_KB,
    PLAN_USER_KB,
    _extract_pytorch_interface,
)


PERTURB_BE_SYSTEM = (
    "You are a careful PyTorch editor. You apply EXACTLY ONE bounded edit "
    "to a verified PyTorch reference implementation, governed by a mutation "
    "contract. You do NOT refactor, rename, reformat, or re-comment beyond "
    "what the contract requires. The edit must be a MINOR functional "
    "perturbation — most fuzz inputs should still produce the same output. "
    "Output ONLY the modified file inside a single ```python ... ``` fence; "
    "the file MUST keep its class name, its `forward` signature, and its "
    "`get_inputs` / `get_init_inputs` bodies BYTE-FOR-BYTE identical to the "
    "original. Insert your edit BEFORE the final `return` in `forward()` "
    "(or, for operators 6-7, swap exactly ONE existing literal). Do NOT "
    "use Python data-dependent control flow (no `if x.item()`, no "
    "data-dependent `for`/`while`, no `.tolist()`/.item()`/.cpu()`) — the "
    "variant must be expressible in Triton."
)

PERTURB_BE_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "Target             : {target_hint}\n"
    "What to do         : {definition}\n"
    "May touch          : {scope}\n"
    "Must NOT change    : {do_not_change}\n"
    "Old property (orig): {old_property}\n"
    "New property (you) : {new_property}\n"
    "=== End contract ===\n\n"
    "Original PyTorch reference:\n"
    "```python\n{ref_src}\n```\n\n"
    "Apply EXACTLY this contract. Output the full modified file in one "
    "```python ... ``` fence. No commentary, no testbench, no extra imports "
    "beyond what the original file already had. The file must be a valid "
    "Python module: `import` block + the `Model` class + `get_inputs()` + "
    "`get_init_inputs()`, all preserved verbatim except for the contract's "
    "edit inside `forward()`."
)


@register_method("kb_perturb_be")
class KBPerturbBEExpansion:
    """Solution-first minor-perturbation method for KernelBench seeds.

    For each (seed, variant), pick one operator from EVOL_OPERATORS_KB
    (hash-rotated by seed.id), have the traj LLM rewrite Model.forward
    under the contract, run the 5-gate validation pipeline, then NL
    back-derive on the variant ref.
    """

    name = "kb_perturb_be"

    MAX_PERTURB_ATTEMPTS = 2
    # Operator fallback DISABLED for the 3x8 8-GPU run: when ~90% of g4
    # attempts fail (Conv-heavy seeds + tight behavioural_lo/hi), the
    # fallback chain doubles per-variant work to 4 LLM rolls + 4 gates,
    # which 3-4x'd the BE wall-clock vs EI in the first 3x8 attempt
    # (dlc55h0juar5s7l3: 59 seeds in 10h vs EI's 169). With
    # num_variants=8 the seed budget is already deep on operator
    # diversity — operator fallback is redundant under that geometry.
    MAX_OPERATOR_FALLBACKS = 0

    async def expand(self, seed: Seed, llm, num_variants: int = 1,
                     **kw) -> List[Expanded]:
        if seed.source_dataset != "kernelbench":
            return []
        ref = seed.reference_solution or ""
        if not ref.strip():
            return []

        orig_interface = _extract_pytorch_interface(ref)
        if orig_interface is None:
            return []

        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))
        cfg = _gate_config_from_kw(kw)

        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)
        # Pick primary operator per variant; fallback chain is constructed
        # inside _perturb_one.
        ops = [operator_for_seed(seed.id, i) for i in range(num_variants)]

        # A-mode: ×m model dimension. Each operator slot is tried with
        # each prompt model. variant_idx = op_slot*m + model_idx makes
        # IDs unique AND chat_prompt rotation lands on model_idx (since
        # (seed_hash + op_slot*m + model_idx) % m = (seed_hash+model_idx) % m,
        # cycles through all m models per op_slot).
        m = router.num_prompt_models
        coros = [
            self._perturb_one(seed, op_primary, op_slot * m + model_idx,
                              ref, orig_interface,
                              router, seed_hash, cfg)
            for op_slot, op_primary in enumerate(ops)
            for model_idx in range(m)
        ]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _perturb_one(self, seed: Seed, op_primary: Dict[str, Any],
                           op_idx: int, ref: str, orig_iface,
                           router: LLMRouter, seed_hash: int,
                           cfg: GateConfig) -> List[Expanded]:
        # Operator fallback chain: primary, then next-in-rotation up to
        # MAX_OPERATOR_FALLBACKS extras. _evolve_forward retries WITHIN one
        # operator MAX_PERTURB_ATTEMPTS times; the chain rolls to the next
        # operator only when ALL attempts on the current one failed.
        from ._perturb_common import EVOL_OPERATORS_KB as _OPS
        primary_idx = _OPS.index(op_primary)
        chain = [
            _OPS[(primary_idx + k) % len(_OPS)]
            for k in range(1 + self.MAX_OPERATOR_FALLBACKS)
        ]

        err_hist: List[str] = []
        for chain_idx, op in enumerate(chain):
            variant_src, traj_wrap, gate, op_err_hist = (
                await self._evolve_forward(
                    seed, op, ref, orig_iface, router, cfg,
                    seed_hash=seed_hash, variant_idx=op_idx))
            err_hist.extend(
                [f"op={op['name']} attempt={i+1}: {e}"
                 for i, e in enumerate(op_err_hist)])
            if variant_src is not None:
                op_used = op
                break
        else:
            # All operators in chain exhausted.
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
        framing = INVERSE_FRAMINGS[op_idx % len(INVERSE_FRAMINGS)]
        variant_iface = _extract_pytorch_interface(variant_src)
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
                "framing_idx": op_idx % len(INVERSE_FRAMINGS),
                "variant_idx": op_idx,
                "operator": op_used["name"],
                "operator_primary": op_primary["name"],
                "operator_chain_index": chain.index(op_used),
                "behavioral_diff_ratio": gate.behavioral_diff,
                "forward_body_diff_ratio": gate.forward_body_diff,
                "original_reference": ref,
                "perturb_attempts": len(err_hist) + 1,
                "assistant_spec": plan,
                "pytorch_interface": asdict(variant_iface) if variant_iface else None,
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, op_idx),
                # Which model actually wrote the variant ref (now rotated
                # across EXPAND_PROMPT_MODELS, not fixed traj). Kept under a
                # distinct name so audits can tell the SFT-assistant traj
                # model apart from the problem-creator rewrite model.
                "rewrite_model": router.prompt_model_name(seed_hash, op_idx),
                "traj_content": traj_wrap,
            })
        # Override reference_solution with the VARIANT ref for downstream
        # teacher rollout to target.
        row.reference_solution = variant_src
        return [row]

    async def _evolve_forward(
        self, seed: Seed, op: Dict[str, Any], ref: str, orig_iface,
        router: LLMRouter, cfg: GateConfig,
        *, seed_hash: int = 0, variant_idx: int = 0,
    ) -> Tuple[Optional[str], str, Any, List[str]]:
        """Up to MAX_PERTURB_ATTEMPTS rolls of contract-bound forward rewrite.
        Returns (variant_src or None, traj_wrap, GateResult, err_history).

        The rewrite is routed through ``chat_prompt_full`` so that with
        ``EXPAND_PROMPT_MODELS="model_a,model_b,model_c"`` set, different
        (seed, variant_idx) tuples land on different problem-creator
        models — this is the 3-models × N-variants diversification used
        to multiply candidate count per seed. The SFT-assistant invariant
        is preserved: the variant ref produced here is NOT the SFT
        assistant text; the SFT assistant comes from teacher_triton_rollout
        downstream, which still uses the single fixed traj model.

        Early stop: if the LLM returns a variant byte-identical to the
        original (forward_body_diff=0), we bail out after the SECOND attempt
        rather than burning the full 4. This is the operator-mismatch
        signal — the LLM saw no way to apply the contract (e.g. eps_perturb
        on a matmul that has no eps literal). Falling back to a different
        operator is faster than re-rolling the same dead one."""
        base_user = PERTURB_BE_USER.format(
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
                    "Tighten the edit so the gate passes — DO NOT change "
                    "the operator's intent or touch anything outside "
                    "forward()'s body.")
            resp = await router.chat_prompt_full(
                PERTURB_BE_SYSTEM, user,
                seed_hash=seed_hash, variant_idx=variant_idx)
            raw = resp.content or ""
            wrap = synthesize_v4pro_wrap(
                raw, resp.reasoning_content, fallback_lang="python")
            last_wrap = wrap
            ok_fmt, fmt_reason = check_v4pro_format(raw)
            if not ok_fmt:
                err_history.append(f"format check: {fmt_reason}")
                continue
            answer = extract_v4pro_answer(strip_dryrun(raw))
            variant_src = extract_python(answer)
            if not variant_src.strip():
                err_history.append("no ```python``` fence in response")
                continue
            is_final = attempt == self.MAX_PERTURB_ATTEMPTS - 1
            gate = await run_gates(ref, variant_src, cfg,
                                   is_final_attempt=is_final,
                                   orig_interface=orig_iface)
            last_gate = gate
            if gate.ok:
                return variant_src, wrap, gate, err_history
            err_history.append(gate.reason)
            # Operator-mismatch short-circuit: two consecutive byte-identical
            # variants → operator doesn't fit this seed; bail to outer
            # operator-fallback loop instead of burning the remaining tries.
            if gate.forward_body_diff == 0.0:
                consecutive_zero_diff += 1
                if consecutive_zero_diff >= 2:
                    err_history.append(
                        "operator-mismatch short-circuit: 2 consecutive "
                        "zero-diff attempts; this operator does not apply "
                        "to this seed")
                    break
            else:
                consecutive_zero_diff = 0
        return None, last_wrap, last_gate, err_history

    async def _back_derive(self, interface, ref_src: str,
                           op: Dict[str, Any],
                           framing: Tuple[str, str], router: LLMRouter,
                           seed_hash: int, variant_idx: int) -> str:
        f_name, f_desc = framing
        user = BACK_DERIVE_USER_KB.format(
            class_name=interface.class_name,
            init_args=interface.init_args or "(none)",
            forward_args=interface.forward_args,
            input_shapes_dtypes_hint=interface.input_shapes_dtypes_hint,
            ref_src=ref_src,
            summary=(
                f"This is a minor perturbation variant of the original "
                f"problem under operator `{op['name']}`. The variant's "
                f"new property: {op['new_property']}. The behavioural "
                f"perturbation is small (5-30% of fuzz inputs affected)."))
        system = BACK_DERIVE_SYSTEM_KB.format(
            framing=f_name, framing_desc=f_desc)
        raw = await router.chat_prompt(
            system, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()

    async def _plan(self, nl: str, ref_src: str, router: LLMRouter,
                    seed_hash: int, variant_idx: int) -> str:
        user = PLAN_USER_KB.format(expanded_prompt=nl, ref_src=ref_src)
        raw = await router.chat_prompt(
            PLAN_SYSTEM_KB, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()


def _gate_config_from_kw(kw: Dict[str, Any]) -> GateConfig:
    """Allow the expand_data CLI / yaml `perturb:` block to tune gate
    thresholds without surfacing every knob through the registry signature.

    P1/P2 knobs (read from yaml `perturb:` block, env-var fallback for
    cache_dir so runner scripts can stamp it per-job):

      behavioral_n_trials_lite      g4-lite trial count (0 = disable)
      behavioral_lo_lite, _hi_lite  lite-stage loose band
      behavioral_timeout_lite       lite-stage timeout
      behavioral_cache_dir          orig-output cache dir
                                    (or KB_G4_CACHE_DIR env var)"""
    perturb = kw.get("perturb") or {}
    cache_dir = (perturb.get("behavioral_cache_dir")
                 or os.environ.get("KB_G4_CACHE_DIR") or None)
    return GateConfig(
        diff_lo=perturb.get("diff_lo", 0.02),
        diff_hi=perturb.get("diff_hi", 0.40),
        diff_lo_final=perturb.get("diff_lo_final", 0.001),
        diff_hi_final=perturb.get("diff_hi_final", 0.50),
        behavioral_lo=perturb.get("behavioral_lo", 0.05),
        behavioral_hi=perturb.get("behavioral_hi", 0.30),
        behavioral_n_trials=perturb.get("n_fuzz_trials", 16),
        behavioral_timeout=perturb.get("behavioral_timeout", 120),
        behavioral_n_trials_lite=perturb.get("n_fuzz_trials_lite", 4),
        behavioral_lo_lite=perturb.get("behavioral_lo_lite", 0.0),
        behavioral_hi_lite=perturb.get("behavioral_hi_lite", 0.60),
        behavioral_timeout_lite=perturb.get("behavioral_timeout_lite", 120),
        behavioral_cache_dir=cache_dir,
        smoke_timeout=perturb.get("smoke_timeout", 60),
    )
