"""kb_perturb_ei — KernelBench × Evol-Instruct-style minor functional perturbation.

Prompt-first: the LLM rewrites the NL prompt to *describe* a small additional
behaviour (under one operator from EVOL_OPERATORS_KB); then a teacher
(traj LLM) reads the evolved prompt + the original PyTorch reference + the
operator contract, and emits a *new* PyTorch ref that implements the
augmented behaviour. The new ref must pass the same 5-gate validation
(format / forward-body diff / smoke / Triton-feasibility lint / behavioural
diff) as BE-style — but the diff comes from a free LLM rewrite (less
contract-bound), so survival is expected to be lower.

Output schema and downstream contract are identical to kb_perturb_be:

    reference_solution   = LLM-generated variant PyTorch ref
    expanded_prompt      = evolved NL prompt (already user-facing)
    metadata.assistant_spec = implementation plan
    metadata.operator
    metadata.behavioral_diff_ratio
    metadata.forward_body_diff_ratio
    metadata.original_reference

so the same teacher Triton rollout + KB verify + SFT pair packer work without
modification.
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
    PLAN_SYSTEM_KB,
    PLAN_USER_KB,
    _extract_pytorch_interface,
)


PERTURB_EI_PROMPT_SYSTEM = (
    "You rewrite a KernelBench-style natural-language request to add ONE "
    "small additional behaviour, governed by a mutation contract. The "
    "rewritten prompt MUST:\n"
    "  - Read like a standalone NL task description that a Triton engineer "
    "    would receive (no operator jargon, no \"contract\" reference).\n"
    "  - Preserve every functional requirement of the original task.\n"
    "  - Add the operator's new property as ONE extra sentence or bullet, "
    "    described in plain math/behavioural terms.\n"
    "  - Still ask explicitly for Triton and the class name `ModelNew`.\n"
    "  - Embed the original PyTorch reference verbatim inside one "
    "    ```python ... ``` block.\n"
    "  - Specify `torch.allclose(atol=1e-2, rtol=1e-2)` as the tolerance.\n\n"
    "Output ONLY the evolved NL prompt — plain text, no commentary, no "
    "wrapping tags."
)

PERTURB_EI_PROMPT_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "What to do         : {definition}\n"
    "Old property (orig): {old_property}\n"
    "New property       : {new_property}\n"
    "=== End contract ===\n\n"
    "Original PyTorch reference (embed verbatim in the evolved prompt):\n"
    "```python\n{ref_src}\n```\n\n"
    "Original NL prompt (for reference; you are REWRITING it, not "
    "appending):\n---\n{original_prompt}\n---\n\n"
    "Write the evolved NL request now."
)


PERTURB_EI_REF_SYSTEM = (
    "You implement a PyTorch reference to spec. You receive the evolved NL "
    "request and the ORIGINAL PyTorch reference (a recently-perturbed "
    "version's starting point). Your job is to produce a new PyTorch "
    "reference that satisfies the evolved request — keeping the original "
    "math, but adding the operator's new property as ONE additional "
    "statement near the final `return` of `forward()`.\n\n"
    "Hard requirements:\n"
    "  - The class name must remain EXACTLY `Model` (NOT `ModelNew` and "
    "    not any other name — the verify harness imports `Model` by "
    "    that exact name).\n"
    "  - `__init__` signature must match the original (same positional "
    "    arg names + defaults).\n"
    "  - `forward` signature must match the original exactly.\n"
    "  - `get_inputs()` and `get_init_inputs()` bodies must be BYTE-FOR-BYTE "
    "    identical to the original — copy them verbatim.\n"
    "  - The new functional change must be expressible in Triton: no "
    "    data-dependent Python control flow, no `.tolist()`/`.item()`/"
    "    `.cpu()`, no list/dict comprehensions over tensor values.\n\n"
    "Output ONLY the full new file inside a single ```python ... ``` fence."
)

PERTURB_EI_REF_USER = (
    "=== Mutation contract (already encoded in the prompt; here for "
    "transparency) ===\n"
    "Operator           : {name}\n"
    "Old property       : {old_property}\n"
    "New property       : {new_property}\n"
    "Must NOT change    : {do_not_change}\n"
    "=== End contract ===\n\n"
    "Evolved NL request:\n---\n{evolved_prompt}\n---\n\n"
    "Original PyTorch reference (use as starting point; copy `get_inputs` "
    "and `get_init_inputs` verbatim):\n```python\n{ref_src}\n```\n\n"
    "Emit the new PyTorch reference file now, in one ```python ... ``` "
    "fence."
)


@register_method("kb_perturb_ei")
class KBPerturbEIExpansion:
    """Prompt-first minor-perturbation method for KernelBench seeds.

    Per (seed, variant): pick an operator (hash-rotated), rewrite the NL
    prompt to describe the additional behaviour, then have the traj LLM
    generate a new PyTorch ref. Same 5-gate validation, same downstream.
    """

    name = "kb_perturb_ei"

    MAX_PERTURB_ATTEMPTS = 2

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
        ops = [operator_for_seed(seed.id, i) for i in range(num_variants)]

        # A-mode: ×m model dimension. Each op slot × each prompt model.
        m = router.num_prompt_models
        coros = [
            self._perturb_one(seed, op, op_slot * m + model_idx,
                              ref, orig_interface,
                              router, seed_hash, cfg)
            for op_slot, op in enumerate(ops)
            for model_idx in range(m)
        ]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _perturb_one(self, seed: Seed, op: Dict[str, Any],
                           op_idx: int, ref: str, orig_iface,
                           router: LLMRouter, seed_hash: int,
                           cfg: GateConfig) -> List[Expanded]:
        # Step 1: prompt-side rewrite (prompt model; PROMPT-side is allowed
        # to vary). Up to 2 prompt rewrites attempted; if the downstream ref
        # rewrite fails the gates we re-roll the REF step, keeping the
        # evolved prompt fixed.
        evolved_prompt = await self._evolve_prompt(seed, op, ref, router,
                                                   seed_hash, op_idx)
        if not evolved_prompt:
            return [base_expanded(
                seed, self.name, op_idx, seed.original_prompt,
                extra_meta={"warn": "perturb_prompt_failed",
                            "operator": op["name"], "framing": "passthrough"})]

        # Step 2: teacher-side ref generation under gates.
        variant_src, traj_wrap, gate, err_hist = await self._evolve_ref(
            seed, op, evolved_prompt, ref, orig_iface, router, cfg,
            seed_hash=seed_hash, variant_idx=op_idx)
        if variant_src is None:
            return [base_expanded(
                seed, self.name, op_idx, evolved_prompt,
                extra_meta={
                    "warn": "perturb_gate_failed",
                    "operator": op["name"],
                    "perturb_attempts": len(err_hist),
                    "perturb_last_err": err_hist[-1] if err_hist else "",
                    "framing": "passthrough",
                })]

        # Step 3: plan-side scaffold for the teacher Triton rollout.
        variant_iface = _extract_pytorch_interface(variant_src)
        plan = await self._plan(evolved_prompt, variant_src, router,
                                seed_hash, op_idx)
        if not plan:
            return []

        row = base_expanded(
            seed, self.name, op_idx, evolved_prompt,
            extra_meta={
                "operator": op["name"],
                "behavioral_diff_ratio": gate.behavioral_diff,
                "forward_body_diff_ratio": gate.forward_body_diff,
                "original_reference": ref,
                "perturb_attempts": len(err_hist) + 1,
                "assistant_spec": plan,
                "pytorch_interface": asdict(variant_iface) if variant_iface else None,
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, op_idx),
                "rewrite_model": router.prompt_model_name(seed_hash, op_idx),
                "traj_content": traj_wrap,
                "framing": "evol_instruct",
                "framing_idx": op_idx,
                "variant_idx": op_idx,
            })
        row.reference_solution = variant_src
        return [row]

    async def _evolve_prompt(self, seed: Seed, op: Dict[str, Any], ref: str,
                             router: LLMRouter, seed_hash: int,
                             variant_idx: int) -> str:
        user = PERTURB_EI_PROMPT_USER.format(
            name=op["name"],
            definition=op["definition"],
            old_property=op["old_property"],
            new_property=op["new_property"],
            ref_src=ref,
            original_prompt=seed.original_prompt)
        raw = await router.chat_prompt(
            PERTURB_EI_PROMPT_SYSTEM, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()

    async def _evolve_ref(
        self, seed: Seed, op: Dict[str, Any], evolved_prompt: str, ref: str,
        orig_iface, router: LLMRouter, cfg: GateConfig,
        *, seed_hash: int = 0, variant_idx: int = 0,
    ) -> Tuple[Optional[str], str, Any, List[str]]:
        base_user = PERTURB_EI_REF_USER.format(
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
                    "Re-emit the FULL new file in one ```python ... ``` "
                    "fence. The evolved request above is unchanged. Tighten "
                    "the edit so the gate passes — keep `get_inputs` / "
                    "`get_init_inputs` byte-for-byte identical to the "
                    "original.")
            resp = await router.chat_prompt_full(
                PERTURB_EI_REF_SYSTEM, user,
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
        return None, last_wrap, last_gate, err_history

    async def _plan(self, evolved_prompt: str, variant_src: str,
                    router: LLMRouter, seed_hash: int,
                    variant_idx: int) -> str:
        user = PLAN_USER_KB.format(
            expanded_prompt=evolved_prompt, ref_src=variant_src)
        raw = await router.chat_prompt(
            PLAN_SYSTEM_KB, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()


def _gate_config_from_kw(kw: Dict[str, Any]) -> GateConfig:
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
