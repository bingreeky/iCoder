"""Evol-Instruct (RTL-domainized) — instruction-first expansion.

Per docs/pipelines/evol-instruct-rtllm.md. Original WizardLM Evol-Instruct
treats prompt evolution as the only step (re-using the seed's response is
fine for NL instruction tuning). On RTLLM the prompt is half-structured
(module name + port table + behaviour) and the response must be a
compilable Verilog module exercised by a self-checking testbench, so once
the evolved prompt diverges from the seed the original (ref, tb) can no
longer be reused. We therefore replace step 4 of the original recipe with
a teacher rollout for the evolved ref AND a teacher rollout for the
matching evolved tb — the same machinery BenchEvolver uses on the
solution-first side, applied here in the prompt-first direction.

Pipeline:

  Step 1   Seed instruction       seed.original_prompt verbatim
  Step 2   Operator selection     pick one of 6 operators (5 in-depth +
                                  1 in-breadth) deterministically by
                                  seed-hash so num_variants=1 spreads
                                  the catalogue uniformly across seeds
  Step 3   Instruction evolution  LLM rewrites the seed prompt under the
                                  operator's directive; up to 3 attempts
                                  with elimination-filter feedback
                                  (template-leak ban + Module-name pin +
                                  port-table preservation)
  Step 4   Reference rollout      teacher writes Verilog from the evolved
                                  prompt alone; iverilog -g2012 standalone
                                  compile gate, up to 3 attempts with
                                  stderr feedback
  Step 5   Testbench rollout      teacher writes a self-checking testbench
                                  for (evolved_prompt, evolved_ref);
                                  iverilog compile-pair gate against the
                                  evolved_ref, up to 3 attempts with
                                  stderr feedback
  Step 6   Emit triplet           (expanded_prompt, evolved_ref,
                                  evolved_tb), evaluator_compatible=True;
                                  end-to-end validation is out-of-band via
                                  scripts/validate_evolved_triplet.py

xcoder.py is style-rephrasing that keeps ref/tb unchanged; this module is
not that, do not conflate.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from ..base import Expanded, Seed
from ..llm import LLMResponse, LLMRouter
from ..registry import register_method
from ._common import (
    check_v4pro_format,
    expand_id,
    extract_v4pro_answer,
    strip_dryrun,
    synthesize_v4pro_wrap,
)
from .benchevolver import (
    extract_verilog,
    force_module_name,
    iverilog_compile_pair,
    iverilog_syntax_check,
)


# ---------- operator catalogue -----------------------------------------
#
# RTL-domainized analogues of the 5 in-depth + 1 in-breadth operators
# from third_party/WizardLM/Evol_Instruct/depth.py and breadth.py. Unlike
# BenchEvolver's mutation contracts (which steer an RTL editor + a
# testbench writer in lockstep), these directives steer a PROMPT
# rewriter; the downstream ref/tb are derived from the rewritten prompt
# alone. The directives are therefore looser by design — they describe a
# *kind* of evolution, not a bit-level invariant.
#
# Fields:
#   name                : short identifier
#   kind                : "in_depth" | "in_breadth"
#   evolution_directive : NL instruction handed to the prompt rewriter
#   rtl_focus           : free-form note on the aspect being evolved

EVOL_OPERATORS: List[Dict[str, Any]] = [
    {
        "name": "add_constraint",
        "kind": "in_depth",
        "evolution_directive": (
            "Add ONE additional behavioural constraint to the module. "
            "Pick exactly one of: (a) require a synchronous active-high "
            "reset signal `rst` that clears all sequential state; "
            "(b) add a 1-bit input `en` that, when low, holds all "
            "sequential state; (c) add a deterministic behaviour clause "
            "for an existing edge case (overflow, all-zero input, "
            "simultaneous control assertion). The new constraint must "
            "either reference an EXISTING port name from the port table, "
            "OR add exactly one new 1-bit input port and extend the "
            "port table to declare it."),
        "rtl_focus": "behaviour rules / control signals",
    },
    {
        "name": "deepen",
        "kind": "in_depth",
        "evolution_directive": (
            "Deepen the behavioural specification: state explicit "
            "behaviour for at least one previously-unspecified edge "
            "case (overflow, all-zero input, simultaneous assertion of "
            "conflicting controls, reset-during-operation, etc.). Do "
            "NOT add new ports. Do NOT change existing port widths or "
            "directions. The added clauses must reference port names "
            "that already appear in the port table."),
        "rtl_focus": "edge-case behaviour",
    },
    {
        "name": "concretize",
        "kind": "in_depth",
        "evolution_directive": (
            "Replace at least one abstract or vague phrase with a "
            "concrete one. Examples: 'process the input' → 'shift left "
            "by 2 then mask the low 4 bits'; 'a counter' → 'a 16-bit "
            "Gray-code counter that wraps at 2**16-1'; 'a reasonable "
            "depth' → 'depth 16'. You may freely tune literal values "
            "as long as the result is internally consistent. Keep the "
            "existing port table; do not add or remove ports."),
        "rtl_focus": "numeric / behavioural specificity",
    },
    {
        "name": "increase_reasoning",
        "kind": "in_depth",
        "evolution_directive": (
            "Re-state the behaviour as an EXPLICIT multi-step "
            "computation: 'step 1 ... step 2 ... step 3 ...'. Each "
            "step must reference a concrete port or named internal "
            "value. Do NOT add new ports. Do NOT change semantics — "
            "only make the algorithmic decomposition explicit so a "
            "reader follows the data flow stage by stage."),
        "rtl_focus": "algorithmic decomposition",
    },
    {
        "name": "complicate_input",
        "kind": "in_depth",
        "evolution_directive": (
            "Increase the cardinality of the primary data input. Pick "
            "one of: (a) widen the primary input port by EXACTLY +N "
            "bits (N in 2..8) and extend the behaviour to cover the new "
            "bits; (b) split a multi-purpose input into two narrower "
            "input ports with distinct, named meanings. Update the port "
            "table to match. The module name is unchanged."),
        "rtl_focus": "primary input port",
    },
    {
        "name": "in_breadth",
        "kind": "in_breadth",
        "evolution_directive": (
            "Replace ONLY the behaviour section with a different but "
            "related behavioural spec in the same domain (e.g. counter "
            "→ finite-state machine producing the same output, FIFO → "
            "stack with the same handshake, adder → multiplier of the "
            "same operand width). Keep the module name and the port "
            "table EXACTLY as in the original prompt — only the "
            "behaviour rules change."),
        "rtl_focus": "behaviour section (port table is preserved)",
    },
]

OP_NAMES = [op["name"] for op in EVOL_OPERATORS]
OP_BY_NAME = {op["name"]: op for op in EVOL_OPERATORS}


# ---------- elimination filter -----------------------------------------

# WizardLM hard-coded keyword ban: any of these in the rewritten prompt
# means the model leaked the rewriting template. We add `Rewriter` and
# the directive's leading verb-form as further heuristics in case the
# model paraphrases.
_BANNED_PHRASES = [
    "#The Given Prompt#",
    "#Rewritten Prompt#",
    "given prompt",
    "rewritten prompt",
    "#Original Prompt#",
    "#Evolved Prompt#",
]


_PORT_KEYWORDS_RE = re.compile(
    r"\b(input|output)\s*(?:\[|reg|wire|\w)", re.IGNORECASE)


# VerilogEval v2 prompts use a different surface form than RTLLM. Module
# name is introduced as `module named TopModule` (no `Module name:` tag),
# port table is a bullet list `- input <name> (N bits)` / `- output ...`.
_VEVAL_MODULE_NAMED_RE = re.compile(
    r"module\s+named\s+(\w+)", re.IGNORECASE)
_VEVAL_PORT_BULLET_RE = re.compile(
    r"^\s*-\s*(input|output)\s+\w+", re.IGNORECASE | re.MULTILINE)


def _filter_legacy(prompt: str, design_name: str, kind: str) -> str:
    """RTLLM-style filter: `Module name:` tag + `Input ports:` label or
    `input/output` declarations. Returns "" on pass, reason on fail."""
    if not prompt or not prompt.strip():
        return "empty rewritten prompt"
    low = prompt.lower()
    for ban in _BANNED_PHRASES:
        if ban.lower() in low:
            return (f"rewritten prompt must not contain the phrase "
                    f"{ban!r}; rephrase without referencing the "
                    f"rewriting process")
    # Module-name tag must be present and (for in-depth) match the seed.
    # RTLLM v2 prompts use "Module name:\n    <name>" as the canonical
    # tag, but some prompts inline as "Module name: <name>" — accept both.
    m = re.search(r"Module\s+name\s*:\s*\n?\s*(\w+)",
                  prompt, re.IGNORECASE)
    if not m:
        return ("rewritten prompt must contain a `Module name:` tag "
                "followed by the module identifier")
    if kind == "in_depth" and m.group(1) != design_name:
        return (f"rewritten prompt must keep `Module name: {design_name}` "
                f"(got {m.group(1)!r}); operator forbids changing the "
                f"module identifier")
    # Port table must be present — heuristic: at least one `Input ports:`
    # or `Output ports:` label, OR at least one `input`/`output` decl.
    has_label = re.search(r"(Input|Output)\s+ports?\s*:",
                          prompt, re.IGNORECASE) is not None
    has_decl = _PORT_KEYWORDS_RE.search(prompt) is not None
    if not (has_label or has_decl):
        return ("rewritten prompt must keep the port table — list "
                "`Input ports:` and `Output ports:` with names, "
                "widths, and directions")
    return ""


def _filter_veval(prompt: str, design_name: str, kind: str) -> str:
    """VerilogEval v2-style filter: `module named <X>` phrasing + bullet
    port list (`- input ...` / `- output ...`). Returns "" on pass."""
    if not prompt or not prompt.strip():
        return "empty rewritten prompt"
    low = prompt.lower()
    for ban in _BANNED_PHRASES:
        if ban.lower() in low:
            return (f"rewritten prompt must not contain the phrase "
                    f"{ban!r}; rephrase without referencing the "
                    f"rewriting process")
    m = _VEVAL_MODULE_NAMED_RE.search(prompt)
    if not m:
        return ("rewritten prompt must keep the original `module named "
                "TopModule` phrasing — do not rename the module or drop "
                "the introductory clause")
    if m.group(1) != design_name:
        return (f"rewritten prompt must keep `module named {design_name}` "
                f"(got {m.group(1)!r}); the harness compiles against "
                f"`{design_name}` and renaming will break it")
    bullets = _VEVAL_PORT_BULLET_RE.findall(prompt)
    if not bullets:
        return ("rewritten prompt must keep the bullet port list (`- "
                "input <name> (<n> bits)` / `- output <name> (<n> "
                "bits)`) — the testbench compares port-by-port against "
                "RefModule and an unlisted port silently breaks the "
                "comparison")
    return ""


# ---------- per-dataset profile ----------------------------------------
#
# Two prompt-style adaptations live alongside each other. RTLLM uses the
# `LEGACY_PROFILE` (canonical "Module name:" tag + `Input ports:` table);
# VerilogEval v2 uses `VEVAL_PROFILE` ("module named TopModule" + bullet
# port list, fixed dual-instantiation testbench). Add new profiles by
# following the same shape — don't fork the operator catalogue. Routing
# happens in `_profile_for(seed)` and is the only switch a new dataset
# needs to wire up. See [docs/pipelines/evol-instruct-verilog-eval.md].

_VEVAL_PORT_PRESERVE = (
    " VEval-specific hard constraint: the testbench compares your "
    "implementation port-by-port against a fixed RefModule that uses "
    "the original port list. You MUST keep every original port "
    "(name AND width) byte-for-byte in the bullet list. You may ADD new "
    "ports if the operator above explicitly authorises it, but you "
    "cannot rename, reorder, or remove any existing port.")


@dataclass
class _Profile:
    """Per-dataset adaptation knobs. Keep this dataclass tight — every
    field is something at least two profiles disagree on."""
    name: str
    # extra clause appended to every operator's evolution_directive when
    # this profile is active. Empty string = no extension.
    extra_operator_constraint: str
    # filter callable: (prompt, design_name, kind) -> "" or reason str.
    filter_fn: Callable[[str, str, str], str]
    # if True, skip the teacher tb rollout and use seed.tests verbatim.
    # VEval has a fixed dual-instantiation tb that compares RefModule
    # against TopModule via "Mismatches: 0" — we don't regenerate it.
    fixed_tb: bool
    # if True, also produce a `module RefModule(...)` text-renamed copy
    # of the evolved ref alongside the canonical TopModule version,
    # stashed in metadata. Downstream verification splices this into
    # the original tb in place of the original RefModule.
    emit_ref_module_copy: bool
    # operator -> weight. None = equal weight round-robin (legacy).
    # When set, operators with higher weight are picked more often via
    # deterministic seed-hash sampling. Pilot data drives this:
    # VEval prefers structural ops (in_breadth / complicate_input /
    # add_constraint) 80% over prose ops (concretize / deepen /
    # increase_reasoning) 20%, because VEval's short structured prompts
    # absorb prose evolution but expose structural changes.
    operator_weights: Dict[str, float] = None


LEGACY_PROFILE = _Profile(
    name="legacy",
    extra_operator_constraint="",
    filter_fn=_filter_legacy,
    fixed_tb=False,
    emit_ref_module_copy=False,
    operator_weights=None,
)


# Pilot data ([eval_results/evol_instruct_veval_pilot.cot_v1/summary.md]):
# in_breadth 3/3, complicate_input 1/1, add_constraint 1/1 (3 structural ops
# all at 100% kept) vs concretize 0/6, deepen 0/5, increase_reasoning 0/4
# (all prose ops at 0% kept). VEval prompts are short + already structured,
# prose-level evolution silently round-trips through the teacher. Bias the
# operator selection 80/20 toward the structural ops to lift kept rate.
_VEVAL_OPERATOR_WEIGHTS = {
    "in_breadth":         0.267,   # ┐ structural ops, 0.80 total
    "complicate_input":   0.267,   # │  (each gets equal share within
    "add_constraint":     0.266,   # ┘   the 0.80 budget)
    "concretize":         0.067,   # ┐ prose ops, 0.20 total
    "deepen":             0.067,   # │
    "increase_reasoning": 0.066,   # ┘
}


VEVAL_PROFILE = _Profile(
    name="verilog_eval_v2",
    extra_operator_constraint=_VEVAL_PORT_PRESERVE,
    filter_fn=_filter_veval,
    fixed_tb=True,
    emit_ref_module_copy=True,
    operator_weights=_VEVAL_OPERATOR_WEIGHTS,
)


def _filter_archx(prompt: str, design_name: str, kind: str) -> str:
    """ArchXBench-style filter. ArchX prompts pin the module name + full
    port signature in a `design-specs` block (surface form differs from
    RTLLM's `Module name:` tag — often `Module Name:\\n- <name>` bullets and
    a literal `module <name> (...)` signature). We only require: non-empty,
    no template leak, the design_name token present somewhere, and at least
    one port declaration. tb is teacher-regenerated (fixed_tb=False), so we
    don't demand a specific tag layout. Returns "" on pass."""
    if not prompt or not prompt.strip():
        return "empty rewritten prompt"
    low = prompt.lower()
    for ban in _BANNED_PHRASES:
        if ban.lower() in low:
            return (f"rewritten prompt must not contain the phrase {ban!r}; "
                    "rephrase without referencing the rewriting process")
    # in_depth must keep the module identity; other ops may extend behaviour
    # but the name should still be discoverable so the teacher pins it.
    if design_name and design_name.lower() not in low:
        return (f"rewritten prompt must name the module `{design_name}` "
                "(the self-check testbench instantiates it by that exact name)")
    if _PORT_KEYWORDS_RE.search(prompt) is None and \
            re.search(r"(input|output)\b", prompt, re.IGNORECASE) is None:
        return ("rewritten prompt must describe the ports (names, widths, "
                "directions) so the implementation matches the tb interface")
    return ""


_ARCHX_PORT_PRESERVE = (
    " ArchXBench hard constraint: keep the EXACT module name and the full "
    "port list (names, directions, widths) from the design specification. "
    "The self-checking testbench instantiates the module by that exact name "
    "and drives those exact ports — renaming or dropping a port breaks it. "
    "You MAY add behaviour the operator authorises, but state the complete "
    "resulting interface explicitly.")


ARCHX_PROFILE = _Profile(
    name="archxbench",
    extra_operator_constraint=_ARCHX_PORT_PRESERVE,
    filter_fn=_filter_archx,
    fixed_tb=False,  # teacher writes a fresh self-check tb for the evolved spec
    emit_ref_module_copy=False,
    operator_weights=None,
)


def _profile_for(seed: "Seed") -> _Profile:
    if seed.source_dataset == "verilog_eval_v2":
        return VEVAL_PROFILE
    if seed.source_dataset in ("archxbench", "realbench"):
        # RealBench change-behaviour evol also uses the lenient ARCHX filter
        # (its evolved prompts pin module+ports but not the RTLLM "Module name:"
        # tag that _filter_legacy demands) + fixed_tb=False so the teacher
        # writes a fresh self-check tb verified by iverilog.
        return ARCHX_PROFILE
    return LEGACY_PROFILE


def _pick_operators(profile: _Profile, seed_hash: int,
                    num_variants: int) -> List[Dict[str, Any]]:
    """Pick operator(s) for a seed run. Deterministic in seed_hash so
    re-runs are reproducible. With ``profile.operator_weights=None``
    falls back to the legacy round-robin starting at ``seed_hash %
    len(EVOL_OPERATORS)``. With weights set, uses a deterministic
    weighted draw per variant index."""
    if not profile.operator_weights:
        start = seed_hash % len(EVOL_OPERATORS)
        return [EVOL_OPERATORS[(start + i) % len(EVOL_OPERATORS)]
                for i in range(num_variants)]
    # Deterministic weighted draw. Build CDF over the operator list,
    # then pick a position in [0, 1) per variant via a stable PRNG seeded
    # by (seed_hash, variant_idx).
    weights = [profile.operator_weights.get(op["name"], 0.0)
               for op in EVOL_OPERATORS]
    total = sum(weights)
    if total <= 0:
        # Pathological — fall back to round-robin to avoid infinite loop
        start = seed_hash % len(EVOL_OPERATORS)
        return [EVOL_OPERATORS[(start + i) % len(EVOL_OPERATORS)]
                for i in range(num_variants)]
    cdf = []
    cum = 0.0
    for w in weights:
        cum += w
        cdf.append(cum / total)
    picks = []
    for i in range(num_variants):
        # LCG-style deterministic mix; we don't need cryptographic
        # quality here, just stable per (seed_hash, i).
        pos = ((seed_hash * 1103515245 + 12345 + i * 524287)
               % 1_000_003) / 1_000_003.0
        idx = next(j for j, c in enumerate(cdf) if pos < c)
        picks.append(EVOL_OPERATORS[idx])
    return picks


# ---------- prompts ----------------------------------------------------

EVOLVE_SYSTEM = (
    "You are a Prompt Rewriter for hardware-design specifications. You "
    "receive an RTL design prompt — module name + port table + "
    "behavioural description — and produce a more complex variant under "
    "a precise rewriting directive.\n\n"
    "HARD CONSTRAINTS:\n"
    "  - Preserve `Module name: <X>` exactly as in the original prompt "
    "    UNLESS the directive explicitly authorises a new module name.\n"
    "  - Preserve every port (name, width, direction) UNLESS the "
    "    directive explicitly authorises adding/widening/splitting one. "
    "    Never silently drop a port.\n"
    "  - Never reference the rewriting process: do not include 'given "
    "    prompt', 'rewritten prompt', '#The Given Prompt#', '#Rewritten "
    "    Prompt#', and do not quote the directive itself.\n"
    "  - Add roughly 10-40 words of new substance, not preambles.\n"
    "  - The result must read as a clean RTLLM-style design spec a fresh "
    "    engineer could implement from scratch.\n\n"
    "Output ONLY the rewritten prompt as plain text — no code fences, no "
    "headings beyond the original prompt's labels, no commentary."
)

EVOLVE_USER = (
    "=== Evolution directive ===\n"
    "Operator   : {op_name}\n"
    "Kind       : {kind}\n"
    "What to do : {evolution_directive}\n"
    "RTL focus  : {rtl_focus}\n"
    "=== End directive ===\n\n"
    "Source specification to evolve (do NOT echo these labels):\n"
    "---BEGIN---\n{original_prompt}\n---END---\n\n"
    "Now emit the new specification."
)

REF_SYSTEM = (
    "You are a Verilog engineer. Read an RTLLM-style design prompt and "
    "implement the module it specifies. Output ONLY the implementation "
    "inside a single ```verilog ... ``` fence — no testbench, no "
    "commentary. The module must compile under `iverilog -g2012` "
    "(Verilog 2005 with -g2012 extensions). Keep the module name "
    "exactly as the prompt says."
)

REF_USER = (
    "Implement the following RTLLM-style design prompt:\n\n"
    "{evolved_prompt}\n\n"
    "Emit the complete implementation as `module {design_name} ... "
    "endmodule` in one ```verilog ... ``` fence."
)

TB_SYSTEM = (
    "You are writing a self-checking Verilog testbench compiled by "
    "`iverilog -g2012`. The testbench's job is to exercise the module "
    "specified by the design prompt and report pass/fail.\n\n"
    "RULES:\n"
    "  - Instantiate the design-under-test using EXACTLY the ports "
    "    declared by the supplied module under test (DUT). Do not "
    "    invent extra signals.\n"
    "  - Drive a deterministic stimulus — fix `integer seed = 32'h1234;` "
    "    if you use `$urandom`.\n"
    "  - Use `$display`. On any mismatch, print 'MISMATCH' and call "
    "    `$finish`. On full success, print 'Your Design Passed' and "
    "    call `$finish`.\n"
    "  - Verilog 2005 / iverilog -g2012 only. No SystemVerilog "
    "    assertions, no UVM, no class.\n"
    "  - Output ONLY the testbench inside a single ```verilog ... ``` "
    "    fence."
)

TB_USER = (
    "Design prompt (the spec the DUT was implemented against; use it to "
    "decide what to test):\n---BEGIN PROMPT---\n{evolved_prompt}\n"
    "---END PROMPT---\n\n"
    "DUT module name (use this EXACT spelling in your instantiation, "
    "even if you think it is a typo): {design_name}\n\n"
    "Module under test (your tb instantiates this; do not modify it):\n"
    "```verilog\n{evolved_ref}\n```\n\n"
    "Write the self-checking testbench."
)


# ---------- method -----------------------------------------------------

@register_method("evol_instruct")
class EvolInstructExpansion:
    """Instruction-first Evol-Instruct adaptation for RTLLM v2.

    ``num_variants`` cycles through the operator catalogue (one operator
    per variant); a seed run with ``num_variants=1`` will use the
    operator deterministically picked from ``hash(seed.id)`` so the
    catalogue is exercised uniformly across seeds.
    """

    name = "evol_instruct"

    async def expand(self, seed: Seed, llm, num_variants: int = 1,
                     **_kw) -> List[Expanded]:
        if not seed.original_prompt.strip():
            return []
        # Wrap a bare LLM in a single-prompt router for backward compat.
        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))

        profile = _profile_for(seed)
        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)
        ops = _pick_operators(profile, seed_hash, num_variants)
        coros = [self._evolve_one(seed, op, router, op_idx, seed_hash)
                 for op_idx, op in enumerate(ops)]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _evolve_one(self, seed: Seed, op: Dict[str, Any],
                          router: LLMRouter, op_idx: int,
                          seed_hash: int) -> List[Expanded]:
        profile = _profile_for(seed)
        # VEval seeds carry `top_module` (canonically "TopModule") rather
        # than `design_name`. RTLLM seeds carry `design_name`. Honour both.
        design_name = (seed.evaluator_info.get("top_module")
                       or seed.evaluator_info.get("design_name")
                       or seed.metadata.get("design_name")
                       or "TopModule")

        # M variants per (seed, op): each variant uses a different
        # prompt-side model for evolve_prompt, then the SAME traj model
        # writes the evolved ref+tb. The prompt model rotates by
        # variant_idx through router.prompt_llms.
        m = router.num_prompt_models
        rows: List[Expanded] = []
        for variant_idx in range(m):
            evolved_prompt = await self._evolve_prompt(
                seed, op, router, design_name, profile,
                seed_hash, variant_idx)
            if not evolved_prompt:
                continue

            if profile.fixed_tb:
                # VEval: tb is the dual-instantiation `<prob>_test.sv` —
                # we don't regenerate. Just rollout the evolved ref via
                # the traj model (assistant must be single-model).
                evolved_ref, traj_content = await self._rollout_ref(
                    evolved_prompt, design_name, router)
                if not evolved_ref:
                    if not traj_content:
                        continue
                    # Soft-fail: keep the traj content (model emitted
                    # reasoning but no parseable ref) by anchoring on the
                    # seed's own reference. Tagged so verify can audit.
                    evolved_ref = seed.reference_solution
                    compile_gate_pass = False
                else:
                    compile_ok, _ = await iverilog_syntax_check(
                        evolved_ref, design_name)
                    compile_gate_pass = bool(compile_ok)
                evolved_tb = seed.tests
                tb_gate_pass = True
            else:
                # RTLLM-style: traj writes ref; prompt model writes tb
                # (tb is verify-side, not part of the SFT assistant).
                evolved_ref, traj_content = await self._rollout_ref(
                    evolved_prompt, design_name, router)
                if not evolved_ref:
                    if not traj_content:
                        continue
                    evolved_ref = seed.reference_solution
                    compile_gate_pass = False
                else:
                    compile_ok, _ = await iverilog_syntax_check(
                        evolved_ref, design_name)
                    compile_gate_pass = bool(compile_ok)
                evolved_tb = await self._rollout_tb(
                    evolved_prompt, evolved_ref, design_name,
                    router, seed_hash, variant_idx)
                if not evolved_tb:
                    evolved_tb = seed.tests
                    tb_gate_pass = False
                else:
                    tb_gate_pass = True

            ev = dict(seed.evaluator_info)
            ev["evaluator_compatible"] = True
            ev["note"] = ("Evolved triplet: prompt evolved under one "
                          "Evol-Instruct operator; ref + tb teacher-rolled "
                          "from the evolved prompt with iverilog gates.")

            md = dict(seed.metadata)
            md.update({
                "operator": op["name"],
                "operator_kind": op["kind"],
                "operator_idx": op_idx,
                "variant_idx": variant_idx,
                "evolution_directive": op["evolution_directive"],
                "original_reference": seed.reference_solution,
                "original_tests": seed.tests,
                "original_tb_path": seed.evaluator_info.get("tb_path"),
                "compile_gate_pass": compile_gate_pass,
                "tb_gate_pass": tb_gate_pass,
                "profile": profile.name,
                "traj_model": router.traj_model_name,
                "traj_content": traj_content,
                "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
            })
            if profile.emit_ref_module_copy:
                md["evolved_ref_module"] = force_module_name(
                    evolved_ref, "RefModule")

            row_idx = op_idx * m + variant_idx
            rows.append(Expanded(
                id=expand_id(seed, "evol_instruct", row_idx),
                source_dataset=seed.source_dataset,
                expansion_method="evol_instruct",
                original_prompt=seed.original_prompt,
                expanded_prompt=evolved_prompt,
                reference_solution=evolved_ref,
                expected_output=seed.expected_output,
                tests=evolved_tb,
                evaluator_info=ev,
                metadata=md,
            ))
        return rows

    async def _evolve_prompt(self, seed: Seed, op: Dict[str, Any],
                             router: LLMRouter, design_name: str,
                             profile: _Profile, seed_hash: int,
                             variant_idx: int) -> str:
        """Step 3: rewrite the seed prompt under the operator. Uses the
        prompt-side model selected by ``variant_idx`` (rotates through
        ``router.prompt_llms``). Up to 3 attempts; reroll with the
        elimination-filter reason fed back. Returns the cleaned prompt
        or "" on terminal failure."""
        directive = op["evolution_directive"] + profile.extra_operator_constraint
        base_user = EVOLVE_USER.format(
            op_name=op["name"],
            kind=op["kind"],
            evolution_directive=directive,
            rtl_focus=op["rtl_focus"],
            original_prompt=seed.original_prompt)
        last_reason = ""
        for _ in range(3):
            user = base_user
            if last_reason:
                user += (
                    "\n\nYour previous attempt was rejected: "
                    f"{last_reason}.\nRe-emit the new specification "
                    "fixing this issue. Output the prompt text only.")
            raw = await router.chat_prompt(
                EVOLVE_SYSTEM, user,
                seed_hash=seed_hash, variant_idx=variant_idx)
            evolved = strip_dryrun(raw).strip()
            reason = profile.filter_fn(evolved, design_name, op["kind"])
            if not reason:
                return evolved
            last_reason = reason
        return ""

    async def _rollout_ref(self, evolved_prompt: str, design_name: str,
                           router: LLMRouter) -> Tuple[str, str]:
        """Step 4: traj-side rollout. Teacher (single fixed v4-pro)
        writes the reference RTL from the evolved prompt.

        Returns ``(extracted_ref, raw_traj_content)``. ``raw_traj_content``
        is the full v4-pro response with ``<think>...</think><answer>
        ```verilog\\n...```</answer>`` wrapping intact, suitable for the
        SFT assistant. The extracted ref is for the iverilog gate.

        iverilog -g2012 syntax gate; up to 3 attempts with stderr
        feedback. v4-pro format check: truncated reasoning (no
        ``</answer>``) is treated as a soft retry."""
        base_user = REF_USER.format(
            evolved_prompt=evolved_prompt, design_name=design_name)
        last_err = ""
        last_traj_content = ""
        last_extracted_ref = ""
        for _ in range(5):
            user = base_user
            if last_err:
                user += (
                    "\n\nYour previous attempt failed `iverilog -g2012` "
                    "compilation:\n```\n" + last_err + "\n```\n"
                    "Fix the syntax error and re-emit the FULL module "
                    "in one ```verilog ... ``` fence. Stick to Verilog "
                    "2005 / iverilog -g2012 (no SystemVerilog-only "
                    "syntax). Keep the module name as "
                    f"`{design_name}`.")
            resp: LLMResponse = await router.chat_traj_full(REF_SYSTEM, user)
            raw_content = resp.content or ""
            wrapped = synthesize_v4pro_wrap(
                raw_content, resp.reasoning_content,
                fallback_lang="verilog")
            last_traj_content = wrapped
            ok_fmt, _reason = check_v4pro_format(raw_content)
            if not ok_fmt:
                last_err = ""
                continue
            answer_body = extract_v4pro_answer(strip_dryrun(raw_content))
            ref = extract_verilog(answer_body)
            ref = force_module_name(ref, design_name)
            if not ref:
                last_err = ""
                continue
            last_extracted_ref = ref
            ok, err = await iverilog_syntax_check(ref, design_name)
            if ok:
                return ref, wrapped
            last_err = err
        # Soft-fail: return the last extracted ref even if it didn't pass
        # the compile gate. Caller tags compile_gate_pass=False so verify
        # can audit. If nothing was extractable, return "" to trigger the
        # caller's hard-drop path.
        return last_extracted_ref, last_traj_content

    async def _rollout_tb(self, evolved_prompt: str, evolved_ref: str,
                          design_name: str, router: LLMRouter,
                          seed_hash: int, variant_idx: int) -> str:
        """Step 5: teacher writes the testbench. tb is verify-side
        scaffolding (not part of the SFT assistant) so it routes to
        a prompt-side model. Compile-pair gate (`tb + ref` together)
        catches port mismatches and tb-side syntax errors. Up to 3
        attempts with stderr feedback. Returns the last non-empty tb
        on terminal failure so a row at least has a tb on disk for
        inspection."""
        base_user = TB_USER.format(
            evolved_prompt=evolved_prompt,
            design_name=design_name,
            evolved_ref=evolved_ref)
        last_err = ""
        last_tb = ""
        for _ in range(3):
            user = base_user
            if last_err:
                user += (
                    "\n\nYour previous testbench failed to compile with "
                    "the DUT under `iverilog -g2012`:\n```\n"
                    + last_err + "\n```\nFix ONLY the testbench. The "
                    "DUT is fixed; do not propose changes to it. "
                    "Re-emit the FULL testbench in one ```verilog ... "
                    "``` fence.")
            raw = await router.chat_prompt(
                TB_SYSTEM, user,
                seed_hash=seed_hash, variant_idx=variant_idx)
            tb = extract_verilog(strip_dryrun(raw))
            if not tb:
                last_err = ""
                continue
            last_tb = tb
            ok, err = await iverilog_compile_pair(
                tb, evolved_ref, design_name)
            if ok:
                return tb
            last_err = err
        return last_tb
