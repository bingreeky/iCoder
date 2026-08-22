"""BenchEvolver — solution-first adaptation.

Prompt-only evolution can cause reference drift when the prompt changes while
the reference RTL and testbench remain fixed. This implementation evolves the
solution and derives the prompt and tests from the same mutation contract.

This rewrite flips the direction. For each seed we:

  1. understand_reference   — DeepSeek summarises the reference RTL
  2. evolve_solution        — apply ONE bounded operator to the reference
  3. derive_prompt          — back-derive a new prompt for the evolved RTL
  4. generate_tests         — write a self-checking testbench that DETECTS
                              the operator-induced behavioural delta

A diff-ratio sanity guard rejects no-op or runaway edits with one re-roll.
The output triplet (expanded_prompt, evolved RTL, evolved testbench) is
designed to be internally consistent — `evaluator_compatible=True`.

Validation of the triplet is done out-of-band by
``scripts/validate_evolved_triplet.py`` (4 checks: self-consistency,
discrimination, evolved-solvability, original-solvability baseline).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

IVERILOG = "/usr/bin/iverilog"

# ---------- operator catalogue (mutation contracts) --------------------
#
# Each operator is a "mutation contract" — a structured description that is
# fed *verbatim* to BOTH the evolve-RTL prompt and the testbench-generation
# prompt. The two LLM calls thus share a single source of truth for what
# the operator means, instead of each independently re-interpreting a
# free-form operation description. This keeps the reference and testbench
# aligned around one explicit behavioral contract.
#
# Fields:
#   name                : short identifier
#   definition          : one-paragraph imperative for the RTL editor
#   scope               : free-form note on what the editor may touch
#   target_signal_hint  : where to focus (port / signal / construct)
#   do_not_change       : explicit list of invariants (read by both prompts)
#   old_property        : property the ORIGINAL module satisfies
#   new_property        : property the EVOLVED module must satisfy
#   test_must_check     : property the testbench MUST exercise. Phrased so
#                         that under it, original would fail and evolved
#                         would pass (or vice versa, by construction).

EVOL_OPERATORS: List[Dict[str, Any]] = [
    {
        "name": "port_widen",
        "definition": (
            "Widen the primary data output port by exactly +1 bit. Add the "
            "new bit at the MSB end (so the original N LSBs remain bit-for-"
            "bit identical). Widen any internal registers/wires that feed "
            "this output so the new high bit is meaningful (i.e. it can be "
            "1 for inputs that previously caused overflow)."),
        "scope": (
            "Output port width and internal accumulator/register widths "
            "feeding it. Nothing else."),
        "target_signal_hint": "the primary data output port",
        "do_not_change": [
            "module name", "all other ports", "control logic",
            "reset behaviour", "instantiation hierarchy",
        ],
        "old_property": (
            "the primary data output is N bits wide; on inputs that cause "
            "the internal computation to exceed N bits the high bit is "
            "silently truncated"),
        "new_property": (
            "the primary data output is N+1 bits wide; the previously-"
            "truncated high bit is now exposed at the MSB; the original N "
            "LSBs are preserved bit-for-bit at the same input"),
        "test_must_check": (
            "Drive a stimulus that would have caused the original module "
            "to overflow / truncate the high bit (operands near the high "
            "end of their representable range, OR a long-enough sequence "
            "for an accumulator). Read the evolved output and assert its "
            "MSB is 1 in at least one such cycle. Independently assert the "
            "low N bits match what the original module would output."),
    },
    {
        "name": "reset_active_high_to_low",
        "definition": (
            "Flip the active level of the existing reset signal: change "
            "active-high reset to active-low (or active-low to active-high "
            "— pick whichever flip direction the seed needs). The reset "
            "signal NAME stays unchanged. The reset's edge sensitivity "
            "(sync vs async) STAYS THE SAME — do not change it."),
        "scope": (
            "Reset polarity comparisons (`if (rst)` ↔ `if (!rst)`, or "
            "`negedge rst` ↔ `posedge rst` if and only if the design "
            "already used edge-sensitive reset). Nothing else."),
        "target_signal_hint": "the existing reset input port",
        "do_not_change": [
            "sync vs async reset (edge sensitivity stays the same)",
            "the reset signal's name", "module name", "port list",
            "port widths", "all non-reset logic",
        ],
        "old_property": (
            "asserting the reset signal at its OLD active level "
            "(e.g. rst=1 if it was active-high) clears state to its "
            "reset value"),
        "new_property": (
            "asserting the reset signal at the NEW active level "
            "(e.g. rst=0 if it is now active-low) clears state to its "
            "reset value; the OPPOSITE level no longer triggers reset"),
        "test_must_check": (
            "Drive the reset signal at the NEW active level for one or "
            "more cycles: the evolved module must enter reset (state == "
            "reset value), while the original module (re-instantiated "
            "with the same stimulus) would NOT reset at that level."),
    },
    {
        "name": "reset_sync_to_async",
        "definition": (
            "Convert the reset between synchronous and asynchronous (pick "
            "whichever direction the seed needs). For sync→async: add the "
            "reset signal to the always-block sensitivity list. For "
            "async→sync: remove it. The reset signal NAME and POLARITY "
            "stay unchanged — only edge sensitivity changes."),
        "scope": (
            "The sensitivity list of the always-blocks that handle reset, "
            "and nothing else."),
        "target_signal_hint": (
            "the reset signal in the always-block sensitivity list"),
        "do_not_change": [
            "reset polarity (active level stays the same)",
            "the reset signal's name", "module name", "port list",
            "port widths", "all non-reset logic",
        ],
        "old_property": (
            "if originally synchronous: reset takes effect only on the "
            "next active clock edge — asserting reset between edges has "
            "no effect until then. If originally asynchronous: reset "
            "takes effect immediately upon assertion"),
        "new_property": (
            "the OPPOSITE of old_property: if it was sync it is now async "
            "(immediate reset on assertion), or vice versa"),
        "test_must_check": (
            "Assert the reset between two consecutive clock edges, leave "
            "it asserted only briefly (no clock edge during the pulse), "
            "then deassert. For sync→async: the evolved module's state "
            "MUST have entered reset, the original's MUST NOT. For "
            "async→sync: the original's state MUST have entered reset, "
            "the evolved's MUST NOT."),
    },
    {
        "name": "add_enable_port",
        "definition": (
            "Add a new input port `en` (1 bit) to the port list. Gate "
            "every non-blocking sequential assignment with `if (en)`. "
            "When en==1 the module behaves byte-for-byte identical to "
            "the original; when en==0 all sequential state holds."),
        "scope": (
            "Port list (one new input `en`), and the gating condition "
            "around every non-blocking assignment in every always-block. "
            "Combinational logic and reset are NOT gated by en."),
        "target_signal_hint": "a new 1-bit input port `en`",
        "do_not_change": [
            "module name", "all other ports", "all combinational logic",
            "reset behaviour (reset still works regardless of en)",
        ],
        "old_property": (
            "all sequential state advances on every active clock edge "
            "(unconditionally, modulo reset)"),
        "new_property": (
            "sequential state advances on a clock edge only when en==1; "
            "when en==0 sequential state HOLDS its previous value; "
            "behaviour at en==1 is identical to the original"),
        "test_must_check": (
            "Hold en=0 for several consecutive clock cycles while changing "
            "other inputs that would normally advance the state: the "
            "evolved module's state must HOLD across that window, the "
            "original module's state must ADVANCE. Independently verify "
            "that for en=1 the evolved output matches the original on "
            "at least one cycle."),
    },
    {
        "name": "pipeline_stage",
        "definition": (
            "Insert exactly one register stage on the primary data output. "
            "This adds exactly +1 cycle of latency between input and that "
            "output. Combinational behaviour and other timing are "
            "unchanged. Reset must also reset the new register."),
        "scope": (
            "One new flip-flop on the primary output and the wiring "
            "around it. Reset list extended to cover the new flop."),
        "target_signal_hint": (
            "an additional output-side flip-flop on the primary data "
            "output"),
        "do_not_change": [
            "module name", "port list", "port widths", "control logic",
            "reset polarity / sensitivity",
        ],
        "old_property": (
            "the primary data output reflects its computation with "
            "latency L (L = original pipeline depth, possibly 0)"),
        "new_property": (
            "the primary data output reflects the same computation with "
            "latency L+1 (one additional cycle of delay due to the "
            "inserted register)"),
        "test_must_check": (
            "Drive a varying input sequence through both modules and "
            "sample the output across many cycles: the original at "
            "cycle N must equal the evolved at cycle N+1, for the same "
            "input sequence and reset state."),
    },
    {
        "name": "parameterize",
        "definition": (
            "Convert exactly ONE hardcoded numeric literal (a width, "
            "depth, threshold, count, …) into a Verilog `parameter` with "
            "a default value EQUAL to the original literal. The parameter "
            "must be overridable at instantiation via `#(...)`. Behaviour "
            "at the default value is byte-for-byte identical to the "
            "original."),
        "scope": (
            "One literal -> one `parameter` declaration, plus updating "
            "any references to that literal."),
        "target_signal_hint": (
            "ONE hardcoded literal that represents a clear design "
            "choice (a width, depth, threshold, count); pick the most "
            "structurally significant one"),
        "do_not_change": [
            "module name", "port list", "all OTHER literals",
            "behaviour at the default parameter value",
        ],
        "old_property": (
            "the chosen literal is hardwired; the module always behaves "
            "with that single value"),
        "new_property": (
            "the literal is replaced by a `parameter` with default equal "
            "to the original literal; the parameter is overridable at "
            "instantiation; behaviour at the default value is identical "
            "to the original"),
        "test_must_check": (
            "Instantiate the evolved module with a NON-default override "
            "for the parameter (e.g. default+1, default*2, or a clearly "
            "different legal value) and exercise it: the evolved produces "
            "an output specific to that overridden value, while the "
            "original (hardwired) module would produce the default-value "
            "output. Separately verify that at the DEFAULT value the "
            "evolved matches the original byte-for-byte."),
    },
]

OP_NAMES = [op["name"] for op in EVOL_OPERATORS]
OP_BY_NAME = {op["name"]: op for op in EVOL_OPERATORS}


def _fmt_invariants(items: List[str]) -> str:
    return "; ".join(items)


# ---------- prompts ----------------------------------------------------

UNDERSTAND_SYSTEM = (
    "You are an expert digital designer. You read a verified Verilog/"
    "SystemVerilog module and summarise what it does. Output 2-4 plain "
    "English sentences only — no headings, no code, no bullet lists. "
    "Mention: the inputs/outputs and their widths, reset polarity and "
    "clocking, and the core behaviour. Do not speculate beyond the code."
)

UNDERSTAND_USER = (
    "Module name: {design_name}\n"
    "Reference RTL:\n---\n{ref_text}\n---"
)

EVOLVE_SYSTEM = (
    "You are a careful RTL editor. You apply EXACTLY ONE bounded edit to "
    "a verified Verilog module, governed by a mutation contract. You do "
    "not refactor, rename, reformat, or re-comment beyond what the "
    "contract requires. Output ONLY the modified module inside a single "
    "```verilog ... ``` fence. Keep the module name unchanged. Keep all "
    "unrelated ports, signals, and logic byte-for-byte identical."
)

EVOLVE_USER = (
    "=== Mutation contract ===\n"
    "Operator           : {name}\n"
    "Target             : {target_signal_hint}\n"
    "What to do         : {definition}\n"
    "May touch          : {scope}\n"
    "Must NOT change    : {do_not_change}\n"
    "Old property (orig): {old_property}\n"
    "New property (you) : {new_property}\n"
    "=== End contract ===\n\n"
    "Original module:\n```verilog\n{ref_text}\n```\n\n"
    "Apply EXACTLY this contract. Output the full modified module in one "
    "```verilog ... ``` fence. No testbench code, no commentary, no "
    "top-level wrapper. The output must compile under "
    "`iverilog -g2012` (Verilog 2005)."
)

DERIVE_PROMPT_SYSTEM = (
    "You are writing a clean design specification that an engineer could "
    "implement from scratch. The reader will NOT see any reference code. "
    "The spec must be unambiguous about: ports (names + widths + "
    "directions), reset polarity, clock edge, parameters with default "
    "values, and behaviour (cycle by cycle when relevant). Output plain "
    "text only — no code, no headings beyond short labels, no commentary."
)

DERIVE_PROMPT_USER = (
    "Module name: {design_name}\n"
    "Implementation under spec (for your reference; do NOT echo or quote "
    "it):\n```verilog\n{evolved_ref}\n```\n\n"
    "Original task summary (background only): {reference_summary}\n\n"
    "Write a precise prompt that, given to a competent RTL engineer, "
    "would result in the implementation above. Include all interface "
    "widths and edge/polarity details. Keep under 300 words."
)

TESTGEN_SYSTEM = (
    "You are writing a self-checking Verilog testbench compiled by "
    "iverilog -g2012. You are governed by a MUTATION CONTRACT: the "
    "testbench's job is to verify the operator's `test_must_check` "
    "property — NOT to reverse-engineer the evolved RTL's behaviour. If "
    "the evolved RTL does something the contract did not authorise, your "
    "tb should still test the contract; we will catch ref/contract "
    "mismatches separately. The testbench MUST:\n"
    "  - Instantiate the design-under-test with the EXACT ports declared "
    "    by the supplied evolved module.\n"
    "  - Drive a deterministic stimulus that EXERCISES the contract's "
    "    `test_must_check` property. The check must distinguish the "
    "    contract's `new_property` from `old_property` — i.e. the original "
    "    (un-evolved) implementation, run on the same stimulus, would "
    "    produce a different observable result.\n"
    "  - The testbench MUST NOT fail for any reason listed in "
    "    `Must NOT change` — those aspects of the design are invariant "
    "    under the operator and should not be probed.\n"
    "  - Use `$display`. On any mismatch, print \"MISMATCH\" and call "
    "    `$finish`. On full success, print \"Your Design Passed\" and "
    "    call `$finish`.\n"
    "  - Use Verilog 2005 / iverilog -g2012 only. No SystemVerilog "
    "    assertions, no UVM, no class. If you use $urandom, fix the seed "
    "    with `integer seed = 32'h1234;`.\n"
    "Output ONLY the testbench inside a single ```verilog ... ``` fence."
)

TESTGEN_USER = (
    "=== Mutation contract (same one used to evolve the RTL) ===\n"
    "Operator           : {name}\n"
    "Target             : {target_signal_hint}\n"
    "Old property (orig): {old_property}\n"
    "New property (evol): {new_property}\n"
    "Must NOT change    : {do_not_change}\n"
    "test_must_check    : {test_must_check}\n"
    "=== End contract ===\n\n"
    "DUT module name (use this EXACT spelling in your instantiation, "
    "even if you think it is a typo): {design_name}\n\n"
    "Module under test (the EVOLVED module — your tb instantiates this):\n"
    "```verilog\n{evolved_ref}\n```\n\n"
    "Original (pre-evolution) module — provided ONLY so you can "
    "construct stimuli that distinguish old vs new behaviour. Your tb is "
    "ONLY for the evolved module; we will run the original through it "
    "separately to confirm discrimination:\n```verilog\n{original_ref}\n"
    "```\n\n"
    "Write the self-checking testbench that verifies "
    "`test_must_check` on the EVOLVED module. The testbench must compile "
    "under `iverilog -g2012` together with the evolved module above."
)


# ---------- helpers ----------------------------------------------------

VERILOG_FENCE_RE = re.compile(
    r"```(?:systemverilog|verilog|sv)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_verilog(raw: str) -> str:
    """Pull a Verilog block out of an LLM response. Mirrors
    eval/rtllm_v2_runner.py:78-87 but kept local to avoid an eval-side
    import dependency from the expansion package."""
    if not raw:
        return ""
    m = VERILOG_FENCE_RE.search(raw)
    if m:
        return m.group(1).strip()
    mm = re.search(r"(module\s+\w+.*?endmodule)", raw, re.DOTALL)
    return mm.group(1).strip() if mm else ""


def force_module_name(code: str, target: str) -> str:
    if not code.strip():
        return code
    return re.sub(r"\bmodule\s+(\w+)", f"module {target}", code, count=1)


def diff_ratio(orig: str, evolved: str) -> float:
    """Character-level diff fraction in [0, 1]. 0 = identical, 1 = no overlap."""
    if not orig and not evolved:
        return 0.0
    return 1.0 - SequenceMatcher(a=orig, b=evolved).ratio()


async def iverilog_syntax_check(code: str, design_name: str,
                                timeout: int = 15) -> Tuple[bool, str]:
    """Standalone iverilog -g2012 compile of an evolved RTL module (no tb).
    Catches syntax errors at expand time so we can re-roll with feedback
    instead of dropping the row downstream at validator pass1.

    Returns (ok, err_tail). err_tail is empty on success and trimmed to
    the last 1024 chars of stderr on failure."""
    if not code.strip():
        return False, "empty module"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / f"{design_name}.v"
        f.write_text(code)
        out_bin = Path(tmp) / "syn.out"
        try:
            proc = await asyncio.create_subprocess_exec(
                IVERILOG, "-g2012", "-o", str(out_bin), str(f),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return False, f"iverilog not found at {IVERILOG}"
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, "compile timeout"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-1024:]
        return True, ""


async def iverilog_compile_pair(tb_code: str, ref_code: str,
                                design_name: str,
                                timeout: int = 30) -> Tuple[bool, str]:
    """Compile a (testbench, evolved_ref) pair together with iverilog.
    Mirrors validator pass1's compile step but only the compile half — we
    don't run vvp here because semantic mismatch retries would risk the
    "tb fits a wrong ref" failure mode the design guards against. We only
    catch tb syntax errors, missing port names, mismatched module types,
    etc. — failures that genuinely belong to the tb side and that the
    iverilog stderr can guide a retry on.

    Returns (ok, err_tail) trimmed to last 1024 chars of stderr on fail."""
    if not tb_code.strip() or not ref_code.strip():
        return False, "empty tb or ref"
    with tempfile.TemporaryDirectory() as tmp:
        tb_f = Path(tmp) / "tb.v"
        ref_f = Path(tmp) / f"{design_name}.v"
        tb_f.write_text(tb_code)
        ref_f.write_text(ref_code)
        out_bin = Path(tmp) / "pair.out"
        try:
            proc = await asyncio.create_subprocess_exec(
                IVERILOG, "-g2012", "-o", str(out_bin),
                str(tb_f), str(ref_f),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return False, f"iverilog not found at {IVERILOG}"
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, "compile timeout"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-1024:]
        return True, ""


# ---------- per-dataset profile ----------------------------------------
#
# RTLLM uses LEGACY_PROFILE: write a self-checking tb from scratch via
# `_generate_testbench`, all 6 operators eligible. VerilogEval v2 uses
# VEVAL_PROFILE: tb is the fixed dual-instantiation `<prob>_test.sv`
# (RefModule vs TopModule comparison), no testgen rollout, narrowed
# operator subset that can survive the fixed stim/check timing. Mirrors
# the same _Profile pattern in expand/methods/evol_instruct.py and
# expand/methods/inversecoder.py — keep the three in sync.
#
# Operator narrowing for VEVAL: out of 6 operators
#   - port_widen      drops    (most VEval refs have 1-2-bit outputs)
#   - reset_*         drops    (most VEval seeds are combinational; no rst)
#   - add_enable_port keep     (compile gate will reject if stim has no en
#                               wire — that's data, not a bug, downstream
#                               summary reports the rejection rate)
#   - pipeline_stage  keep     (timing change manifests as sim_fail in tb,
#                               which is exactly the pass2 signal we want)
#   - parameterize    keep     (default-value evolution is sim-trivial,
#                               but operator-injected non-default makes
#                               pass2 detectable as long as ports survive)


@dataclass
class _Profile:
    """Per-dataset adaptation knobs."""
    name: str
    # if True, skip the teacher tb rollout and use seed.tests verbatim.
    # VEval has the dual-instantiation `<prob>_test.sv` we don't regenerate.
    fixed_tb: bool
    # if True, also stash a `module RefModule(...)` text-renamed copy of
    # the evolved ref alongside the canonical TopModule version, in
    # metadata.evolved_ref_module. Downstream verification splices this
    # into the original tb compilation slot in place of the old RefModule.
    emit_ref_module_copy: bool
    # operator names this profile is willing to dispatch. Empty = all 6.
    allowed_operators: Tuple[str, ...]


LEGACY_PROFILE = _Profile(
    name="legacy",
    fixed_tb=False,
    emit_ref_module_copy=False,
    allowed_operators=(),
)


VEVAL_PROFILE = _Profile(
    name="verilog_eval_v2",
    fixed_tb=True,
    emit_ref_module_copy=True,
    allowed_operators=("add_enable_port", "pipeline_stage", "parameterize"),
)


def _profile_for(seed: "Seed") -> _Profile:
    if seed.source_dataset == "verilog_eval_v2":
        return VEVAL_PROFILE
    return LEGACY_PROFILE


def _operators_for(profile: _Profile) -> List[Dict[str, Any]]:
    """Return the operator catalogue filtered by the profile's allow-list.
    Empty allow-list = all operators."""
    if not profile.allowed_operators:
        return EVOL_OPERATORS
    allow = set(profile.allowed_operators)
    return [op for op in EVOL_OPERATORS if op["name"] in allow]


# ---------- method -----------------------------------------------------

@register_method("benchevolver")
class BenchEvolverExpansion:
    """Solution-first BenchEvolver. ``num_variants`` cycles through the
    operator catalogue (one operator per variant); a seed run with
    num_variants=1 will use the operator at index 0 (port_widen)."""

    name = "benchevolver"

    # diff-ratio sanity bounds; outside these, evolution is rerolled once
    DIFF_LO = 0.025   # below: model produced no real change (no-op)
    DIFF_HI = 0.30    # above: model rewrote too much (over-edit)
    # Final-attempt widened range. When we hit the last try, accept any
    # non-trivial diff so hard seeds (RTLLM dual-instantiation tb, etc.)
    # don't silent-drop.
    DIFF_LO_FINAL = 0.001
    DIFF_HI_FINAL = 0.50
    MAX_EVOLVE_ATTEMPTS = 5

    async def expand(self, seed: Seed, llm, num_variants: int = 1,
                     **_kw) -> List[Expanded]:
        # If reference is missing, this method is undefined (we cannot
        # evolve a non-existent solution). Skip the seed entirely.
        if not seed.reference_solution.strip():
            return []

        # Wrap a bare LLM in a single-prompt router for backward compat.
        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))

        profile = _profile_for(seed)
        ops_pool = _operators_for(profile)

        # Pick the operator(s) deterministically from seed.id so that running
        # `num_variants=1` over N seeds spreads the operators uniformly
        # (otherwise every seed gets ops_pool[0] and only one operator
        # ever gets exercised in the pilot).
        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)
        start = seed_hash % len(ops_pool)
        ops = [ops_pool[(start + i) % len(ops_pool)]
               for i in range(num_variants)]

        # Step 1 (once per seed): reference understanding. This is a
        # prompt-side step (no part of the SFT assistant), so it can use
        # any prompt-side model. We pick variant_idx=0 for stability.
        ref_summary_raw = await router.chat_prompt(
            UNDERSTAND_SYSTEM,
            UNDERSTAND_USER.format(
                design_name=seed.evaluator_info.get("design_name", seed.id),
                ref_text=seed.reference_solution),
            seed_hash=seed_hash, variant_idx=0,
        )
        ref_summary = strip_dryrun(ref_summary_raw).strip()

        # Steps 2-4 are per-operator. Each operator chain emits up to
        # ``M = router.num_prompt_models`` prompt-side variants per
        # (seed, op): the traj model writes the evolved RTL ONCE
        # (assistant side fixed), then the prompt model writes M
        # different derived prompts so the (USER, ASSISTANT) pairs
        # diverge on the user side. Different seeds get gather()'d at
        # the caller level if num_variants==1.
        coros = [self._evolve_one(seed, ref_summary, op, router,
                                  op_idx, profile, seed_hash)
                 for op_idx, op in enumerate(ops)]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _evolve_one(self, seed: Seed, ref_summary: str,
                          op: Dict[str, Any], router: LLMRouter,
                          op_idx: int, profile: _Profile, seed_hash: int
                          ) -> List[Expanded]:
        op_name = op["name"]
        # VEval seeds carry `top_module` (canonically "TopModule") rather
        # than `design_name`. RTLLM seeds carry `design_name`. Honour both.
        design_name = (seed.evaluator_info.get("top_module")
                       or seed.evaluator_info.get("design_name")
                       or seed.metadata.get("design_name")
                       or "TopModule")

        # Step 2: solution evolution — TRAJ side (assistant must be a
        # single fixed model). Returns (extracted_code, raw_traj_content,
        # diff_ratio). raw_traj_content keeps the v4-pro <think>...
        # </think><answer>...</answer> wrapper for SFT.
        evolved_ref, traj_content, ratio = await self._evolve_solution(
            seed, op, router, design_name)
        if not evolved_ref:
            # All attempts produced unparseable output. Soft-fail with a
            # bare row carrying the last traj content (if any) and the
            # original ref so downstream verify can audit. If traj is
            # also empty (LLM raised on every attempt), give up.
            if not traj_content:
                return []
            evolved_ref = seed.reference_solution
            compile_gate_pass = False
        else:
            # Re-check compile gate to set the metadata flag. The evolve
            # loop already accepted this ref, but its widened final-
            # attempt range may have let through a non-compiling ref.
            compile_ok, _ = await iverilog_syntax_check(
                evolved_ref, design_name)
            compile_gate_pass = bool(compile_ok)

        # Step 4: testbench. Either teacher-rolled (RTLLM-style, write a new
        # self-checking tb from scratch) or fixed (VEval-style, the original
        # dual-instantiation `<prob>_test.sv` is reused verbatim). The tb
        # is verify-side scaffolding only (not part of the SFT assistant)
        # so it can use a prompt-side model.
        if profile.fixed_tb:
            evolved_tb = seed.tests
            tb_gate_pass = True
        else:
            evolved_tb = await self._generate_testbench(
                seed, op, evolved_ref, design_name, router, seed_hash)
            if not evolved_tb:
                # Fall back to original tb so the row still ships; verify
                # will flag the mismatch downstream.
                evolved_tb = seed.tests
                tb_gate_pass = False
            else:
                tb_gate_pass = True

        # Step 3: derive M prompts (one per prompt-side model). Each
        # variant becomes its own SFT row (USER differs per prompt
        # model; ASSISTANT is the SAME evolved_ref/traj_content).
        m = router.num_prompt_models
        rows: List[Expanded] = []
        for variant_idx in range(m):
            new_prompt = await self._derive_prompt(
                evolved_ref, design_name, ref_summary,
                router, seed_hash, variant_idx)
            if not new_prompt:
                continue

            ev = dict(seed.evaluator_info)
            ev["evaluator_compatible"] = True
            ev["note"] = ("Evolved triplet: prompt/ref/tests are mutually "
                          "derived via solution-first benchevolver under a "
                          "shared mutation contract.")

            md = dict(seed.metadata)
            md.update({
                "operator": op_name,
                "operator_idx": op_idx,
                "variant_idx": variant_idx,
                "original_reference": seed.reference_solution,
                "original_tests": seed.tests,
                "original_tb_path": seed.evaluator_info.get("tb_path"),
                "reference_summary": ref_summary,
                "evolution_diff_ratio": round(ratio, 4),
                "compile_gate_pass": compile_gate_pass,
                "tb_gate_pass": tb_gate_pass,
                "profile": profile.name,
                "traj_model": router.traj_model_name,
                "traj_content": traj_content,  # raw <think>/<answer>
                "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
                "contract": {
                    "old_property": op["old_property"],
                    "new_property": op["new_property"],
                    "test_must_check": op["test_must_check"],
                    "do_not_change": list(op["do_not_change"]),
                },
            })
            if profile.emit_ref_module_copy:
                md["evolved_ref_module"] = force_module_name(
                    evolved_ref, "RefModule")

            # Unique row id per (op, variant) — old code used a single
            # variant_idx stream; we keep the same shape but bump the
            # last index by op_idx*M + variant_idx so re-runs collide
            # cleanly with the previous schema.
            row_idx = op_idx * m + variant_idx
            rows.append(Expanded(
                id=expand_id(seed, "benchevolver", row_idx),
                source_dataset=seed.source_dataset,
                expansion_method="benchevolver",
                original_prompt=seed.original_prompt,
                expanded_prompt=new_prompt,
                reference_solution=evolved_ref,
                expected_output=seed.expected_output,
                tests=evolved_tb,
                evaluator_info=ev,
                metadata=md,
            ))
        return rows

    async def _evolve_solution(self, seed, op: Dict[str, Any],
                               router: LLMRouter, design_name
                               ) -> Tuple[str, str, float]:
        """Step 2: traj-side evolution. Returns (evolved_ref_extracted,
        raw_traj_content, diff_ratio) or ("", "", 0.0) on terminal
        failure.

        ``raw_traj_content`` is the v4-pro response's full ``content``
        (including ``<think>...</think><answer>...</answer>``) and is
        what the SFT pack stores as the assistant message. The
        extracted ``evolved_ref`` is for downstream verify (iverilog
        compile / pass1 / pass2).

        Up to 3 attempts. On compile failure we feed the iverilog error
        tail back to the traj LLM so it can fix the syntax instead of
        guessing. Diff-ratio failures retry with the same prompt."""
        base_user = EVOLVE_USER.format(
            name=op["name"],
            target_signal_hint=op["target_signal_hint"],
            definition=op["definition"],
            scope=op["scope"],
            do_not_change=_fmt_invariants(op["do_not_change"]),
            old_property=op["old_property"],
            new_property=op["new_property"],
            ref_text=seed.reference_solution)
        last_err = ""
        last_traj_content = ""
        last_evolved_ref = ""
        last_ratio = 0.0
        for attempt in range(self.MAX_EVOLVE_ATTEMPTS):
            user = base_user
            if last_err:
                user += (
                    "\n\nYour previous attempt failed `iverilog -g2012` "
                    "compilation with this error:\n```\n"
                    f"{last_err}\n```\n"
                    "Fix the syntax error and re-emit the FULL module in a "
                    "single ```verilog ... ``` fence. Stick to Verilog 2005 / "
                    "iverilog -g2012 (no SystemVerilog-only syntax). The "
                    "fix MUST still satisfy the mutation contract above — "
                    "do not change the operator's intent to avoid the error.")
            resp: LLMResponse = await router.chat_traj_full(EVOLVE_SYSTEM, user)
            raw_content = resp.content or ""
            # Synthesise the <think>/<answer> wrap from prose+fence (most
            # endpoints don't return tags natively). reasoning_content is
            # used as <think> body when present (qwen/kimi); else the
            # prose-before-fence is used.
            wrapped = synthesize_v4pro_wrap(
                raw_content, resp.reasoning_content,
                fallback_lang="verilog")
            last_traj_content = wrapped
            ok_fmt, fmt_reason = check_v4pro_format(raw_content)
            if not ok_fmt:
                last_err = ""
                continue
            answer_body = extract_v4pro_answer(strip_dryrun(raw_content))
            evolved_ref = extract_verilog(answer_body)
            evolved_ref = force_module_name(evolved_ref, design_name)
            if not evolved_ref:
                last_err = ""
                continue
            # Keep the most recent extracted ref as a fallback — even if
            # all attempts fail the compile gate, we want to ship a row
            # rather than silent-drop.
            last_evolved_ref = evolved_ref
            ratio = diff_ratio(seed.reference_solution, evolved_ref)
            last_ratio = ratio
            is_final = attempt == self.MAX_EVOLVE_ATTEMPTS - 1
            diff_lo = self.DIFF_LO_FINAL if is_final else self.DIFF_LO
            diff_hi = self.DIFF_HI_FINAL if is_final else self.DIFF_HI
            diff_ok = diff_lo <= ratio <= diff_hi
            compile_ok, err = await iverilog_syntax_check(
                evolved_ref, design_name)
            if compile_ok and diff_ok:
                return evolved_ref, wrapped, ratio
            last_err = err if not compile_ok else ""
        # Soft-fail: all attempts failed the gate. Return the last
        # extracted ref + traj so the caller can ship a row tagged
        # ``compile_gate_pass=False`` (audited downstream by verify/pack).
        return last_evolved_ref, last_traj_content, last_ratio

    async def _derive_prompt(self, evolved_ref: str, design_name: str,
                             ref_summary: str, router: LLMRouter,
                             seed_hash: int, variant_idx: int) -> str:
        """Step 3: prompt-side reverse derivation. Uses one of the
        prompt models (selected by ``variant_idx``) to write a clean NL
        spec for ``evolved_ref``. Strips any leading ``<think>``."""
        derived_raw = await router.chat_prompt(
            DERIVE_PROMPT_SYSTEM,
            DERIVE_PROMPT_USER.format(
                design_name=design_name,
                evolved_ref=evolved_ref,
                reference_summary=ref_summary or "(unavailable)"),
            seed_hash=seed_hash, variant_idx=variant_idx,
        )
        return strip_dryrun(derived_raw).strip()

    async def _generate_testbench(self, seed: Seed, op: Dict[str, Any],
                                  evolved_ref: str, design_name: str,
                                  router: LLMRouter, seed_hash: int = 0
                                  ) -> str:
        """Run step 4; reroll on (tb+ref) iverilog compile failure.

        The tb is verify-side scaffolding (not in the SFT assistant) so
        we route to a prompt-side model. ``seed_hash`` selects which
        prompt model deterministically; we always pin variant_idx=0
        because the tb is shared across the M prompt-side variants of
        ``derive_prompt``.

        Up to 3 attempts. The retry is deliberately tb-only: we feed
        back the iverilog stderr so the model fixes the testbench's
        syntax / port wiring / module-name typo, but we KEEP the
        contract bound so the tb cannot drift into "fitting whatever
        the ref happens to do" (an LLM-oracle failure mode the design
        notes warn against).

        Returns the extracted tb source, or "" if all 3 attempts fail."""
        base_user = TESTGEN_USER.format(
            name=op["name"],
            target_signal_hint=op["target_signal_hint"],
            old_property=op["old_property"],
            new_property=op["new_property"],
            do_not_change=_fmt_invariants(op["do_not_change"]),
            test_must_check=op["test_must_check"],
            design_name=design_name,
            evolved_ref=evolved_ref,
            original_ref=seed.reference_solution)
        last_err = ""
        last_tb = ""
        for attempt in range(3):
            user = base_user
            if last_err:
                user += (
                    "\n\nYour previous testbench failed to compile with "
                    "the evolved module under `iverilog -g2012`:\n"
                    f"```\n{last_err}\n```\n"
                    "Fix ONLY the testbench. The mutation contract above "
                    "is unchanged — the tb must still verify "
                    "`test_must_check`. The evolved module is fixed; do "
                    "not propose changes to it. Stick to Verilog 2005 / "
                    "iverilog -g2012 (no SystemVerilog-only syntax). "
                    "Re-emit the FULL testbench in one ```verilog ... ``` "
                    "fence.")
            tb_raw = await router.chat_prompt(
                TESTGEN_SYSTEM, user,
                seed_hash=seed_hash, variant_idx=0)
            evolved_tb = extract_verilog(strip_dryrun(tb_raw))
            if not evolved_tb:
                last_err = ""
                continue
            last_tb = evolved_tb
            compile_ok, err = await iverilog_compile_pair(
                evolved_tb, evolved_ref, design_name)
            if compile_ok:
                return evolved_tb
            last_err = err
        # all attempts failed; return last non-empty tb so the row at least
        # has a tb on disk (validator will mark it failed downstream and
        # we can inspect it)
        return last_tb
