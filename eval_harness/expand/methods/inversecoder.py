"""InverseCoder style expansion — spec-as-CoT (assistant-side) variant.

Background. The contract-v1 behaviour-only pipeline still hit drift = 50 %
(P_orig 40 % / P_inv 80 %) on RTLLM v2 because the spec, even leak-linted,
ships the verbatim port table and implementation-aware behaviour rules.
The drift gate (≤10 % per docs/inversecoder_design.md §2) measures spec
*against* the original prompt — and a clean Verilog port list reading
"output reg [9:0] data_out" alongside the original NL "output ports:
... data_out[9:0]" is automatically more compile-friendly for a fresh
model. There is no way to satisfy the gate while keeping spec on the
user side without crippling the spec.

This revision moves spec from the user side to the assistant side as a
chain-of-thought / design draft. The user side becomes a back-derived
NL prompt produced from the (spec + ref_code) "answer block" alone —
the original RTLLM prompt is NOT seen by any step of the pipeline,
otherwise the drift metric (fresh model on original_prompt vs fresh
model on the back-derived prompt) is being fed by its own bias source.

Pipeline:

  Step 1   Code collection      carries through from seed
  Step 2   Code understanding   black-box behavioural summary, ref source
                                leaves the working set after this step;
                                feeds spec generation only.
  Step 3   Spec generation      uses ONLY the step-2 summary + framing,
                                no ref source code in the prompt
  Step 4   Multi-framing        4 framings as before
  Step 5   Spec leak lint       static check that spec doesn't leak ref
                                submodule names / internal wire-reg names
                                / banned structural vocabulary; one re-roll
  Step 6   Back-derive prompt   NEW: from (spec + ref_code) — no original
                                prompt, no summary, no port_list — produce
                                the NL request that the answer block
                                responds to. Same internal-name lint
                                applies to the new prompt with one re-roll.
  Step 7   SFT pair assembly    spec → metadata.assistant_spec, scripts/
                                convert_to_sft.py prepends it to the
                                code-fenced reference so the assistant
                                turn reads "<spec>\\n\\n```verilog\\n
                                <code>\\n```".

Drift now measures fresh-model pass on (original NL prompt) vs fresh
model on (back-derived NL prompt). Both sides are NL; the back-derive
sees only the answer block (clean spec + verified code), so any drift
gap reflects genuine difficulty difference rather than a contaminating
peek at one side's prompt.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from ..base import Expanded
from ..llm import LLMResponse, LLMRouter
from ..registry import register_method
from ._common import (
    Seed,
    base_expanded,
    check_v4pro_format,
    extract_v4pro_answer,
    strip_dryrun,
    synthesize_v4pro_wrap,
)
from .benchevolver import force_module_name

# ---------- step 2: code understanding (LEGACY: RTLLM/VEval) ----------
#
# RTLLM and VEval ship with this compressed-summary prompt. CVDP needed
# something different — see the *_CVDP variants below and the
# `_InvProfile` selector. KB has its own *_KB variants too. **Don't
# touch these legacy strings without re-running drift on the data sets
# that depend on them.**

UNDERSTAND_SYSTEM = (
    "You are an expert digital designer. Read a verified Verilog/"
    "SystemVerilog module and write a BLACK-BOX behavioural summary.\n\n"
    "HARD RULES:\n"
    "  - Describe the module's externally observable behaviour: per-"
    "cycle / per-input behaviour rules, reset behaviour, clocking. "
    "YES.\n"
    "  - Describe the module from the perspective of someone who has "
    "ONLY seen its ports, not its body. They could implement the same "
    "spec totally differently. YES.\n"
    "  - The public interface (port names, widths, directions, reset "
    "polarity, clock edge) is given to you separately as `Public "
    "interface`. You MUST refer to ports by the EXACT names listed "
    "there — no renaming, no abbreviating. YES.\n"
    "  - Mention internal submodules, internal hierarchy, decomposition "
    "strategy, intermediate wire/reg names, or any specific implementation "
    "structure. NEVER. If the module is hierarchical, describe what it "
    "does as a whole; do not enumerate the parts.\n"
    "  - Counter-example: if the reference implements 16-bit add as two "
    "8-bit adds plus carry chaining, your summary describes 16-bit "
    "addition with carry-in / carry-out — NOT the decomposition.\n\n"
    "Output 2-5 plain-English sentences. No bullet lists, no headings, "
    "no code, no implementation language."
)

UNDERSTAND_USER = (
    "Module name: {design_name}\n\n"
    "Public interface (use these EXACT names + widths in your summary):\n"
    "---\n{port_list}\n---\n\n"
    "Reference RTL (for your eyes only — do NOT echo or quote it):\n"
    "---\n{code}\n---\n\n"
    "Write the black-box behavioural summary now."
)

# ---------- step 2: code understanding (CVDP cid003 variant) ----------
#
# Drift study (see docs/pipelines/inversecoder-cvdp.md §3 Phase 2 改动 A)
# found that protocol-rich CVDP modules (FSM transition tables, encoders
# with lookup tables) lost detail in the 2-5 sentence compression above
# before step 3 could even see them. Compression was redundant safety —
# leak protection is what step 5 lint does. CVDP path uses an exhaustive
# rule listing instead.

UNDERSTAND_SYSTEM_CVDP = (
    "You are an expert digital designer. Read a verified Verilog/"
    "SystemVerilog module and write an EXHAUSTIVE black-box behaviour "
    "specification organised as bullet rules.\n\n"
    "HARD RULES:\n"
    "  - List EVERY externally observable rule the design follows. Be "
    "thorough. If the ref has a 256-entry lookup table or a 16-state "
    "FSM, list every entry / every transition (or, if truly identical, "
    "summarise the pattern that covers them — but never silently drop "
    "rules).\n"
    "  - Organise rules by group: per-input rules, per-state rules, "
    "per-edge rules, reset behaviour, output assertion timing. Use "
    "bullet lists with headings; this is NOT a free-prose paragraph.\n"
    "  - The public interface (port names, widths, directions, reset "
    "polarity, clock edge) is given to you separately as `Public "
    "interface`. You MUST refer to ports by the EXACT names listed "
    "there — no renaming, no abbreviating.\n"
    "  - Banned: internal submodule names, intermediate wire/reg names, "
    "decomposition strategy, implementation structure, language like "
    "'split into', 'cascade', 'submodule', 'register', 'pipeline stage'. "
    "Describe what someone holding the testbench would observe — not "
    "how the ref happens to organise its body.\n"
    "  - Counter-example (BAD): \"the encoder uses a 4-stage pipeline "
    "with intermediate registers q1..q4 carrying carry bits\".\n"
    "    Counter-example (GOOD): \"on each rising edge of clk, when "
    "i_valid is high, the output o_data presents the unsigned sum of "
    "all IN_DATA_NS elements of i_data two cycles later\".\n\n"
    "Output: bullet list with short group headings. Aim for "
    "comprehensive coverage of behaviour, not for brevity. No code, "
    "no implementation language."
)

UNDERSTAND_USER_CVDP = (
    "Module name: {design_name}\n\n"
    "Public interface (use these EXACT names + widths in your rules):\n"
    "---\n{port_list}\n---\n\n"
    "Reference RTL (for your eyes only — do NOT echo or quote it):\n"
    "---\n{code}\n---\n\n"
    "Write the exhaustive behaviour-rule list now."
)

# ---------- step 3+4: framings & spec generation -----------------------

# LEGACY framings (RTLLM/VEval ship). Don't tweak without a drift re-run
# on those data sets — VEval's B+A pipeline depends on these strings.
INVERSE_FRAMINGS = [
    ("formal_spec",
     "Write a precise interface-level specification: ports / parameters "
     "(names, widths, directions), reset polarity, clock edge, then "
     "behaviour stated as input → output rules. No prose introduction."),
    ("docstring",
     "Write the spec as if it were the module-level docstring an "
     "engineer would read first. Friendly, clear, but covers all "
     "interface and behaviour points."),
    ("requirement_list",
     "Write a numbered list of unambiguous behavioural requirements "
     "the design must satisfy. Each requirement is one fact about the "
     "interface or per-cycle behaviour."),
    ("design_brief",
     "Write a short design brief: 1) goal, 2) interface, 3) behaviour, "
     "4) any non-functional constraints visible from the interface "
     "(timing, reset)."),
]

# CVDP framings — `formal_spec` rewritten so the rules section is the
# bulk of the spec instead of an interface table. See
# docs/pipelines/inversecoder-cvdp.md §3 Phase 2 改动 D.
INVERSE_FRAMINGS_CVDP = [
    ("formal_spec",
     "Write a precise specification in three sections: (1) a one-line "
     "interface line stating the public ports are exactly the ones "
     "given (do NOT re-list the port table — it's elsewhere in the "
     "answer), (2) clocking + reset preface (one line), (3) "
     "EXHAUSTIVE input→output behaviour rules covering every case in "
     "the rule list — this section is the BULK of the spec. Use "
     "structured formatting (numbered cases, condition→effect lines, "
     "tables for lookup-style rules). The spec is precise but its "
     "precision lives in the rules, not in the interface paragraph."),
    ("docstring",
     "Write the spec as if it were the module-level docstring an "
     "engineer would read first. Friendly, clear, but covers all "
     "interface and behaviour points."),
    ("requirement_list",
     "Write a numbered list of unambiguous behavioural requirements "
     "the design must satisfy. Each requirement is one fact about the "
     "interface or per-cycle behaviour."),
    ("design_brief",
     "Write a short design brief: 1) goal, 2) interface, 3) behaviour, "
     "4) any non-functional constraints visible from the interface "
     "(timing, reset)."),
]

# TritonBench-G framings — operator-level NL request, not RTL ports/
# reset. Reuses the four shape names so the framing-loop in _expand_tbg
# can iterate the same way, but rewrites every description so the
# back-derive model doesn't drift into "clock edge / reset polarity"
# phrasing the way it does with the LEGACY (Verilog) framings.
INVERSE_FRAMINGS_TBG = [
    ("formal_spec",
     "Write a precise operator-level specification: function name, "
     "tensor argument names with shapes and dtypes, output shape and "
     "dtype, then behaviour stated as input → output rules in plain "
     "English. No section headers about clocks, ports, or reset; this "
     "is a GPU kernel specification, not a hardware module."),
    ("docstring",
     "Write the request as if it were the docstring an engineer would "
     "read first. Friendly, clear, naming the entry-point function "
     "and arguments, summarising the math behaviour and expected "
     "output."),
    ("requirement_list",
     "Write a numbered list of unambiguous requirements the kernel "
     "must satisfy: function name, argument names + shape/dtype, "
     "output shape/dtype, math operation performed, numerical "
     "tolerance. Each requirement is one fact."),
    ("design_brief",
     "Write a short design brief in four sections: (1) what the "
     "kernel computes (one-line math summary), (2) entry-point "
     "function + arguments, (3) expected output, (4) correctness "
     "tolerance. No discussion of internal block size or scheduling."),
]

# LEGACY spec gen (RTLLM/VEval ship)
SPEC_SYSTEM = (
    "You are an InverseCoder. You write a clean specification for a "
    "Verilog module given a black-box behavioural summary written by a "
    "designer. The specification is what an engineer would read to "
    "implement the module from scratch.\n\n"
    "HARD RULES (these are the same rules the behavioural summary was "
    "written under — do not violate them):\n"
    "  - Describe externally observable behaviour. YES.\n"
    "  - Mention internal submodules, internal hierarchy, decomposition "
    "strategy, intermediate wire/reg names, or any specific implementation "
    "structure. NEVER.\n"
    "  - Banned vocabulary: 'decompose', 'split into', 'cascade', "
    "'submodule', 'internal stage', or any phrasing that prescribes how "
    "the implementation is organised internally.\n"
    "  - Counter-example: a leaky spec says \"the module decomposes "
    "16-bit addition into two 8-bit adds, the lower add produces a "
    "carry that chains into the upper\". A clean spec says \"the module "
    "computes a 16-bit unsigned sum a + b + Cin and exposes the high "
    "carry on Co\".\n\n"
    "Output ONLY the specification text — no code blocks, no headings "
    "beyond short labels, no commentary."
)

SPEC_USER = (
    "Write the specification in the **{framing}** framing — "
    "{framing_desc}\n\n"
    "Module name to use in the spec: {design_name}\n\n"
    "Public interface (use these EXACT port names, widths, and "
    "directions — no renaming, no abbreviating):\n"
    "---\n{port_list}\n---\n\n"
    "Behavioural summary (treat as ground truth — your spec must be "
    "consistent with it):\n"
    "---\n{summary}\n---\n\n"
    "Output the specification now."
)

# CVDP spec gen — adds "Cover EVERY rule" hard rule because the input
# is the exhaustive rule list (UNDERSTAND_SYSTEM_CVDP) instead of the
# 2-5 sentence summary, and we need the spec to honour it.
SPEC_SYSTEM_CVDP = (
    "You are an InverseCoder. You write a clean specification for a "
    "Verilog module given an exhaustive list of black-box behaviour "
    "rules written by a designer. The specification is what an engineer "
    "would read to implement the module from scratch.\n\n"
    "HARD RULES (these are the same rules the behaviour-rule list was "
    "written under — do not violate them):\n"
    "  - Describe externally observable behaviour. YES.\n"
    "  - Cover EVERY rule from the input rule list — your spec must be "
    "implementable into a module that satisfies all of them. Do not "
    "silently drop rules; if a rule is too detailed for your framing's "
    "form, restructure or summarise the PATTERN faithfully but never "
    "lose its observable consequence.\n"
    "  - Mention internal submodules, internal hierarchy, decomposition "
    "strategy, intermediate wire/reg names, or any specific implementation "
    "structure. NEVER.\n"
    "  - Banned vocabulary: 'decompose', 'split into', 'cascade', "
    "'submodule', 'internal stage', or any phrasing that prescribes how "
    "the implementation is organised internally.\n"
    "  - Counter-example: a leaky spec says \"the module decomposes "
    "16-bit addition into two 8-bit adds, the lower add produces a "
    "carry that chains into the upper\". A clean spec says \"the module "
    "computes a 16-bit unsigned sum a + b + Cin and exposes the high "
    "carry on Co\".\n\n"
    "Output ONLY the specification text — no code blocks, no headings "
    "beyond short labels, no commentary."
)

SPEC_USER_CVDP = (
    "Write the specification in the **{framing}** framing — "
    "{framing_desc}\n\n"
    "Module name to use in the spec: {design_name}\n\n"
    "Public interface (use these EXACT port names, widths, and "
    "directions — no renaming, no abbreviating):\n"
    "---\n{port_list}\n---\n\n"
    "Behaviour rules (exhaustive — treat as ground truth, your spec "
    "must cover every rule listed; the spec may regroup or paraphrase "
    "but must not silently drop any observable consequence):\n"
    "---\n{summary}\n---\n\n"
    "Output the specification now."
)

# ---------- step 6: back-derive new NL prompt --------------------------
#
# Input is the ANSWER BLOCK only — the cleaned spec (already lint-passed
# in step 5) and the verified reference Verilog. NO original_prompt and
# no summary flow into this step: the back-derive must not see anything
# from the user side, otherwise the metric we use to grade it (drift =
# fresh model on original_prompt vs fresh model on new prompt) is being
# fed by its own bias source.
#
# The output is the new ``expanded_prompt`` — the user-side request that
# the answer block (spec + code) is the natural response to. It must
# stay at high-level engineering-ask specificity, not echo the spec.

BACK_DERIVE_SYSTEM = (
    "You are reverse-engineering a USER REQUEST from a complete answer "
    "block. The answer is a Verilog design specification followed by the "
    "matching reference implementation. You must imagine what plain-"
    "English request a hardware engineer would have written so that this "
    "answer block IS the natural response.\n\n"
    "A deterministic public-port table (module name, port names with "
    "exact widths and directions, reset polarity / sync style if any) "
    "is computed from the reference and PREPENDED to your output "
    "automatically. Your job is the BEHAVIOUR-DESCRIPTION portion only. "
    "DO NOT repeat the port table or the module name in your prose — "
    "that information is already present.\n\n"
    "HARD RULES:\n"
    "  - Output ~3-10 lines describing functional intent. No port table, "
    "no module-name header, no code fences, no commentary outside the "
    "request.\n"
    "  - Strictly LESS DETAILED than the spec. If you find yourself "
    "transcribing per-cycle behaviour rules, FSM states, internal "
    "counter widths, or accumulator semantics in the request — stop, "
    "that detail belongs in the answer.\n"
    "  - DO NOT mention any internal signal name, register name, sub-"
    "module name, or hierarchy detail visible inside the reference "
    "module body. Only the public port names from the prepended table "
    "are allowed.\n"
    "  - DO NOT use the words 'spec', 'specification', 'design draft', "
    "'as the spec says', etc. — the engineer writing this request does "
    "not yet have a spec; they are asking for one (and the code).\n"
    "  - DO NOT echo the spec verbatim. Paraphrase the high-level intent "
    "in your own words.\n\n"
    "CONVENTION-PRECISION CHECKLIST. Examine the reference. Where the "
    "reference exhibits any of the patterns below, state the convention "
    "explicitly in plain English. Failing to state a present pattern is "
    "the most common drift mode — be defensive:\n\n"
    "  1. Bit-slice control mapping. If a control bit at index k writes "
    "or selects a specific data slice [a:b], spell the mapping out.\n"
    "       BAD : \"byte-wide write enables\"\n"
    "       GOOD: \"`byteena[1]` gates writes to `d[15:8]`; `byteena[0]` "
    "gates `d[7:0]`\"\n"
    "  2. Concatenation / packing order. If outputs are formed by "
    "{a, b, c, ...}, state which input occupies the most-significant "
    "bits and which occupies the least; mention any constant padding "
    "and where it sits.\n"
    "  3. Shift-register output tap. If the output is tied to a specific "
    "stage of an internal shift chain, state which stage (first / last / "
    "specific position counted from the input).\n"
    "  4. Shift / count direction. State left vs right; up vs down.\n"
    "  5. Priority / scan direction. For encoders / first-set-bit / "
    "leading-zero detectors, state whether priority is given to the LSB "
    "or the MSB end.\n"
    "  6. Sign. State explicitly when arithmetic is signed or unsigned, "
    "and whether sign-extension applies.\n\n"
    "Output ONLY the behaviour-description text."
)

BACK_DERIVE_USER = (
    "ANSWER BLOCK\n\n"
    "Design specification (the assistant's design draft):\n"
    "---\n{spec}\n---\n\n"
    "Reference implementation (the assistant's final code):\n"
    "---\n{ref_code}\n---\n\n"
    "Public-port table that will be prepended to your output (DO NOT "
    "repeat in your prose):\n"
    "---\n{port_block}\n---\n\n"
    "Now write the behaviour-description portion of the user request, "
    "obeying the convention-precision checklist."
)

# CVDP back-derive — length cap removed (the "~3-10 lines" /
# "strictly LESS DETAILED" rule was forcing protocol-heavy modules to
# drop testbench-checked rules from the user-side prompt). See
# docs/pipelines/inversecoder-cvdp.md §3 Phase 2 改动 C.
BACK_DERIVE_SYSTEM_CVDP = (
    "You are reverse-engineering a USER REQUEST from a complete answer "
    "block. The answer is a Verilog design specification followed by the "
    "matching reference implementation. You must imagine what plain-"
    "English request a hardware engineer would have written so that this "
    "answer block IS the natural response.\n\n"
    "A deterministic public-port table (module name, port names with "
    "exact widths and directions, reset polarity / sync style if any) "
    "is computed from the reference and PREPENDED to your output "
    "automatically. Your job is the BEHAVIOUR-DESCRIPTION portion only. "
    "DO NOT repeat the port table or the module name in your prose — "
    "that information is already present.\n\n"
    "HARD RULES:\n"
    "  - Output the BEHAVIOUR DESCRIPTION ONLY. No port table, no module-"
    "name header, no code fences, no commentary outside the request. "
    "Length: be as concise as possible WHILE preserving every "
    "behavioural constraint a fresh engineer needs to solve the "
    "problem and pass the testbench. Simple modules naturally land "
    "around 3-10 lines; protocol-heavy modules (FSMs, encoders with "
    "lookup tables, multi-stage state machines) can need 20-50 lines. "
    "DO NOT trim away rules to hit a length target.\n"
    "  - The request stays at engineering-ASK altitude — it states WHAT "
    "the module must do, not the per-bit implementation details. But "
    "rules that are part of the contract (per-input cases, per-state "
    "behaviour, encoding tables that the testbench checks) ARE part "
    "of WHAT and MUST be stated.\n"
    "  - DO NOT mention any internal signal name, register name, sub-"
    "module name, or hierarchy detail visible inside the reference "
    "module body. Only the public port names from the prepended table "
    "are allowed.\n"
    "  - DO NOT use the words 'spec', 'specification', 'design draft', "
    "'as the spec says', etc. — the engineer writing this request does "
    "not yet have a spec; they are asking for one (and the code).\n"
    "  - DO NOT echo the spec verbatim. Paraphrase the high-level intent "
    "in your own words.\n\n"
    "CONVENTION-PRECISION CHECKLIST. Examine the reference. Where the "
    "reference exhibits any of the patterns below, state the convention "
    "explicitly in plain English. Failing to state a present pattern is "
    "the most common drift mode — be defensive:\n\n"
    "  1. Bit-slice control mapping. If a control bit at index k writes "
    "or selects a specific data slice [a:b], spell the mapping out.\n"
    "       BAD : \"byte-wide write enables\"\n"
    "       GOOD: \"`byteena[1]` gates writes to `d[15:8]`; `byteena[0]` "
    "gates `d[7:0]`\"\n"
    "  2. Concatenation / packing order. If outputs are formed by "
    "{a, b, c, ...}, state which input occupies the most-significant "
    "bits and which occupies the least; mention any constant padding "
    "and where it sits.\n"
    "  3. Shift-register output tap. If the output is tied to a specific "
    "stage of an internal shift chain, state which stage (first / last / "
    "specific position counted from the input).\n"
    "  4. Shift / count direction. State left vs right; up vs down.\n"
    "  5. Priority / scan direction. For encoders / first-set-bit / "
    "leading-zero detectors, state whether priority is given to the LSB "
    "or the MSB end.\n"
    "  6. Sign. State explicitly when arithmetic is signed or unsigned, "
    "and whether sign-extension applies.\n\n"
    "Output ONLY the behaviour-description text."
)

# ---------- profile selector (LEGACY vs CVDP variants) ----------------
#
# RTLLM/VEval are SHIPPED on the LEGACY prompts above. The CVDP work
# changed step-2 / spec-gen / back-derive in ways that break drift on
# those data sets, so the changes are gated behind a profile and only
# kick in when ``seed.source_dataset == "cvdp_cid003"``. Adding a new
# data set's variant means adding another profile here, NOT mutating
# the LEGACY constants.

@dataclass
class _InvProfile:
    understand_system: str
    understand_user: str
    spec_system: str
    spec_user: str
    framings: List[Tuple[str, str]]
    back_derive_system: str
    enable_reset_lint: bool = False  # CVDP-only: hard-pin reset polarity/sync


LEGACY_PROFILE = _InvProfile(
    understand_system=UNDERSTAND_SYSTEM,
    understand_user=UNDERSTAND_USER,
    spec_system=SPEC_SYSTEM,
    spec_user=SPEC_USER,
    framings=INVERSE_FRAMINGS,
    back_derive_system=BACK_DERIVE_SYSTEM,
    enable_reset_lint=False,
)

CVDP_PROFILE = _InvProfile(
    understand_system=UNDERSTAND_SYSTEM_CVDP,
    understand_user=UNDERSTAND_USER_CVDP,
    spec_system=SPEC_SYSTEM_CVDP,
    spec_user=SPEC_USER_CVDP,
    framings=INVERSE_FRAMINGS_CVDP,
    back_derive_system=BACK_DERIVE_SYSTEM_CVDP,
    enable_reset_lint=True,
)


def _profile_for(seed: "Seed") -> _InvProfile:
    if seed.source_dataset == "cvdp_cid003":
        return CVDP_PROFILE
    return LEGACY_PROFILE


# ---------- step 5: structural-leak lint -------------------------------

# Verilog identifiers we want to extract from the reference, then
# blacklist in the spec text. Matches `module <name>` and
# `wire/reg/logic [w:0] <name>` declarations.
_MODULE_DECL_RE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)", re.MULTILINE)
_WIRE_DECL_RE = re.compile(
    r"\b(?:wire|reg|logic)\s*(?:\[[^\]]+\])?\s*([A-Za-z_]\w*)\s*[;,=]")
_INSTANCE_DECL_RE = re.compile(
    # `<UserModule> <inst_name> (` — common Verilog instantiation
    r"\b([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
# Pulls everything between `module foo (` and the matching `);`. Captures
# the port-list body so we can extract public port names — those are part
# of the interface and MUST be mentionable in the spec. Allows an optional
# parameter block (`module foo #(parameter A=1, ...) (...)`) — common in
# CVDP cid003 refs and any parameterised SystemVerilog module.
_PORT_LIST_RE = re.compile(
    r"\bmodule\s+\w+\s*"
    r"(?:#\s*\(.*?\)\s*)?"
    r"\(([^;]*?)\)\s*;",
    re.DOTALL,
)
_VERILOG_KEYWORDS_IN_PORTS = {
    "input", "output", "inout", "wire", "reg", "logic",
    "signed", "unsigned", "parameter",
}

BANNED_STRUCTURAL_PHRASES = [
    "decompose", "decomposition",
    "split into", "splits into",
    "cascade", "cascaded",
    "submodule", "sub-module",
    "internal stage", "intermediate stage",
    "instantiat",            # 'instantiate', 'instantiated', etc.
    "two halves", "lower half", "upper half",
]

# Whitelist of identifiers that show up everywhere in Verilog and
# shouldn't trigger a leak alarm even if they appear in ref. Anything in
# here is treated as generic English rather than as a leaked identifier.
_GENERIC_NAMES = {
    # clocks / reset / common control
    "clk", "clock", "rst", "reset", "rst_n", "en", "enable",
    # generic IO
    "in", "out", "data", "valid", "ready", "addr", "we",
    "a", "b", "c", "d", "x", "y", "z",
    "i", "j", "k", "n", "m",
    # adder/arith
    "cin", "cout", "co", "ci",
    "sum", "result", "output", "input", "prod", "diff",
    # common state/counter words (also valid English nouns in a spec)
    "count", "counter", "state", "next", "current",
    "buf", "buffer", "tmp", "temp", "idx", "index",
    # control protocol
    "start", "done", "busy",
    "Verilog", "module",
}


def _extract_port_names(ref: str) -> set[str]:
    """Return the set of identifier tokens appearing inside the
    `module foo (...)` port list. Those are part of the public
    interface and MUST be allowed in the spec — never blacklisted."""
    m = _PORT_LIST_RE.search(ref)
    if not m:
        return set()
    body = m.group(1)
    names: set[str] = set()
    for tok in re.findall(r"\b[A-Za-z_]\w*\b", body):
        if tok in _VERILOG_KEYWORDS_IN_PORTS:
            continue
        names.add(tok)
    return names


def _extract_port_list_text(ref: str) -> str:
    """Return the raw text of the `module foo (...)` port list, lightly
    cleaned. This is the literal interface that downstream tb / SFT
    consumers will check against — passing it verbatim into the spec-
    generation prompt is how we preserve port names + widths +
    directions without leaking the module body's internal structure."""
    m = _PORT_LIST_RE.search(ref)
    if not m:
        return ""
    body = m.group(1)
    # collapse extra whitespace within each port line, keep newlines
    cleaned_lines = []
    for line in body.splitlines():
        stripped = line.strip().rstrip(",")
        if stripped:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


# ---------- structured port-table extraction (Path B) -----------------
#
# back-derive's "MUST list every port by exact name + width" rule was
# routinely violated on hard cases (the LLM dropped widths or paraphrased
# 'byteena (2 bits)' to 'byte-enable signal'). Path B fixes this by
# computing the port table deterministically and prepending a uniform
# NL-formatted block to the back-derived NL — the LLM only writes the
# behaviour-description portion.

def _width_expr_to_text(width_expr: Optional[str]) -> str:
    """Convert a Verilog width expression to a clean NL form.

    `[7:0]`     -> "8 bits"
    bare port   -> "1 bit"
    `[W-1:0]`   -> "W bits"  (parameterised — common in CVDP)
    fallback    -> "[<expr>] bits"
    """
    if not width_expr:
        return "1 bit"
    we = width_expr.strip()
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", we)
    if m:
        hi, lo = int(m.group(1)), int(m.group(2))
        n = abs(hi - lo) + 1
        return "1 bit" if n == 1 else f"{n} bits"
    m = re.match(r"^\s*(\w+)\s*-\s*1\s*:\s*0\s*$", we)
    if m:
        return f"{m.group(1)} bits"
    return f"[{we}] bits"


def _parse_port_decl(decl: str) -> Optional[Tuple[str, str, str]]:
    """Parse a single ANSI port declaration like
    `input [7:0] a` / `output reg [9:0] data_out` / `output overflow`.

    Returns (direction, name, width_text) or None on parse failure.
    Allows reg/wire/logic/signed/unsigned modifiers in any reasonable
    position relative to the bracketed width.
    """
    m = re.match(
        r"^(input|output|inout)\s+"
        r"(?:(?:reg|wire|logic|signed|unsigned)\s+)*"
        r"(?:\[([^\]]+)\])?\s*"
        r"(?:(?:reg|wire|logic|signed|unsigned)\s+)*"
        r"([A-Za-z_]\w*)\s*$",
        decl,
    )
    if not m:
        return None
    direction = m.group(1)
    width_expr = m.group(2)
    name = m.group(3)
    return (direction, name, _width_expr_to_text(width_expr))


def _extract_port_table(ref: str) -> List[Tuple[str, str, str]]:
    """Return [(direction, name, width_text), ...] for each public port.

    Strategy:
      1. Pull the `module foo(...)` body via _PORT_LIST_RE.
      2. If the body contains input/output keywords (ANSI style),
         comma-split each declaration and parse with _parse_port_decl.
      3. Otherwise (non-ANSI), scan the rest of the file for top-level
         input/output/inout declarations.

    Returns [] on parse failure — caller falls back to the old behaviour
    where the LLM is responsible for writing the port table itself.
    """
    m = _PORT_LIST_RE.search(ref)
    if not m:
        return []
    body = m.group(1)
    table: List[Tuple[str, str, str]] = []

    # strip Verilog comments before splitting
    body_clean = re.sub(r"//[^\n]*", "", body)
    body_clean = re.sub(r"/\*.*?\*/", "", body_clean, flags=re.DOTALL)

    if re.search(r"\b(input|output|inout)\b", body_clean):
        # ANSI: each port is one chunk between commas
        for chunk in body_clean.split(","):
            chunk = " ".join(chunk.split())  # collapse whitespace incl. newlines
            if not chunk:
                continue
            entry = _parse_port_decl(chunk)
            if entry is not None:
                table.append(entry)
        return table

    # non-ANSI fallback: scan from `module foo(...)` to `endmodule`
    mod_m = re.search(r"\bmodule\s+\w+", ref)
    end_m = re.search(r"\bendmodule\b", ref)
    if not mod_m:
        return []
    tail = ref[mod_m.end(): end_m.start() if end_m else None]
    # match each `input|output|inout [W:0]? name`
    for decl_m in re.finditer(
        r"\b(input|output|inout)\s+"
        r"(?:(?:reg|wire|logic|signed|unsigned)\s+)*"
        r"(?:\[([^\]]+)\])?\s*"
        r"([A-Za-z_]\w*)\s*[;,]",
        tail,
    ):
        direction = decl_m.group(1)
        width_expr = decl_m.group(2)
        name = decl_m.group(3)
        # skip if we accidentally re-matched a name twice (non-ANSI repeats)
        if any(t[1] == name for t in table):
            continue
        table.append((direction, name, _width_expr_to_text(width_expr)))
    return table


def _format_port_table_nl(
    table: List[Tuple[str, str, str]],
    design_name: str,
    reset: Optional[Tuple[str, str, str]] = None,
) -> str:
    """Render the port table as a clean NL block to prepend to the
    back-derived prompt. Includes module name + reset summary.

    Returns "" when ``table`` is empty — caller treats that as a signal
    to fall back to LLM-generated port phrasing (the previous behaviour).
    """
    if not table:
        return ""
    lines = [
        f"Module name: `{design_name}`",
        "Public ports:",
    ]
    for direction, name, width in table:
        lines.append(f"  - {direction} `{name}` ({width})")
    if reset is not None:
        sync, polarity, name = reset
        lines.append(f"Reset: `{name}` is {polarity} and {sync}.")
    return "\n".join(lines)


# ---------- reset-style detection (fix 2 for CVDP drift) --------------
#
# Drift study on CVDP found that when ref uses async reset, the inverted
# spec/prompt sometimes flips it to sync (or vice-versa). One word change
# but the cocotb test asserts reset between clock edges and expects
# immediate response — sync reset misses it. Same risk on active-high vs
# active-low. Both are byte-mechanical to detect from the ref.
_ASYNC_RESET_RE = re.compile(
    r"always(?:_ff)?\s*@\s*\(\s*"
    r"(?:posedge|negedge)\s+\w+\s*"
    r"or\s+(posedge|negedge)\s+(\w+)\s*\)",
    re.IGNORECASE,
)
_SYNC_RESET_HEAD_RE = re.compile(
    r"always(?:_ff)?\s*@\s*\(\s*posedge\s+\w+\s*\)",
    re.IGNORECASE,
)
_SYNC_RESET_TEST_NEG_RE = re.compile(r"if\s*\(\s*[!~]\s*(\w+)\s*\)")
_SYNC_RESET_TEST_POS_RE = re.compile(r"if\s*\(\s*(\w+)\s*\)")


def detect_reset_style(ref: str) -> Optional[Tuple[str, str, str]]:
    """Return (sync, polarity, name) tuple or None if no reset detected.

    sync     ∈ {"async", "sync"}
    polarity ∈ {"active-high", "active-low"}
    name     = identifier the ref uses (e.g. "rst_n")
    """
    m = _ASYNC_RESET_RE.search(ref)
    if m:
        edge = m.group(1).lower()
        name = m.group(2)
        polarity = "active-low" if edge == "negedge" else "active-high"
        return ("async", polarity, name)
    m2 = _SYNC_RESET_HEAD_RE.search(ref)
    if m2:
        body = ref[m2.end(): m2.end() + 800]
        m3 = _SYNC_RESET_TEST_NEG_RE.search(body)
        if m3:
            return ("sync", "active-low", m3.group(1))
        m3 = _SYNC_RESET_TEST_POS_RE.search(body)
        if m3 and m3.group(1).lower() not in {"clk", "clock", "i_valid",
                                                "valid", "en", "enable"}:
            return ("sync", "active-high", m3.group(1))
    return None


def _reset_constraint_text(reset: Optional[Tuple[str, str, str]]) -> str:
    """Build the line that gets appended to ``Public interface``."""
    if reset is None:
        return ""
    sync, polarity, name = reset
    return (f"\nReset behaviour: signal `{name}` is {polarity} and "
            f"{sync}; spec MUST state these EXACTLY.")


def lint_spec_reset_preserved(
    spec: str, reset: Optional[Tuple[str, str, str]]
) -> Tuple[bool, str]:
    """Hard check that spec preserves both reset polarity and sync style.

    Only checks if a reset was detected. Looks for the polarity word
    (``active-low`` / ``active-high``) AND the sync word (``async`` /
    ``synchronous``) in the spec text, case-insensitive. Either missing
    or contradicted -> fail.
    """
    if reset is None:
        return True, ""
    sync, polarity, _name = reset
    spec_low = spec.lower()
    pol_word = "active-low" if polarity == "active-low" else "active-high"
    pol_anti = "active-high" if polarity == "active-low" else "active-low"
    sync_word = "asynchron" if sync == "async" else "synchron"
    sync_anti = "synchron" if sync == "async" else "asynchron"

    if pol_word not in spec_low:
        return False, f"reset polarity missing: expected '{pol_word}'"
    if pol_anti in spec_low:
        return False, f"reset polarity contradicted: contains '{pol_anti}'"
    if sync_word not in spec_low:
        return False, f"reset sync missing: expected '{sync_word}'"
    if (sync == "async" and "synchronous" in spec_low
            and "asynchronous" not in spec_low):
        return False, "reset sync wrong: only 'synchronous' present"
    if sync == "sync" and sync_anti in spec_low:
        return False, f"reset sync contradicted: contains '{sync_anti}'"
    return True, ""


def _ref_identifiers(ref: str) -> Tuple[List[str], List[str]]:
    """Pull module names (excluding the top one) and internal
    wire/reg/logic names out of a Verilog reference.

    The top-level module name is part of the public interface and is
    NOT blacklisted. Names that appear in the top module's port list
    are also NOT blacklisted (they are public ports even if they have
    a `reg <name>` body declaration shadowing them — common pattern
    for registered outputs)."""
    if not ref:
        return [], []

    modules = _MODULE_DECL_RE.findall(ref)
    submodules = modules[1:] if len(modules) > 1 else []
    public_ports = _extract_port_names(ref)

    wire_names = []
    for m in _WIRE_DECL_RE.finditer(ref):
        nm = m.group(1)
        if (nm and nm not in _GENERIC_NAMES and nm not in public_ports
                and len(nm) > 2):
            wire_names.append(nm)

    # also pick up <UserModule> <inst_name> ( instantiation forms
    inst_names = []
    for m in _INSTANCE_DECL_RE.finditer(ref):
        mod, inst = m.group(1), m.group(2)
        if (mod in modules and mod != modules[0]
                and inst not in _GENERIC_NAMES
                and inst not in public_ports
                and len(inst) > 2):
            inst_names.append(inst)

    # dedup, keep order
    seen = set()
    blacklist_modules = [m for m in submodules
                         if m not in seen and not seen.add(m)]
    seen = set()
    blacklist_signals = [s for s in wire_names + inst_names
                         if s not in seen and not seen.add(s)]
    return blacklist_modules, blacklist_signals


def lint_spec_for_leaks(spec: str, ref: str) -> Tuple[bool, str]:
    """Static check that a spec doesn't leak ref's internal structure.

    Returns (ok, reason). ok=False means the spec mentions a submodule
    name, an internal signal name, or a banned structural phrase. The
    caller may re-roll spec generation on a non-ok result.
    """
    if not spec.strip():
        return False, "empty spec"

    blacklist_modules, blacklist_signals = _ref_identifiers(ref)
    spec_low = spec.lower()

    # 1. ref-derived identifiers — case-sensitive whole-word match
    leaked_ids = []
    for ident in blacklist_modules + blacklist_signals:
        # word-boundary match so 'add8' wouldn't match accidentally
        if re.search(r"\b" + re.escape(ident) + r"\b", spec):
            leaked_ids.append(ident)
    if leaked_ids:
        return False, f"leaked ref identifiers: {leaked_ids[:6]}"

    # 2. banned structural vocabulary — case-insensitive substring match
    leaked_words = [w for w in BANNED_STRUCTURAL_PHRASES
                    if w.lower() in spec_low]
    if leaked_words:
        return False, f"banned structural phrases: {leaked_words}"

    return True, ""


# ====================================================================== #
# KernelBench branch — steps 2–7 of docs/pipelines/kernelbench.md         #
# (interface extraction → understand → expanded prompt → implementation  #
# plan → 4-framing → consistency filter). Verilog branch above unchanged. #
# ====================================================================== #

# ---------- step 2: PyTorch interface extraction ----------------------

@dataclass
class PyTorchInterface:
    class_name: str
    init_args: List[str] = field(default_factory=list)
    forward_args: List[str] = field(default_factory=list)
    get_inputs_src: str = ""
    get_init_inputs_src: str = ""
    input_shapes_dtypes_hint: str = ""


def _func_def(tree: ast.AST, name: str) -> Optional[ast.FunctionDef]:
    """Find a top-level FunctionDef by name, or None."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_nn_module_class(node: ast.ClassDef) -> bool:
    """True iff the ClassDef has a base that resolves to nn.Module
    (either `nn.Module`, `Module`, or `torch.nn.Module`)."""
    for b in node.bases:
        # nn.Module
        if (isinstance(b, ast.Attribute) and b.attr == "Module"
                and isinstance(b.value, ast.Name) and b.value.id == "nn"):
            return True
        # bare Module (rare, but tolerate)
        if isinstance(b, ast.Name) and b.id == "Module":
            return True
        # torch.nn.Module
        if (isinstance(b, ast.Attribute) and b.attr == "Module"
                and isinstance(b.value, ast.Attribute)
                and b.value.attr == "nn"):
            return True
    return False


def _func_arg_names(fn: ast.FunctionDef) -> List[str]:
    """All positional / kw-only arg names for fn, excluding `self`."""
    out = [a.arg for a in fn.args.args if a.arg != "self"]
    out += [a.arg for a in fn.args.kwonlyargs if a.arg != "self"]
    return out


def _render_arg(node: ast.AST) -> str:
    """Render a single arg of a torch.<X>(...) call as a short string.
    Literals → their literal repr; Name nodes → the name verbatim;
    anything else → ast.unparse fallback or '?'."""
    try:
        return repr(ast.literal_eval(node))
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _render_torch_call(call: ast.Call) -> Optional[str]:
    """If `call` is `torch.<rand|randn|randint|zeros|ones|empty>(...)`,
    render a brief 'torch.rand(batch_size, dim) float32' style hint string.
    Otherwise return None."""
    if not (isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"):
        return None
    fn = call.func.attr
    if fn not in {"rand", "randn", "randint", "zeros", "ones", "empty",
                  "arange", "full"}:
        return None

    args_s = ", ".join(_render_arg(a) for a in call.args)

    # dtype kwarg
    dtype = None
    for kw in call.keywords:
        if kw.arg == "dtype":
            try:
                dtype = ast.unparse(kw.value)
            except Exception:
                dtype = None
            break
    base = f"torch.{fn}({args_s})"
    return f"{base} {dtype}" if dtype else f"{base} float32"


def _render_input_hint(get_inputs_fn: ast.FunctionDef) -> str:
    """Walk the body of get_inputs() and render each torch.<X>(...)
    call as one line. Best-effort; falls back to '(see source)' on
    anything we don't recognise."""
    hints: List[str] = []
    for node in ast.walk(get_inputs_fn):
        if isinstance(node, ast.Call):
            rendered = _render_torch_call(node)
            if rendered:
                hints.append(rendered)
    if not hints:
        return "(see get_inputs source above)"
    return "\n".join(f"- {h}" for h in hints)


def _extract_pytorch_interface(ref_src: str) -> Optional[PyTorchInterface]:
    """Parse a KernelBench reference source. Pull the class name + init/
    forward signatures + get_inputs / get_init_inputs sources + a
    rendered input shapes/dtypes hint.

    Returns None if any required piece is missing — caller surfaces a
    warn flag and falls back to passthrough rather than emitting a
    half-formed row.
    """
    if not ref_src or not ref_src.strip():
        return None
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return None

    # find first nn.Module subclass
    cls: Optional[ast.ClassDef] = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_nn_module_class(node):
            cls = node
            break
    if cls is None:
        return None

    init_fn = forward_fn = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init_fn = node
            elif node.name == "forward":
                forward_fn = node
    if forward_fn is None:
        return None

    init_args = _func_arg_names(init_fn) if init_fn is not None else []
    forward_args = _func_arg_names(forward_fn)

    get_inputs_fn = _func_def(tree, "get_inputs")
    get_init_inputs_fn = _func_def(tree, "get_init_inputs")
    if get_inputs_fn is None or get_init_inputs_fn is None:
        return None

    try:
        get_inputs_src = ast.unparse(get_inputs_fn)
        get_init_inputs_src = ast.unparse(get_init_inputs_fn)
    except Exception:
        return None

    return PyTorchInterface(
        class_name=cls.name,
        init_args=init_args,
        forward_args=forward_args,
        get_inputs_src=get_inputs_src,
        get_init_inputs_src=get_init_inputs_src,
        input_shapes_dtypes_hint=_render_input_hint(get_inputs_fn),
    )


# ---------- step 7 helpers: ref smoke test (subprocess) --------------

_REF_SMOKE_DRIVER = textwrap.dedent("""
    import importlib.util, sys, os, traceback, torch
    p = sys.argv[1]
    spec = importlib.util.spec_from_file_location("kb_ref_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        init_args = mod.get_init_inputs() if hasattr(mod, 'get_init_inputs') else []
        m = mod.Model(*init_args)
        inputs = mod.get_inputs()
        with torch.no_grad():
            y = m(*inputs)
    except Exception:
        traceback.print_exc()
        sys.exit(2)

    def _all_finite(x):
        if isinstance(x, torch.Tensor):
            return x.isfinite().all().item()
        if isinstance(x, (list, tuple)):
            return all(_all_finite(v) for v in x)
        return True

    if not _all_finite(y):
        print("non-finite tensor in output", file=sys.stderr)
        sys.exit(3)
    print("OK")
""").strip()


async def _ref_smoke_test(ref_src: str, timeout: int = 60) -> Tuple[bool, str]:
    """Run the ref: import → instantiate Model → run forward → assert
    output is finite. Returns (ok, err_tail).

    Routes through the persistent gate worker pool when KB_GATE_WORKERS>0
    (default); falls back to a per-call subprocess otherwise.

    Per-call cold subprocess torch import alone is ~10-15 s; the worker
    amortizes that across many calls.
    """
    if not ref_src.strip():
        return False, "empty source"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        ref_p = Path(tmp) / "ref_under_test.py"
        ref_p.write_text(ref_src)

        # --- Try persistent worker pool first ---
        from ._perturb_common import _get_gate_pool  # noqa: PLC0415
        pool = await _get_gate_pool()
        if pool is not None:
            res = await pool.call(
                "kb_smoke", timeout=timeout, ref_path=str(ref_p))
            if res.get("ok"):
                return True, ""
            return False, res.get("err", "")[-1024:]

        # --- Fallback: legacy subprocess driver ---
        try:
            from ._perturb_common import _next_gpu_env  # noqa: PLC0415
            env = _next_gpu_env()
        except Exception:
            env = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _REF_SMOKE_DRIVER, str(ref_p),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            return False, f"python launch failed: {e}"
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return False, f"smoke test timeout ({timeout}s)"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-1024:]
        return True, ""


# ---------- step 7: consistency filter ---------------------------------

_LEAK_TOKENS_KB = [
    "@triton.jit", "tl.program_id", "tl.load", "tl.store",
    "tl.arange", "tl.constexpr", "tl.dot", "tl.where",
    "load_inline", "__global__", "__device__", "cpp_extension",
    "torch.utils.cpp_extension",
    "num_warps", "num_stages",
    "tile_size", "block_size", "BLOCK_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K",
]

_FILTER_FAILED_PATH = Path(
    "data/expansion/pilot/kernelbench_inversecoder.filter_failed.jsonl")


async def filter_kb_row(row_record: dict, interface: PyTorchInterface,
                        ref_src: str) -> Tuple[bool, str]:
    """Structured 7-item check from docs/pipelines/kernelbench.md §7.

    `row_record` is the dict form of an Expanded row. The function does NOT
    mutate it; caller decides whether to emit / record-as-failed.
    """
    p = row_record.get("expanded_prompt", "")
    s = row_record.get("metadata", {}).get("assistant_spec", "")

    # 1. expanded_prompt 含 Triton/CUDA/kernel + ModelNew framing
    p_low = p.lower()
    if not (("triton" in p_low or "cuda" in p_low) and "modelnew" in p_low):
        return False, "prompt missing triton/modelnew framing"

    # 2. expanded_prompt embeds PyTorch reference (block) or names it
    has_ref_block = "```python" in p or "class Model" in p
    has_ref_word = bool(re.search(r"\b(reference|baseline)\b", p, re.I))
    if not (has_ref_block or has_ref_word):
        return False, "prompt does not present PyTorch reference"

    # 3. expanded_prompt lists each forward arg name
    missing = [a for a in interface.forward_args if a not in p]
    if missing:
        return False, f"prompt missing forward args: {missing}"

    # 4. assistant_spec carries high-level constraints
    if not s:
        return False, "spec is empty"
    if not re.search(r"ModelNew|forward|allclose|tolerance", s, re.I):
        return False, "spec missing high-level constraints"

    # 5. assistant_spec does not leak Triton-level impl tokens
    leaked = [t for t in _LEAK_TOKENS_KB if t in s]
    if leaked:
        return False, f"spec leaked impl tokens: {leaked[:3]}"

    # 6. interface non-None (defensive)
    if interface is None:
        return False, "interface is None"

    # 7. reference still imports + forwards + finite
    ok, err = await _ref_smoke_test(ref_src)
    if not ok:
        return False, f"ref smoke fail: {err[:160]}"

    return True, ""


def _append_filter_failed(row_record: dict, reason: str) -> None:
    """Persist a filter-rejected candidate to a side jsonl for debugging.
    Best-effort — never raises into caller."""
    try:
        _FILTER_FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = dict(row_record)
        record.setdefault("metadata", {})
        record["metadata"]["filter_reject_reason"] = reason
        with _FILTER_FAILED_PATH.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------- KB prompt templates ---------------------------------------

UNDERSTAND_SYSTEM_KB = (
    "You read a PyTorch nn.Module reference and write a 2-4 sentence "
    "BLACK-BOX behavioural summary in plain English.\n\n"
    "HARD RULES:\n"
    "  - Mention: input tensor shapes + dtypes, the forward operator "
    "chain (matmul / reduction / broadcast / activation / normalisation "
    "/ indexing etc.), output shape, numerical-precision considerations.\n"
    "  - Do NOT name an implementation language (do not say 'PyTorch', "
    "'Triton', or 'CUDA').\n"
    "  - Do NOT describe internal partitioning, kernel count, tile size, "
    "shared memory, scheduling, etc.\n"
    "  - When you reference an interface name, use the EXACT name listed "
    "under `Public interface` below.\n\n"
    "Output 2-4 plain-English sentences. No bullet lists, no headings, "
    "no code, no implementation language."
)

UNDERSTAND_USER_KB = (
    "Class name: {class_name}\n\n"
    "Public interface (use these EXACT names):\n"
    "---\n"
    "__init__ args: {init_args}\n"
    "forward args:  {forward_args}\n"
    "inputs (shape/dtype hints):\n{input_shapes_dtypes_hint}\n"
    "---\n\n"
    "Reference (for your eyes only — do NOT echo or quote):\n"
    "---\n{ref_src}\n---\n\n"
    "Write the black-box behavioural summary now."
)

BACK_DERIVE_SYSTEM_KB = (
    "You write a NL request that a hardware-aware engineer would send to "
    "an LLM coding assistant. The expected answer is (a) a high-level "
    "implementation plan and (b) a Triton / CUDA `class ModelNew(nn.Module)` "
    "implementation. Imagine the request that produces that answer.\n\n"
    "MUST INCLUDE in the request:\n"
    "  (a) The PyTorch baseline / reference is given (embed the code "
    "      block, or clearly say 'the PyTorch reference is provided below' "
    "      and embed it). Use the exact reference shown to you below.\n"
    "  (b) Implement `class ModelNew(nn.Module)` whose `forward` "
    "      signature matches the reference EXACTLY (same parameter "
    "      names, in the same order).\n"
    "  (c) The hot compute path MUST live inside one or more "
    "      `@triton.jit` kernels (or `load_inline` CUDA). It must NOT "
    "      be pure PyTorch.\n"
    "  (d) Output tensors must match the PyTorch reference under "
    "      `torch.allclose(atol=1e-2, rtol=1e-2)`.\n"
    "  (e) Spell out the input tensor shapes / dtypes / parameter "
    "      names (mirror what `get_inputs()` returns).\n\n"
    "Express the request in this framing: **{framing}** — {framing_desc}. "
    "Whatever the framing, all five items above must remain present "
    "and visible.\n\n"
    "MUST NOT:\n"
    "  - Echo the behavioural summary verbatim.\n"
    "  - Mention any low-level Triton implementation token: "
    "`@triton.jit` body, `tl.program_id`, `tl.load`, `tl.store`, "
    "`tl.arange`, tile size, block size, num_warps, num_stages.\n"
    "  - Use the words 'spec' / 'specification' / 'design draft'. The "
    "user is asking for an implementation, not a spec.\n\n"
    "Output the request text only. No code fences except the embedded "
    "reference block. No commentary outside the request."
)

BACK_DERIVE_USER_KB = (
    "Class name: {class_name}\n"
    "Public interface:\n"
    "---\n"
    "__init__ args: {init_args}\n"
    "forward args:  {forward_args}\n"
    "inputs:\n{input_shapes_dtypes_hint}\n"
    "---\n\n"
    "PyTorch reference (embed this in the request):\n"
    "```python\n{ref_src}\n```\n\n"
    "Behavioural summary (background only — paraphrase or use to "
    "structure the request, do not echo it verbatim):\n"
    "---\n{summary}\n---\n\n"
    "Now write the user's NL request."
)

PLAN_SYSTEM_KB = (
    "You are a Triton-kernel engineer's design partner. Given a user "
    "request (PyTorch → Triton task) and the PyTorch reference, write a "
    "high-level implementation plan. This plan will be the OPENING of "
    "an assistant reply, BEFORE the actual code — it teaches the model "
    "to think about the task before it writes Triton.\n\n"
    "The plan MUST cover:\n"
    "  1. The math behaviour (one-line summary).\n"
    "  2. The ModelNew interface contract: class name, forward "
    "signature (parameter names + shape/dtype), output shape/dtype.\n"
    "  3. Which computations live inside Triton/CUDA kernels — at the "
    "level of 'matmul main loop' / 'softmax two-pass scan' / 'fused "
    "reduction'. NOT at the level of per-thread arithmetic.\n"
    "  4. Output shape and dtype.\n"
    "  5. Correctness tolerance: torch.allclose(atol=1e-2, rtol=1e-2).\n\n"
    "The plan MUST NOT contain:\n"
    "  - Any Triton implementation token: @triton.jit, tl.program_id, "
    "tl.load, tl.store, tl.arange, tl.constexpr, tl.dot.\n"
    "  - tile size / block size / num_warps / num_stages / kernel "
    "launch grid configuration.\n"
    "  - load_inline, cpp_extension, __global__, __device__.\n"
    "  - Per-thread/block/grid index arithmetic.\n\n"
    "Output 5-15 lines of plain English or simple markdown. No code "
    "fences. No section heading prefixes like '## Plan'."
)

PLAN_USER_KB = (
    "User request:\n"
    "---\n{expanded_prompt}\n---\n\n"
    "PyTorch reference:\n"
    "```python\n{ref_src}\n```\n\n"
    "Write the implementation plan."
)


# ---------- the method -------------------------------------------------

@register_method("inversecoder")
class InverseCoderExpansion:
    """Spec-as-CoT inversecoder. Per seed:
      1. understand_reference  — black-box behaviour summary (LLM, once
         per seed; feeds spec generation only).
      2. for each variant:
           a. generate_spec from summary + framing (LLM)
           b. lint each spec for structural leaks; re-roll once if fail.
           c. back_derive_prompt from (spec, ref_code) (LLM); lint the
              new prompt for the same internal-name leaks; re-roll once.
      3. emit row: ``expanded_prompt`` is the back-derived NL,
         ``metadata.assistant_spec`` carries the spec for the SFT
         converter to prepend before the reference code.

    The back-derive step deliberately NEVER reads ``original_prompt``
    — the drift metric (fresh on original_prompt vs fresh on
    expanded_prompt) must not be biased by the pipeline.
    """

    name = "inversecoder"
    MAX_LINT_ATTEMPTS = 2

    async def expand(self, seed: Seed, llm, num_variants: int = 3,
                     **_kw) -> List[Expanded]:
        # Wrap a bare LLM in a single-prompt router for backward compat.
        router = (llm if isinstance(llm, LLMRouter)
                  else LLMRouter(traj_llm=llm, prompt_llms=[llm]))

        if seed.source_dataset == "kernelbench":
            return await self._expand_kb(seed, router, num_variants)

        if seed.source_dataset == "tritonbench_g":
            return await self._expand_tbg(seed, router, num_variants)

        if seed.source_dataset == "tritonbench_t":
            return await self._expand_tbt(seed, router, num_variants)

        profile = _profile_for(seed)

        ref = seed.reference_solution or ""
        if not ref.strip():
            # No ref to inverse-engineer; emit a passthrough record so
            # the seed isn't silently dropped.
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "no_reference_solution",
                            "framing": "passthrough"})]

        design_name = (seed.evaluator_info.get("design_name")
                       or seed.evaluator_info.get("top_module")
                       or seed.metadata.get("design_name") or "TopModule")
        port_list = _extract_port_list_text(ref)
        if not port_list:
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "no_port_list_parsed",
                            "framing": "passthrough"})]

        if profile.enable_reset_lint:
            reset_style = detect_reset_style(ref)
            port_list = port_list + _reset_constraint_text(reset_style)
        else:
            reset_style = None

        ref = force_module_name(ref, design_name)

        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)

        # Step 2: behaviour summary (once per seed). prompt-side step
        # (not visible in SFT assistant), uses any prompt model.
        summary_raw = await router.chat_prompt(
            profile.understand_system,
            profile.understand_user.format(design_name=design_name,
                                           port_list=port_list, code=ref),
            seed_hash=seed_hash, variant_idx=0,
        )
        summary = strip_dryrun(summary_raw).strip()
        if not summary:
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "empty_understanding",
                            "framing": "passthrough"})]

        # Steps 3-4-5-6 per (framing × prompt_variant). spec is generated
        # ONCE per framing on the traj model (it's part of the SFT
        # assistant CoT). back-derive then runs M times per framing,
        # one per prompt-side model, so the (USER, ASSISTANT) pairs
        # diverge on the user side.
        framings = [profile.framings[i % len(profile.framings)]
                    for i in range(num_variants)]
        coros = [self._generate_one(seed, summary, design_name,
                                    port_list, ref, framing, f_idx,
                                    router, seed_hash,
                                    reset_style, profile)
                 for f_idx, framing in enumerate(framings)]
        nested = await asyncio.gather(*coros)
        return [r for rs in nested for r in (rs or [])]

    async def _generate_one(self, seed: Seed, summary: str,
                            design_name: str, port_list: str,
                            ref: str, framing: Tuple[str, str],
                            f_idx: int, router: LLMRouter,
                            seed_hash: int,
                            reset_style: Optional[Tuple[str, str, str]],
                            profile: _InvProfile) -> List[Expanded]:
        f_name, f_desc = framing

        # ---- Steps 3-5: spec gen on TRAJ side (single fixed model) ----
        # spec is part of the SFT assistant CoT, so it must come from
        # ONE traj model. We capture the v4-pro raw content so the SFT
        # pack can construct the assistant as
        # ``<think>{spec_think}</think><answer>{spec}\n```verilog\n{ref}\n```</answer>``.
        spec, spec_traj_content, spec_lint_warn, spec_lint_attempts = (
            await self._generate_spec(summary, design_name, port_list,
                                      ref, framing, router, reset_style,
                                      profile))
        if spec is None:
            return []

        # ---- Step 6: back-derive M times, one per prompt model -------
        m = router.num_prompt_models
        rows: List[Expanded] = []
        for variant_idx in range(m):
            nl_prompt, nl_lint_warn, nl_lint_attempts = (
                await self._back_derive_prompt(
                    spec, ref, design_name, router,
                    seed_hash, variant_idx,
                    reset_style, profile))
            if not nl_prompt:
                continue

            row_idx = f_idx * m + variant_idx
            meta = {
                "framing": f_name,
                "framing_idx": f_idx,
                "variant_idx": variant_idx,
                "behaviour_summary": summary,
                "spec_lint_attempts": spec_lint_attempts,
                "nl_lint_attempts": nl_lint_attempts,
                "assistant_spec": spec,
                "spec_traj_content": spec_traj_content,
                "ref_for_assistant": ref,
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
            }
            if reset_style is not None:
                sync, polarity, name = reset_style
                meta["reset_style"] = f"{name}:{polarity}:{sync}"
            warns = [w for w in (spec_lint_warn, nl_lint_warn) if w]
            if warns:
                meta["warn"] = ";".join(warns)
            rows.append(base_expanded(
                seed, "inversecoder", row_idx, nl_prompt,
                extra_meta=meta))
        return rows

    async def _generate_spec(self, summary: str, design_name: str,
                             port_list: str, ref: str,
                             framing: Tuple[str, str],
                             router: LLMRouter,
                             reset_style: Optional[Tuple[str, str, str]],
                             profile: _InvProfile
                             ) -> Tuple[Optional[str], str, str, int]:
        """Run steps 3-5 on the TRAJ model: generate a spec under the
        given framing, lint for structural leaks AND (when CVDP profile)
        reset-style preservation, re-roll once on lint fail.

        Returns (spec_extracted, raw_traj_content, warn, attempts).
        spec_extracted is None only if every attempt produced empty
        text. raw_traj_content is the full v4-pro response (including
        ``<think>...</think><answer>{spec}</answer>`` wrap) so the SFT
        pack stage can reconstruct the assistant.
        """
        f_name, f_desc = framing
        last_lint_reason = ""
        last_traj_content = ""
        spec = ""
        for attempt in range(self.MAX_LINT_ATTEMPTS):
            user = profile.spec_user.format(
                framing=f_name, framing_desc=f_desc,
                summary=summary, design_name=design_name,
                port_list=port_list)
            if last_lint_reason:
                user += (
                    "\n\nYour previous attempt failed a lint check:\n"
                    f"  {last_lint_reason}\n"
                    "Re-write the specification: avoid internal submodule "
                    "names, intermediate signal names, or implementation "
                    "structure. Describe only externally observable "
                    "behaviour. Keep public port names exact.")
                if profile.enable_reset_lint:
                    user += (
                        " If the Reset behaviour line above pins polarity "
                        "/ asynchronous-vs-synchronous, your spec MUST "
                        "contain those exact words verbatim.")

            resp = await router.chat_traj_full(profile.spec_system, user)
            raw_content = resp.content or ""
            wrapped = synthesize_v4pro_wrap(
                raw_content, resp.reasoning_content,
                fallback_lang="verilog")
            last_traj_content = wrapped
            ok_fmt, _reason = check_v4pro_format(raw_content)
            if not ok_fmt:
                last_lint_reason = "v4pro format truncated"
                continue
            # Peel <answer> wrap; spec text is what's inside.
            spec = extract_v4pro_answer(strip_dryrun(raw_content)).strip()
            if not spec:
                last_lint_reason = "empty spec"
                continue

            ok, reason = lint_spec_for_leaks(spec, ref)
            if not ok:
                last_lint_reason = reason
                continue
            if profile.enable_reset_lint:
                ok, reason = lint_spec_reset_preserved(spec, reset_style)
                if not ok:
                    last_lint_reason = reason
                    continue
            return spec, wrapped, "", attempt + 1

        if not spec.strip():
            return None, last_traj_content, "", self.MAX_LINT_ATTEMPTS
        return (spec, last_traj_content,
                f"spec_lint_failed:{last_lint_reason}",
                self.MAX_LINT_ATTEMPTS)

    async def _back_derive_prompt(self, spec: str, ref: str,
                                  design_name: str, router: LLMRouter,
                                  seed_hash: int, variant_idx: int,
                                  reset_style: Optional[Tuple[str, str, str]],
                                  profile: _InvProfile
                                  ) -> Tuple[str, str, int]:
        """Run step 6: produce the new user-side NL prompt from
        (spec, ref_code). PROMPT-SIDE step — uses the prompt model
        selected by ``variant_idx`` (rotates through router.prompt_llms),
        so M variants produce M different USER prompts for the same
        ASSISTANT (spec + ref).

        The public port table is computed deterministically from the
        ref and PREPENDED to whatever the LLM writes — Path B from
        docs/inversecoder_design.md.

        Lints (internal-name leak + optional reset-style preservation)
        run on the combined prepend + prose; one re-roll on lint fail.

        Returns (nl_prompt, warn, attempts). nl_prompt may be empty
        string only if every LLM attempt returned blank — caller drops
        the row in that case.
        """
        # Deterministic port-table block. Empty if parser fails — fall
        # back to the old behaviour (LLM writes the port list itself).
        port_table = _extract_port_table(ref)
        port_block = _format_port_table_nl(
            port_table, design_name, reset_style)
        port_block_warn = "" if port_block else "port_table_parse_failed"

        last_lint_reason = ""
        nl_prompt = ""
        for attempt in range(self.MAX_LINT_ATTEMPTS):
            user = BACK_DERIVE_USER.format(
                spec=spec, ref_code=ref,
                port_block=port_block or "(none — describe ports in your prose)")
            if last_lint_reason:
                user += (
                    "\n\nYour previous attempt failed a lint check:\n"
                    f"  {last_lint_reason}\n"
                    "Re-write the request: no internal signal name, "
                    "register name, sub-module name, or hierarchy "
                    "detail from the reference. Public port names are "
                    "the only identifiers allowed. Keep the request "
                    "high-level and shorter than the spec.")
                if profile.enable_reset_lint:
                    user += (
                        " If the spec states reset polarity or "
                        "asynchronous/synchronous behaviour, your "
                        "request MUST repeat those exact words.")

            nl_raw = await router.chat_prompt(
                profile.back_derive_system, user,
                seed_hash=seed_hash, variant_idx=variant_idx)
            prose = strip_dryrun(nl_raw).strip()
            if not prose:
                last_lint_reason = "empty nl prompt"
                continue
            # Combine deterministic port block + LLM-written prose.
            nl_prompt = (f"{port_block}\n\n{prose}"
                         if port_block else prose)

            ok, reason = lint_spec_for_leaks(nl_prompt, ref)
            if not ok:
                last_lint_reason = reason
                continue
            if profile.enable_reset_lint:
                ok, reason = lint_spec_reset_preserved(nl_prompt, reset_style)
                if not ok:
                    last_lint_reason = reason
                    continue
            warn = port_block_warn
            return nl_prompt, warn, attempt + 1

        if not nl_prompt.strip():
            return "", port_block_warn, self.MAX_LINT_ATTEMPTS
        warn_parts = [w for w in
                      (port_block_warn,
                       f"nl_lint_failed:{last_lint_reason}") if w]
        return nl_prompt, ";".join(warn_parts), self.MAX_LINT_ATTEMPTS

    # ---------- KernelBench branch (per docs/pipelines/kernelbench.md) ----

    async def _expand_kb(self, seed: Seed, router: LLMRouter,
                         num_variants: int) -> List[Expanded]:
        ref = seed.reference_solution or ""
        if not ref.strip():
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "no_reference_solution",
                            "framing": "passthrough"})]

        interface = _extract_pytorch_interface(ref)
        if interface is None:
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "kb_interface_extraction_failed",
                            "framing": "passthrough"})]

        seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)

        # Step 3: black-box understanding (once per seed) — prompt-side.
        # KB summaries don't appear in the SFT assistant; teacher_triton
        # _rollout uses ``assistant_spec`` (= plan) as the in-prompt
        # scaffold, not the summary.
        summary_raw = await router.chat_prompt(
            UNDERSTAND_SYSTEM_KB,
            UNDERSTAND_USER_KB.format(
                class_name=interface.class_name,
                init_args=interface.init_args or "(none)",
                forward_args=interface.forward_args,
                input_shapes_dtypes_hint=interface.input_shapes_dtypes_hint,
                ref_src=ref),
            seed_hash=seed_hash, variant_idx=0,
        )
        summary = strip_dryrun(summary_raw).strip()
        if not summary:
            return [base_expanded(
                seed, "inversecoder", 0, seed.original_prompt,
                extra_meta={"warn": "empty_understanding",
                            "framing": "passthrough"})]

        # Steps 4-7 per (framing × prompt_variant). Both back_derive and
        # plan are PROMPT-SIDE (the SFT assistant for KB comes from
        # teacher_triton_rollout's v4-pro response, not from these
        # steps). Different (framing, variant) → different (NL, plan)
        # pairs, each fed independently into the downstream Triton
        # rollout.
        m = router.num_prompt_models
        framings = [INVERSE_FRAMINGS[i % len(INVERSE_FRAMINGS)]
                    for i in range(num_variants)]
        coros = [self._generate_one_kb(seed, summary, interface,
                                       ref, framing, f_idx,
                                       variant_idx, router, seed_hash)
                 for f_idx, framing in enumerate(framings)
                 for variant_idx in range(m)]
        rows = await asyncio.gather(*coros)
        return [r for r in rows if r is not None]

    async def _generate_one_kb(self, seed: Seed, summary: str,
                               interface: PyTorchInterface, ref: str,
                               framing: Tuple[str, str],
                               f_idx: int, variant_idx: int,
                               router: LLMRouter, seed_hash: int
                               ) -> Optional[Expanded]:
        m = router.num_prompt_models
        # Step 4 + 6: expanded prompt under the chosen framing — prompt-side.
        nl = await self._back_derive_kb(
            interface, ref, summary, framing, router,
            seed_hash, variant_idx)
        if not nl:
            return None

        # Step 5: implementation plan — prompt-side (becomes the
        # ``assistant_spec`` scaffold that teacher_triton_rollout sees
        # in its user prompt; not in the SFT assistant directly).
        plan = await self._plan_kb(nl, ref, router, seed_hash, variant_idx)
        if not plan:
            return None

        # Step 7: structured consistency filter
        row_idx = f_idx * m + variant_idx
        candidate = base_expanded(
            seed, "inversecoder", row_idx, nl,
            extra_meta={
                "framing": framing[0],
                "framing_idx": f_idx,
                "variant_idx": variant_idx,
                "behaviour_summary": summary,
                "assistant_spec": plan,
                "pytorch_interface": asdict(interface),
                "traj_model": router.traj_model_name,
                "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
            })
        rec = candidate.to_record()
        ok, reason = await filter_kb_row(rec, interface, ref)
        if not ok:
            _append_filter_failed(rec, reason)
            return None
        return candidate

    async def _back_derive_kb(self, interface: PyTorchInterface,
                              ref: str, summary: str,
                              framing: Tuple[str, str],
                              router: LLMRouter,
                              seed_hash: int, variant_idx: int) -> str:
        f_name, f_desc = framing
        user = BACK_DERIVE_USER_KB.format(
            class_name=interface.class_name,
            init_args=interface.init_args or "(none)",
            forward_args=interface.forward_args,
            input_shapes_dtypes_hint=interface.input_shapes_dtypes_hint,
            ref_src=ref,
            summary=summary)
        system = BACK_DERIVE_SYSTEM_KB.format(
            framing=f_name, framing_desc=f_desc)
        raw = await router.chat_prompt(
            system, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()

    async def _plan_kb(self, expanded_prompt: str, ref: str,
                       router: LLMRouter,
                       seed_hash: int, variant_idx: int) -> str:
        user = PLAN_USER_KB.format(
            expanded_prompt=expanded_prompt, ref_src=ref)
        raw = await router.chat_prompt(
            PLAN_SYSTEM_KB, user,
            seed_hash=seed_hash, variant_idx=variant_idx)
        return strip_dryrun(raw).strip()


# ======================================================================
#                       TritonBench-G branch
# ======================================================================
#
# Why a separate branch (not just a TBG_PROFILE):
#   - Interface extraction is Triton-AST, not Verilog regex (LEGACY) and
#     not PyTorch class+forward (KB).
#   - Spec leak-check tokens differ (no ModelNew check; we still ban
#     low-level tl.* and BLOCK_SIZE).
#   - Filter has only 6 items vs KB's 7 (no ModelNew framing), and the
#     ref smoke is `result_gold = test_xxx()` rather than
#     `Model(*get_init_inputs())(*get_inputs())`.
#   - Most importantly: ref language == target language. Ref Triton is
#     the SFT answer verbatim; no teacher rollout downstream. Same shape
#     as RTLLM/VEval but operator-level instead of nn.Module.

@dataclass
class TritonInterface:
    """Parsed Triton ref interface.

    A G-split file has one or more `@triton.jit` kernels plus a Python
    wrapper that launches the kernel and returns a tensor. The test
    block (`def test_xxx()`) calls the wrapper, never the kernel
    directly. We capture both names plus the wrapper signature so the
    back-derive prompt can constrain the user-side request.
    """
    kernel_names: List[str] = field(default_factory=list)
    wrapper_names: List[str] = field(default_factory=list)
    primary_wrapper: str = ""
    primary_signature: str = ""        # e.g. "puzzle1(x: torch.Tensor)"
    primary_arg_names: List[str] = field(default_factory=list)
    test_func_name: str = ""           # e.g. "test_puzzle"
    has_class_wrapper: bool = False    # True if ref uses class-based API


def _is_triton_jit_decorator(d: ast.expr) -> bool:
    # Plain forms: `@triton.jit`, `@jit`
    if isinstance(d, ast.Attribute) and d.attr == "jit":
        v = d.value
        if isinstance(v, ast.Name) and v.id == "triton":
            return True
    if isinstance(d, ast.Name) and d.id == "jit":
        return True
    # Call forms: `@triton.jit()`, `@triton.jit(launch_metadata=...)`,
    # `@triton.jit(do_not_specialize=[...])`. Same identity, just
    # parameterised — recurse into d.func.
    if isinstance(d, ast.Call):
        return _is_triton_jit_decorator(d.func)
    return False


def _func_signature_text(fn: ast.FunctionDef) -> str:
    try:
        return f"{fn.name}({ast.unparse(fn.args)})".replace("\n", " ")
    except Exception:
        return fn.name + "(...)"


def _function_calls_in(fn: ast.FunctionDef) -> List[str]:
    """Return the dotted names of every Call node in fn body
    (e.g. ['add_wrapper', 'puzzle1_kernel[grid]'])."""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            out.append(f.id)
        elif isinstance(f, ast.Attribute):
            try:
                out.append(ast.unparse(f))
            except Exception:
                pass
        elif isinstance(f, ast.Subscript):
            # kernel[grid](...) form — kernel launches
            try:
                out.append(ast.unparse(f.value))
            except Exception:
                pass
    return out


def _extract_triton_interface(ref_src: str) -> Optional[TritonInterface]:
    """Parse a TritonBench-G ref. Pull kernel names, wrapper names,
    primary wrapper signature (whichever wrapper is called from the
    test function), and test_xxx name.

    Returns None if no @triton.jit kernel found or no test function
    found — both indicate a malformed seed.
    """
    if not ref_src or not ref_src.strip():
        return None
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return None

    kernel_names: List[str] = []
    candidate_wrappers: Dict[str, ast.FunctionDef] = {}
    test_fn: Optional[ast.FunctionDef] = None
    has_class_wrapper = False

    # walk top-level + class-body funcdefs
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            is_jit = any(_is_triton_jit_decorator(d) for d in node.decorator_list)
            if is_jit:
                kernel_names.append(node.name)
            else:
                if node.name.startswith("test_"):
                    test_fn = node
                else:
                    candidate_wrappers[node.name] = node
        elif isinstance(node, ast.ClassDef):
            # class-based wrapper (e.g. nn.Module)
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    if any(_is_triton_jit_decorator(d) for d in sub.decorator_list):
                        kernel_names.append(f"{node.name}.{sub.name}")
                    elif sub.name == "forward":
                        has_class_wrapper = True
                        candidate_wrappers[f"{node.name}.forward"] = sub

    if not kernel_names:
        return None
    if test_fn is None:
        return None

    # decide primary wrapper: pick the candidate name called from test_fn
    test_calls = _function_calls_in(test_fn)
    primary_wrapper = ""
    primary_fn: Optional[ast.FunctionDef] = None
    for name, fn in candidate_wrappers.items():
        # match by leaf name (`Cls.forward` matched if test calls `cls(...)`
        # via instantiation — best-effort, prefer direct name match)
        leaf = name.split(".")[-1]
        if name in test_calls or leaf in test_calls:
            primary_wrapper = name
            primary_fn = fn
            break
    # fallback: first wrapper that is NOT a kernel
    if primary_fn is None and candidate_wrappers:
        primary_wrapper, primary_fn = next(iter(candidate_wrappers.items()))

    if primary_fn is None:
        # extreme corner: only kernels and a test, no wrapper. Fall
        # back to the first kernel (the test must be calling the
        # kernel directly via name[grid](...) — rare).
        primary_wrapper = kernel_names[0]
        primary_signature = primary_wrapper + "(...)"
        primary_arg_names = []
    else:
        primary_signature = _func_signature_text(primary_fn)
        primary_arg_names = _func_arg_names(primary_fn)

    return TritonInterface(
        kernel_names=kernel_names,
        wrapper_names=list(candidate_wrappers.keys()),
        primary_wrapper=primary_wrapper,
        primary_signature=primary_signature,
        primary_arg_names=primary_arg_names,
        test_func_name=test_fn.name,
        has_class_wrapper=has_class_wrapper,
    )


# ---------- TBG ref smoke test (subprocess) ---------------------------

# Run the ref unchanged. The ref's last statement is
# `result_gold = test_xxx()`; if that returns / completes without
# exception, the ref is healthy.
_TBG_REF_SMOKE_DRIVER = textwrap.dedent("""
    import importlib.util, sys, traceback
    p = sys.argv[1]
    spec = importlib.util.spec_from_file_location("tbg_ref_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    rg = getattr(mod, 'result_gold', None)
    if rg is None:
        print('result_gold attribute missing', file=sys.stderr)
        sys.exit(3)
    print('OK')
""").strip()


async def _tbg_ref_smoke_test(ref_src: str,
                              timeout: int = 90) -> Tuple[bool, str]:
    """Run the ref in an isolated subprocess: import → execute module
    body (which runs `result_gold = test_xxx()`). 90s default — Triton
    JIT compile of complex kernels (lightning attention etc.) can take
    30-60s on first hit.
    """
    if not ref_src.strip():
        return False, "empty source"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        ref_p = Path(tmp) / "ref_under_test.py"
        ref_p.write_text(ref_src)
        try:
            from ._perturb_common import _next_gpu_env  # noqa: PLC0415
            env = _next_gpu_env()
        except Exception:
            env = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _TBG_REF_SMOKE_DRIVER, str(ref_p),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as e:
            return False, f"python launch failed: {e}"
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return False, f"smoke timeout ({timeout}s)"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-1024:]
        return True, ""


# ---------- TBG consistency filter ------------------------------------

# Same as KB but without the ModelNew framing requirement (TBG is
# operator-level). We still ban low-level Triton tokens in the spec
# (the spec is part of the SFT assistant CoT; leaking BLOCK_SIZE there
# defeats the spec-as-CoT scaffold).
_LEAK_TOKENS_TBG = [
    "@triton.jit", "tl.program_id", "tl.load", "tl.store",
    "tl.arange", "tl.constexpr", "tl.dot", "tl.where",
    "num_warps", "num_stages",
    "tile_size", "block_size", "BLOCK_SIZE", "BLOCK_M", "BLOCK_N",
    "BLOCK_K", "BLOCK_P",
]

_FILTER_FAILED_PATH_TBG = Path(
    "data/expansion/pilot/tritonbench_g_inversecoder.filter_failed.jsonl")


async def filter_tbg_row(row_record: dict, interface: TritonInterface,
                         ref_src: str,
                         skip_smoke: bool = False) -> Tuple[bool, str]:
    """6-item structured check (KB has 7; TBG drops the ModelNew check).

    skip_smoke=True is the per-framing fast path: the caller already
    ran ``_tbg_ref_smoke_test`` once for the seed in ``_expand_tbg``
    (the ref doesn't change across framings, so re-running it 16x
    serialises Triton compile on a single GPU and blows past the
    expand_data.py 600s per-seed timeout — see logs/tbg_full/run.log
    where every seed timed out at >600s before this hoist).
    """
    p = row_record.get("expanded_prompt", "")
    s = row_record.get("metadata", {}).get("assistant_spec", "")

    # 1. expanded_prompt mentions Triton + the kernel/wrapper interface
    p_low = p.lower()
    if "triton" not in p_low:
        return False, "prompt missing triton mention"

    # 2. expanded_prompt names the primary entry point (kernel or wrapper)
    if interface.primary_wrapper:
        leaf = interface.primary_wrapper.split(".")[-1]
        if leaf not in p:
            return False, f"prompt missing entry-point name: {leaf}"

    # 3. expanded_prompt names each wrapper arg (lets test harness wire
    #    candidate to the same call site)
    missing = [a for a in (interface.primary_arg_names or [])
               if a not in p and a not in ("self",)]
    if missing:
        return False, f"prompt missing wrapper args: {missing}"

    # 4. assistant_spec is non-empty and carries shape/dtype constraint
    if not s:
        return False, "spec is empty"
    if not re.search(r"shape|dtype|tensor|input|output", s, re.I):
        return False, "spec missing shape/dtype constraints"

    # 5. assistant_spec does not leak Triton-level impl tokens
    leaked = [t for t in _LEAK_TOKENS_TBG if t in s]
    if leaked:
        return False, f"spec leaked impl tokens: {leaked[:3]}"

    # 6. reference still imports + runs `result_gold = test_xxx()`
    if not skip_smoke:
        ok, err = await _tbg_ref_smoke_test(ref_src)
        if not ok:
            return False, f"ref smoke fail: {err[:160]}"

    return True, ""


def _append_filter_failed_tbg(row_record: dict, reason: str) -> None:
    try:
        _FILTER_FAILED_PATH_TBG.parent.mkdir(parents=True, exist_ok=True)
        record = dict(row_record)
        record.setdefault("metadata", {})
        record["metadata"]["filter_reject_reason"] = reason
        with _FILTER_FAILED_PATH_TBG.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------- TBG prompt templates --------------------------------------

UNDERSTAND_SYSTEM_TBG = (
    "You read a Triton kernel reference and write a 2-4 sentence "
    "BLACK-BOX behavioural summary in plain English.\n\n"
    "HARD RULES:\n"
    "  - Mention: input tensor shapes + dtypes, the operation "
    "performed (matmul / reduction / scan / elementwise / attention "
    "etc.), output shape/dtype. State numerical-precision concerns "
    "if any (fp16 accumulation, mixed-precision reductions, etc.).\n"
    "  - Use the EXACT entry-point name listed under `Public "
    "interface` (the wrapper or kernel function the test harness "
    "calls).\n"
    "  - Do NOT name the implementation language (do not say 'Triton', "
    "'CUDA', 'GPU').\n"
    "  - Do NOT describe internal partitioning, block size, grid "
    "shape, kernel count, shared memory, scheduling, num_warps.\n\n"
    "Output 2-4 plain-English sentences. No bullet lists, no headings, "
    "no code, no implementation language."
)

UNDERSTAND_USER_TBG = (
    "Public interface (use these EXACT names):\n"
    "---\n"
    "kernels:           {kernel_names}\n"
    "primary entry:     {primary_wrapper}\n"
    "entry signature:   {primary_signature}\n"
    "test function:     {test_func_name}\n"
    "---\n\n"
    "Reference (for your eyes only — do NOT echo or quote):\n"
    "---\n{ref_src}\n---\n\n"
    "Write the black-box behavioural summary now."
)

BACK_DERIVE_SYSTEM_TBG = (
    "You write a NL request that an engineer would send to an LLM "
    "coding assistant. The expected answer is (a) a high-level "
    "implementation plan and (b) a Triton kernel implementation that "
    "matches a fixed entry-point signature so an external test harness "
    "can invoke it. Imagine the request that produces that answer.\n\n"
    "MUST INCLUDE in the request:\n"
    "  (a) The TARGET kernel must be implemented in Triton — name the "
    "      language explicitly. The compute must live inside one or "
    "      more `@triton.jit` kernels (mention the @triton.jit "
    "      requirement in plain English; do not write the literal "
    "      '@triton.jit' token).\n"
    "  (b) The entry-point function the test harness calls is named "
    "      exactly `{primary_wrapper}` and takes the arguments listed "
    "      in `entry signature` (in the same order). DO NOT rename the "
    "      function or reorder arguments.\n"
    "  (c) Each kernel function in the `kernels` list must keep its "
    "      EXACT name — those are what the wrapper launches.\n"
    "  (d) The output must be numerically equivalent to a clear "
    "      reference behaviour (described in your own words from the "
    "      summary). Allow `torch.allclose(atol=1e-2, rtol=1e-2)`.\n"
    "  (e) Spell out input tensor shapes / dtypes implied by the "
    "      summary (e.g. 'predictions: Tensor[N, M] float32').\n\n"
    "Express the request in this framing: **{framing}** — {framing_desc}. "
    "Whatever the framing, all five items above must remain present.\n\n"
    "MUST NOT:\n"
    "  - Echo the behavioural summary verbatim.\n"
    "  - Mention any low-level Triton implementation token: "
    "`tl.program_id`, `tl.load`, `tl.store`, `tl.arange`, "
    "`tl.constexpr`, tile size, block size (BLOCK_SIZE/BLOCK_M/...), "
    "num_warps, num_stages.\n"
    "  - Use the words 'spec' / 'specification' / 'design draft'. The "
    "user is asking for an implementation, not a spec.\n\n"
    "Output the request text only. No code fences. No commentary "
    "outside the request."
)

BACK_DERIVE_USER_TBG = (
    "Public interface:\n"
    "---\n"
    "kernels:         {kernel_names}\n"
    "primary entry:   {primary_wrapper}\n"
    "entry signature: {primary_signature}\n"
    "test function:   {test_func_name}\n"
    "---\n\n"
    "Reference Triton (for your eyes only — do NOT embed verbatim):\n"
    "```python\n{ref_src}\n```\n\n"
    "Behavioural summary (background only — paraphrase or use to "
    "structure the request, do not echo it verbatim):\n"
    "---\n{summary}\n---\n\n"
    "Now write the user's NL request."
)

PLAN_SYSTEM_TBG = (
    "You are a Triton-kernel engineer's design partner. Given a user "
    "request and the Triton reference, write a high-level "
    "implementation plan. This plan will be the OPENING of an "
    "assistant reply, BEFORE the actual Triton code — it teaches the "
    "model to think about the task before it writes the kernel.\n\n"
    "The plan MUST cover:\n"
    "  1. The math behaviour (one-line summary).\n"
    "  2. Entry-point contract: function name, argument names + "
    "shape/dtype, output shape/dtype.\n"
    "  3. Which computations live in the kernel(s) — at the level of "
    "'matmul main loop' / 'softmax two-pass scan' / 'fused reduction' "
    "/ 'cumulative scan along last dim'. NOT at the level of per-"
    "thread arithmetic.\n"
    "  4. Output shape and dtype.\n"
    "  5. Correctness tolerance: torch.allclose(atol=1e-2, rtol=1e-2).\n\n"
    "The plan MUST NOT contain:\n"
    "  - Any Triton implementation token: @triton.jit, tl.program_id, "
    "tl.load, tl.store, tl.arange, tl.constexpr, tl.dot.\n"
    "  - tile size / block size / num_warps / num_stages / kernel "
    "launch grid configuration.\n"
    "  - load_inline, cpp_extension, __global__, __device__.\n"
    "  - Per-thread / per-block / per-grid index arithmetic.\n\n"
    "Output 5-15 lines of plain English or simple markdown. No code "
    "fences. No section heading prefixes like '## Plan'."
)

PLAN_USER_TBG = (
    "User request:\n"
    "---\n{expanded_prompt}\n---\n\n"
    "Triton reference (for your eyes only):\n"
    "```python\n{ref_src}\n```\n\n"
    "Now write the implementation plan."
)


# ---------- TBG class methods (attached to InverseCoderExpansion) ----
#
# Defined at module level then bound to the class below. Avoids
# re-opening the class body and keeps the TBG addition surgical.

async def _expand_tbg(self, seed: "Seed", router: "LLMRouter",
                      num_variants: int) -> List["Expanded"]:
    ref = seed.reference_solution or ""
    if not ref.strip():
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "no_reference_solution",
                        "framing": "passthrough"})]

    interface = _extract_triton_interface(ref)
    if interface is None:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "tbg_interface_extraction_failed",
                        "framing": "passthrough"})]

    # Step 1.5: ref smoke test ONCE per seed (Triton JIT compile is
    # ~30-90s cold; running it once per framing × 16 framings × 1 GPU
    # serialises and exceeds the expand_data.py 600s per-seed cap, see
    # logs/tbg_full/run.log first attempt: every seed timed out >600s).
    smoke_ok, smoke_err = await _tbg_ref_smoke_test(ref)
    if not smoke_ok:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "tbg_ref_smoke_failed",
                        "smoke_error": smoke_err[:200],
                        "framing": "passthrough"})]

    seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)

    # Step 2: black-box understanding (once per seed). prompt-side step
    # (not visible in SFT assistant — assistant_spec serves as the CoT).
    summary_raw = await router.chat_prompt(
        UNDERSTAND_SYSTEM_TBG,
        UNDERSTAND_USER_TBG.format(
            kernel_names=", ".join(interface.kernel_names),
            primary_wrapper=interface.primary_wrapper,
            primary_signature=interface.primary_signature,
            test_func_name=interface.test_func_name,
            ref_src=ref),
        seed_hash=seed_hash, variant_idx=0,
    )
    summary = strip_dryrun(summary_raw).strip()
    if not summary:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "empty_understanding",
                        "framing": "passthrough"})]

    m = router.num_prompt_models
    framings = [INVERSE_FRAMINGS_TBG[i % len(INVERSE_FRAMINGS_TBG)]
                for i in range(num_variants)]
    coros = [_generate_one_tbg(self, seed, summary, interface,
                               ref, framing, f_idx,
                               variant_idx, router, seed_hash)
             for f_idx, framing in enumerate(framings)
             for variant_idx in range(m)]
    rows = await asyncio.gather(*coros)
    return [r for r in rows if r is not None]


async def _generate_one_tbg(self, seed: "Seed", summary: str,
                            interface: TritonInterface, ref: str,
                            framing: Tuple[str, str], f_idx: int,
                            variant_idx: int, router: "LLMRouter",
                            seed_hash: int) -> Optional["Expanded"]:
    nl = await _back_derive_tbg(self, interface, ref, summary,
                                framing, router, seed_hash, variant_idx)
    if not nl:
        return None
    plan = await _plan_tbg(self, nl, ref, router, seed_hash, variant_idx)
    if not plan:
        return None

    m = router.num_prompt_models
    row_idx = f_idx * m + variant_idx
    candidate = base_expanded(
        seed, "inversecoder", row_idx, nl,
        extra_meta={
            "framing": framing[0],
            "framing_idx": f_idx,
            "variant_idx": variant_idx,
            "behaviour_summary": summary,
            "assistant_spec": plan,
            "triton_interface": asdict(interface),
            "traj_model": router.traj_model_name,
            "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
        })
    rec = candidate.to_record()
    # smoke is hoisted in _expand_tbg; pass skip_smoke=True so we don't
    # re-run it 16x per seed.
    ok, reason = await filter_tbg_row(rec, interface, ref, skip_smoke=True)
    if not ok:
        _append_filter_failed_tbg(rec, reason)
        return None
    return candidate


async def _back_derive_tbg(self, interface: TritonInterface, ref: str,
                           summary: str, framing: Tuple[str, str],
                           router: "LLMRouter",
                           seed_hash: int, variant_idx: int) -> str:
    f_name, f_desc = framing
    user = BACK_DERIVE_USER_TBG.format(
        kernel_names=", ".join(interface.kernel_names),
        primary_wrapper=interface.primary_wrapper,
        primary_signature=interface.primary_signature,
        test_func_name=interface.test_func_name,
        ref_src=ref,
        summary=summary)
    system = BACK_DERIVE_SYSTEM_TBG.format(
        framing=f_name, framing_desc=f_desc,
        primary_wrapper=interface.primary_wrapper)
    raw = await router.chat_prompt(
        system, user,
        seed_hash=seed_hash, variant_idx=variant_idx)
    return strip_dryrun(raw).strip()


async def _plan_tbg(self, expanded_prompt: str, ref: str,
                    router: "LLMRouter",
                    seed_hash: int, variant_idx: int) -> str:
    user = PLAN_USER_TBG.format(
        expanded_prompt=expanded_prompt, ref_src=ref)
    raw = await router.chat_prompt(
        PLAN_SYSTEM_TBG, user,
        seed_hash=seed_hash, variant_idx=variant_idx)
    return strip_dryrun(raw).strip()


# Bind onto the class.
InverseCoderExpansion._expand_tbg = _expand_tbg


# ===================================================================
# TBT (TritonBench-T) branch
# ===================================================================
#
# Shape vs TBG:
#   - TBG ref = Triton kernel + wrapper        → target = Triton (same)
#   - TBT ref = PyTorch standalone function    → target = Triton
#   - TBT test block runs the func + saves a `test_results` dict
#     (compare-against = `test_results`, not `result_gold`).
#
# Reuses the framings + plan templates (target is still Triton). The
# back-derive prompt is rewritten because the source language is PyTorch,
# not Triton. No elaborate ref-smoke: TBT seeds are pre-validated by the
# dataset (each ships a working test_block; we trust that signal).

# Smaller smoke: import the func + invoke the test block to ensure it
# produces a non-empty test_results dict. Cheap (1 short subprocess).
_TBT_REF_SMOKE_DRIVER = textwrap.dedent("""
    import importlib.util, sys, traceback, torch
    if torch.cuda.is_available():
        torch.cuda.init()
        _d = torch.randn(1, device='cuda') + 1.0
        torch.cuda.synchronize(); del _d
    spec = importlib.util.spec_from_file_location("tbt_seed", sys.argv[1])
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    test_func_name = sys.argv[2]
    try:
        fn = getattr(mod, test_func_name)
        result = fn()
    except Exception:
        traceback.print_exc()
        sys.exit(3)
    # Accept dict-of-tensors or namedtuple
    if isinstance(result, dict):
        if not result:
            print("EMPTY_RESULTS"); sys.exit(4)
        print(f"OK keys={len(result)}")
    else:
        # Fallback for tuple/namedtuple — test_block might assign to a global
        try:
            tr = getattr(mod, 'test_results', None)
            if isinstance(tr, dict) and tr:
                print(f"OK keys={len(tr)}")
            else:
                print("NO_TEST_RESULTS"); sys.exit(5)
        except Exception:
            traceback.print_exc(); sys.exit(6)
""").strip()


async def _tbt_ref_smoke_test(seed_src: str, test_func_name: str,
                              timeout: int = 60
                              ) -> Tuple[bool, str]:
    """Run the seed file + its test func in a subprocess. Returns
    (ok, err)."""
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        p = Path(tmp) / "seed.py"
        p.write_text(seed_src)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _TBT_REF_SMOKE_DRIVER,
                str(p), test_func_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return False, f"launch failed: {e}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill(); await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return False, f"smoke timeout ({timeout}s)"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-512:]
        return True, ""


# ---------- TBT prompt templates --------------------------------------

UNDERSTAND_SYSTEM_TBT = (
    "You read a PyTorch standalone-function reference and write a 2-4 "
    "sentence BLACK-BOX behavioural summary in plain English.\n\n"
    "HARD RULES:\n"
    "  - Mention: input tensor shapes + dtypes (from the test block "
    "calls), the operation performed (matmul / reduction / elementwise "
    "/ attention etc.), output shape/dtype.\n"
    "  - Use the EXACT entry-point name given under `Public interface`.\n"
    "  - Do NOT name the source or target implementation language. Do "
    "NOT say 'PyTorch', 'Triton', 'CUDA', or 'GPU'.\n"
    "  - Do NOT describe Triton internals (block size, grid shape, "
    "num_warps, etc.).\n\n"
    "Output 2-4 plain-English sentences. No bullet lists, no headings, "
    "no code."
)

UNDERSTAND_USER_TBT = (
    "Public interface (use these EXACT names):\n"
    "---\n"
    "primary function: {primary_func_name}\n"
    "test function:    {test_func_name}\n"
    "---\n\n"
    "PyTorch reference (for your eyes only — do NOT echo or quote):\n"
    "```python\n{ref_src}\n```\n\n"
    "Write the black-box behavioural summary now."
)

BACK_DERIVE_SYSTEM_TBT = (
    "You write a NL request that an engineer would send to an LLM "
    "coding assistant. The expected answer is (a) a high-level "
    "implementation plan and (b) a Triton kernel implementation that "
    "matches a fixed entry-point signature so an external test harness "
    "can invoke it. Imagine the request that produces that answer.\n\n"
    "MUST INCLUDE in the request:\n"
    "  (a) The TARGET function must be implemented in Triton — name "
    "      the language explicitly. The compute must live inside one "
    "      or more `@triton.jit` kernels (mention the @triton.jit "
    "      requirement in plain English; do not write the literal "
    "      '@triton.jit' token).\n"
    "  (b) The entry-point function the test harness calls is named "
    "      exactly `{primary_func_name}` and takes the SAME arguments "
    "      (names + order) as the reference function. DO NOT rename or "
    "      reorder.\n"
    "  (c) The output must be numerically equivalent to a clear "
    "      reference behaviour (described in your own words from the "
    "      summary). Allow `torch.allclose(atol=1e-2, rtol=1e-2)`.\n"
    "  (d) Spell out input tensor shapes / dtypes implied by the "
    "      summary (e.g. 'x: Tensor[N, M] float32').\n\n"
    "Express the request in this framing: **{framing}** — {framing_desc}. "
    "Whatever the framing, all four items above must remain present.\n\n"
    "MUST NOT:\n"
    "  - Echo the behavioural summary verbatim.\n"
    "  - Mention low-level Triton implementation tokens: `tl.program_id`, "
    "`tl.load`, `tl.store`, `tl.arange`, `tl.constexpr`, tile size, "
    "block size, num_warps, num_stages.\n"
    "  - Reveal the source language (do not say 'PyTorch reference', "
    "'translate from PyTorch', etc.). Speak from the engineer-asking-"
    "for-an-implementation perspective.\n\n"
    "Output the request text only. No code fences. No commentary "
    "outside the request."
)

BACK_DERIVE_USER_TBT = (
    "Public interface:\n"
    "---\n"
    "primary function: {primary_func_name}\n"
    "test function:   {test_func_name}\n"
    "---\n\n"
    "Reference function (for your eyes only — do NOT embed verbatim):\n"
    "```python\n{ref_src}\n```\n\n"
    "Behavioural summary (background only — paraphrase or use to "
    "structure the request, do not echo verbatim):\n"
    "---\n{summary}\n---\n\n"
    "Now write the user's NL request."
)


# ---------- TBT class methods --------------------------------------

async def _expand_tbt(self, seed: "Seed", router: "LLMRouter",
                      num_variants: int) -> List["Expanded"]:
    ei = seed.evaluator_info or {}
    primary_func_name = ei.get("primary_func_name", "")
    test_func_name = ei.get("test_func_name", "")
    func_src = ei.get("func_src", "") or (seed.reference_solution or "")
    if not func_src.strip() or not primary_func_name or not test_func_name:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "tbt_missing_evaluator_info",
                        "framing": "passthrough"})]

    # Ref smoke (1× per seed): import + run test_func, ensure it produces
    # a non-empty result. Cheap CPU/GPU smoke.
    smoke_ok, smoke_err = await _tbt_ref_smoke_test(
        seed.reference_solution or func_src, test_func_name)
    if not smoke_ok:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "tbt_ref_smoke_failed",
                        "smoke_error": smoke_err[:200],
                        "framing": "passthrough"})]

    seed_hash = int(hashlib.md5(seed.id.encode()).hexdigest()[:8], 16)

    summary_raw = await router.chat_prompt(
        UNDERSTAND_SYSTEM_TBT,
        UNDERSTAND_USER_TBT.format(
            primary_func_name=primary_func_name,
            test_func_name=test_func_name,
            ref_src=func_src),
        seed_hash=seed_hash, variant_idx=0,
    )
    summary = strip_dryrun(summary_raw).strip()
    if not summary:
        return [base_expanded(
            seed, "inversecoder", 0, seed.original_prompt,
            extra_meta={"warn": "tbt_empty_understanding",
                        "framing": "passthrough"})]

    m = router.num_prompt_models
    framings = [INVERSE_FRAMINGS_TBG[i % len(INVERSE_FRAMINGS_TBG)]
                for i in range(num_variants)]
    coros = [_generate_one_tbt(self, seed, summary, primary_func_name,
                               test_func_name, func_src, framing, f_idx,
                               variant_idx, router, seed_hash)
             for f_idx, framing in enumerate(framings)
             for variant_idx in range(m)]
    rows = await asyncio.gather(*coros)
    return [r for r in rows if r is not None]


async def _generate_one_tbt(self, seed: "Seed", summary: str,
                            primary_func_name: str, test_func_name: str,
                            ref_src: str,
                            framing: Tuple[str, str], f_idx: int,
                            variant_idx: int, router: "LLMRouter",
                            seed_hash: int) -> Optional["Expanded"]:
    nl = await _back_derive_tbt(self, primary_func_name, test_func_name,
                                ref_src, summary, framing, router,
                                seed_hash, variant_idx)
    if not nl:
        return None
    plan = await _plan_tbg(self, nl, ref_src, router, seed_hash, variant_idx)
    if not plan:
        return None

    m = router.num_prompt_models
    row_idx = f_idx * m + variant_idx
    # Provide a TBG-shaped triton_interface dict so
    # scripts/teacher_triton_rollout_tbg.py picks up `primary_wrapper`
    # without modification.
    triton_interface = {
        "kernel_names": [],
        "wrapper_names": [primary_func_name],
        "primary_wrapper": primary_func_name,
        "primary_signature": f"{primary_func_name}(...)",
        "primary_arg_names": [],
        "test_func_name": test_func_name,
        "has_class_wrapper": False,
    }
    return base_expanded(
        seed, "inversecoder", row_idx, nl,
        extra_meta={
            "framing": framing[0],
            "framing_idx": f_idx,
            "variant_idx": variant_idx,
            "behaviour_summary": summary,
            "assistant_spec": plan,
            "triton_interface": triton_interface,
            "traj_model": router.traj_model_name,
            "prompt_model": router.prompt_model_name(seed_hash, variant_idx),
            "rewrite_model": router.prompt_model_name(seed_hash, variant_idx),
        })


async def _back_derive_tbt(self, primary_func_name: str,
                           test_func_name: str, ref_src: str, summary: str,
                           framing: Tuple[str, str], router: "LLMRouter",
                           seed_hash: int, variant_idx: int) -> str:
    f_name, f_desc = framing
    user = BACK_DERIVE_USER_TBT.format(
        primary_func_name=primary_func_name,
        test_func_name=test_func_name,
        ref_src=ref_src,
        summary=summary)
    system = BACK_DERIVE_SYSTEM_TBT.format(
        framing=f_name, framing_desc=f_desc,
        primary_func_name=primary_func_name)
    raw = await router.chat_prompt(
        system, user,
        seed_hash=seed_hash, variant_idx=variant_idx)
    return strip_dryrun(raw).strip()


# Bind onto the class.
InverseCoderExpansion._expand_tbt = _expand_tbt
