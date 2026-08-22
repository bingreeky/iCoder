# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""Robust re-extraction helpers for the rescore pipeline.

These exist because the production extractors grab the FIRST fenced block (VE
sv-generate:544-580 / ArchX run_archx.py:135 / TBG teacher_triton_rollout_tbg:174),
which drops the real answer when a model emits an explanation/skeleton fence
first and the answer fence second. The rescore pipeline re-parses the raw
logged response with `pick_verilog` / `pick_triton` to find a *complete*
candidate the production extractor missed, then re-runs the existing verifier
on it.

A row is a RECOVERY CANDIDATE iff the robust extractor yields a complete
candidate that differs from the on-disk extracted code (and the on-disk code is
incomplete/shorter). Phase-1 reports candidate counts (an upper bound — prose
that mentions `module`/`@triton.jit` can overcount); Phase-2 re-verify is the
ground truth that filters to actual flips (fail→pass).
"""
from __future__ import annotations
import re

# ---- shared with production extractors (kept identical so behaviour matches) ----
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TAG_RE = re.compile(r"</?(?:answer|think)>")
# closed fenced block (lang optional); non-greedy so each ``` pair = one block
_FENCE_RE = re.compile(r"```(?:[a-zA-Z+./-]+)?[ \t]*\n?(.*?)```", re.DOTALL)
# complete module ... endmodule (non-greedy: one module per match)
_MOD_COMPLETE_RE = re.compile(r"\bmodule\b.*?endmodule", re.DOTALL)
# truncated module (no endmodule) — greedy to EOF
_MOD_TRUNC_RE = re.compile(r"\bmodule\b.*", re.DOTALL)


def extract_v4pro_answer(content: str) -> str:
    """Peel the ``<answer>...</answer>`` wrapper if present (mirrors
    expand/methods/_common.py:extract_v4pro_answer). Plain content unchanged."""
    if not content:
        return content
    m = _ANSWER_BLOCK_RE.search(content)
    return m.group(1).strip() if m else content


def _regions(raw: str) -> list[str]:
    """Collect candidate code regions from a raw response: every closed
    fenced block, every [BEGIN]/[DONE] body (internal ``` lines stripped), the
    unclosed-fence tail (truncation), and the raw text as a last resort."""
    if not raw:
        return []
    t = _THINK_RE.sub("", raw)
    t = _TAG_RE.sub("", t)
    regions: list[str] = []
    for m in _FENCE_RE.finditer(t):
        regions.append(m.group(1))
    for m in re.finditer(r"\[BEGIN\](.*?)\[DONE\]", t, re.DOTALL):
        # strip internal ``` fence lines so the [BEGIN]/[DONE] body is pure code
        body = re.sub(r"^[ \t]*```.*$", "", m.group(1), flags=re.MULTILINE)
        regions.append(body)
    # unclosed fence tail (odd number of ``` => last opener has no closer).
    # Only the tail of the LAST opener — single block, no cross-block span.
    ticks = [mt.start() for mt in re.finditer(r"```", t)]
    if len(ticks) % 2 == 1:
        regions.append(t[ticks[-1] + 3:])
    # NOTE: deliberately NO raw-text last-resort region — it lets
    # `module.*?endmodule` span prose + a later code block and produce
    # prose-contaminated false-positive "complete" modules. Only search
    # WITHIN each individual fenced block / [BEGIN]/[DONE] body.
    return regions


def pick_verilog(raw: str) -> tuple[str, str]:
    """Robust Verilog extractor. Returns (code, status).

    status:
      'complete'  — a ``module ... endmodule`` was found (the LAST complete one,
                    since models explain-then-answer; tie-break: longest among
                    the last region's matches).
      'truncated' — a ``module`` was found but no ``endmodule`` (max_tokens cut);
                    returns the last truncated module. NOT a recovery candidate.
      'empty'     — no ``module`` at all (model produced no code).
    """
    if not raw:
        return "", "empty"
    complete: list[str] = []
    truncated: list[str] = []
    for region in _regions(raw):
        for mm in _MOD_COMPLETE_RE.finditer(region):
            complete.append(mm.group(0))
        # truncated only where no endmodule follows a module in THIS region
        if "endmodule" not in region:
            for mm in _MOD_TRUNC_RE.finditer(region):
                truncated.append(mm.group(0))
    if complete:
        return complete[-1].strip(), "complete"
    if truncated:
        return truncated[-1].strip(), "truncated"
    return "", "empty"


def pick_verilog_candidates(raw: str) -> list[str]:
    """Return ALL complete ``module ... endmodule`` blocks the raw response
    contains (deduped, last-first) — so a VE-spec R-fail (on-disk = a complete
    module that compiles but gives WRONG output) can be re-verified against
    alternative complete modules the model also emitted. A flip = the model
    wrote a CORRECT module that production's first-block-grab dropped in favour
    of a wrong one. Model-agnostic upper bound on VE-spec, same legitimacy as
    pick_ve_cc_candidates (the code IS the model's output)."""
    if not raw:
        return []
    cands: list[str] = []
    for region in _regions(raw):
        for mm in _MOD_COMPLETE_RE.finditer(region):
            cands.append(mm.group(0).strip())
    # last-first (the real answer is usually the last module)
    cands.reverse()
    seen = set()
    out = []
    for c in cands:
        k = c.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


# ---- TBG / Triton ----
_TRITON_FENCE_RE = re.compile(r"```(?:[a-zA-Z+./-]+)?[ \t]*\n?(.*?)```", re.DOTALL)


def pick_triton(traj_content: str) -> tuple[str, str]:
    """Robust Triton kernel extractor from a rollout row's teacher_traj_content
    (the <answer> wrap; raw resp.content is NOT persisted on the row).

    Returns (code, status):
      'complete' — a fenced block with ``@triton.jit`` AND a ``def`` (the
                   longest such block — the real kernel, not an explanation).
      'partial'  — fenced blocks exist but none has both @triton.jit+def
                   (explanation/fragment — NOT a recovery candidate).
      'empty'    — no fenced block retained in the traj wrap.
    """
    if not traj_content:
        return "", "empty"
    body = extract_v4pro_answer(traj_content)
    blocks = _TRITON_FENCE_RE.findall(body)
    if not blocks:
        return "", "empty"
    jit_blocks = [b for b in blocks if "@triton.jit" in b and re.search(r"\bdef\b", b)]
    if jit_blocks:
        return max(jit_blocks, key=len).strip(), "complete"
    # fall back to any block with @triton.jit (fragment)
    jit_only = [b for b in blocks if "@triton.jit" in b]
    if jit_only:
        return max(jit_only, key=len).strip(), "partial"
    return "", "partial"


def is_verilog_recovery_candidate(new_code: str, new_status: str,
                                  old_code: str) -> bool:
    """A row is a recovery candidate iff the robust extractor found a COMPLETE
    module that differs from the on-disk extracted code, AND the on-disk code
    is incomplete (lacks endmodule) or shorter. This separates:
      - truncation (new_status='truncated' -> NOT a candidate, unrecoverable)
      - dropped-complete-block (new complete, old incomplete -> candidate)
      - clean (new == old -> not a candidate)
    """
    if new_status != "complete" or not new_code:
        return False
    if not old_code:
        return True  # on-disk empty but raw has a complete module
    if new_code.strip() == old_code.strip():
        return False  # same extraction -> nothing to recover
    old_complete = "endmodule" in old_code
    # recover when on-disk is incomplete, or on-disk is a different/shorter module
    if not old_complete:
        return True
    return len(new_code) > len(old_code) * 1.05


def is_triton_recovery_candidate(new_code: str, new_status: str,
                                 old_code: str) -> bool:
    """TBG recovery candidate iff robust extractor found a complete @triton.jit
    kernel that differs from the on-disk teacher_code."""
    if new_status != "complete" or not new_code:
        return False
    if not old_code:
        return True
    if new_code.strip() == old_code.strip():
        return False
    return len(new_code) > len(old_code) * 1.05


# ---- VerilogEval code-complete (body+endmodule, NOT full module) ----
# The code-complete task tells the model: "Do NOT include module/input/output
# definitions" — emit only the body ending in `endmodule`; the harness prepends
# the module header (interface). So the model almost never emits `module`; the
# production sv-generate extractor (:544-580) takes the FIRST fenced block or
# the text-up-to-endmodule, which — when the model explains-then-answers with
# abnormal fences / prose first — yields an EMPTY or PROSE-contaminated body
# (false S-fail). pick_ve_cc recovers by walking back from the LAST endmodule.

_VE_CC_CODE_RE = re.compile(
    r'^\s*(assign|always|reg|wire|integer|real|parameter|localparam|input|'
    r'output|inout|module|endmodule|begin|end|case|endcase|casex|casez|if|else|'
    r'for|while|generate|endgenerate|genvar|defparam|initial|function|'
    r'endfunction|task|endtask|//|\$|[}\;]|`)')


def _ve_cc_is_code(s: str) -> bool:
    """Heuristic code-vs-prose line for the code-complete body walk-back.
    Conservative: blank lines and //comments count as code (harmless in
    Verilog); the iverilog verify is the ground-truth filter for false hits.
    NOTE: deliberately does NOT treat ``*`` / ``/*`` as code — markdown bold
    ``**text**`` and block-comment continuations both start with ``*`` and would
    leak prose into the body (a real S-fail cause). Inline ``/* */`` on one line
    is kept via the assignment/keyword paths; rare standalone ``/*`` blocks make
    the walk-back stop early (harmless — verify filters)."""
    s = s.strip()
    if not s:
        return True
    if s.startswith('//'):
        return True
    if _VE_CC_CODE_RE.match(s):
        return True
    if re.match(r'^[a-zA-Z_]\w*\s*(\[.*?\])?\s*(<=|=)\s', s):
        return True
    return False


def pick_ve_cc(raw: str, interface: str = "") -> tuple[str, str]:
    """Robust VerilogEval code-complete re-extractor. Returns the reconstructed
    `.sv` text and a status:

      'full'            — model emitted a complete ``module TopModule...endmodule``;
      'interface+body'  — reconstructed ``<interface>\\n<body>\\nendmodule``;
      'body_only'       — body+endmodule found but no interface given;
      'no_endmodule'    — no endmodule in the raw (unrecoverable);
      'empty'           — no raw.

    Strategy: PREFER the LAST fenced block that looks like real Verilog (contains
    assign/always/endmodule/module + a ``;`` or endmodule) — the model's real
    answer is usually in a fence, and this avoids prose-equation lines (e.g.
    "Or more elegantly: f = ...") that the raw walk-back would leak. If no
    usable fenced block, FALL BACK to walking back from the LAST ``endmodule``,
    stopping at the first prose line. Fence delimiter lines are stripped."""
    if not raw:
        return "", "empty"
    body = ""
    # --- prefer a fenced block containing real code ---
    fences = list(_FENCE_RE.finditer(raw))  # closed ``` blocks
    for m in reversed(fences):
        blk = m.group(1)
        has_kw = re.search(r'\b(assign|always|endmodule|module|case|initial|wire|reg)\b', blk)
        has_term = ('endmodule' in blk) or (';' in blk) or ('end' in blk)
        if has_kw and has_term:
            body = blk
            break
    # --- fallback: walk back from the last endmodule ---
    if not body:
        i = raw.rfind("endmodule")
        if i < 0:
            return "", "no_endmodule"
        bl = []
        for ln in reversed(raw[:i].split("\n")):
            if _ve_cc_is_code(ln):
                bl.append(ln)
            else:
                break
        bl = [l for l in bl if l.strip() != ""] or [""]
        bl = [l for l in bl if not re.match(r'^\s*```', l)]
        body = "\n".join(reversed(bl)) if bl else ""
    # strip any fence delimiter lines that slipped through (markdown, not Verilog)
    body = "\n".join(l for l in body.split("\n") if not re.match(r'^\s*```', l))
    code = (body.rstrip() + "\nendmodule") if body.strip() else "endmodule"
    if re.search(r'(?m)^\s*module\b', body):
        return code, "full"
    if not interface:
        return code, "body_only"
    return interface.rstrip() + "\n\n" + code, "interface+body"


def _reconstruct(body: str, interface: str) -> str:
    """Reconstruct a .sv from a body blob: strip fence lines, append endmodule,
    prepend interface unless the body already has a module."""
    body = "\n".join(l for l in body.split("\n") if not re.match(r'^\s*```', l))
    if not body.strip():
        body = ""
    if re.search(r'(?m)^\s*module\b', body):
        return body.rstrip() + "\n"  # full module as-is (may already end in endmodule)
    code = (body.rstrip() + "\nendmodule") if body.strip() else "endmodule"
    if interface:
        return interface.rstrip() + "\n\n" + code
    return code


def pick_ve_cc_candidates(raw: str, interface: str = "") -> list[str]:
    """Return an ORDERED, DEDUPED list of candidate .sv reconstructions for a
    VE code-complete raw response — so the re-verify can try each and take the
    first that passes (the model DID emit a correct body somewhere; the
    extractor's job is to find it).

    Order (best-guess first): the LAST fenced block that looks like real Verilog,
    then any EARLIER code-looking fenced blocks, then the raw walk-back from the
    last endmodule. Some models put their real answer in a fence; others usually
    have it as raw body — trying both per sample is what makes the extractor
    model-agnostic. Duplicates (same stripped code) collapse to the first."""
    if not raw:
        return []
    cands: list[str] = []
    # fenced blocks (reverse: last answer-first), filtered to code-looking
    fences = list(_FENCE_RE.finditer(raw))
    for m in reversed(fences):
        blk = m.group(1)
        has_kw = re.search(r'\b(assign|always|endmodule|module|case|initial|wire|reg)\b', blk)
        has_term = ('endmodule' in blk) or (';' in blk) or ('end' in blk)
        if has_kw and has_term:
            cands.append(_reconstruct(blk, interface))
    # walk-back from last endmodule (the raw-body strategy — gemini's usual case)
    i = raw.rfind("endmodule")
    if i >= 0:
        bl = []
        for ln in reversed(raw[:i].split("\n")):
            if _ve_cc_is_code(ln):
                bl.append(ln)
            else:
                break
        bl = [l for l in bl if l.strip() != ""] or [""]
        bl = [l for l in bl if not re.match(r'^\s*```', l)]
        if bl:
            cands.append(_reconstruct("\n".join(reversed(bl)), interface))
    # dedup by stripped content, preserve order
    seen = set()
    out = []
    for c in cands:
        k = c.strip()
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out
