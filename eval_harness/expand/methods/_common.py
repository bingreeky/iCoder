"""Common helpers for method adapters."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..base import Expanded, Seed

CODE_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z+]+)?\s*\n?(.*?)```", re.DOTALL)

# reasoning emits `<think>...</think><answer>...</answer>` wrapping. Both
# tags are consumed by SFT training (assistant content stays raw), but
# verify-side code needs to peel the wrapper before fence extraction.
_THINK_BLOCK_RE = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_BLOCK_RE = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def strip_fences(text: str) -> str:
    """If response is wrapped in a single code fence, return the inner body.
    Otherwise return text unchanged."""
    if not text:
        return text
    m = CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def strip_dryrun(text: str) -> str:
    """For DryRunLLM outputs: remove the leading marker so downstream code
    sees the raw user input back. Real LLM responses pass through unchanged."""
    t = text.lstrip()
    if t.startswith("[DRYRUN"):
        # drop first line + trailing [END]
        lines = t.splitlines()
        if lines and lines[-1].strip() == "[END]":
            lines = lines[:-1]
        return "\n".join(lines[1:]).strip()
    return text


def expand_id(seed: Seed, method: str, idx: int) -> str:
    return f"{seed.id}::{method}::v{idx}"


def base_expanded(seed: Seed, method: str, idx: int,
                  expanded_prompt: str,
                  extra_meta: dict | None = None) -> Expanded:
    """Build an Expanded record carrying through evaluator info / refs."""
    md = dict(seed.metadata)
    md["variant_idx"] = idx
    if extra_meta:
        md.update(extra_meta)
    return Expanded(
        id=expand_id(seed, method, idx),
        source_dataset=seed.source_dataset,
        expansion_method=method,
        original_prompt=seed.original_prompt,
        expanded_prompt=expanded_prompt,
        reference_solution=seed.reference_solution,
        expected_output=seed.expected_output,
        tests=seed.tests,
        evaluator_info=dict(seed.evaluator_info),
        metadata=md,
    )


# ---------- reasoning answer extraction -----------------------------------

def extract_v4pro_answer(content: str) -> str:
    """Peel the reasoning ``<answer>...</answer>`` wrapper if present.

    Plain (non-reasoning) outputs return unchanged. The returned string
    keeps any code fences inside; downstream callers run their own
    fence extractor on top (e.g. ``extract_verilog`` or
    ``extract_code``)."""
    if not content:
        return content
    m = _ANSWER_BLOCK_RE.search(content)
    return m.group(1).strip() if m else content


def check_v4pro_format(content: str) -> Tuple[bool, str]:
    """Validate a traj response well enough to either pass through or
    feed to :func:`synthesize_v4pro_wrap`.

    Two acceptable shapes:

    1. Native reasoning wrap — ``<think>...</think>`` AND ``<answer>...
       </answer>`` both present and closed.
    2. Plain content + (eventually) a fenced code block — the
       endpoint emits prose then ```lang\\n...```, no tags. We accept
       this and let :func:`synthesize_v4pro_wrap` build the tags.

    Hard fail only on truncation: opened ``<think>`` without
    ``</think>`` (reasoning ate the budget), or opened ``<answer>``
    without ``</answer>`` (answer cut mid-stream)."""
    if not content or not content.strip():
        return False, "empty content"
    low = content.lower()
    if "<think>" in low and "</think>" not in low:
        return False, ("reasoning response truncated: opened <think> but "
                       "no </think>; reasoning was likely cut by "
                       "max_tokens — set traj_max_tokens=None")
    if "<answer>" in low and "</answer>" not in low:
        return False, ("reasoning response truncated: opened <answer> but "
                       "no </answer>; reasoning ate the budget. Set "
                       "traj_max_tokens=None.")
    return True, ""


def extract_think_answer(content: str) -> Tuple[str, str]:
    """Split a reasoning response into ``(think_text, answer_body)``. Both
    are empty if the corresponding tag is absent. Used for diagnostic
    inspection — verify-side code generally only needs
    :func:`extract_v4pro_answer`."""
    think_m = _THINK_BLOCK_RE.search(content or "")
    answer_m = _ANSWER_BLOCK_RE.search(content or "")
    return (
        think_m.group(1).strip() if think_m else "",
        answer_m.group(1).strip() if answer_m else "",
    )


# Code fence (any language) used to split content into prose-before vs
# fenced answer when synthesising the reasoning wrap.
_FIRST_FENCE_RE = re.compile(
    r"```[A-Za-z+]*\s*\n.*?```", re.DOTALL)


def synthesize_v4pro_wrap(content: str,
                          reasoning_content: str = "",
                          fallback_lang: str = "verilog") -> str:
    """Produce a synthetic think/answer
    wrapping over an LLM response that did not emit the tags
    natively (most providers return prose + a fenced code block without tag wrapping).

    Strategy:

    * If ``content`` already contains both ``<think>`` AND ``<answer>``,
      return it unchanged (model emitted the format itself).
    * If a fenced code block exists in ``content``, the **first** fence
      becomes the ``<answer>`` body. Anything before that fence becomes
      the ``<think>`` body, unless ``reasoning_content`` is non-empty,
      in which case ``reasoning_content`` wins (providers that
      expose reasoning in a parallel stream).
    * If no fence is present, the whole ``content`` becomes the
      ``<answer>`` body and ``reasoning_content`` (or a placeholder)
      goes into ``<think>``.

    The returned string is suitable as the SFT ``assistant`` message
    and matches the reference format expected by the
    downstream trainer."""
    raw = (content or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if "<think>" in low and "</think>" in low and "<answer>" in low and "</answer>" in low:
        return raw

    m = _FIRST_FENCE_RE.search(raw)
    if m is None:
        # No code fence — wrap the entire content as <answer> body.
        think = (reasoning_content or "").strip() \
            or "(no reasoning trace exposed by provider)"
        return f"<think>{think}</think><answer>{raw}</answer>"

    fence = m.group(0)
    prose_before = raw[:m.start()].strip()
    # Prefer the parallel reasoning_content stream when present (some
    # providers expose chain-of-thought there). Fall back to
    # the in-content prose-before-fence (some providers emit prose first,
    # then the fence).
    think = (reasoning_content or "").strip()
    if not think:
        think = prose_before or "(direct answer; no reasoning emitted)"
    return f"<think>{think}</think><answer>{fence}</answer>"


__all__ = [
    "strip_fences",
    "strip_dryrun",
    "expand_id",
    "base_expanded",
    "extract_v4pro_answer",
    "check_v4pro_format",
    "extract_think_answer",
    "synthesize_v4pro_wrap",
    "Expanded",
    "Seed",
    "List",
]

