"""X-Coder style expansion (pilot).

Strategy: take the seed's ``original_prompt`` and produce ``num_variants``
diverse rephrasings. Each rephrasing must preserve the *semantic intent* of
the original task — same module interface / same target function — but vary
wording, structure, and angle (style: terse-spec, narrative, bulleted).

Output ``reference_solution`` is preserved unchanged: only the prompt is
changed, so existing testbenches / harnesses still apply.
"""

from __future__ import annotations

import asyncio
from typing import List

from ..registry import register_method
from ._common import Expanded, Seed, base_expanded, strip_dryrun

XCODER_STYLES = [
    ("terse_spec",
     "Concise bullet-point spec. Keep all interface signals exactly. "
     "Drop narrative; keep behaviour rules."),
    ("narrative",
     "A flowing narrative description, as a senior engineer might explain "
     "to a junior. Keep all interface signals; rephrase behaviour."),
    ("constraint_first",
     "Lead with the hard constraints (timing, widths, reset behaviour). "
     "Then describe behaviour. Keep all interface signals."),
    ("interface_first",
     "Open with a clean port table, then a numbered behavioural list. "
     "Keep semantics; vary format."),
    ("usage_focused",
     "Frame as a downstream user requirement: what should the module DO "
     "when used in a larger system. Keep all interface signals."),
]

SYSTEM_PROMPT = (
    "You rewrite hardware-design / kernel-implementation specifications "
    "for data augmentation. You preserve the underlying task exactly: same "
    "module name, same ports, same semantics. You only vary surface form."
)

USER_TMPL = (
    "Rewrite the following specification in this style: **{style}** — "
    "{style_desc}\n\n"
    "Constraints:\n"
    "- Keep the same module / class name and the same input/output signature.\n"
    "- Keep the same intended behaviour. Do NOT add or drop functional "
    "requirements.\n"
    "- Output only the rewritten specification, no commentary, no code.\n\n"
    "Original specification:\n---\n{prompt}\n---"
)


@register_method("xcoder")
class XCoderExpansion:
    name = "xcoder"

    async def expand(self, seed: Seed, llm, num_variants: int = 3,
                     **_kw) -> List[Expanded]:
        styles = [XCODER_STYLES[i % len(XCODER_STYLES)]
                  for i in range(num_variants)]
        coros = [
            llm.chat(
                SYSTEM_PROMPT,
                USER_TMPL.format(
                    style=name, style_desc=desc,
                    prompt=seed.original_prompt))
            for name, desc in styles
        ]
        completions = await asyncio.gather(*coros)
        out: List[Expanded] = []
        for i, ((style_name, _), raw) in enumerate(zip(styles, completions)):
            text = strip_dryrun(raw).strip() or seed.original_prompt
            out.append(base_expanded(
                seed, "xcoder", i, text,
                extra_meta={"style": style_name},
            ))
        return out
