"""RTLLM v2 adapter.

Layout (per design):
  <root>/<Category>/<...>/<design_name>/
    design_description.txt
    testbench.v
    verified_<design_name>.v   (reference RTL)
    makefile

Mirrors eval/rtllm_v2_runner.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_ROOT = Path(
    os.environ.get(
        "RTLLM_ROOT",
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "rtllm" / "RTLLM"),
    )
)
SKIP_DIRS = {"_chatgpt35", "_chatgpt4", "_pic"}


@register_dataset("rtllm_v2")
class RTLLMv2Adapter:
    name = "rtllm_v2"

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        n = 0
        for desc in sorted(self.root.rglob("design_description.txt")):
            # skip helper / reference subtrees
            if any(p in SKIP_DIRS for p in desc.parts):
                continue
            d = desc.parent
            tb = d / "testbench.v"
            if not tb.exists():
                continue
            if limit is not None and n >= limit:
                return
            design_name = d.name
            ref = d / f"verified_{design_name}.v"
            ref_text = ref.read_text() if ref.exists() else ""
            # category = first dir under root
            try:
                rel = d.relative_to(self.root)
                category = rel.parts[0] if rel.parts else ""
            except ValueError:
                category = ""
            yield Seed(
                id=f"rtllm_v2/{category}/{design_name}",
                source_dataset="rtllm_v2",
                original_prompt=desc.read_text(),
                reference_solution=ref_text,
                expected_output=None,
                tests=tb.read_text(),
                evaluator_info={
                    "kind": "rtllm_v2",
                    "design_name": design_name,
                    "category": category,
                    "design_dir": str(d),
                    "prompt_path": str(desc),
                    "tb_path": str(tb),
                    "ref_path": str(ref) if ref.exists() else None,
                    "harness": "iverilog testbench.v + gen.v → vvp; pass markers / 'Test completed with 0/N failures'",
                },
                metadata={
                    "design_name": design_name,
                    "category": category,
                },
            )
            n += 1
