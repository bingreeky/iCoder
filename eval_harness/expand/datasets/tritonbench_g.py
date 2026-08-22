"""TritonBench-G adapter.

TritonBench (thunlp/TritonBench, paper 2502.14752) ships two splits:
  G — 184 problems, ref language = Triton, target = Triton.
  T — 166 problems, ref language = PyTorch, target = Triton.

This adapter handles only the G split (T can reuse the KB pipeline
as-is, since it has the same PyTorch→Triton shape).

G layout:
  data/TritonBench_G_v1/<file>.py   → ref source (each file embeds
                                       a ``def test_xxx()`` plus
                                       ``result_gold = test_xxx()``)
  data/TritonBench_G_v1.json        → metadata array, one entry per
                                       file with simp_instru/comp_instru/
                                       difficulty/repo/output.

For inversecoder we keep:
  - reference_solution = the full ref .py source (kernel + wrapper +
    test block)
  - evaluator_info     = {kind, file_path, ref_split_at, has_test} so
    the verify wrapper can split kernel-vs-test and substitute the
    candidate kernel under the ref's test harness.
  - metadata           = the row from TritonBench_G_v1.json
    (simp_instru/comp_instru/difficulty/repo) for downstream cross-
    reference.

Why ref language = target language matters: ref directly serves as
the SFT answer (same as RTLLM/VEval inversecoder). No teacher rollout
is needed; the verify step is just a smoke test that our extraction
didn't break the ref.
"""

from __future__ import annotations
import os

import json
from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset


DEFAULT_ROOT = Path(os.environ.get("TRITONBENCH_ROOT", str(Path(__file__).resolve().parents[2] / "benchmarks" / "TritonBench")))

# A line of `#` separators inside each .py file marks where the kernel
# code ends and the test block begins. Used to split ref kernel from
# `def test_xxx()` so the verify wrapper can swap candidate code under
# the same harness without re-parsing.
_TEST_SEP_NEEDLE = "#" * 50


def _split_kernel_and_test(src: str) -> tuple[str, str]:
    """Split a TritonBench-G ref into (kernel_source, test_block).

    Falls back to (whole src, "") if the long-`#` separator is missing.
    """
    if _TEST_SEP_NEEDLE in src:
        # Split on the first long-# line; everything before is kernel,
        # everything after is the test harness.
        idx = src.find(_TEST_SEP_NEEDLE)
        # Walk left/right past adjacent #'s and a single trailing
        # newline to keep both halves syntactically clean.
        kernel = src[:idx].rstrip()
        rest = src[idx:]
        # Drop the leading separator line then any blank lines.
        rest = rest.lstrip("#").lstrip("\n")
        return kernel + "\n", rest
    return src, ""


def _build_prompt(name: str, ref_src: str) -> str:
    """Build a default prompt referencing the Triton ref.

    Mirrors the KB adapter's ``_build_prompt`` but frames the ref as
    Triton (not PyTorch). Inversecoder will replace this with a
    back-derived NL request anyway; this is just the fallback /
    passthrough text.
    """
    return (
        f"You are given a Triton kernel named `{name}`. "
        f"Re-implement the same computation as a Triton kernel that is "
        f"functionally equivalent to the reference. Reference Triton "
        f"below:\n\n"
        f"```python\n{ref_src.rstrip()}\n```\n"
    )


@register_dataset("tritonbench_g")
class TritonBenchGAdapter:
    name = "tritonbench_g"

    def __init__(self, root: Optional[str] = None,
                 metadata_filename: str = "data/TritonBench_G_v1.json",
                 data_subdir: str = "data/TritonBench_G_v1"):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.data_dir = self.root / data_subdir
        self.metadata_path = self.root / metadata_filename

    def _load_metadata(self) -> dict[str, dict]:
        """Read TritonBench_G_v1.json → {filename: row}."""
        if not self.metadata_path.exists():
            return {}
        rows = json.loads(self.metadata_path.read_text())
        return {r["file"]: r for r in rows}

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        meta = self._load_metadata()
        n = 0
        for f in sorted(self.data_dir.glob("*.py")):
            if limit is not None and n >= limit:
                return
            src = f.read_text()
            name = f.stem
            kernel_src, test_block = _split_kernel_and_test(src)
            has_test = bool(test_block.strip())

            row_meta = meta.get(f.name, {})

            yield Seed(
                id=f"tritonbench_g/{name}",
                source_dataset="tritonbench_g",
                original_prompt=_build_prompt(name, kernel_src),
                reference_solution=src,
                expected_output=None,
                tests="",
                evaluator_info={
                    "kind": "tritonbench_g",
                    "file_path": str(f),
                    "kernel_src": kernel_src,
                    "test_block": test_block,
                    "has_test": has_test,
                    "harness": "TritonBench-G result_gold dict allclose",
                },
                metadata={
                    "filename": f.name,
                    "difficulty": row_meta.get("difficulty"),
                    "repo": row_meta.get("repo"),
                    "simp_instru": row_meta.get("simp_instru"),
                    "comp_instru": row_meta.get("comp_instru"),
                    "output_triton_len": row_meta.get("output_triton_len"),
                },
            )
            n += 1
