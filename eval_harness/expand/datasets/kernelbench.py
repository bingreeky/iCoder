"""KernelBench adapter.

KernelBench layout:
  <root>/level{1..4}/<NN>_<Name>.py

Each .py defines:
  - ``Model`` (nn.Module) with ``forward``  → reference PyTorch implementation
  - ``get_inputs()``                         → list of input tensors
  - ``get_init_inputs()``                    → constructor args

For RTL/triton work, the file body is the spec + reference; correctness +
speed evaluation is handled by KernelBench's own harness. We keep:
  - reference_solution = full python source (the canonical reference)
  - evaluator_info     = {file_path, level, problem_index} so the KB harness
                         can be re-pointed at this seed.
"""

from __future__ import annotations
import os

from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_ROOT = Path(os.environ.get("KERNELBENCH_ROOT", str(Path(__file__).resolve().parents[2] / "benchmarks" / "KernelBench")))
DEFAULT_LEVELS = ("level1", "level2", "level3", "level4")


def _build_prompt(name: str, source: str) -> str:
    return (
        f"You are given a PyTorch reference implementation named `{name}`.\n"
        f"Re-implement the same computation as a fast, numerically-equivalent "
        f"CUDA / Triton kernel. Reference (PyTorch) below:\n\n"
        f"```python\n{source.rstrip()}\n```\n"
    )


@register_dataset("kernelbench")
class KernelBenchAdapter:
    name = "kernelbench"

    def __init__(self, root: Optional[str] = None,
                 levels=DEFAULT_LEVELS):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.levels = tuple(levels)

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        n = 0
        for level in self.levels:
            level_dir = self.root / level
            if not level_dir.exists():
                continue
            for f in sorted(level_dir.glob("*.py")):
                if limit is not None and n >= limit:
                    return
                src = f.read_text()
                # filename pattern "NN_Name.py" → split index/name
                stem = f.stem
                idx, _, name = stem.partition("_")
                yield Seed(
                    id=f"kernelbench/{level}/{stem}",
                    source_dataset="kernelbench",
                    original_prompt=_build_prompt(name or stem, src),
                    reference_solution=src,
                    expected_output=None,
                    tests="",  # KernelBench uses functional + speed harness
                    evaluator_info={
                        "kind": "kernelbench",
                        "file_path": str(f),
                        "level": level,
                        "problem_id": idx,
                        "problem_name": name,
                        "harness": "KernelBench correctness + speed eval",
                    },
                    metadata={
                        "level": level,
                        "filename": f.name,
                    },
                )
                n += 1
