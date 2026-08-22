"""TritonBench-T adapter.

T split layout differs subtly from G:
  - G ref language: Triton (@triton.jit kernel + wrapper + test_xxx)
  - T ref language: PyTorch (standalone `def foo(...)` calling torch ops)
  - Both use the 50-char `#` separator and a `test_xxx()` returning a
    dict of test cases.
  - Variable name differs: G uses `result_gold = test_xxx()`,
    T uses `test_results = test_xxx()`.

T pipeline shape: PyTorch ref → Triton SFT target (like KB), but with
TBG-style discrete test_xxx() verification (NOT KB-style random fuzz).

For inversecoder / perturb we keep:
  - reference_solution = full ref .py source (function + test block)
  - evaluator_info     = {kind, file_path, func_src, test_block,
                          test_func_name, test_var_name="test_results",
                          has_test, harness} so downstream verify can
                          run the candidate's Triton against the ref's
                          test harness.
  - metadata           = row from TritonBench_T_v1.jsonl (difficulty,
                          repo, simp_instru, comp_instru, output_triton_len)
                          for cross-reference.
"""

from __future__ import annotations
import os

import ast
import json
import re
from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset


DEFAULT_ROOT = Path(os.environ.get("TRITONBENCH_ROOT", str(Path(__file__).resolve().parents[2] / "benchmarks" / "TritonBench")))

_TEST_SEP_NEEDLE = "#" * 50


def _split_func_and_test(src: str) -> tuple[str, str]:
    """Split a TritonBench-T ref into (function_source, test_block).

    Same logic as G — `#` × 50 separator marks the boundary.
    """
    if _TEST_SEP_NEEDLE in src:
        idx = src.find(_TEST_SEP_NEEDLE)
        func = src[:idx].rstrip()
        rest = src[idx:]
        rest = rest.lstrip("#").lstrip("\n")
        return func + "\n", rest
    return src, ""


def _extract_test_func_name(test_block: str) -> Optional[str]:
    """Find the `def test_xxx()` name in the test block.

    Returns the function name (e.g. "test_abs") or None if not found.
    """
    m = re.search(r"^def\s+(test_\w+)\s*\(", test_block, re.MULTILINE)
    return m.group(1) if m else None


def _extract_primary_func_name(func_src: str) -> Optional[str]:
    """Find the FIRST top-level `def <name>(...)` in the function source.

    Returns the function name or None on parse error / no def found.
    """
    try:
        tree = ast.parse(func_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _build_prompt(name: str, func_src: str) -> str:
    """Default passthrough prompt — inversecoder / perturb replaces this."""
    return (
        f"You are given a PyTorch function `{name}` that calls "
        f"`torch` ops directly. Re-implement the same computation as a "
        f"Triton kernel that is functionally equivalent to the reference. "
        f"Reference PyTorch implementation:\n\n"
        f"```python\n{func_src.rstrip()}\n```\n"
    )


def _load_metadata(jsonl_path: Path) -> dict[str, dict]:
    """Load TritonBench_T_v1.jsonl → {filename: row}.

    The TBT manifest is JSONL (vs TBG's JSON array). Each line is one row.
    """
    if not jsonl_path.exists():
        return {}
    out = {}
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if "file" in r:
                out[r["file"]] = r
        except json.JSONDecodeError:
            continue
    return out


@register_dataset("tritonbench_t")
class TritonBenchTAdapter:
    name = "tritonbench_t"

    def __init__(self, root: Optional[str] = None,
                 metadata_filename: str = "data/TritonBench_T_v1.jsonl",
                 data_subdir: str = "data/TritonBench_T_v1"):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.data_dir = self.root / data_subdir
        self.metadata_path = self.root / metadata_filename

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        meta = _load_metadata(self.metadata_path)
        n = 0
        for f in sorted(self.data_dir.glob("*.py")):
            if limit is not None and n >= limit:
                return
            src = f.read_text()
            name = f.stem
            func_src, test_block = _split_func_and_test(src)
            has_test = bool(test_block.strip())
            test_func_name = _extract_test_func_name(test_block)
            primary_func_name = _extract_primary_func_name(func_src)

            # Skip rows we can't parse — better to silently drop than emit
            # a half-formed seed.
            if primary_func_name is None or test_func_name is None:
                continue

            row_meta = meta.get(f.name, {})

            yield Seed(
                id=f"tritonbench_t/{name}",
                source_dataset="tritonbench_t",
                original_prompt=_build_prompt(primary_func_name, func_src),
                reference_solution=src,
                expected_output=None,
                tests="",
                evaluator_info={
                    "kind": "tritonbench_t",
                    "file_path": str(f),
                    "func_src": func_src,
                    "test_block": test_block,
                    "primary_func_name": primary_func_name,
                    "test_func_name": test_func_name,
                    "test_var_name": "test_results",  # T uses this; G uses result_gold
                    "has_test": has_test,
                    "harness": "TritonBench-T test_results dict allclose",
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
