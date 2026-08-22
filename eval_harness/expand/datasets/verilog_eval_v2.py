"""VerilogEval v2 (spec-to-rtl) adapter.

Layout (per problem ``Prob<NNN>_<slug>``):
  <root>/<prob>_prompt.txt   spec / module interface
  <root>/<prob>_ref.sv       reference RTL (canonical solution)
  <root>/<prob>_test.sv      testbench

Mirrors what eval/verilog_eval_v2_runner.py consumes.
"""

from __future__ import annotations
import os

from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_ROOT = Path(os.environ.get("VERILOGEVAL_ROOT", str(Path(__file__).resolve().parents[2] / "benchmarks" / "verilog-eval" / "dataset_spec-to-rtl")))


@register_dataset("verilog_eval_v2")
class VerilogEvalV2Adapter:
    name = "verilog_eval_v2"

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        n = 0
        for prompt_file in sorted(self.root.glob("*_prompt.txt")):
            prob = prompt_file.stem.replace("_prompt", "")
            ref = self.root / f"{prob}_ref.sv"
            test = self.root / f"{prob}_test.sv"
            if not (ref.exists() and test.exists()):
                continue
            if limit is not None and n >= limit:
                return
            yield Seed(
                id=f"verilog_eval_v2/{prob}",
                source_dataset="verilog_eval_v2",
                original_prompt=prompt_file.read_text(),
                reference_solution=ref.read_text(),
                expected_output=None,
                tests=test.read_text(),
                evaluator_info={
                    "kind": "verilog_eval_v2",
                    "prob_id": prob,
                    "prompt_path": str(prompt_file),
                    "ref_path": str(ref),
                    "test_path": str(test),
                    "top_module": "TopModule",
                    "harness": "iverilog test.sv + ref.sv + gen.sv → vvp; pass marker 'Mismatches: 0'",
                },
                metadata={"prob_id": prob},
            )
            n += 1
