"""RealBench adapter.

RealBench (arXiv 2507.16200, IPRC-DIP) — real-world IP design tasks harvested
from three open-source projects. 3 systems / 60 module-level tasks:
  aes (6) / sdc (14) / e203_hbirdv2 (40).

Layout:
  <root>/problems/<sys>/problems.jsonl   {"task": <module>, "problem": <md doc>}
  <root>/<sys>/<module>/
      <module>.v                          Golden RTL (original open-source code)
      <module>.md                         design document (source of `problem`)
      verification/
          <module>_ref.sv                 Ref (golden, renamed ref_<module>)
          <module>_top.sv                 DUT slot (candidate goes here)
          <module>_stimulus_gen.sv        stimulus
          <module>_testbench.sv           dual-instantiation compare (ref vs top)
          Makefile                        verilator `make all`
  <root>/benchmark_info.py                per-module dependency lists

Maps onto the 4-part decomposition:
  Prompt  = functional part of <module>.md   (in problems.jsonl `problem`)
  Spec    = interface table + injected defines + include reqs + dep tree
  Golden  = <sys>/<module>/<module>.v ; Ref = verification/<module>_ref.sv
  Test    = verilator dual-instantiation (make all) [+ jaspergold formal, opt]

Mirrors RealBench/run_verify.py (testbench_verification / formal_verification).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_ROOT = Path(
    os.environ.get(
        "RB_ROOT",
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "RealBench"),
    )
)
SYSTEMS = ("aes", "sdc", "e203_hbirdv2")


def _load_deps(root: Path) -> Dict[str, List[str]]:
    """Flatten benchmark_info.py's per-module dependency lists (best-effort)."""
    info_path = root / "benchmark_info.py"
    if not info_path.exists():
        return {}
    ns: Dict[str, object] = {}
    try:
        exec(compile(info_path.read_text(), str(info_path), "exec"), ns)
    except Exception:
        return {}
    deps: Dict[str, List[str]] = {}
    for _sys, comps in (ns.get("benchmark_info") or {}).items():
        if isinstance(comps, dict):
            for mod, d in comps.items():
                deps[mod] = list(d) if isinstance(d, (list, tuple)) else []
    return deps


@register_dataset("realbench")
class RealBenchAdapter:
    name = "realbench"

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def iter_seeds(self, limit: Optional[int] = None,
                   systems: Optional[str] = None,
                   max_ref_lines: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        """Yield one Seed per module task.

        systems: comma-separated filter over {aes,sdc,e203_hbirdv2}. None = all.
        max_ref_lines: skip modules whose golden RTL exceeds this many lines
            (the e203 core/cpu/exu giants — 800..1843 lines — are only usable
            for direct / InverseCoder-rephrase, not perturbation).
        """
        wanted = None
        if systems:
            wanted = {s.strip() for s in systems.split(",")}
        deps = _load_deps(self.root)
        n = 0
        for sys_name in SYSTEMS:
            if wanted is not None and sys_name not in wanted:
                continue
            prob_file = self.root / "problems" / sys_name / "problems.jsonl"
            if not prob_file.exists():
                continue
            for line in prob_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                module = rec["task"]
                prompt = rec["problem"]
                mod_dir = self.root / sys_name / module
                ref = mod_dir / f"{module}.v"
                golden = ref.read_text() if ref.exists() else ""
                verif_dir = mod_dir / "verification"
                tb = verif_dir / f"{module}_testbench.sv"
                ref_lines = golden.count("\n") + 1 if golden else 0
                if max_ref_lines is not None and ref_lines > max_ref_lines:
                    continue
                if limit is not None and n >= limit:
                    return
                yield Seed(
                    id=f"realbench/{sys_name}/{module}",
                    source_dataset="realbench",
                    original_prompt=prompt,
                    reference_solution=golden,  # Golden ships (open-source RTL)
                    expected_output=None,
                    tests=tb.read_text() if tb.exists() else "",
                    evaluator_info={
                        "kind": "realbench",
                        "system": sys_name,
                        "module": module,
                        "module_dir": str(mod_dir),
                        "ref_path": str(ref) if ref.exists() else None,
                        "verification_dir": str(verif_dir) if verif_dir.exists() else None,
                        "ref_sv_path": str(verif_dir / f"{module}_ref.sv"),
                        "top_sv_path": str(verif_dir / f"{module}_top.sv"),
                        "dependencies": deps.get(module, []),
                        "top_module": module,
                        "harness": (
                            "verilator `make all` in verification/ "
                            "(dual-instantiation ref vs top, 'no mismatches'); "
                            "optional yosys+jaspergold -sec formal equivalence"
                        ),
                    },
                    metadata={
                        "system": sys_name,
                        "module": module,
                        "ref_lines": ref_lines,
                        "dependencies": deps.get(module, []),
                    },
                )
                n += 1
