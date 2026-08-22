"""ArchXBench adapter.

Layout (per design, 71 designs across level-{0,1a,1b,1c,2,3,4,5,6}):
  <root>/level-<L>/<design>/
    problem-description.txt   functional description  → Seed.original_prompt (part)
    design-specs.txt          adaptation constraints (module signature pinned)
    tb.v | tb_<name>.v        self-checking testbench (Passed:N Failed:M)
    scripts/compare_outputs.py + outputs/golden_output.json   (19 DSP designs only)

Maps onto the 4-part decomposition:
  Prompt  = problem-description.txt
  Spec    = design-specs.txt (pins module name / ports / signature)
  Golden  = NONE (ArchXBench ships no reference RTL — teacher must bootstrap it)
  Test    = self-check tb.v (iverilog) or golden_output.json compare (DSP)

Mirrors eval_harness/benches/run_archx.py (find_designs / tb_file / build_prompt).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_ROOT = Path(
    os.environ.get(
        "ARCHX_ROOT",
        str(Path(__file__).resolve().parents[2] / "benchmarks" / "ArchXBench"),
    )
)


def _tb_file(design_dir: Path) -> Optional[Path]:
    """Pick the testbench file. Mirrors run_archx.py:tb_file()."""
    cands = [f for f in os.listdir(design_dir)
             if f.endswith(".v") and (f == "tb.v" or f.startswith("tb")
                                      or "testbench" in f)]
    for pref in ("tb.v", "testbench.v"):
        if pref in cands:
            return design_dir / pref
    return (design_dir / cands[0]) if cands else None


def _top_module(specs: str, tb_text: str, fallback: str) -> str:
    """Extract the pinned top-module name.

    design-specs.txt carries a ``module <name> (`` signature; the tb also
    instantiates ``<name> dut (...)``. Prefer the spec signature, then a
    ``Module Name:\n- <name>`` bullet, then the tb DUT instantiation.
    """
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)\s*[#(]", specs)
    if m:
        return m.group(1)
    m = re.search(r"Module\s+Name\s*:\s*\n\s*-\s*([A-Za-z_]\w*)", specs)
    if m:
        return m.group(1)
    # tb instantiation: "<name> dut (" or "<name> uut ("
    m = re.search(r"^\s*([A-Za-z_]\w*)\s+(?:dut|uut|u_dut|DUT)\s*\(",
                  tb_text, re.MULTILINE)
    if m:
        return m.group(1)
    return fallback


@register_dataset("archxbench")
class ArchXBenchAdapter:
    name = "archxbench"

    def __init__(self, root: Optional[str] = None,
                 golden_file: Optional[str] = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        # Optional bootstrapped-golden map: {seed_id or design_name: rtl}.
        # ArchXBench ships no golden, so benchevolver/inversecoder (which
        # require seed.reference_solution) need one injected from a prior
        # bootstrap verify. Build with scripts/archx_build_golden_map.py.
        self.golden: Dict[str, str] = {}
        gf = golden_file or os.environ.get("ARCHX_GOLDEN_FILE")
        if gf and Path(gf).exists():
            for line in Path(gf).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                code = (rec.get("golden") or rec.get("code")
                        or rec.get("reference_solution") or "")
                for key in (rec.get("seed_id"), rec.get("design_name"),
                            rec.get("id")):
                    if key and code:
                        self.golden[key] = code
        # Optional design allowlist (env ARCHX_ONLY_DESIGNS = path to a file with
        # one design name per line). Lets evol/bench focus v4-pro on the designs
        # the teacher CAN solve, instead of wasting timeouts on unsolvable ones
        # (the ~35 hard DSP/arith designs that plateau the evol pool).
        self.only_designs: Optional[set] = None
        odf = os.environ.get("ARCHX_ONLY_DESIGNS")
        if odf and Path(odf).exists():
            names = {ln.strip() for ln in Path(odf).read_text().splitlines()
                     if ln.strip()}
            if names:
                self.only_designs = names

    def iter_seeds(self, limit: Optional[int] = None,
                   levels: Optional[str] = None,
                   only_selfcheck: bool = False,
                   **_kw) -> Iterator[Seed]:
        """Yield one Seed per design.

        levels: comma-separated level filter, e.g. "0,1a,2" (matches the
            ``level-<L>`` dir suffix). None = all.
        only_selfcheck: if True, skip DSP designs whose oracle is the
            golden_output.json compare (scripts/compare_outputs.py present),
            keeping only the lighter iverilog self-check tb designs.
        """
        wanted = None
        if levels:
            wanted = {f"level-{l.strip()}" for l in levels.split(",")}
        n = 0
        for lvl_dir in sorted(self.root.glob("level-*")):
            if not lvl_dir.is_dir():
                continue
            if wanted is not None and lvl_dir.name not in wanted:
                continue
            for design_dir in sorted(p for p in lvl_dir.iterdir() if p.is_dir()):
                if self.only_designs is not None and \
                        design_dir.name not in self.only_designs:
                    continue
                desc = design_dir / "problem-description.txt"
                if not desc.exists():
                    continue
                tb = _tb_file(design_dir)
                if tb is None:
                    continue
                has_golden = (design_dir / "scripts" / "compare_outputs.py").exists()
                if only_selfcheck and has_golden:
                    continue
                if limit is not None and n >= limit:
                    return
                specs_path = design_dir / "design-specs.txt"
                specs = specs_path.read_text() if specs_path.exists() else ""
                desc_text = desc.read_text()
                tb_text = tb.read_text()
                top = _top_module(specs, tb_text, design_dir.name)
                level = lvl_dir.name.replace("level-", "")
                seed_id = f"archxbench/level-{level}/{design_dir.name}"
                golden = (self.golden.get(seed_id)
                          or self.golden.get(design_dir.name) or "")
                prompt = desc_text
                if specs.strip():
                    prompt = f"{desc_text}\n\n## Design Specification\n{specs}"
                yield Seed(
                    id=seed_id,
                    source_dataset="archxbench",
                    original_prompt=prompt,
                    reference_solution=golden,  # "" unless a bootstrap golden is injected
                    expected_output=None,
                    tests=tb_text,
                    evaluator_info={
                        "kind": "archxbench",
                        "design_name": design_dir.name,
                        "level": level,
                        "top_module": top,
                        "design_dir": str(design_dir),
                        "prompt_path": str(desc),
                        "spec_path": str(specs_path) if specs_path.exists() else None,
                        "tb_path": str(tb),
                        "has_golden": has_golden,
                        "golden_dir": str(design_dir / "outputs") if has_golden else None,
                        "harness": (
                            "iverilog -g2012 gen.v + tb → vvp; self-check "
                            "'Passed:N Failed:M'" if not has_golden else
                            "iverilog gen.v + tb → vvp dump; "
                            "scripts/compare_outputs.py vs outputs/golden_output.json (tol ±1)"
                        ),
                    },
                    metadata={
                        "design_name": design_dir.name,
                        "level": level,
                        "top_module": top,
                        "has_golden": has_golden,
                    },
                )
                n += 1
