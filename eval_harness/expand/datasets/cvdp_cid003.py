"""CVDP cid003 adapter.

Source jsonl: ``cvdp_v1.0.4_nonagentic_code_generation_no_commercial.nonjudge.jsonl``
contains multiple categories. We filter to ``cid003`` (RTL code generation).

Each row schema (per CVDP docs / inspection):
  {
    "id": str,
    "categories": ["cid003", "<difficulty>"],
    "input":   {"prompt": str, "context": dict},
    "output":  {"response": str, "context": {"<rtl path>": str, ...}},
    "harness": {"files": {"docker-compose.yml": str, "src/.env": str,
                          "src/test_*.py": str, "src/test_runner.py": str, ...}},
  }

We keep:
  - reference_solution = first non-empty value in output.context (RTL).
                          For cid003 this is empty in the public dataset
                          (placeholder paths only); pass ``teacher_refs_path``
                          to merge in teacher-rolled refs (see
                          scripts/cvdp_extract_teacher_refs.py).
  - tests              = a packed JSON view of harness.files
  - evaluator_info     = enough for the existing CVDP run_samples.py harness
                          (the source jsonl path + id) to re-run this row.
                          Adds ``design_name`` extracted from the first
                          ``expected_rtl_paths`` entry — inversecoder uses
                          it to anchor the spec/back-derive output to the
                          correct module name.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator, Optional

from ..base import Seed
from ..registry import register_dataset

DEFAULT_PATH = Path(
    os.environ.get(
        "CVDP_DATASET",
        str(
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "CVDP"
            / "datasets"
            / "cvdp_v1.0.4_nonagentic_code_generation_no_commercial.nonjudge.jsonl"
        ),
    )
)


def _pick_reference(output: dict) -> str:
    ctx = (output or {}).get("context") or {}
    for _, v in ctx.items():
        if isinstance(v, str) and v.strip():
            return v
    resp = (output or {}).get("response") or ""
    return resp if isinstance(resp, str) else ""


_RTL_PATH_RE = re.compile(r"(?:^|/)([A-Za-z_]\w*)\.s?v$")
_TOPLEVEL_RE = re.compile(r"^\s*TOPLEVEL\s*=\s*(\S+)\s*$", re.MULTILINE)


def _design_name_from_paths(paths: list) -> Optional[str]:
    """Extract module name from the first ``rtl/<name>.sv`` path.

    e.g. ``["rtl/encoder_64b66b.sv"]`` -> ``"encoder_64b66b"``. Returns
    None if nothing matches. NOTE: the file name is NOT always the same
    as the harness's TOPLEVEL — prefer ``_design_name_from_env`` and use
    this only as a fallback.
    """
    for p in paths or []:
        m = _RTL_PATH_RE.search(p)
        if m:
            return m.group(1)
    return None


def _design_name_from_env(harness_files: dict) -> Optional[str]:
    """Read the harness's ``src/.env`` and return the ``TOPLEVEL`` value.

    The cocotb harness binds DUT lookup to TOPLEVEL — if the model emits
    ``module <X>`` for a different X, elaboration fails before a single
    test runs. CVDP has a handful of seeds where the file basename and
    TOPLEVEL diverge (e.g. file ``priority_encoder.v`` but TOPLEVEL
    ``priority_encoder_8x3``); for those, file name is wrong.
    """
    env = (harness_files or {}).get("src/.env", "")
    if not env:
        return None
    m = _TOPLEVEL_RE.search(env)
    return m.group(1) if m else None


def _load_teacher_refs(path: Path) -> dict[str, str]:
    """Map raw row_id -> verified RTL string from a teacher_refs jsonl."""
    refs: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = (row.get("evaluator_info") or {}).get("row_id")
            if not row_id:
                # Tolerate the SFT-side id form ``cvdp_cid003/<row_id>``.
                full = row.get("id") or ""
                if "/" in full:
                    row_id = full.split("/", 1)[1]
            ref = row.get("reference_solution") or ""
            if row_id and ref.strip():
                refs[row_id] = ref
    return refs


@register_dataset("cvdp_cid003")
class CVDPcid003Adapter:
    name = "cvdp_cid003"

    def __init__(self, path: Optional[str] = None,
                 category: str = "cid003",
                 teacher_refs_path: Optional[str] = None,
                 require_ref: bool = False):
        self.path = Path(path) if path else DEFAULT_PATH
        self.category = category
        self.teacher_refs_path = (
            Path(teacher_refs_path) if teacher_refs_path else None)
        self.require_ref = require_ref
        self._refs: dict[str, str] = {}
        if self.teacher_refs_path:
            if not self.teacher_refs_path.is_file():
                raise FileNotFoundError(
                    f"teacher_refs_path not found: {self.teacher_refs_path}")
            self._refs = _load_teacher_refs(self.teacher_refs_path)

    def iter_seeds(self, limit: Optional[int] = None,
                   **_kw) -> Iterator[Seed]:
        n = 0
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cats = row.get("categories") or []
                if self.category not in cats:
                    continue

                row_id = row["id"]
                inp = row.get("input") or {}
                out = row.get("output") or {}
                harness = row.get("harness") or {}
                harness_files = harness.get("files") or {}

                ref = _pick_reference(out)
                ref_source = "output.context"
                teacher = None
                if self._refs.get(row_id):
                    ref = self._refs[row_id]
                    ref_source = "teacher_refs"
                    teacher = self.teacher_refs_path.name

                if self.require_ref and not ref.strip():
                    continue
                if limit is not None and n >= limit:
                    return

                expected_rtl_paths = list((out.get("context") or {}).keys())
                # TOPLEVEL from harness/.env is the cocotb DUT name, which
                # is what `module <X>` MUST match for elaboration. File
                # basename (from expected_rtl_paths) is sometimes wrong;
                # use it only as fallback when .env is absent.
                design_name = (_design_name_from_env(harness_files)
                               or _design_name_from_paths(expected_rtl_paths))

                yield Seed(
                    id=f"cvdp_cid003/{row_id}",
                    source_dataset="cvdp_cid003",
                    original_prompt=inp.get("prompt", "") or "",
                    reference_solution=ref,
                    expected_output=None,
                    tests=json.dumps(harness_files, ensure_ascii=False),
                    evaluator_info={
                        "kind": "cvdp_cid003",
                        "row_id": row_id,
                        "categories": cats,
                        "source_jsonl": str(self.path),
                        "expected_rtl_paths": expected_rtl_paths,
                        "harness_file_names": list(harness_files.keys()),
                        "design_name": design_name,
                        "harness": (
                            "CVDP run_samples.py + Icarus cocotb harness "
                            "(docker-compose.yml + src/test_runner.py)"),
                    },
                    metadata={
                        "row_id": row_id,
                        "categories": cats,
                        "ref_source": ref_source,
                        "teacher": teacher,
                    },
                )
                n += 1
