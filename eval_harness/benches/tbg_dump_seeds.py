#!/usr/bin/env python3
"""tbg_dump_seeds.py — dump original TritonBench G/T seeds to a jsonl that
teacher_triton_rollout_tbg.py / tbg_verify_correctness.py can consume.

Each line is a Seed.to_record() dict (id, original_prompt, reference_solution,
evaluator_info{test_block, test_var_name, ...}, metadata{triton_interface,...}).
The teacher script does its own passthrough/missing-spec filtering, so this
emits raw seeds from the bench dataset PLUS the triton_interface the teacher's
format_check + prompt need (primary_wrapper etc.) computed from the ref.

Usage:
  PYTHONPATH=<SFT_ROOT> python tbg_dump_seeds.py <split=g|t> <out.jsonl> [root] [limit]
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Requires the SFT expand package on the path (set PYTHONPATH=$SFT_ROOT).
from expand.datasets.tritonbench_g import TritonBenchGAdapter
from expand.datasets.tritonbench_t import TritonBenchTAdapter
from expand.methods.inversecoder import _extract_triton_interface


def _attach_interface(split: str, rec: dict) -> dict:
    """Compute the Triton interface (primary_wrapper, signature, kernel_names,
    test_func_name, ...) from the ref source and store it as
    metadata.triton_interface — exactly where teacher_triton_rollout_tbg.py
    reads it (`row.metadata.triton_interface or row.evaluator_info`). Without
    this the teacher's format_check has an empty primary_wrapper and flags
    every row missing_wrapper, and the prompt never tells the model which
    wrapper name the test_block calls."""
    ref = rec.get("reference_solution", "")
    iface = None
    try:
        if split == "g":
            iface = _extract_triton_interface(ref)
        else:  # t — uses a different extractor
            from expand.methods._perturb_common import _extract_tbt_func_interface  # was `inversecoder` (wrong module) → ImportError → swallowed by except → triton_interface=None → TBT 0/166 (teacher prompt had no primary_wrapper → model output kernel-only → format_fail). FIXED 2026-07-20.
            iface = _extract_tbt_func_interface(rec.get("evaluator_info", {}).get("func_src", ""))
    except Exception:
        iface = None
    md = rec.setdefault("metadata", {})
    # _extract_tbt_func_interface returns a plain dict; _extract_triton_interface
    # returns a TritonInterface dataclass. asdict() only works on dataclasses;
    # calling it on a dict raises TypeError (was crashing the TBT dump → 0 seeds
    # → triton_interface=None → TBT 0/166). Handle both.
    if iface is None:
        md["triton_interface"] = None
    elif isinstance(iface, dict):
        md["triton_interface"] = iface
    else:
        md["triton_interface"] = asdict(iface)
    return rec


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "g"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("seeds.jsonl")
    root = sys.argv[3] if len(sys.argv) > 3 else None
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

    cls = {"g": TritonBenchGAdapter, "t": TritonBenchTAdapter}.get(split)
    if cls is None:
        raise SystemExit(f"unknown split {split!r} (use g|t)")
    adapter = cls(root=root)

    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_iface = 0
    with out.open("w") as f:
        for seed in adapter.iter_seeds(limit=limit):
            rec = seed.to_record()
            # Teacher's load_rows drops rows without metadata.assistant_spec
            # (treats them as un-expanded passthrough) and reads `expanded_prompt`
            # for the prompt. Raw bench seeds have `original_prompt` + no plan,
            # so for EVAL we synthesize both: prompt <- original_prompt, and a
            # non-empty assistant_spec marker to pass the filter (the plan slot
            # in the teacher prompt ends up with this text — harmless).
            if not rec.get("expanded_prompt"):
                rec["expanded_prompt"] = rec.get("original_prompt", "")
            md = rec.setdefault("metadata", {})
            if not md.get("assistant_spec"):
                md["assistant_spec"] = "(direct implementation from spec)"
            # Attach the triton interface so format_check + the prompt know the
            # wrapper contract.
            _attach_interface(split, rec)
            if md.get("triton_interface"):
                n_iface += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[tbg-dump-seeds] split={split} wrote {n} seeds "
          f"({n_iface} with triton_interface) -> {out}", flush=True)


if __name__ == "__main__":
    main()
