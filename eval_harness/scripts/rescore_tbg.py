#!/usr/bin/env python3
# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""rescore_tbg.py — Phase 1+2+3 re-extraction+re-verify for TritonBench-G.

Why: the production TBG extractor (teacher_triton_rollout_tbg:174
CODE_FENCE_RE) grabs the FIRST fenced block. When the model emits an
explanation/skeleton fence first and the answer fence second, the real
@triton.jit kernel is dropped -> teacher_code is a near-empty fragment
(often ~3 chars) -> the row is classified request_failure (model_execution)
and scored correct=False. This re-parses teacher_traj_content (the v4-pro
<answer> wrap that retains the dropped block) with pick_triton (longest
@triton.jit+def block) and, for complete candidates the extractor missed,
re-runs the existing verify_tbg() to see if the failure flips to correct.

Only FAILED, non-infra rows can recover. Some models = 0 candidates (their
traj retains only the single already-extracted block). Others may have several
candidates (teacher_code is short, traj holds the real kernel).

MUST run on venv-vllm (triton 3.6) — verify_tbg spawns CUDA subprocesses.
Each call is cold-init, so candidates run sequentially (only 3).

Usage:
  eval_harness/.venv-vllm/bin/python scripts/rescore_tbg.py --model <model>
"""
from __future__ import annotations
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # eval_harness/
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)        # _rescore_extract
sys.path.insert(0, ROOT)          # verify.*

from _rescore_extract import pick_triton  # noqa: E402
from verify.tritonbench import verify_tbg  # noqa: E402
from verify.core import is_infra          # noqa: E402

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))


def load_rows(model: str):
    p = os.path.join(RESULTS_DIR, model, "tritonbench_g", "verified.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()], p


def phase1(rows):
    """Failed, non-infra rows whose traj yields a complete @triton.jit kernel
    differing from the on-disk teacher_code."""
    cands = []
    for r in rows:
        if r.get("correct"):
            continue
        if is_infra(r):
            continue
        tc = r.get("teacher_code") or ""
        tj = r.get("teacher_traj_content") or ""
        new, status = pick_triton(tj)
        if status == "complete" and new and new.strip() != tc.strip():
            cands.append((r, new))
    return cands


def phase2(cands, timeout=180):
    """Re-verify each candidate via verify_tbg on the current (venv-vllm)
    runtime. Compare new correct to old correct."""
    rows_out = []
    flips = 0
    for r, new in cands:
        ref = r.get("reference_solution") or ""
        ei = r.get("evaluator_info") or {}
        tb = ei.get("test_block") or ""
        atol = float(ei.get("atol") or 1e-2)
        rtol = float(ei.get("rtol") or 1e-2)
        old_correct = bool(r.get("verify_correct"))
        err = ""
        if not ref or not tb or not new:
            res = {"correct": False, "compiled": False,
                   "info": "missing ref/test_block/new"}
        else:
            try:
                res = verify_tbg(ref, new, tb, timeout=timeout,
                                 atol=atol, rtol=rtol, seed=0,
                                 measure_perf=False)
            except Exception as e:
                err = f"verify_tbg raised: {e}"
                res = {"correct": False, "compiled": False, "info": err}
        new_correct = bool(res.get("correct"))
        new_compiled = bool(res.get("compiled"))
        flipped = new_correct and not old_correct
        if flipped:
            flips += 1
        rows_out.append({"id": r["id"],
                         "old_correct": old_correct, "new_correct": new_correct,
                         "new_compiled": new_compiled, "flipped": flipped,
                         "old_tc_len": len(r.get("teacher_code") or ""),
                         "new_tc_len": len(new),
                         "info": (res.get("info") or "")[:160]})
        flag = "FLIP" if flipped else "    "
        print(f"  [{flag}] {r['id']}: correct {old_correct}->{new_correct} "
              f"compiled={new_compiled} | {(res.get('info') or '')[:70]}", flush=True)
    return flips, rows_out


def phase3(rows, flips, model):
    """Recompute corr@1 = 100*corr/n (n = non-infra rows), with flipped
    candidates counted correct. Mirrors summarize.sh tbg()."""
    n = corr = nskip = 0
    flipped_ids = {r["id"] for r in rows if r.get("flipped")}
    for r in rows:
        if is_infra(r):
            nskip += 1
            continue
        n += 1
        if r.get("verify_correct") or r["id"] in flipped_ids:
            corr += 1
    new_corr1 = round(100.0 * corr / n, 2) if n else 0.0
    old_corr = sum(1 for r in rows if r.get("verify_correct"))
    # n/corr from the original (without flips) for the old number
    old_n = n
    old_corr1 = round(100.0 * old_corr / old_n, 2) if old_n else 0.0
    return old_corr1, new_corr1, n, old_corr, nskip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    print(f"\n=== TBG re-extract+re-verify: {a.model} ===")
    rows, verified_path = load_rows(a.model)
    print(f"loaded {len(rows)} rows from {verified_path}")
    cands = phase1(rows)
    print(f"Phase-1: {len(cands)} recovery candidates (failed non-infra rows "
          f"with a complete dropped kernel in traj)")
    if not cands:
        print("-> 0 candidates; TBG score stands. (some models' traj retains only "
              "the single already-extracted block.)")
        return
    print("Phase-2: re-verifying on venv-vllm (triton 3.6) ...")
    flips, rows_out = phase2(cands, timeout=a.timeout)
    old, new, n, oc, nskip = phase3(rows, rows_out, a.model)
    out_path = os.path.join(RESULTS_DIR, a.model, "tritonbench_g", "tbg.rescored.jsonl")
    meta = {"model": a.model, "candidates": len(cands), "flips": flips,
            "old_corr_at_1": old, "new_corr_at_1": new,
            "n_scored": n, "old_correct": oc, "skip": nskip}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[tbg {a.model}] candidates={len(cands)} flips(fail->pass)={flips}  "
          f"corr@1 {old}->{new}  (n={n} skip={nskip})  -> {out_path}")


if __name__ == "__main__":
    main()
