#!/usr/bin/env python3
"""
Compute the KernelBench "fast" (effective speedup) metric — and the canonical
comp/corr KB column — from eval_results.json.

``compute_kb_level()`` is the SINGLE source of truth for one level's KB numbers;
both this CLI and ``summarize.sh``'s table ``kb()`` call it, so the table and
the standalone tool can never disagree on sample set / denominator / failure
classification.

For each CORRECT sample's stored model runtime (eval_results.json ->
runtime_stats.mean, A100/cuda/100-trial/fp32) paired with the A100 reference
runtime (_kb_ref_timing_A100.json -> runtime_stats.mean, same config),
effective_speedup = ref_runtime / model_runtime. Reported:
  n              — scored denominator (all rows minus generate-stage infra)
  comp / corr    — compiled / correct counts (the table's KB column)
  n_with_ref     — correct samples that also have a ref timing (some ref
                   kernels fail to time -> excluded)
  fast_1x        — share of n_with_ref with speedup > 1x  (model beats eager)
  fast_2x        — share of n_with_ref with speedup > 2x
  median/mean_speedup

Speedup>1 means the model's generated kernel beats the PyTorch eager reference
on the SAME A100 with the SAME inputs — i.e. a genuine "fast" kernel.

Read-only: parses JSON only. It never executes model-generated code.

Handles both sample-container shapes seen in eval_results.json:
  - list:  [{sample_id, correctness, runtime_stats, ...}, ...]   (most models)
  - dict:  {"1": {...}, "2": {...}, ...}                           (a reasoning
            model) — values are sample dicts keyed by sample-id string.
"""
import json
import os
import statistics
import sys
from pathlib import Path

REF_FILE = os.environ.get(
    "KB_REF_TIMING_FILE",
    str(Path(__file__).resolve().parents[1] / "results" / "_kb_ref_timing_A100.json"),
)
RES_ROOT = os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parents[1] / "results"))
LEVELS = [1, 2, 3]


def _discover_models(res_root: str) -> list[str]:
    """Auto-discover model dirs under RES_ROOT that have a kernelbench/
    subdir. Override with EVAL_MODELS=a,b,c. Empty list if none found."""
    env = os.environ.get("EVAL_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    if not os.path.isdir(res_root):
        return []
    out = []
    for name in sorted(os.listdir(res_root)):
        if os.path.isdir(os.path.join(res_root, name, "kernelbench")):
            out.append(name)
    return out


def _rows_with_pid(eval_results):
    """Yield (pid, sample_row) for every sample dict carrying a 'compiled' key,
    tolerant of the list/dict sample-container shapes KB produces. Mirrors
    summarize.sh's KB-column walker so both count one row per problem."""
    if not isinstance(eval_results, dict):
        return
    for pid, samps in eval_results.items():
        if isinstance(samps, list):
            for s in samps:
                if isinstance(s, dict) and "compiled" in s:
                    yield pid, s
        elif isinstance(samps, dict):
            for s in samps.values():
                if isinstance(s, dict) and "compiled" in s:
                    yield pid, s


def _is_infra_row(r):
    """'Kernel not found ...' = generate-stage infra failure (no kernel file
    produced — gateway errored during gen). Excluded from the denominator: not
    a model compile failure. Same rule as summarize.sh's KB column."""
    e = str((r.get("metadata") or {}).get("error", ""))
    return e.startswith("Kernel not found")


def load_ref(ref_data, level, pid):
    """Return ref mean (ms) or None."""
    blk = (ref_data or {}).get(f"level{level}", {})
    e = blk.get(str(pid)) or blk.get(pid)
    if not e:
        return None
    rs = e.get("runtime_stats")
    if not rs:
        return None
    m = rs.get("mean")
    return m if isinstance(m, (int, float)) and m > 0 else None


def compute_kb_level(eval_results, ref_data, level):
    """Compute the KB column metrics for one level from an eval_results dict.

    Returns a dict: n (scored denom), comp, corr, drop (generate-stage infra),
    n_correct, n_with_ref, n_fast1x, fast1x, n_fast2x, fast2x,
    median_speedup, mean_speedup.

    Failure classification matches summarize.sh's KB column: rows whose
    ``metadata.error`` starts with "Kernel not found" are dropped from the
    denominator. (Such rows are never ``correct``, so this is also a no-op on
    the correct set that fast is computed over — but applied for consistency so
    the fast denominator and the comp/corr denominator cannot drift.)

    Read-only: parses the already-loaded dict only.
    """
    rows = list(_rows_with_pid(eval_results))
    scored = [(pid, r) for pid, r in rows if not _is_infra_row(r)]
    drop = len(rows) - len(scored)
    n = len(scored)
    comp = sum(1 for _, r in scored if r.get("compiled"))
    corr = sum(1 for _, r in scored if r.get("correctness") is True)
    speedups = []
    n_correct = 0
    n_with_ref = 0
    for pid, r in scored:
        if r.get("correctness") is not True:
            continue
        n_correct += 1
        mrs = r.get("runtime_stats") if isinstance(r.get("runtime_stats"), dict) else {}
        mmean = mrs.get("mean") if isinstance(mrs, dict) else None
        if not (isinstance(mmean, (int, float)) and mmean > 0):
            continue
        rmean = load_ref(ref_data, level, pid)
        if rmean is None:
            continue
        n_with_ref += 1
        speedups.append(rmean / mmean)
    if speedups:
        n_fast1x = sum(1 for x in speedups if x > 1)
        n_fast2x = sum(1 for x in speedups if x > 2)
        fast1x = n_fast1x / len(speedups)
        fast2x = n_fast2x / len(speedups)
        med = statistics.median(speedups)
        avg = statistics.mean(speedups)
    else:
        n_fast1x = n_fast2x = fast1x = fast2x = med = avg = 0
    return {
        "n": n, "comp": comp, "corr": corr, "drop": drop,
        "n_correct": n_correct, "n_with_ref": n_with_ref,
        "n_fast1x": n_fast1x, "fast1x": fast1x,
        "n_fast2x": n_fast2x, "fast2x": fast2x,
        "median_speedup": med, "mean_speedup": avg,
    }


def _self_test():
    """Regression test on a synthetic eval_results + ref timing sample. Pins
    the comp/corr/denominator + fast speedup math so the table and CLI stay
    aligned. Run: python3 scripts/kb_fast_metric.py --selftest"""
    # 4 problems at level 1. ref timing: 10ms each.
    ref = {"level1": {1: {"runtime_stats": {"mean": 10.0}},
                      2: {"runtime_stats": {"mean": 10.0}},
                      3: {"runtime_stats": {"mean": 10.0}},
                      4: {"runtime_stats": {"mean": 10.0}}}}
    # eval_results: pid -> [sample]. scored denom excludes the infra row (p4).
    er = {
        1: [{"compiled": True, "correctness": True,
             "runtime_stats": {"mean": 4.0}}],   # speedup 2.5 -> fast1x + fast2x
        2: [{"compiled": True, "correctness": True,
             "runtime_stats": {"mean": 20.0}}],  # speedup 0.5 -> not fast
        3: [{"compiled": True, "correctness": True,
             "runtime_stats": {"mean": 8.0}}],   # speedup 1.25 -> fast1x only
        4: [{"compiled": False, "correctness": False,
             "metadata": {"error": "Kernel not found: no .py written"}}],  # infra
    }
    m = compute_kb_level(er, ref, 1)
    # denominator: 4 rows, 1 generate-stage infra dropped (p4) -> n=3
    assert m["n"] == 3, m
    assert m["drop"] == 1, m                  # the "Kernel not found" row
    assert m["comp"] == 3, m                  # p1,p2,p3 compiled=True (p4 dropped)
    assert m["corr"] == 3, m                  # p1,p2,p3 correctness=True
    assert m["n_correct"] == 3, m             # p1,p2,p3 correct
    assert m["n_with_ref"] == 3, m           # all 3 correct have a ref + timing
    assert m["n_fast1x"] == 2, m             # p1 (2.5>1) + p3 (1.25>1); p2 0.5 no
    assert m["n_fast2x"] == 1, m             # only p1 (2.5>2)
    assert abs(m["fast1x"] - 2 / 3) < 1e-9, m
    # missing ref for some pids -> those excluded from n_with_ref
    ref2 = {"level1": {1: {"runtime_stats": {"mean": 10.0}}}}  # only pid 1
    m2 = compute_kb_level(er, ref2, 1)
    assert m2["n"] == 3 and m2["n_with_ref"] == 1 and m2["n_fast1x"] == 1, m2
    # no ref data at all -> fast zeroed, comp/corr unaffected
    m3 = compute_kb_level(er, None, 1)
    assert m3["comp"] == 3 and m3["corr"] == 3 and m3["n_with_ref"] == 0, m3
    print("kb_fast_metric self-test: passed (comp/corr/drop + fast speedup math OK)")


def main():
    if "--selftest" in sys.argv:
        _self_test()
        return
    if not os.path.exists(REF_FILE):
        print(f"ERROR: ref timing file not found: {REF_FILE}\n"
              f"Run kb_ref_timing_A100.py first (or set KB_REF_TIMING_FILE).")
        sys.exit(1)
    with open(REF_FILE) as fh:
        ref = json.load(fh)
    ref_ok = sum(1 for lk in ref for v in ref[lk].values() if v.get("runtime_stats"))
    ref_tot = sum(len(ref[lk]) for lk in ref)
    print(f"# ref timing: {ref_ok}/{ref_tot} problems timed OK "
          f"(A100, fp32, cuda_event, 100 trials)")
    print()

    models = _discover_models(RES_ROOT)
    if not models:
        print(f"ERROR: no model results found under {RES_ROOT}/<model>/kernelbench/.\n"
              f"Run summarize.sh <key> first, or set EVAL_MODELS=model1,model2.")
        sys.exit(1)

    rows = []
    for mdl in models:
        for L in LEVELS:
            fp = os.path.join(RES_ROOT, mdl, "kernelbench", f"level{L}",
                              "eval_results.json")
            if not os.path.exists(fp):
                rows.append((mdl, L, "MISSING"))
                continue
            with open(fp) as fh:
                d = json.load(fh)
            m = compute_kb_level(d, ref, L)
            rows.append((mdl, L, m))

    hdr = (f"{'model':<20} {'L':>2} {'comp/n':>7} {'corr/n':>7} {'n_ref':>6} "
           f"{'fast1x':>7} {'fast2x':>7} {'med_sp':>7} {'mean_sp':>7}")
    print(hdr)
    print("-" * len(hdr))
    for mdl, L, m in rows:
        if m == "MISSING":
            print(f"{mdl:<20} {L:>2} {'MISSING':>7}")
            continue
        f1s = f"{m['fast1x']*100:5.1f}%" if m["n_with_ref"] else "  n/a "
        f2s = f"{m['fast2x']*100:5.1f}%" if m["n_with_ref"] else "  n/a "
        mds = f"{m['median_speedup']:7.2f}" if m["n_with_ref"] else "    n/a"
        avs = f"{m['mean_speedup']:7.2f}" if m["n_with_ref"] else "    n/a"
        print(f"{mdl:<20} {L:>2} {m['comp']}/{m['n']:<5} {m['corr']}/{m['n']:<5} "
              f"{m['n_with_ref']:>6} {f1s:>7} {f2s:>7} {mds:>7} {avs:>7}")

    print()
    print("# fast_1x (share of correct+ref kernels beating PyTorch eager) "
          "per model/level:")
    for mdl in models:
        cells = []
        for L in LEVELS:
            for r in rows:
                if r[0] == mdl and r[1] == L and r[2] != "MISSING":
                    c = r[2]["fast1x"] if r[2]["n_with_ref"] else None
                    cells.append(c)
                    break
            else:
                cells.append(None)
        print(f"  {mdl:<20} " + " | ".join(
            (f"L{i+1}: {c*100:.0f}%" if isinstance(c, (int, float)) else f"L{i+1}: n/a")
            for i, c in enumerate(cells)))


if __name__ == "__main__":
    main()
