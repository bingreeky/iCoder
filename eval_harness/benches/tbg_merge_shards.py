#!/usr/bin/env python3
"""tbg_merge_shards.py — merge sharded TBG/TBT verify outputs into one
verified.jsonl + print fast_0 stats. Mirrors kb_merge_shards.py but for the
jsonl-of-rows schema (each line = rollout row + verify_* fields).

Inputs : <run_dir>/shard_verified_{0..N}.jsonl  (one per GPU shard)
Output : <run_dir>/verified.jsonl  (dedup by id, last wins)

Usage: python tbg_merge_shards.py <run_dir>
"""
import json
import sys
import glob
from pathlib import Path
from collections import Counter

# Single source of truth for "is this row an infra failure excluded from the
# denominator?". Imported from verify.core so the merge script and summarize.sh
# can NEVER disagree on the TBG skip set (the original 137/139 drift was three
# divergent _skipped definitions: here, summarize.sh:105-108, summarize.sh:246-255).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify.core import is_infra as _is_infra  # noqa: E402


def main() -> None:
    rd = Path(sys.argv[1])
    merged = {}
    shards = sorted(glob.glob(str(rd / "shard_verified_*.jsonl")))
    for sf in shards:
        try:
            with open(sf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    merged[r.get("id")] = r
        except Exception as e:
            print(f"  [warn] {sf}: {e}")

    if not merged:
        print(f"[tbg-merge] no shard fragments found in {rd} -> "
              f"refuse to overwrite verified.jsonl")
        sys.exit(0)

    out = rd / "verified.jsonl"
    with out.open("w") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Skipped = rows unwinnable for ANY model: the reference itself doesn't
    # compile/run under the installed Triton+CUDA (ref_smoke_failed / verify_skipped),
    # i.e. an infrastructure failure. These must NOT count against the
    # denominator — otherwise the metric is a harness ceiling (4 models all
    # scored 26-27/184 because 46 refs were unrunnable; see
    # eval_results/COMPARISON.md TBG non-discrimination audit).
    #
    # NOTE: "skipped_no_teacher_code" (model produced no parseable Triton) is a
    # MODEL failure, NOT a skip — it stays in the denominator as compiled=False.
    # This is the 2026-07-20 narrow口径; verify.core.is_infra encodes it once.
    def _skipped(r):
        return _is_infra(r)

    n = len(merged)
    n_scored = sum(1 for r in merged.values() if not _skipped(r))
    comp = sum(1 for r in merged.values() if r.get("verify_compiled"))
    corr = sum(1 for r in merged.values() if r.get("verify_correct"))
    n_skip = n - n_scored
    c = Counter()
    for r in merged.values():
        if r.get("verify_correct"):
            c["correct"] += 1
        elif _skipped(r):
            c["skipped"] += 1
        elif r.get("verify_compiled"):
            c["comp_nc"] += 1
        else:
            c["comp_err"] += 1
    print(f"[tbg-merge] {rd.name}: n={n} scored={n_scored} skip={n_skip} "
          f"compiled={comp} correct={corr} "
          f"fast_0={100*corr/n_scored if n_scored else 0:.1f}%  {dict(c)}  "
          f"-> {out}")


if __name__ == "__main__":
    main()
