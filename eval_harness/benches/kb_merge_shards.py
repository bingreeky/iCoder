#!/usr/bin/env python3
# ============================================================
# kb_merge_shards.py — merge KernelBench shard fragments
# runs/<run>/eval_results_gpu{0..N}.json -> eval_results.json + print stats.
# Refuses to overwrite eval_results.json if no fragments are found.
#   python kb_merge_shards.py <run_name> [kernelbench_root]
# ============================================================
import json, os, sys, glob
from collections import Counter

RUN = sys.argv[1]
KB = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.environ.get("CODERBENCH_ROOT", "."), "KernelBench")
rd = f"{KB}/runs/{RUN}"

merged = {}
for sf in sorted(glob.glob(f"{rd}/eval_results_gpu*.json")):
    try:
        d = json.load(open(sf))
    except Exception as e:
        print(f"  [warn] {sf}: {e}"); continue
    for pid, v in d.items():
        merged[pid] = v
if not merged:
    print(f"[{RUN}] no shard fragments found -> refuse to overwrite eval_results.json")
    sys.exit(0)
json.dump(merged, open(f"{rd}/eval_results.json", "w"), indent=2)

rows = []
def walk(o):
    if isinstance(o, dict):
        if "compiled" in o:
            rows.append(o); return
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(merged)

n = len(rows)
comp = sum(1 for r in rows if r.get("compiled"))
corr = sum(1 for r in rows if r.get("correctness"))
c = Counter()
for r in rows:
    md = r.get("metadata") or {}
    e = str(md.get("compilation_error") or md.get("runtime_error") or md.get("error") or "")
    if r.get("correctness"): c["correct"] += 1
    elif "timed out" in e.lower(): c["timeout"] += 1
    elif r.get("compiled"): c["comp_nc"] += 1
    else: c["comp_err"] += 1
print(f"[{RUN}] n={n} compiled={comp} correct={corr} "
      f"fast_0={100*corr/n if n else 0:.1f}%  {dict(c)}")
