#!/usr/bin/env python3
"""
Measure reference-kernel runtime for every KernelBench L1-L3 problem on THIS
A100, with the SAME timing config the model evals used (cuda_event, 100 trials,
fp32, use_torch_compile=False, get_inputs() from the problem's ref source).

Output is saved INCREMENTALLY (after every problem) so a hang/crash on one
problem never loses prior results. Failed/skipped problems are recorded with
runtime_stats=null.

The resulting `mean` per problem is the ref_runtime to pair with each model's
stored `runtime_stats.mean` (also A100/cuda/100-trial/fp32) -> effective_speedup
= ref_runtime / model_runtime -> fast_1x (>1x) / fast_2x (>2x).
"""
import os, sys, json, time, traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODER_ROOT = os.environ.get("CODER_ROOT", str(_REPO_ROOT / "benchmarks"))
KB = os.path.join(_CODER_ROOT, "benchmark", "KernelBench")
sys.path.insert(0, os.path.join(KB, "src"))
sys.path.insert(0, os.path.join(KB, "scripts"))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch
from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.timing import measure_ref_program_time

OUT = os.environ.get(
    "KB_REF_TIMING_FILE",
    str(Path(os.environ.get("RESULTS_DIR", str(_REPO_ROOT / "results"))) / "_kb_ref_timing_A100.json"),
)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"[ref-time] device={torch.cuda.get_device_name(device)} precision=fp32 "
      f"timing=cuda_event trials=100 use_compile=False", flush=True)

# Resume from incremental file if present
if os.path.exists(OUT):
    results = json.load(open(OUT))
    print(f"[ref-time] resuming from {OUT}: "
          f"L1={len(results.get('level1',{}))} L2={len(results.get('level2',{}))} "
          f"L3={len(results.get('level3',{}))}", flush=True)
else:
    results = {"level1": {}, "level2": {}, "level3": {}}

for level in [1, 2, 3]:
    dataset = construct_kernelbench_dataset(level)
    pids = list(dataset.get_problem_ids())
    key = f"level{level}"
    done = set(results[key].keys())
    todo = [p for p in pids if p not in done]
    print(f"[ref-time] level{level}: {len(pids)} problems, {len(done)} done, {len(todo)} todo", flush=True)
    for i, pid in enumerate(todo):
        t0 = time.time()
        try:
            ref_arch_path, ref_arch_name, ref_arch_src = fetch_ref_arch_from_dataset(dataset, pid)
            rs = measure_ref_program_time(
                ref_arch_name=ref_arch_name,
                ref_arch_src=ref_arch_src,
                use_torch_compile=False,
                device=device,
                verbose=False,
                precision="fp32",
            )
            results[key][pid] = {"ref_arch_name": ref_arch_name, "runtime_stats": rs}
            status = f"mean={rs.get('mean'):.3f}ms" if rs else "None (failed)"
        except Exception as e:
            results[key][pid] = {"ref_arch_name": "?", "runtime_stats": None,
                                 "error": f"{type(e).__name__}: {e}"}
            status = f"EXC {type(e).__name__}"
        # incremental save every problem
        with open(OUT, "w") as f:
            json.dump(results, f)
        print(f"[ref-time]   L{level} pid={pid} {status} ({time.time()-t0:.1f}s) "
              f"[{i+1+len(done)}/{len(pids)}]", flush=True)

n_ok = sum(1 for lk in ["level1","level2","level3"] for v in results[lk].values() if v.get("runtime_stats"))
n_tot = sum(len(results[lk]) for lk in ["level1","level2","level3"])
print(f"[ref-time] DONE: {n_ok}/{n_tot} problems timed OK -> {OUT}", flush=True)
