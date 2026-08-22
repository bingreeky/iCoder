#!/usr/bin/env python3
"""benches/kb_verify_correctness.py — hardened KernelBench eval driver.

Replaces (or complements) scripts/eval_from_generations.py with the hardened
verify.kernelbench.verify_kb path: deterministic seeding, 32 correctness trials,
triton completed-launch counter, identity_hack + framework_delegation anti-cheat
gates, and the principled "every post-bootstrap exception is a model failure"
classification. Produces eval_results_hardened.json keyed by
"{problem_id}_sample{sample_id}", carrying the verify.core classify fields
(failure_origin / infra_code / triton_launched / identity_hack /
framework_delegation) so summarize.sh / kb_merge_shards.py read the same shape.

Per-row subprocess isolation (verify_kb spawns a fresh KB-venv python per
candidate) means a Triton/MLIR compile crash on problem N cannot poison N+1's
CUDA context — mirrors kb_sharded_eval.sh's one-card-per-shard isolation, taken
to the per-problem boundary.

Refs are read straight off disk (KernelBench/KernelB/level{N}/{pid}_*.py); the
_INIT_ADAPTER appended by verify_kb patches Model/ModelNew.__init__ to the KB
harness call shape. Gens are read from runs/<run>/level_{N}_problem_{pid}_
sample{s}_kernel.py (the path eval_from_generations.py writes).

Usage (single shard):
    KERNELBENCH_PY=$KB/.venv/bin/python \\
    python benches/kb_verify_correctness.py \\
        --run-name your-model_level1 --level 1 --gpu 0 \\
        --num-correct-trials 32 --timeout 300

Sharded: run one process per GPU (CUDA_VISIBLE_DEVICES + --gpu), then merge the
per-shard eval_results_hardened_gpuN.json with kb_merge_shards.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify.kernelbench import verify_kb, _resolve_kb_src  # noqa: E402
from verify.core import is_infra, model_failure, infra_failure  # noqa: E402
from verify.core import finalize_failure_classification  # noqa: E402


def _dataset_root(kb_src: str) -> Path:
    # kb_src is .../KernelBench/src; dataset lives at .../KernelBench/KernelBench
    return Path(kb_src).resolve().parent / "KernelBench"


def _ref_for_problem(kb_src: str, level: int, problem_id: int) -> str | None:
    ds = _dataset_root(kb_src)
    # {pid}_*.py ; pid is zero-padded? KB uses bare int filenames: 10_3D_....py
    for pat in (f"{ds}/level{level}/{problem_id}_*.py",
                f"{ds}/level{level}/{problem_id:02d}_*.py"):
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0]).read_text(encoding="utf-8")
    return None


def _gen_path(run_dir: str, level: int, problem_id: int, sample_id: int) -> Path:
    # matches eval_from_generations.fetch_kernel_from_disk
    return Path(run_dir) / f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"


def _iter_problems(run_dir: str, level: int, samples: int,
                   subset: tuple[int, int] | None,
                   problem_ids: list[int] | None):
    """Yield (problem_id, sample_id) for every kernel.py present under run_dir,
    optionally filtered to a subset range or explicit id list."""
    import re
    pat = re.compile(rf"level_{level}_problem_(\d+)_sample_(\d+)_kernel\.py$")
    found: list[tuple[int, int]] = []
    for p in glob.glob(f"{run_dir}/level_{level}_problem_*_sample_*_kernel.py"):
        m = pat.search(Path(p).name)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    found.sort()
    for pid, sid in found:
        if problem_ids is not None and pid not in problem_ids:
            continue
        if samples and sid >= samples:
            continue
        if subset and not (subset[0] <= pid <= subset[1]):
            continue
        yield pid, sid


async def main_async(args) -> None:
    kb_python = args.kb_python or os.environ.get("KERNELBENCH_PY") or sys.executable
    kb_src = _resolve_kb_src(args.kb_src)
    run_dir = args.run_dir or str(
        Path(kb_src).resolve().parent / "runs" / args.run_name)
    out_path = Path(args.output) if args.output else (
        Path(run_dir) / f"eval_results_hardened_gpu{args.gpu}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # resume: load existing results, skip done keys
    results: dict[str, dict] = {}
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
        except Exception:
            results = {}
    if results:
        print(f"[kb-verify] resume: {len(results)} keys already done", flush=True)

    subset = (args.subset_start, args.subset_end) if args.subset_start else None
    pids = [int(x) for x in args.problem_ids.split(",")] if args.problem_ids else None
    work = list(_iter_problems(run_dir, args.level, args.samples, subset, pids))
    print(f"[kb-verify] gpu={args.gpu} run={args.run_name} level={args.level} "
          f"trials={args.num_correct_trials} work={len(work)} (after resume)",
          flush=True)

    n_done = n_comp = n_corr = 0
    for i, (pid, sid) in enumerate(work, 1):
        key = f"{pid}_sample{sid}"
        if key in results:
            continue
        ref = _ref_for_problem(kb_src, args.level, pid)
        gpath = _gen_path(run_dir, args.level, pid, sid)
        if ref is None:
            res = finalize_failure_classification(
                infra_failure(
                    infra_code="trusted_reference_failure",
                    eval_backend="kernelbench",
                    info=f"no_ref_for_problem:{pid}",
                    error_type="reference_harness_failure",
                    verify_meta={"stage": "ref_smoke_failed", "error": "no_ref"},
                )
            )
        elif not gpath.is_file():
            res = finalize_failure_classification(
                model_failure(reason="no_generated_kernel",
                              eval_backend="kernelbench",
                              info=f"skipped_no_kernel:{pid}")
            )
        else:
            gen = gpath.read_text(encoding="utf-8")
            res = verify_kb(
                ref, gen, gpu=args.gpu, kb_python=kb_python, kb_src=kb_src,
                num_correct_trials=args.num_correct_trials,
                num_perf_trials=args.num_perf_trials,
                measure_perf=args.measure_perf,
                timeout=args.timeout, seed=0,
            )

        # synthesize the legacy eval_results.json row shape so
        # kb_merge_shards.py (which walks for dict-with-"compiled") keeps working
        row = {
            "compiled": bool(res.get("compiled")),
            "correctness": bool(res.get("correct")),
            "metadata": {
                "error": (res.get("verify_meta") or {}).get("error") or res.get("info"),
                "stage": (res.get("verify_meta") or {}).get("stage"),
            },
            # hardened classify fields (the real signal)
            "verify_meta": res.get("verify_meta") or {"stage": res.get("info")},
            "failure_origin": res.get("failure_origin"),
            "infra_code": res.get("infra_code"),
            "triton_launched": res.get("triton_launched"),
            "identity_hack": res.get("identity_hack"),
            "framework_delegation": res.get("framework_delegation"),
            "speedup": res.get("speedup"),
        }
        results[key] = row
        # incremental flush
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        n_done += 1
        if row["compiled"]:
            n_comp += 1
        if row["correctness"]:
            n_corr += 1
        extra = ""
        if row.get("triton_launched") is not None:
            extra = f" launches={row['triton_launched']}"
        if row.get("identity_hack"):
            extra += " IDENTITY_HACK"
        if row.get("framework_delegation"):
            extra += " FRAMEWORK_DELEGATION"
        if is_infra(res):
            extra += " INFRA"
        print(f"[kb-verify] {i}/{len(work)} pid={pid} s={sid} "
              f"compiled={row['compiled']} correct={row['correctness']}{extra}",
              flush=True)

    print(f"[kb-verify] DONE gpu={args.gpu}: {n_done} done, {n_comp} compiled, "
          f"{n_corr} correct ({n_corr*100/max(1,n_done):.1f}%)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--run-dir", default=None,
                   help="defaults to <KB>/runs/<run_name>")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--subset-start", type=int, default=None)
    p.add_argument("--subset-end", type=int, default=None)
    p.add_argument("--problem-ids", default=None,
                   help="comma-separated explicit problem ids")
    p.add_argument("--num-correct-trials", type=int, default=32)
    p.add_argument("--num-perf-trials", type=int, default=10)
    p.add_argument("--measure-perf", action="store_true")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--kb-python", default=None)
    p.add_argument("--kb-src", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main() -> None:
    import asyncio
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
