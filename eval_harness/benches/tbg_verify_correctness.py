#!/usr/bin/env python3
"""TritonBench-G/T correctness verify (standalone, shard-friendly).

Reads a teacher-rollout JSONL (output of teacher_triton_rollout_tbg.py with
--no-verify, or any rollout missing verify_* fields) and writes back the same
rows annotated with verify_compiled / verify_correct / verify_meta, plus the
hardened classify fields (infra_code / failure_origin / triton_launched /
identity_hack / framework_delegation).

The hardened path delegates to verify.tritonbench.verify_tbg, which combines
the task-native numerical oracle with deterministic seeding, adaptive timeout
handling, launch evidence, anti-cheat diagnostics, and the shared
infrastructure/model classification policy.

Each row gets its own disposable subprocess (verify_tbg spawns a fresh python
per variant), so a Triton compile crash on row N cannot poison row N+1's CUDA
context (mirrors kb_verify_correctness.py isolation guarantee).

Usage (single shard):
    python benches/tbg_verify_correctness.py \\
        --input  rollout/.../rollout.jsonl \\
        --output rollout/.../shard_0.verified.jsonl \\
        --gpu 0 --timeout 180

Sharded parallel: see benches/run_tritonbench.sh (eval stage splits rollout
round-robin across N_GPU, one shard per card).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benches"))

# Hardened verify (verify.tritonbench) + classification (verify.core). The old
# path imported verify_one from teacher_triton_rollout_tbg; that function is
# now superseded — kept around only for the inline gen+verify rollout path.
from verify.tritonbench import verify_tbg  # noqa: E402
from verify.core import is_infra, model_failure, finalize_failure_classification  # noqa: E402


async def main_async(args) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pin this entire process to one CUDA device. verify_tbg's subprocesses
    # inherit env, so each variant runs on the same GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Resume: skip ids already in output.
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line).get("id"))
                except Exception:
                    pass
    if done_ids:
        print(f"[tbg-verify] resume: {len(done_ids)} ids already done", flush=True)

    rows = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("id") in done_ids:
                continue
            rows.append(r)
    print(f"[tbg-verify] gpu={args.gpu} input={in_path} "
          f"timeout={args.timeout}s rows={len(rows)} (after resume)", flush=True)

    n_done = n_compiled = n_correct = 0
    with out_path.open("a") as fout:
        for i, row in enumerate(rows, 1):
            tc = row.get("teacher_code", "")
            ref = row.get("reference_solution", "")
            ei = row.get("evaluator_info") or {}
            test_block = ei.get("test_block", "")
            # atol/rtol may be carried per-row by the payload-style seeds;
            # default to the upstream TBG tolerance.
            atol = float(ei.get("atol", 1e-2))
            rtol = float(ei.get("rtol", 1e-2))

            if not tc or row.get("teacher_status") != "ok":
                # Teacher format failed (no usable code) — MODEL failure, NOT
                # a skip: stays in the denominator as compiled=false (the
                # 2026-07-20 narrow口径, now encoded in verify.core.is_infra).
                res = finalize_failure_classification(
                    model_failure(reason="no_teacher_code",
                                  eval_backend="tritonbench",
                                  info="skipped_no_teacher_code")
                )
            elif not test_block:
                res = finalize_failure_classification(
                    model_failure(reason="no_test_block",
                                  eval_backend="tritonbench",
                                  info="skipped_no_test_block")
                )
            else:
                res = verify_tbg(ref, tc, test_block,
                                 timeout=args.timeout,
                                 atol=atol, rtol=rtol, seed=0,
                                 measure_perf=args.measure_perf)

            # Synthesize the legacy verify_* aliases on top of the hardened
            # classify dict, so tbg_merge_shards.py / summarize.sh / SFT
            # pack_sft_v4pro.py all keep reading the same fields.
            out = {**row, **res,
                   "verify_compiled": bool(res.get("compiled")),
                   "verify_correct": bool(res.get("correct")),
                   "verify_skipped": is_infra(res),
                   "verify_meta": res.get("verify_meta") or
                       {"stage": res.get("info", "")}}

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if out.get("verify_compiled"):
                n_compiled += 1
            if out.get("verify_correct"):
                n_correct += 1

            issue = ""
            if not out.get("verify_correct"):
                vm = out.get("verify_meta") or {}
                issue = vm.get("stage") or str(vm.get("error", ""))[:60]
                if not issue:
                    issue = str(out.get("info", ""))[:60]
            extra = ""
            if out.get("triton_launched") is not None:
                extra = f" launches={out.get('triton_launched')}"
            if out.get("identity_hack"):
                extra += " IDENTITY_HACK"
            if out.get("framework_delegation"):
                extra += " FRAMEWORK_DELEGATION"
            if is_infra(out):
                extra += " INFRA"
            print(f"[tbg-verify] {i}/{len(rows)}: "
                  f"compiled={out.get('verify_compiled')} "
                  f"correct={out.get('verify_correct')} "
                  f"id={row.get('id', '?')[-60:]}{extra} "
                  f"{('issue=' + issue) if issue else ''}", flush=True)

    print(f"[tbg-verify] DONE on shard gpu={args.gpu}: {n_done} rows, "
          f"{n_compiled} compiled, {n_correct} correct "
          f"({n_correct*100/max(1,n_done):.1f}%)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=int, default=180,
                   help="per-variant budget; effective per-variant timeout "
                        "= max(180, timeout/2) (autotune floor)")
    p.add_argument("--measure-perf", action="store_true",
                   help="time ref+candidate (cuda-event median, 20 trials) and "
                        "write speedup + fast (= correct AND speedup>1.05) per "
                        "row. The `fast` metric per reward_kernel_v2.py:706-707.")
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
