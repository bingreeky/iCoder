#!/usr/bin/env python3
# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""
ArchX summary recovery — for when a run_archx.py run is killed or a few candidates
never finished (reasoning models can run away on hard designs: fft/matmul/aes/fir).

run_archx writes summary.json only after ALL (design, k) candidates finish, so a
handful of runaway candidates block the whole summary even though the other ~99%
of *_sN.v files are already on disk and fully usable.

This script:
  1. re-verifies every saved *_sN.v candidate from disk (fast, no engine needed),
  2. OPTIONALLY regenerates missing candidates via the engine (--regen), or treats
     them as syntax=0 fail (default — matches "a runaway that never produced code"),
  3. writes summary.json in the IDENTICAL schema run_archx would have (so
     summarize.sh reads it the same as a fresh run: n/t + n_clean/t_clean/n_infra).

verify_candidate can raise subprocess.TimeoutExpired (iverilog 120s cap on a huge
generated.v); we catch it so one pathological file can't crash the whole recovery.

Adapted from the eval harness internals: reuses this repo's run_archx primitives
(find_designs, tb_file, build_prompt, query, extract_verilog, verify_candidate,
apply_result) and replicates run_archx.main()'s summary computation so the output
schema matches exactly. In recovery there is no infra concept — every sample is
scored (missing-without-regen = syn=0 model failure, not infra) — so n_infra=0.

Usage (finalize from disk, missing -> fail; no engine needed):
  python archx_recover_summary.py --model M --out results/M/archxbench --num-samples 5
Usage (also regenerate missing candidates against a live engine):
  python archx_recover_summary.py --model M --out results/M/archxbench --num-samples 5 \
      --regen --base-url http://localhost:8000/v1 --max-tokens 65536 --temperature 0.8
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_archx as R  # reuse find_designs, tb_file, build_prompt, query, extract_verilog, verify_candidate, apply_result


def recover_one(design_dir, k, out_dir, regen, base_url, model, max_tokens, temperature):
    name = os.path.basename(design_dir)
    tb = R.tb_file(design_dir)
    # infra=False: in recovery every sample is scored. A missing candidate with
    # no --regen is a model failure (runaway -> syn=0), NOT an infra gap, so it
    # stays in the denominator — matches run_archx's model-failure semantics.
    r = {"design": name, "k": k, "syntax": 0, "t": 0.0, "error": "", "infra": False}
    if tb is None:
        r["error"] = "no testbench"
        r["infra"] = True  # missing testbench = harness gap, not model failure
        return r
    has_golden = os.path.exists(os.path.join(design_dir, "scripts", "compare_outputs.py"))
    vpath = os.path.join(out_dir, f"{name}_s{k}.v")
    if os.path.exists(vpath) and os.path.getsize(vpath) > 0:
        code = open(vpath).read()
    elif regen:
        # missing candidate -> regenerate with the retried query (retry+backoff
        # is inside R.query; ARCHX_QUERY_TIMEOUT caps the per-attempt wall).
        temp = 0.0 if k == 0 else temperature
        try:
            resp = R.query(base_url, model, R.build_prompt(design_dir), max_tokens=max_tokens, temperature=temp)
        except Exception as e:
            r["error"] = f"query: {e}"
            return r
        code = R.extract_verilog(resp)
        with open(vpath, "w") as f:
            f.write(code)
        with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
            f.write(resp)
    else:
        # missing and not regenerating -> a runaway that never produced code = fail
        r["error"] = "missing (runaway -> syn=0)"
        return r
    try:
        syntax, pct, err = R.verify_candidate(code, design_dir, tb, has_golden)
        r["syntax"], r["t"], r["error"] = syntax, pct, err[:120]
    except Exception as e:  # e.g. iverilog subprocess.TimeoutExpired on a huge file
        r["error"] = f"verify-timeout/err: {type(e).__name__}"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--regen", action="store_true",
                    help="regenerate missing candidates against a live engine (needs --base-url)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    designs = R.find_designs()
    K = args.num_samples
    # per-dict shape identical to run_archx.main() so apply_result + the summary
    # computation below match a fresh run exactly.
    per = {os.path.basename(d): {"n": 0, "bt": 0.0, "infra": 0, "s0s": 0, "s0f": 0, "s0e": ""}
           for d in designs}
    done = 0
    total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(recover_one, d, k, args.out, args.regen, args.base_url,
                          args.model, args.max_tokens, args.temperature)
                for d in designs for k in range(K)]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            R.apply_result(per, r)
            print(f"  [{done}/{total}] {r['design']:30s} k={r['k']} syn={r['syntax']} t={r['t']:5.0f} {r['error'][:38]}", flush=True)

    ndes = len(designs)
    results = []
    for d in designs:
        name = os.path.basename(d); p = per[name]
        scored = K - p["infra"]
        results.append({
            "design": name, "num_samples": K, "n": p["n"],
            "n_clean": round(p["n"]/scored, 4) if scored > 0 else 0.0,
            "t": round(p["bt"], 2), "n_infra": p["infra"],
            "syntax": p["s0s"], "func": p["s0f"], "error": p["s0e"],
        })
    syn = sum(r["syntax"] for r in results); fun = sum(r["func"] for r in results)
    n_infra = sum(r["n_infra"] for r in results)
    avg_n = sum(r["n"] for r in results)/ndes if ndes else 0
    avg_t = sum(r["t"] for r in results)/ndes if ndes else 0
    scored_des = [r for r in results if (K - r["n_infra"]) > 0]
    total_compile = sum(r["n"] for r in results)
    total_scored = K * ndes - n_infra
    avg_t_clean = (sum(r["t"] for r in scored_des)/len(scored_des)) if scored_des else 0.0
    avg_n_clean = (total_compile / total_scored) if total_scored > 0 else 0.0
    summary = {"model": args.model, "num_designs": ndes, "num_samples": K,
               "n": round(avg_n, 2), "t": round(avg_t, 2),
               "n_clean": round(avg_n_clean, 4), "t_clean": round(avg_t_clean, 2),
               "n_infra": n_infra,
               "syntax_pass@1": round(100*syn/ndes, 2) if ndes else 0,
               "func_pass@1": round(100*fun/ndes, 2) if ndes else 0,
               "syntax_count": syn, "func_count": fun}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[archx-recover] {args.model}: n={summary['n']}/{K} t={summary['t']}% "
          f"(clean: compile={100*avg_n_clean:.1f}% of scored, t={summary['t_clean']}%, n_infra={n_infra})", flush=True)


if __name__ == "__main__":
    main()
