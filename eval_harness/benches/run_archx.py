#!/usr/bin/env python3
"""
ArchXBench evaluation with iverilog. 71 RTL designs across level-0..level-6.
Default (--num-samples 1): single greedy sample -> syntax/func pass@1.
Optional n/t mode (--num-samples K --temperature T, K>1): draw K candidates
(k=0 greedy, 1..K-1 sampled), report per-design averaged:
  n = #candidates that compile (0..K); t = best compiling candidate's assertion-pass %.
Flattened (design,k) tasks + as_completed so a slow long-think sample never blocks.

t = 100*passed/(passed+failed) for "Passed:N,Failed:M"/JSON self-check tb;
    binary 0/100 for golden-compare and plain PASS/FAIL tb.

  python run_archx.py --base-url http://localhost:8000/v1 --model base-9b \
      --out results/base-9b/archxbench --workers 64 --num-samples 5 --temperature 0.8
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Closed-blacklist classification adapter (verify.archxbench). The infra
# decision (no testbench / query failure) is routed through verify.core so the
# same gate as the other hardened benches vets it — candidate-controllable text
# can never authorize an infra exclusion. Denominator math is unchanged; the
# gated r["infra"] replaces the old free bool. See verify/archxbench.py.
try:
    import sys as _sys
    _BINFRA = os.environ.get("BENCHINFRA_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _sys.path.insert(0, _BINFRA)
    from verify.archxbench import classify_archx_sample  # noqa: E402
    _HAVE_CLASSIFY = True
except Exception:
    _HAVE_CLASSIFY = False

ARCHX_ROOT = os.environ.get("ARCHX_ROOT",
    os.path.join(os.environ.get("CODERBENCH_ROOT", "."), "ArchXBench"))
SYS_MSG = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."


def find_designs():
    designs = []
    for lvl in sorted(os.listdir(ARCHX_ROOT)):
        lvl_dir = os.path.join(ARCHX_ROOT, lvl)
        if not (lvl.startswith("level-") and os.path.isdir(lvl_dir)):
            continue
        for name in sorted(os.listdir(lvl_dir)):
            d = os.path.join(lvl_dir, name)
            if not os.path.isdir(d):
                continue
            if os.path.exists(os.path.join(d, "problem-description.txt")):
                designs.append(d)
    return designs


def tb_file(design_dir):
    cands = [f for f in os.listdir(design_dir)
             if f.endswith(".v") and (f == "tb.v" or f.startswith("tb") or "testbench" in f)]
    for pref in ("tb.v", "testbench.v"):
        if pref in cands:
            return pref
    return cands[0] if cands else None


def build_prompt(design_dir):
    desc = open(os.path.join(design_dir, "problem-description.txt")).read()
    spec_path = os.path.join(design_dir, "design-specs.txt")
    spec = open(spec_path).read() if os.path.exists(spec_path) else ""
    return (f"{desc}\n\n## Design Specification\n{spec}\n\n"
            "Write the complete, synthesizable Verilog module(s) implementing the "
            "design above. Match the module name and port list exactly as specified. "
            "Enclose your code with ```verilog and ```.")


def query(base_url, model, prompt, max_tokens=32768, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    # Direct-gateway support: if an API key is in env, send the Authorization
    # header so run_archx.py can hit an OpenAI-compatible gateway directly
    # without needing the local proxy_rr.py / serve_vllm.sh.
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("EXTERNAL_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = base_url.rstrip("/") + "/chat/completions"
    # Retry transient 5xx / connection errors (vLLM queue overflow under load
    # returns HTTP 500/503; a dropped connection raises URLError). Backoff like
    # the upstream harness. We do NOT retry on TimeoutError: with the
    # 600s default cap a timeout almost always means a silently-dropped gateway
    # connection (DeepSeek) or a runaway — retrying would block 6×600s and hang
    # the run, exactly what the cap prevents. A local-vLLM reasoning run that
    # legitimately needs >10min can raise ARCHX_QUERY_TIMEOUT=3600, in which
    # case a timeout still isn't retried (genuine runaway -> fail, don't loop).
    timeout = int(os.environ.get("ARCHX_QUERY_TIMEOUT", "600"))
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            return resp["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500:   # 4xx is a real error (bad request/auth), don't loop
                raise
        except (urllib.error.URLError, ConnectionError) as e:
            last = e
        time.sleep(min(2 ** attempt, 30))
    raise last


def extract_verilog(text):
    if not text:
        return ""
    # Strip reasoning wrappers (<think> proxy already removes; <answer> it does
    # not) and handle the unclosed fence from truncated max_tokens output —
    # otherwise the wrappers/raw text leak into the code and every compile fails
    # at 1:1.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?(?:answer|think)>", "", text)
    m = re.search(r"```(?:verilog|systemverilog)?\s*(.*?)```", text, re.DOTALL)
    if not m:  # truncated output: opening fence with no close
        m = re.search(r"```(?:verilog|systemverilog)?\s*(.*)$", text, re.DOTALL)
    code = m.group(1) if m else text
    mods = re.search(r"(module\b.*endmodule)", code, re.DOTALL)
    if mods:
        return mods.group(1)
    mods = re.search(r"(module\b.*)", code, re.DOTALL)  # truncated, no endmodule
    return mods.group(1) if mods else code


def assertion_pct(out):
    m = re.search(r'"passed"\s*:\s*(\d+).*?"failed"\s*:\s*(\d+)', out, re.DOTALL)
    if not m:
        m = re.search(r'Passed:\s*(\d+).*?Failed:\s*(\d+)', out, re.DOTALL | re.IGNORECASE)
    if m:
        passed, failed = int(m.group(1)), int(m.group(2))
        tot = passed + failed
        return (100.0 * passed / tot) if tot > 0 else 0.0
    up = out.upper()
    if "PASS" in up and "FAIL" not in up and "MISMATCH" not in up and not re.search(r"\bERROR\b", up):
        return 100.0
    return 0.0


def result_path(out_dir, name, k):
    return os.path.join(out_dir, f"{name}_s{k}_result.json")


def apply_result(per, r):
    """Fold one sample result into the per-design accumulator (n / best-t /
    infra count / k=0 snapshot). Used for both fresh and resumed samples."""
    p = per[r["design"]]
    p["n"] += r["syntax"]
    if r.get("infra"):
        p["infra"] += 1
    if r["syntax"] and r["t"] > p["bt"]:
        p["bt"] = r["t"]
    if r["k"] == 0:
        p["s0s"] = r["syntax"]
        p["s0f"] = 1 if r["t"] >= 100.0 else 0
        p["s0e"] = r["error"]


def verify_candidate(code, design_dir, tb, has_golden):
    with tempfile.TemporaryDirectory() as wd:
        for item in os.listdir(design_dir):
            if item.endswith(".v") or item.startswith(".") or item == "outputs":
                continue
            os.symlink(os.path.join(design_dir, item), os.path.join(wd, item))
        os.makedirs(os.path.join(wd, "outputs"), exist_ok=True)
        has_make = os.path.exists(os.path.join(wd, "Makefile"))
        if has_make:
            # v1.5: delegate the full pipeline (pregen -> compile -> compare) to
            # the per-design Makefile. It knows the correct pre-generation target
            # name (pyref / generate / golden / stimuli) and script, which differs
            # per design — hardcoding generate_golden.py misses fft_streaming_64pt
            # (`generate`), multich_conv2d (`golden`), etc. outputs/ is a fresh
            # per-worker dir so parallel workers never race in the source tree.
            with open(os.path.join(wd, "code.v"), "w") as f:
                f.write(code)
            make = subprocess.run(["make", "all"], capture_output=True, text=True,
                                  errors="replace", timeout=300, cwd=wd)
            out = (make.stdout + "\n" + make.stderr)
            if "[PASS]" in out:
                return 1, 100.0, ""
            if "[FAIL]" in out:  # compiled + simulated; compare mismatch
                return 1, 0.0, "golden mismatch: " + out[-200:]
            return 0, 0.0, "make: " + out[-300:]
        shutil.copy(os.path.join(design_dir, tb), os.path.join(wd, tb))
        gen = os.path.join(wd, "generated.v")
        with open(gen, "w") as f:
            f.write(code)
        simv = os.path.join(wd, "simv")
        # v1.5: some no-Makefile designs (the L2 one) still ship
        # scripts/generate_golden.py — run it before iverilog so tb.v finds
        # outputs/python_golden.json. No-op for level 0-3 self-check designs.
        gen_golden = os.path.join(wd, "scripts", "generate_golden.py")
        if os.path.exists(gen_golden):
            gg = subprocess.run(
                ["python3", "scripts/generate_golden.py"],
                capture_output=True, text=True, errors="replace", timeout=180, cwd=wd)
            if gg.returncode != 0:
                return 0, 0.0, "pyref: " + (gg.stderr or gg.stdout)[-300:]
        comp = subprocess.run(
            ["iverilog", "-g2012", "-Wno-timescale", "-o", simv, gen, os.path.join(wd, tb)],
            capture_output=True, text=True, errors="replace", timeout=120, cwd=wd)
        if comp.returncode != 0:
            return 0, 0.0, "compile: " + comp.stderr[-300:]
        try:
            run = subprocess.run(["vvp", simv], capture_output=True, text=True,
                                 errors="replace", timeout=180, cwd=wd)
            out = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return 1, 0.0, "sim timeout"
        if has_golden:
            cmp = subprocess.run(["python3", "scripts/compare_outputs.py"],
                                 capture_output=True, text=True, errors="replace", timeout=60, cwd=wd)
            if cmp.returncode == 0 and "PASS" in (cmp.stdout + cmp.stderr).upper():
                return 1, 100.0, ""
            return 1, 0.0, "golden mismatch: " + (cmp.stdout + cmp.stderr)[-200:]
        pct = assertion_pct(out)
        return 1, pct, "" if pct >= 100.0 else f"func partial/fail: {pct:.0f}%"


def gen_verify_one(design_dir, k, base_url, model, out_dir, max_tokens, temperature):
    name = os.path.basename(design_dir)
    tb = tb_file(design_dir)
    # infra=True marks a gateway/upstream failure (no valid generation). Such
    # samples must NOT deflate n/t — they're infra, not model failures. The
    # _infra_trigger is a trusted harness-set key (NOT candidate text); the
    # closed-blacklist gate in verify.archxbench vets it before it can exclude
    # from the denominator.
    r = {"design": name, "k": k, "syntax": 0, "t": 0.0, "error": "", "infra": False}
    if tb is None:
        r["error"] = "no testbench"
        r["_infra_trigger"] = "no_testbench"  # missing testbench = harness gap
        if _HAVE_CLASSIFY: r = classify_archx_sample(r)
        return r
    has_golden = os.path.exists(os.path.join(design_dir, "scripts", "compare_outputs.py"))
    temp = 0.0 if k == 0 else temperature
    try:
        resp = query(base_url, model, build_prompt(design_dir), max_tokens=max_tokens, temperature=temp)
    except Exception as e:
        r["error"] = f"query: {e}"
        r["_infra_trigger"] = "query_failed"  # gateway/5xx — model never answered
        if _HAVE_CLASSIFY: r = classify_archx_sample(r)
        return r
    code = extract_verilog(resp)
    with open(os.path.join(out_dir, f"{name}_s{k}.v"), "w") as f:
        f.write(code)
    with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
        f.write(resp)
    syntax, pct, err = verify_candidate(code, design_dir, tb, has_golden)
    r["syntax"], r["t"], r["error"] = syntax, pct, err[:120]
    # Stamp closed-blacklist classification (model result; infra stays False).
    # Idempotent on resumed checkpoints (skips if already classified).
    if _HAVE_CLASSIFY: r = classify_archx_sample(r)
    # Checkpoint (atomic temp+rename) so a mid-run restart + rerun skips
    # this sample. Only real results are checkpointed — infra failures (no tb /
    # query error) are NOT, so a rerun retries them instead of inheriting the
    # network/infra failure forever.
    rp = result_path(out_dir, name, k)
    try:
        tmp = rp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(r, f)
        os.replace(tmp, rp)
    except Exception:
        pass
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    designs = find_designs()
    if args.limit:
        designs = designs[:args.limit]
    K = args.num_samples
    print(f"[archx] {len(designs)} designs x {K} = {len(designs)*K} tasks, "
          f"model={args.model}, workers={args.workers}, temp={args.temperature}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    per = {os.path.basename(d): {"n": 0, "bt": 0.0, "infra": 0, "s0s": 0, "s0f": 0, "s0e": ""} for d in designs}
    # Resume: fold any existing per-sample checkpoints (written atomically by
    # gen_verify_one) into `per`, and remember which (design, k) are already
    # done so they aren't re-submitted. Real results skip on rerun; infra
    # failures (not checkpointed) are retried. Survives a mid-run restart.
    done_keys = set()
    for fn in os.listdir(args.out):
        if not fn.endswith("_result.json"):
            continue
        try:
            with open(os.path.join(args.out, fn)) as f:
                r = json.load(f)
            apply_result(per, r)
            done_keys.add((r["design"], r["k"]))
        except Exception:
            pass
    if done_keys:
        print(f"[archx] resumed {len(done_keys)} completed samples from {args.out}", flush=True)
    done = len(done_keys); total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_verify_one, d, k, args.base_url, args.model, args.out,
                          args.max_tokens, args.temperature)
                for d in designs for k in range(K)
                if (os.path.basename(d), k) not in done_keys]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            apply_result(per, r)
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
    # clean: t averaged only over designs that got at least one real
    # generation (excludes all-infra designs); n_clean = POOLED compile rate
    # among scored samples (total_compiling / total_scored) — a ratio 0-1,
    # NOT a per-design average (which would diverge from the pooled rate via
    # Simpson's paradox when designs have uneven infra counts).
    scored_des = [r for r in results if (K - r["n_infra"]) > 0]
    total_compile = sum(r["n"] for r in results)  # #compiling across ALL K samples
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
    print(f"[archx] {args.model}: n={summary['n']}/{K} t={summary['t']}% "
          f"(clean: compile={100*avg_n_clean:.1f}% of scored, t={summary['t_clean']}%, n_infra={n_infra})", flush=True)


if __name__ == "__main__":
    main()
