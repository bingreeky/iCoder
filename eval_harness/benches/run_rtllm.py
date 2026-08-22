#!/usr/bin/env python3
"""
RTLLM v2 evaluation with iverilog (RTLLM ships Synopsys-VCS makefiles which we
don't have). Default (--num-samples 1): single greedy sample -> pass@1.
Optional average@K (--num-samples K --temperature T, K>1): draw K samples
(k=0 greedy, 1..K-1 sampled), report syntax_avg@K / func_avg@K = mean over
designs of (#passing candidates / K). Flattened (design,k) tasks + as_completed
so a slow long-think sample never blocks others.

  python run_rtllm.py --base-url http://localhost:8000/v1 --model base-9b \
      --out results/base-9b/rtllm --workers 64 --num-samples 4 --temperature 0.8
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

RTLLM_ROOT = os.environ.get("RTLLM_ROOT",
    os.path.join(os.environ.get("CODERBENCH_ROOT", "."), "RTLLM"))
SYS_MSG = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."

# --- hardened stdout-hash judgment (opt-in via RTLLM_VERDICT_PROFILE) ---
# When a precomputed verdict profile is present (see
# verify/profiles/build_profiles.py), a candidate's correctness is the SHA256
# of its vvp stdout matching a golden hash set — not the fragile
# `re.search(r"\b(pass|passed)\b")` regex, which a model can game by
# $display-ing "pass" and which flips on failure messages containing "passed".
# Falls back to the legacy regex when no profile is configured, so nothing
# breaks before the one-time profile precompute.
_RTLLM_PROFILE_PATH = os.environ.get("RTLLM_VERDICT_PROFILE", "")
_RTLLM_PROFILE: dict[str, list[str]] = {}
if _RTLLM_PROFILE_PATH and os.path.isfile(_RTLLM_PROFILE_PATH):
    try:
        _RTLLM_PROFILE = {k: list(v) for k, v in
                          json.load(open(_RTLLM_PROFILE_PATH)).items()}
    except Exception as _e:
        print(f"[rtllm] WARN: bad RTLLM_VERDICT_PROFILE={_RTLLM_PROFILE_PATH}: "
              f"{_e}; falling back to regex judgment", flush=True)

try:
    import sys as _sys
    _BINFRA = os.environ.get("BENCHINFRA_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _sys.path.insert(0, _BINFRA)
    from verify.icarus import execute_icarus, judge_against_profile  # noqa: E402
    _HAVE_ICARUS = True
except Exception as _e:
    _HAVE_ICARUS = False



def find_designs():
    designs = []
    for root, _, files in os.walk(RTLLM_ROOT):
        if "design_description.txt" in files and "testbench.v" in files:
            if "_chatgpt" in root:
                continue
            designs.append(root)
    return sorted(designs)


def query(base_url, model, prompt, max_tokens=32768, temperature=0.0):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": prompt +
             "\n\nGive the complete Verilog code. Enclose your code with ```verilog and ```."},
        ],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=86400) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"].get("content") or ""


def extract_verilog(text):
    if not text:
        return ""
    # Strip reasoning wrappers (<think> proxy already removes; <answer> it does not)
    # and handle the unclosed fence from truncated max_tokens output — otherwise the
    # wrappers/raw text leak into the code and every compile fails at 1:1.
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


def verify_candidate(gen_path, tb_path, design_name=None):
    """Return (syntax, func, error). syntax=1 if compiles; func=1 if correct.

    Hardened path (when _RTLLM_PROFILE has an entry for this design): compile
    + run via verify.icarus.execute_icarus, judge by stdout SHA256 match
    against the golden hash set — anti-cheat ($display/$dumpvars/DPI-C ban)
    included. Legacy path (no profile): the pass/passed regex.
    """
    # hardened path
    if _HAVE_ICARUS and design_name and design_name in _RTLLM_PROFILE:
        expected = frozenset(_RTLLM_PROFILE[design_name])
        # empty set = golden harness itself failed to produce a stdout
        # (build_profiles marked it broken) → infra, NOT a model fail.
        if not expected:
            return 0, 0, "infra:broken_golden_harness"
        try:
            src = open(gen_path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            return 0, 0, f"read: {e}"
        res = execute_icarus(src, testbench_path=tb_path,
                             timeout=60, eval_backend="rtllm")
        res = judge_against_profile(res, expected)
        compiled = bool(res.get("compiled"))
        correct = bool(res.get("correct"))
        if res.get("failure_origin") == "infrastructure":
            return compiled, correct, "infra:" + str(res.get("info"))
        if not compiled:
            return 0, 0, "compile: " + str(res.get("error_type", ""))[-120:]
        return 1, (1 if correct else 0), ("hash_pass" if correct else
                                          "hash_mismatch")
    # legacy regex path
    with tempfile.TemporaryDirectory() as wd:
        simv = os.path.join(wd, "simv")
        comp = subprocess.run(
            ["iverilog", "-g2012", "-Wno-timescale", "-o", simv, gen_path, tb_path],
            capture_output=True, text=True, timeout=60)
        if comp.returncode != 0:
            return 0, 0, "compile: " + comp.stderr[-300:]
        try:
            run = subprocess.run(["vvp", simv], capture_output=True, text=True, timeout=60)
            out = run.stdout + run.stderr
        except subprocess.TimeoutExpired:
            return 1, 0, "sim timeout"
        if re.search(r"\b(pass|passed)\b", out, re.IGNORECASE) and \
           not re.search(r"failure", out, re.IGNORECASE):
            return 1, 1, ""
        return 1, 0, "func fail"


def gen_verify_one(design_dir, k, base_url, model, out_dir, max_tokens, temperature):
    name = os.path.basename(design_dir)
    desc = open(os.path.join(design_dir, "design_description.txt")).read()
    tb = os.path.join(design_dir, "testbench.v")
    temp = 0.0 if k == 0 else temperature
    # infra=True marks a gateway/upstream failure (the sample never got a
    # valid generation). Such samples must NOT be counted as 0 in the
    # average — they're infra, not model failures. See COMPARISON.md audit.
    r = {"design": name, "k": k, "syntax": 0, "func": 0, "error": "", "infra": False}
    try:
        resp = query(base_url, model, desc, max_tokens=max_tokens, temperature=temp)
    except Exception as e:
        r["error"] = f"query: {e}"
        r["infra"] = True
        return r
    code = extract_verilog(resp)
    gen_path = os.path.join(out_dir, f"{name}_s{k}.v")
    with open(gen_path, "w") as f:
        f.write(code)
    with open(os.path.join(out_dir, f"{name}_s{k}_response.txt"), "w") as f:
        f.write(resp)
    syntax, func, err = verify_candidate(gen_path, tb, design_name=name)
    r["syntax"], r["func"], r["error"] = syntax, func, err[:120]
    if err.startswith("infra:"):
        r["infra"] = True  # broken golden harness / iverilog missing → not a model fail
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    designs = find_designs()
    K = args.num_samples
    print(f"[rtllm] {len(designs)} designs x {K} = {len(designs)*K} tasks, "
          f"model={args.model}, workers={args.workers}, temp={args.temperature}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    per = {os.path.basename(d): {"syn": 0, "fun": 0, "infra": 0, "s0s": 0, "s0f": 0, "s0e": "", "errors": []} for d in designs}
    done = 0; total = len(designs) * K
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(gen_verify_one, d, k, args.base_url, args.model, args.out,
                          args.max_tokens, args.temperature)
                for d in designs for k in range(K)]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            p = per[r["design"]]; p["syn"] += r["syntax"]; p["fun"] += r["func"]
            if r.get("infra"):
                p["infra"] += 1
            if r["error"]:
                p["errors"].append(f"k{r['k']}: {r['error']}")
            if r["k"] == 0:
                p["s0s"] = r["syntax"]; p["s0f"] = r["func"]; p["s0e"] = r["error"]
            print(f"  [{done}/{total}] {r['design']:28s} k={r['k']} syn={r['syntax']} func={r['func']} {r['error'][:40]}", flush=True)

    n = len(designs)
    results = []
    for d in designs:
        name = os.path.basename(d); p = per[name]
        scored = K - p["infra"]  # samples that got a real generation
        results.append({
            "design": name, "num_samples": K,
            "syntax_avg": round(p["syn"]/K, 4),
            "func_avg": round(p["fun"]/K, 4),
            # clean avg excludes infra-errored samples from the denominator
            "syntax_avg_clean": round(p["syn"]/scored, 4) if scored > 0 else 0.0,
            "func_avg_clean": round(p["fun"]/scored, 4) if scored > 0 else 0.0,
            "n_infra": p["infra"],
            "syntax": p["s0s"], "func": p["s0f"], "error": p["s0e"],
            "errors": p["errors"],
        })
    syn = sum(r["syntax"] for r in results); fun = sum(r["func"] for r in results)
    n_infra = sum(r["n_infra"] for r in results)
    syn_avg = sum(r["syntax_avg"] for r in results)/n if n else 0
    fun_avg = sum(r["func_avg"] for r in results)/n if n else 0
    # clean: re-average per-design clean avgs (designs with all-infra contribute 0)
    syn_avg_clean = sum(r["syntax_avg_clean"] for r in results)/n if n else 0
    fun_avg_clean = sum(r["func_avg_clean"] for r in results)/n if n else 0
    summary = {"model": args.model, "num_designs": n, "num_samples": K,
               "syntax_avg@%d" % K: round(100*syn_avg, 2), "func_avg@%d" % K: round(100*fun_avg, 2),
               "syntax_avg_clean@%d" % K: round(100*syn_avg_clean, 2),
               "func_avg_clean@%d" % K: round(100*fun_avg_clean, 2),
               "n_infra": n_infra,
               "syntax_pass@1": round(100*syn/n, 2) if n else 0,
               "func_pass@1": round(100*fun/n, 2) if n else 0,
               "syntax_count": syn, "func_count": fun}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"[rtllm] {args.model}: func_avg@{K}={round(100*fun_avg,2)}% "
          f"(clean={round(100*fun_avg_clean,2)}% excl {n_infra} infra samples)  "
          f"legacy func@1={summary['func_pass@1']}%", flush=True)


if __name__ == "__main__":
    main()
