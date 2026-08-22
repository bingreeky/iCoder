#!/usr/bin/env python3
"""
RealBench generation layer. Writes samples/<model>/<system>.jsonl for run_verify.py.
Default (--num-samples 1): 1 sample/module (codeid=1, greedy) -> pass@1.
Optional (--num-samples K --temperature T): K records/module (codeid 1..K,
codeid 1 greedy, 2..K sampled) so run_verify --num_samples K reports pass@1/@K.

Run with system python3 (stdlib + HTTP only).
  python gen_realbench.py --base-url http://localhost:8000/v1 --model base-9b \
      --workers 60 --num-samples 5 --temperature 0.8
"""
import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RB_ROOT = os.environ.get("RB_ROOT",
    os.path.join(os.environ.get("CODERBENCH_ROOT", "."), "RealBench"))
SYS_MSG = ("You are an expert Verilog/SystemVerilog RTL designer. You write "
           "complete, synthesizable modules with correct syntax.")
SYSTEMS = ["sdc", "aes", "e203_hbirdv2"]


def query(base_url, model, prompt, max_tokens=32768, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYS_MSG},
                     {"role": "user", "content": prompt +
                      "\n\nWrite the complete module implementation. Keep the "
                      "module name and port list exactly as specified. Enclose "
                      "your code with ```verilog and ```."}],
        "temperature": temperature,
    }
    if max_tokens and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    # timeout=900 (not the prior 86400=24h): a hung HTTP request must FAIL and
    # let the threadpool drain, else one stuck request blocks gen_realbench's
    # `for fu in futs` join indefinitely -> the whole UP group stalls for up to
    # 24h (seen on a model (long idle): model idle 14+min, gen never returned). 900s is
    # >10x any real RTL generation; code that hasn't responded by then is stuck.
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": resp.get("usage", {}),
    }


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


def gen_one(task, problem, base_url, model, max_tokens, codeid, temperature):
    try:
        r = query(base_url, model, problem, max_tokens=max_tokens, temperature=temperature)
        code = extract_verilog(r["content"])
        trace = {"ok": True, "codeid": codeid, "content": r["content"],
                 "reasoning_content": r["reasoning_content"],
                 "finish_reason": r["finish_reason"], "usage": r["usage"],
                 "content_len": len(r["content"]), "code_len": len(code), "error": ""}
        return task, codeid, code, trace
    except Exception as e:
        return task, codeid, "", {"ok": False, "codeid": codeid, "content": "",
                          "reasoning_content": "", "finish_reason": None, "usage": {},
                          "content_len": 0, "code_len": 0, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--systems", nargs="*", default=SYSTEMS)
    args = ap.parse_args()

    out_dir = os.path.join(RB_ROOT, "samples", args.model)
    os.makedirs(out_dir, exist_ok=True)
    trace_dir = os.path.join(RB_ROOT, "traces", args.model)
    os.makedirs(trace_dir, exist_ok=True)
    K = args.num_samples

    for system in args.systems:
        prob_file = os.path.join(RB_ROOT, "problems", system, "problems.jsonl")
        if not os.path.exists(prob_file):
            print(f"[realbench] missing {prob_file}; run generate_problem.py first")
            continue
        tasks = [json.loads(l) for l in open(prob_file) if l.strip()]
        print(f"[realbench] {system}: {len(tasks)} tasks x {K} samples, model={args.model}", flush=True)
        records = {}; traces = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for t in tasks:
                for codeid in range(1, K + 1):
                    temp = 0.0 if codeid == 1 else args.temperature
                    futs.append(ex.submit(gen_one, t["task"], t["problem"],
                                          args.base_url, args.model, args.max_tokens, codeid, temp))
            for fu in futs:
                task, codeid, code, trace = fu.result()
                records[(task, codeid)] = code
                traces[(task, codeid)] = trace
                u = trace.get("usage", {})
                ct = u.get("completion_tokens", "?")
                fr = trace.get("finish_reason")
                print(f"  {task:28s} #{codeid} code={len(code):5d} fin={str(fr):8s} ctok={ct} {trace.get('error','')[:40]}", flush=True)
        with open(os.path.join(out_dir, f"{system}.jsonl"), "w") as f:
            for t in tasks:
                for codeid in range(1, K + 1):
                    f.write(json.dumps({"task": t["task"], "codeid": codeid,
                                        "code": records.get((t["task"], codeid), "")}) + "\n")
        with open(os.path.join(trace_dir, f"{system}.jsonl"), "w") as f:
            for t in tasks:
                for codeid in range(1, K + 1):
                    tr = traces.get((t["task"], codeid), {})
                    tr = {"task": t["task"], **tr}
                    f.write(json.dumps(tr) + "\n")
        print(f"[realbench] wrote {out_dir}/{system}.jsonl", flush=True)


if __name__ == "__main__":
    main()
