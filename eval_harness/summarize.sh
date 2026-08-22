#!/bin/bash
# ============================================================
# summarize.sh — print metrics for a served model from results/<key>/.
#   bash summarize.sh <served_name>            # per-bench breakdown
#   bash summarize.sh --table <served_name>    # one TSV row in the 11-col
#                                              # deliverable table column order
# Robust to partial runs (missing files -> "-"/empty).
# ============================================================
source "$(dirname "$0")/config.sh"
if [ "${1:-}" = "--table" ]; then
  TABLE=1; KEY=$2
else
  TABLE=0; KEY=$1
fi
[ -z "$KEY" ] && { echo "usage: summarize.sh [--table] <served_name>"; exit 1; }
R="$RESULTS_DIR/$KEY"

if [ "$TABLE" = 1 ]; then
"$SYS_PY" - "$R" "$KEY" <<'PY'
import json, os, sys, re, glob, subprocess
# Single source of truth for infra/skip exclusion (verify.core.is_infra), so the
# table path and the merge script and the breakdown path can never disagree.
sys.path.insert(0, os.environ.get("BENCHINFRA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verify.core import is_infra as _is_infra
from verify.verilogeval import recompute_pass_rate_clean as _ve_clean
# scripts/ holds the single KB column + fast metric source of truth so the
# table and the standalone kb_fast_metric.py CLI can never disagree.
sys.path.insert(0, os.path.join(os.environ.get("BENCHINFRA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from kb_fast_metric import compute_kb_level as _kb_level
R, KEY = sys.argv[1], sys.argv[2]

# KB reference timing (A100 eager): one file shared across models at
# $RESULTS_DIR/_kb_ref_timing_A100.json (one level up from R = $RESULTS_DIR/$KEY).
# Honors KB_REF_TIMING_FILE. None if absent (fast omitted, not an error).
def _kb_ref_data():
    p = os.environ.get("KB_REF_TIMING_FILE") or os.path.join(os.path.dirname(R), "_kb_ref_timing_A100.json")
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None
_KB_REF = _kb_ref_data()

def ve(task):
    task_dir = os.path.join(R, task)
    f = os.path.join(task_dir, "summary.txt")
    if not os.path.exists(f): return ""
    # Closed-blacklist infra carve-out (verify.verilogeval): prefer the clean
    # pass_rate that excludes gateway-retry-exhausted samples from the
    # denominator; for clean runs n_infra==0 and clean==raw. Falls back to the
    # raw pass_rate line only if the recompute can't parse summary.txt.
    try: rc = _ve_clean(task_dir)
    except Exception: rc = None
    if rc:
        v = rc["pass_rate_clean"] if rc["n_infra"] else rc["pass_rate"]
        return f"{v}" + (f" (infra={rc['n_infra']})" if rc["n_infra"] else "")
    for line in open(f):
        m = re.search(r"pass_rate\s*=\s*([\d.]+)", line)
        if m: return f"{float(m.group(1))}"
    return ""

def rtllm_func():
    f = os.path.join(R, "rtllm", "summary.json")
    if not os.path.exists(f): return ""
    try:
        s = json.load(open(f))["summary"]; k = s.get("num_samples", 1)
        # Prefer the "clean" avg that excludes gateway-errored samples from
        # the denominator (n_infra of them); fall back to the legacy /K avg
        # for runs produced before the infra fix.
        v = s.get(f"func_avg_clean@{k}", s.get(f"func_avg@{k}", s.get("func_pass@1")))
        return "" if v is None else str(v)
    except Exception: return ""

def cvdp_pass():
    # CVDP raw_result.json is {problem_id: {category, difficulty, tests, errors},
    # 'metadata': {...}} — iterate problem keys, not treat the file as 1 problem.
    # A problem whose tests ONLY show model-response timeouts ("Request timed
    # out" / "Unable to get response") is an INFRA failure, not a model coding
    # failure — exclude from the denominator (else it deflates pass@1 like the
    # RTLLM silent-zero; see COMPARISON.md CVDP audit). Cocotb ran for the rest.
    rr = glob.glob(os.path.join(R, "cvdp", "**", "raw_result.json"), recursive=True)
    n = passed = n_infra = 0
    def _is_infra(tests):
        msgs = [str(t.get("error_msg", "")).lower() for t in tests if t.get("error_msg")]
        if not msgs: return False  # no error msg = genuine compile/sim fail
        return all(("timed out" in m or "unable to get response" in m
                    or "request timed out" in m) for m in msgs)
    for f in rr:
        try: d = json.load(open(f))
        except Exception: continue
        for k, v in d.items():
            if k == "metadata" or not isinstance(v, dict): continue
            tests = v.get("tests", []) or []
            errs = v.get("errors")
            is_pass = errs == 0 and tests and all(t.get("result", 1) == 0 for t in tests)
            if is_pass:
                passed += 1; n += 1
            elif _is_infra(tests):
                n_infra += 1  # excluded from denominator
            else:
                n += 1  # genuine fail
    if not n and not n_infra: return ""
    return f"{passed}/{n}" + (f" (infra={n_infra})" if n_infra else "") if n else f"(infra={n_infra})"

def kb(level):
    f = os.path.join(R, "kernelbench", f"level{level}", "eval_results.json")
    if not os.path.exists(f): return ""
    try: er = json.load(open(f))
    except Exception: return ""
    m = _kb_level(er, _KB_REF, level)
    # denominator = scored rows (generate-stage "Kernel not found" infra dropped);
    # same function the standalone CLI uses, so the table and CLI agree.
    if not m["n"]: return "" if not m["drop"] else f"(drop={m['drop']})"
    s = f"{m['comp']}/{m['n']} {m['corr']}/{m['n']}"
    if m["drop"]: s += f" (drop={m['drop']})"
    # fast = share of correct+ref-timed kernels beating PyTorch eager (speedup>1).
    # Read-only (JSON only). Omitted entirely when no ref timing file is
    # available; "n/a" when the file exists but no correct kernel was timed.
    if m["n_with_ref"]:
        s += f" fast={m['n_fast1x']}/{m['n_with_ref']}"
    elif _KB_REF is not None:
        s += " fast=n/a"
    return s

def tbg():
    f = os.path.join(R, "tritonbench_g", "verified.jsonl")
    if not os.path.exists(f): return ""
    n = comp = corr = nskip = 0
    # is_infra is the single denominator predicate (verify.core). It honours
    # both the post-finalize shape (failure_origin=="infrastructure") and the
    # legacy verify_skipped/ref_smoke_failed flag on old TBG rows.
    for line in open(f):
        line=line.strip()
        if not line: continue
        r = json.loads(line)
        if _is_infra(r): nskip += 1; continue   # unwinnable ref, exclude from denom
        n += 1
        if r.get("verify_compiled"): comp += 1
        if r.get("verify_correct"): corr += 1
    if not n and not nskip: return ""
    # denominator = runnable refs only; (skip=K) flags unrunnable refs that
    # would otherwise cap the metric at a non-discriminating ceiling.
    return f"{comp}/{n} {corr}/{n} (skip={nskip})" if (n or nskip) else ""

def realbench_pair():
    # verify.log line "task_level:module s1 s5 f1 f5 fo1 fo5" -> Syn@5, Func@5
    f = os.path.join(R, "realbench", "verify.log")
    if not os.path.exists(f): return "", ""
    for line in open(f):
        if line.startswith("task_level:module"):
            p = line.split("task_level:module", 1)[1].strip().split()
            if len(p) >= 4:
                return p[1], p[3]   # syntax_5, function_5
    return "", ""

def archx_t():
    f = os.path.join(R, "archxbench", "summary.json")
    if not os.path.exists(f): return ""
    try:
        s = json.load(open(f))["summary"]
        # Prefer t_clean (averaged only over designs that got a real
        # generation, excluding gateway-errored samples); fall back to t.
        return str(s.get("t_clean", s.get("t", "")))
    except Exception: return ""

# 11 columns: VEval-spec, VEval-code, RTLLM, CVDP, RealBench-Syn@5, RealBench-Func@5,
# ArchXBench-t, KB-L1, KB-L2, KB-L3, TritonBench-G
rb_syn, rb_func = realbench_pair()
cells = [ve("spec-to-rtl"), ve("code-complete-iccad2023"), rtllm_func(), cvdp_pass(),
         rb_syn, rb_func, archx_t(),
         kb(1), kb(2), kb(3), tbg()]
print("\t".join([KEY] + cells))
PY
exit 0
fi

"$SYS_PY" - "$R" "$KEY" <<'PY'
import json, os, sys, re
sys.path.insert(0, os.environ.get("BENCHINFRA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verify.core import is_infra as _is_infra
from verify.verilogeval import recompute_pass_rate_clean as _ve_clean
# scripts/ holds the single KB column + fast metric source of truth.
sys.path.insert(0, os.path.join(os.environ.get("BENCHINFRA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from kb_fast_metric import compute_kb_level as _kb_level
R, KEY = sys.argv[1], sys.argv[2]
print(f"\n=== {KEY} ===  ({R})\n")

# KB ref timing (A100 eager), shared across models. None if absent.
def _kb_ref_data():
    p = os.environ.get("KB_REF_TIMING_FILE") or os.path.join(os.path.dirname(R), "_kb_ref_timing_A100.json")
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None
_KB_REF = _kb_ref_data()

# VerilogEval: summary.txt pass_rate, with the closed-blacklist infra carve-out
# (verify.verilogeval). Clean excludes gateway-retry-exhausted samples from the
# denominator; for clean runs n_infra==0 and clean==raw.
def ve(task):
    task_dir = os.path.join(R, task)
    f = os.path.join(task_dir, "summary.txt")
    if not os.path.exists(f): return "-"
    rc = None
    try: rc = _ve_clean(task_dir)
    except Exception: rc = None
    raw = None
    for line in open(f):
        m = re.search(r"pass_rate\s*=\s*([\d.]+)", line)
        if m: raw = float(m.group(1)); break
    if rc and rc["n_infra"]:
        return (f"{rc['pass_rate_clean']:.2f}  (raw={rc['pass_rate']:.2f},"
                f" infra={rc['n_infra']} excluded)")
    if rc:
        return f"{rc['pass_rate']:.2f}"
    return "-" if raw is None else f"{raw:.2f}"
print(f"VerilogEval  spec-to-rtl     : {ve('spec-to-rtl')}")
print(f"VerilogEval  code-complete   : {ve('code-complete-iccad2023')}")

# RTLLM: summary.json
def js(path):
    return json.load(open(path))["summary"] if os.path.exists(path) else None
s = js(os.path.join(R, "rtllm", "summary.json"))
if s:
    k = s.get("num_samples", 1)
    print(f"RTLLM  func_avg@{k}           : {s.get('func_avg@%d'%k, s.get('func_pass@1'))}"
          f"  (clean={s.get('func_avg_clean@%d'%k, '-')}, n_infra={s.get('n_infra', 0)})")
    print(f"RTLLM  syntax_avg@{k}         : {s.get('syntax_avg@%d'%k, s.get('syntax_pass@1'))}"
          f"  (clean={s.get('syntax_avg_clean@%d'%k, '-')})")
else:
    print("RTLLM                        : -")

# ArchX: summary.json (n/t)
s = js(os.path.join(R, "archxbench", "summary.json"))
if s:
    print(f"ArchXBench  n                : {s.get('n')}/{s.get('num_samples')}"
          f"  (clean compile={int(round(100*float(s.get('n_clean', 0))))}% of scored)")
    print(f"ArchXBench  t                : {s.get('t')}%"
          f"  (clean={s.get('t_clean', '-')}%, n_infra={s.get('n_infra', 0)})")
else:
    print("ArchXBench                   : -")

# RealBench: verify.log last "task_level:module s1 s5 f1 f5 ..." line
f = os.path.join(R, "realbench", "verify.log")
rb = "-"
if os.path.exists(f):
    for line in open(f):
        if line.startswith("task_level:module"):
            rb = line.split("task_level:module", 1)[1].strip()
if rb != "-":
    p = rb.split()
    # order: syntax_1 syntax_5 function_1 function_5 formal_1 formal_5
    def pct(x):
        try: return f"{float(x)*100:.1f}"
        except: return x
    if len(p) >= 4:
        print(f"RealBench  Syn@1 / Syn@5     : {pct(p[0])} / {pct(p[1])}")
        print(f"RealBench  Func@1 / Func@5   : {pct(p[2])} / {pct(p[3])}")
    else:
        print(f"RealBench                    : {rb}")
else:
    print("RealBench                    : -")

# KernelBench: level dirs. Uses the single KB source of truth (compute_kb_level)
# so the breakdown and the --table path agree on denominator / infra / fast.
kb = os.path.join(R, "kernelbench")
if os.path.isdir(kb):
    for lv in sorted(os.listdir(kb)):
        f = os.path.join(kb, lv, "eval_results.json")
        if not os.path.exists(f): continue
        try:
            lvl = int(lv.replace("level", ""))
        except ValueError:
            continue  # not a level dir
        try: er = json.load(open(f))
        except Exception: continue
        m = _kb_level(er, _KB_REF, lvl)
        n = m["n"]; drop = m["drop"]; comp = m["comp"]; corr = m["corr"]
        dropstr = f"  drop={drop}(infra)" if drop else ""
        print(f"KernelBench {lv}  correct%   : {100*corr/n if n else 0:.1f}%  (compiled {comp}/{n}{dropstr})")
        if m["n_with_ref"]:
            print(f"KernelBench {lv}  fast_1x    : {100*m['fast1x']:.1f}%  "
                  f"({m['n_fast1x']}/{m['n_with_ref']} correct kernels beat eager; "
                  f"median speedup {m['median_speedup']:.2f}x)")
        elif _KB_REF is not None:
            print(f"KernelBench {lv}  fast_1x    : n/a (no timed correct kernels)")

# TritonBench G/T: verified.jsonl rows (verify_correct field).
# Skipped rows (ref unrunnable / no teacher code) are excluded from the
# denominator — they're unwinnable for any model and cap the metric at a
# non-discriminating ceiling otherwise.
def tbg(split):
    f = os.path.join(R, f"tritonbench_{split}", "verified.jsonl")
    if not os.path.exists(f): return None, 0, 0
    n = corr = comp = nskip = 0
    # is_infra is the single denominator predicate (verify.core): only a true
    # reference-unrunnable (ref_smoke_failed / verify_skipped / failure_origin
    # == infrastructure) is excluded. skipped_no_teacher_code is a MODEL fail
    # and stays in the denominator — the 2026-07-20 narrow口径, now in one place.
    for line in open(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if _is_infra(r): nskip += 1; continue
        n += 1
        if r.get("verify_correct"): corr += 1
        if r.get("verify_compiled"): comp += 1
    return corr, n, nskip
for sp in ("g", "t"):
    corr, n, nskip = tbg(sp)
    if n or nskip:
        sk = f" skip={nskip}" if nskip else ""
        print(f"TritonBench-{sp}  fast_0     : {100*corr/n if n else 0:.1f}%  (correct {corr}/{n}{sk})")
    else:
        print(f"TritonBench-{sp}              : -")

# CVDP: prefer the official run_reporter.py; fall back to raw_result.json count.
import glob, subprocess
cvdp_out = os.path.join(R, "cvdp")
cvdp_run = os.path.join(os.environ.get("CVDP_ROOT", ""), "runs", KEY)
reported = False
if os.path.isdir(cvdp_run):
    reporter = os.path.join(os.environ.get("CVDP_ROOT", ""), "run_reporter.py")
    venv_py = os.environ.get("CVDP_VENV", "python3")
    if os.path.exists(reporter):
        try:
            out = subprocess.run([venv_py, reporter, "--run-dir", cvdp_run,
                                  "--pass-at-k", "1"],
                                 capture_output=True, text=True, timeout=120)
            for line in out.stdout.splitlines():
                if "pass" in line.lower() and "@" in line:
                    print(f"CVDP  {line.strip()}")
                    reported = True
        except Exception:
            pass
if not reported and os.path.isdir(cvdp_out):
    rr = glob.glob(os.path.join(cvdp_out, "**", "raw_result.json"), recursive=True)
    n = passed = n_infra = 0
    def _is_infra(tests):
        msgs = [str(t.get("error_msg", "")).lower() for t in tests if t.get("error_msg")]
        if not msgs: return False
        return all(("timed out" in m or "unable to get response" in m
                    or "request timed out" in m) for m in msgs)
    for f in rr:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if k == "metadata" or not isinstance(v, dict): continue
            tests = v.get("tests", []) or []
            errs = v.get("errors")
            if errs == 0 and tests and all(t.get("result", 1) == 0 for t in tests):
                passed += 1; n += 1
            elif _is_infra(tests):
                n_infra += 1
            else:
                n += 1
    if n or n_infra:
        infr = f", infra={n_infra} excluded" if n_infra else ""
        print(f"CVDP  pass@1                : {100*passed/n if n else 0:.1f}%  (passed {passed}/{n}{infr})")
    elif not reported:
        print("CVDP                         : -")
elif not reported and not os.path.isdir(cvdp_out):
    print("CVDP                         : -")
print()
PY
