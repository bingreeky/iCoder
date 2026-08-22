#!/usr/bin/env python3
# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""Regenerate VerilogEval *-sv-iv-test.log files by running compile+sim in /tmp
(overlay fs) to dodge the dop-fuse VCD-write D-state hang, then let the OFFICIAL
sv-iv-analyze compute pass_rate for identical scoring.

WHY THIS EXISTS: on a FUSE-backed working dir (e.g. dop-fuse), vvp processes that
$dumpvars a VCD can wedge in uninterruptible D-state while writing, hanging the
whole VerilogEval `make` verify. Re-running just the stuck sims in a real /tmp
overlay sidesteps it. Only (re)tests samples whose iv-test log is missing or a
TIMEOUT placeholder; existing valid logs are kept. Writes compile stdout+stderr
then vvp stdout+stderr into the log, exactly as the Makefile's
`iverilog ... &> log; timeout 30 ./sim &>> log`.

Usage: retest_ve_tmp.py <task_dir> <dataset_dir> <problems_file> [jobs]
  K (samples per problem) is read from env K, default 4 (matches average@4).

Ported from the eval harness verbatim — the FUSE VCD D-state hang is a
runtime/toolchain property, not an infra-shape one, so no adaptation is needed.
"""
import os, sys, subprocess, tempfile, concurrent.futures as cf

TASK_DIR = os.path.abspath(sys.argv[1])
DS = os.path.abspath(sys.argv[2])
PROB_FILE = sys.argv[3]
JOBS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
K = int(os.environ.get("K", "4"))
probs = [l.split()[0] for l in open(PROB_FILE) if l.strip()]

def one(p, s):
    sv = os.path.join(TASK_DIR, p, f"{p}_sample{s:02d}.sv")
    log = os.path.join(TASK_DIR, p, f"{p}_sample{s:02d}-sv-iv-test.log")
    tb = os.path.join(DS, f"{p}_test.sv")
    rf = os.path.join(DS, f"{p}_ref.sv")
    buf = []
    if not os.path.exists(sv) or os.path.getsize(sv) < 25:
        buf.append("TIMEOUT\n")  # treated as fail
        open(log, "w").write("".join(buf)); return
    with tempfile.TemporaryDirectory(dir="/tmp") as wd:
        g = os.path.join(wd, "gen.sv"); open(g, "w").write(open(sv).read())
        simv = os.path.join(wd, "sim")
        try:
            c = subprocess.run(["iverilog", "-Wall", "-Winfloop", "-Wno-timescale",
                                "-g2012", "-s", "tb", "-o", simv, g, tb, rf],
                               capture_output=True, text=True, timeout=60, cwd=wd)
            buf.append(c.stdout); buf.append(c.stderr)
            if c.returncode == 0:
                try:
                    r = subprocess.run(["vvp", simv], capture_output=True, text=True,
                                       timeout=30, cwd=wd)
                    buf.append(r.stdout); buf.append(r.stderr)
                except subprocess.TimeoutExpired:
                    buf.append("\nTIMEOUT\n")
                    subprocess.run(["pkill", "-9", "-f", simv], capture_output=True)
        except subprocess.TimeoutExpired:
            buf.append("\nTIMEOUT\n")
    open(log, "w").write("".join(buf))

tasks = [(p, s) for p in probs for s in range(1, K + 1)]
# only redo missing / TIMEOUT-only logs
todo = []
for p, s in tasks:
    log = os.path.join(TASK_DIR, p, f"{p}_sample{s:02d}-sv-iv-test.log")
    if not os.path.exists(log):
        todo.append((p, s)); continue
    txt = open(log).read()
    if txt.strip() == "TIMEOUT" or "0.8/" in txt or "check TMP" in txt or not txt.strip():
        todo.append((p, s))
print(f"retesting {len(todo)}/{len(tasks)} samples in /tmp with {JOBS} workers", flush=True)
with cf.ThreadPoolExecutor(max_workers=JOBS) as ex:
    list(ex.map(lambda a: one(*a), todo))
print("done regenerating logs", flush=True)
