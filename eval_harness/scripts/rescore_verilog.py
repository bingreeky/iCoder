#!/usr/bin/env python3
# ⚠️ Diagnostic tool — NOT part of the standard eval pipeline. This is a
#    recovery/rescore helper (not invoked by run_all.sh / summarize.sh).
"""rescore_verilog.py — Phase 1 (diagnose) + Phase 2 (re-verify) + Phase 3
(recompute) for the Verilog benches (VerilogEval code-complete, ArchXBench)
of the re-extraction pipeline. VE spec-to-rtl is a regression guard (must
report 0 recoverable).

Why: the production extractors grab the FIRST fenced block (VE sv-generate,
ArchX run_archx.extract_verilog). When a model emits an explanation/skeleton
fence first and the answer fence second, the real (complete) module is
dropped and the .sv/.v on disk is a fragment -> false fail. This re-parses
the raw logged response with pick_verilog (takes the LAST complete
module...endmodule) and, for complete candidates the extractor missed,
re-runs the existing verifier to see if the failure flips to pass.

Phase 1 (CPU, read-only, --phase diagnose): walk the result dir, for each
FAILED sample read the raw response, run pick_verilog, classify
{truncated (unrecoverable), recoverable-candidate, clean}. Print counts +
candidate list. No verification.

Phase 2 (CPU, iverilog): for each candidate, re-verify — VE via the existing
`make sv-iv-analyze` in the BUILD dir (touch the re-extracted .sv); ArchX via
the pure verify_candidate() function. Count fail->pass flips.

Phase 3: recompute pass_rate / t / func@1 with recovered rows; report old->new.

Original on-disk .sv/.v/_result.json are NEVER overwritten in place; recovered
verdicts land in a `<task>.rescored.jsonl` side-file + a printed table.

Usage:
  python3 rescore_verilog.py diagnose --model <model> --bench ve-cc
  python3 rescore_verilog.py diagnose --model <model> --bench archx
  python3 rescore_verilog.py diagnose --model <model> --bench ve-spec   # guard
  python3 rescore_verilog.py reverify --model <model> --bench archx
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # eval_harness/
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)  # for _rescore_extract
sys.path.insert(0, os.path.join(ROOT, "benches"))  # for run_archx (ArchX re-verify)
sys.path.insert(0, ROOT)  # for verify.*

from _rescore_extract import (pick_verilog, is_verilog_recovery_candidate,  # noqa: E402
                              pick_verilog_candidates,
                              pick_ve_cc, pick_ve_cc_candidates)

RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
ARCHX_ROOT = os.environ.get("ARCHX_ROOT", os.path.join(os.environ.get("CODER_ROOT", os.path.join(ROOT, "benches")), "bench", "ArchXBench"))
# iverilog v12 (same binary the eval used) — config.local.sh points here.
IVERILOG12_BIN = os.environ.get("IVERILOG12_BIN", os.path.join(os.environ.get("CODER_ROOT", os.path.join(ROOT, "benches")), "env", "iverilog12", "bin"))
if os.path.isdir(IVERILOG12_BIN):
    os.environ["PATH"] = IVERILOG12_BIN + os.pathsep + os.environ.get("PATH", "")

BENCH_TASK = {
    "ve-spec": "spec-to-rtl",
    "ve-cc": "code-complete-iccad2023",
    "archx": "archxbench",
}


# --------------------------------------------------------------------------
# raw-response readers
# --------------------------------------------------------------------------
def ve_raw_from_log(path: str) -> str:
    """Extract the raw model response from a VE *-sv-generate.log. The
    response sits between the 'Response' header line and the 'Statistics'
    header line (sv-generate:493-509 prints both; resp.content at :497)."""
    text = open(path, errors="ignore").read()
    i = text.find("\nResponse\n")
    if i < 0:
        return ""
    j = text.find("\nStatistics\n", i)
    body = text[i + 1:] if j < 0 else text[i + 1:j]
    # drop the 'Response' header line itself + following '---' separators/blanks
    lines = body.split("\n")
    out = []
    started = False
    for ln in lines:
        if not started:
            if ln.strip() in ("Response", "") or set(ln.strip()) <= {"-"}:
                continue
            started = True
        out.append(ln)
    # trim trailing separators/blanks
    while out and (out[-1].strip() in ("",) or set(out[-1].strip()) <= {"-"}):
        out.pop()
    return "\n".join(out)


def archx_raw_from_resp(path: str) -> str:
    """ArchX *_response.txt is the raw model response verbatim
    (run_archx.py:267 writes resp)."""
    return open(path, errors="ignore").read()


# --------------------------------------------------------------------------
# Phase 1: diagnose
# --------------------------------------------------------------------------
def diagnose_ve(model: str, task: str):
    base = os.path.join(RESULTS_DIR, model, task)
    logs = sorted(glob.glob(os.path.join(base, "Prob*", "*-sv-generate.log")))
    counts = {"total": 0, "clean": 0, "recoverable": 0, "truncated": 0,
              "empty": 0, "no_log": 0}
    cands = []
    for lg in logs:
        counts["total"] += 1
        sv = lg.replace("-sv-generate.log", ".sv")
        old = open(sv, errors="ignore").read() if os.path.exists(sv) else ""
        raw = ve_raw_from_log(lg)
        new, status = pick_verilog(raw)
        if status == "empty":
            counts["empty"] += 1
            continue
        if status == "truncated":
            counts["truncated"] += 1
            continue
        if is_verilog_recovery_candidate(new, status, old):
            counts["recoverable"] += 1
            cands.append((os.path.basename(lg).replace("-sv-generate.log", ""),
                          len(old), len(new), "old_endmod" if "endmodule" in old else "old_NO_endmod"))
        else:
            counts["clean"] += 1
    return counts, cands


def diagnose_archx(model: str):
    base = os.path.join(RESULTS_DIR, model, "archxbench")
    resps = sorted(glob.glob(os.path.join(base, "*_response.txt")))
    counts = {"total": 0, "passed": 0, "clean": 0, "recoverable": 0,
              "truncated": 0, "empty": 0, "no_result": 0}
    cands = []
    for rp in resps:
        jp = rp.replace("_response.txt", "_result.json")
        vp = rp.replace("_response.txt", ".v")
        if not os.path.exists(jp):
            counts["no_result"] += 1
            continue
        try:
            r = json.load(open(jp))
        except Exception:
            counts["no_result"] += 1
            continue
        counts["total"] += 1
        # only FAILED samples can recover (passed samples already = pass)
        if r.get("syntax") and r.get("t", 0) >= 100.0:
            counts["passed"] += 1
            continue
        old = open(vp, errors="ignore").read() if os.path.exists(vp) else ""
        raw = archx_raw_from_resp(rp)
        new, status = pick_verilog(raw)
        if status == "empty":
            counts["empty"] += 1
            continue
        if status == "truncated":
            counts["truncated"] += 1
            continue
        if is_verilog_recovery_candidate(new, status, old):
            counts["recoverable"] += 1
            cands.append((os.path.basename(rp).replace("_response.txt", ""),
                          len(old), len(new),
                          "old_endmod" if "endmodule" in old else "old_NO_endmod",
                          r.get("t", 0), r.get("syntax", 0)))
        else:
            counts["clean"] += 1
    return counts, cands


def cmd_diagnose(args):
    bench = args.bench
    print(f"\n=== Phase-1 diagnose: {args.model} / {bench} ===")
    if bench in ("ve-spec", "ve-cc"):
        counts, cands = diagnose_ve(args.model, BENCH_TASK[bench])
    elif bench == "archx":
        counts, cands = diagnose_archx(args.model)
    else:
        print(f"unknown bench {bench}"); return
    print("counts:", json.dumps(counts, indent=2))
    print(f"recoverable candidates: {len(cands)}")
    for c in cands[:20]:
        print("  ", c)
    if len(cands) > 20:
        print(f"   ... (+{len(cands)-20} more)")
    return counts, cands


# --------------------------------------------------------------------------
# design-dir index (ArchX)
# --------------------------------------------------------------------------
_ARCHX_INDEX = None


def archx_design_index():
    """Map design-name -> design_dir by walking ARCHX_ROOT/level-*/<name>."""
    global _ARCHX_INDEX
    if _ARCHX_INDEX is not None:
        return _ARCHX_INDEX
    idx = {}
    if not os.path.isdir(ARCHX_ROOT):
        print(f"[archx] ARCHX_ROOT not found: {ARCHX_ROOT}")
        _ARCHX_INDEX = {}
        return _ARCHX_INDEX
    for lvl in sorted(os.listdir(ARCHX_ROOT)):
        ld = os.path.join(ARCHX_ROOT, lvl)
        if not (lvl.startswith("level-") and os.path.isdir(ld)):
            continue
        for n in sorted(os.listdir(ld)):
            d = os.path.join(ld, n)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "problem-description.txt")):
                idx.setdefault(n, d)  # first level wins if duplicated
    _ARCHX_INDEX = idx
    return idx


# --------------------------------------------------------------------------
# Phase 2: re-verify
# --------------------------------------------------------------------------
def reverify_archx(model: str):
    """Re-verify each ArchX recovery candidate via the pure verify_candidate()
    function (run_archx.py:179). No model call, no network — just iverilog on
    the re-extracted code in a temp dir. Count fail->pass flips."""
    from run_archx import verify_candidate, tb_file  # benches/ on path
    base = os.path.join(RESULTS_DIR, model, "archxbench")
    idx = archx_design_index()
    out_path = os.path.join(base, "archxbench.rescored.jsonl")
    flips = 0
    tested = 0
    rows = []
    for rp in sorted(glob.glob(os.path.join(base, "*_response.txt"))):
        jp = rp.replace("_response.txt", "_result.json")
        vp = rp.replace("_response.txt", ".v")
        if not os.path.exists(jp):
            continue
        try:
            old_r = json.load(open(jp))
        except Exception:
            continue
        # only FAILED samples can recover
        if old_r.get("syntax") and old_r.get("t", 0) >= 100.0:
            continue
        old = open(vp, errors="ignore").read() if os.path.exists(vp) else ""
        raw = archx_raw_from_resp(rp)
        new, status = pick_verilog(raw)
        if not is_verilog_recovery_candidate(new, status, old):
            continue
        name = old_r.get("design")
        design_dir = idx.get(name)
        if not design_dir:
            print(f"  [skip] design dir not found for '{name}'")
            continue
        tb = tb_file(design_dir)
        if tb is None:
            print(f"  [skip] no testbench for '{name}'")
            continue
        has_golden = os.path.exists(os.path.join(design_dir, "scripts", "compare_outputs.py"))
        tested += 1
        try:
            syntax, t, err = verify_candidate(new, design_dir, tb, has_golden)
        except Exception as e:
            err = f"verify_candidate raised: {e}"
            syntax, t = 0, 0.0
        old_t = old_r.get("t", 0)
        old_syn = old_r.get("syntax", 0)
        flipped = (t >= 100.0 and old_t < 100.0) or (syntax == 1 and old_syn == 0 and t >= 100.0)
        if flipped:
            flips += 1
        rows.append({"design": name, "k": old_r.get("k"), "file": os.path.basename(rp),
                     "old_syntax": old_syn, "old_t": old_t,
                     "new_syntax": syntax, "new_t": round(t, 2),
                     "flipped": flipped, "error": (err or "")[:120]})
        flag = "FLIP" if flipped else "    "
        print(f"  [{flag}] {name}_s{old_r.get('k')}: t {old_t}->{t:.1f} syn {old_syn}->{syntax} | {err[:60]}")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[archx {model}] tested={tested} flips(fail->pass)={flips}  -> {out_path}")
    return flips, tested, rows


def _ve_sample_pass(log_path: str) -> bool:
    """A VE *-sv-iv-test.log records 'Mismatches: N in M samples' (sv-iv-test
    wrapper). N==0 => functional pass. Absent / compile-fail => not pass."""
    if not os.path.exists(log_path):
        return False
    txt = open(log_path, errors="ignore").read()
    m = re.search(r"Mismatches:\s*(\d+)\s+in\s+\d+\s+samples", txt)
    return bool(m) and int(m.group(1)) == 0


def _ve_read_summary_csv(base: str):
    """Read summary.csv -> {problem: (nsamples, pass_rate, [markers])}.
    Columns: problem,level,nsamples,pass_rate,m1..m4. summary.csv is written by
    sv-iv-analyze and is langchain-free to read."""
    scv = os.path.join(base, "summary.csv")
    out = {}
    if not os.path.exists(scv):
        return out
    for ln in open(scv, errors="ignore"):
        p = ln.strip().split(",")
        if len(p) < 5 or not p[0].startswith("Prob"):
            continue
        out[p[0]] = (int(p[2]), float(p[3]), p[4:])
    return out


def _ve_recompute_pass_rate(base: str, flips_by_prob: dict):
    """Recompute the aggregate pass_rate from the original summary.csv with the
    flipped samples applied. flips_by_prob: {problem: set(sample_idx_1based)}.
    A flip raises that problem's npass by 1 -> pass_rate += 1/nsamples. The
    aggregate = mean over problems of per-problem pass_rate (matches the
    sv-iv-analyze :174 formula, verified to reproduce 45.51 from the csv)."""
    sm = _ve_read_summary_csv(base)
    if not sm:
        return None, None
    total = 0.0
    for prob, (ns, pr, marks) in sm.items():
        if prob in flips_by_prob:
            pr = min(1.0, pr + len(flips_by_prob[prob]) / ns)
        total += pr
    return round(100.0 * total / len(sm), 2), len(sm)


def reverify_ve_cc(model: str, task: str = "code-complete-iccad2023"):
    """Re-verify each VE-cc recovery candidate via the existing `make
    <prob>/<name>-sv-iv-test.log` in the BUILD dir (Makefile.in:165-172 re-runs
    iverilog on the re-extracted .sv). Original .sv/.log/summary.txt are backed
    up + regenerated so the original run is preserved; recovered verdicts land
    in a side-report.

    NOTE: `make sv-iv-analyze` itself needs langchain (now installed); the
    aggregate pass_rate is recomputed directly from summary.csv (the
    langchain-free scored artifact). The OLD verdict is read from summary.csv's
    per-sample markers (col 4+), NOT from the .log — because this very function
    mutates the .log on a prior run, the .log is not a reliable old-truth."""
    base = os.path.join(RESULTS_DIR, model, task)
    if not os.path.isdir(base):
        print(f"[ve-cc] BUILD dir not found: {base}")
        return 0, 0, [], None
    sm = _ve_read_summary_csv(base)  # original verdicts (never mutated)
    backups = []  # (sv_path, orig_bytes)
    flips = 0
    tested = 0
    rows = []
    flips_by_prob = {}
    for lg in sorted(glob.glob(os.path.join(base, "Prob*", "*-sv-generate.log"))):
        sv = lg.replace("-sv-generate.log", ".sv")
        old = open(sv, errors="ignore").read() if os.path.exists(sv) else ""
        raw = ve_raw_from_log(lg)
        new, status = pick_verilog(raw)
        if not is_verilog_recovery_candidate(new, status, old):
            continue
        prob_dir = os.path.dirname(sv)
        name = os.path.basename(sv)[:-3]  # strip .sv
        prob = prob_dir.split("/")[-1]
        test_log = os.path.join(prob_dir, name + "-sv-iv-test.log")
        # OLD verdict from summary.csv marker (authoritative, unmutated)
        ms = re.search(r"_sample(\d+)$", name)
        sidx = int(ms.group(1)) if ms else 1
        marks = sm.get(prob, (0, 0.0, []))[2]
        old_pass = (sidx - 1 < len(marks)) and (marks[sidx - 1] == ".")
        # backup original .sv + stale bin/log so make rebuilds
        backups.append((sv, open(sv, "rb").read() if os.path.exists(sv) else b""))
        open(sv, "w").write(new + "\n")
        binf = os.path.join(prob_dir, name)
        for stale in (binf, test_log):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        tested += 1
        tgt = os.path.relpath(test_log, base)
        try:
            r = subprocess.run(["make", tgt], cwd=base,
                                capture_output=True, text=True, timeout=180)
            err = (r.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            err = "make timeout 180s"
        new_pass = _ve_sample_pass(test_log)
        flipped = (new_pass and not old_pass)
        if flipped:
            flips += 1
            flips_by_prob.setdefault(prob, set()).add(sidx)
        rows.append({"sample": name, "old_pass": old_pass, "new_pass": new_pass,
                     "flipped": flipped, "old_len": len(old), "new_len": len(new),
                     "old_marker": marks[sidx - 1] if sidx - 1 < len(marks) else "?",
                     "error": err[:120]})
        flag = "FLIP" if flipped else "    "
        print(f"  [{flag}] {name}: {old_pass}->{new_pass} (mark {marks[sidx-1] if sidx-1<len(marks) else '?'}) | {err[:60]}")
    # recompute aggregate from summary.csv (no langchain)
    new_pr, n_prob = _ve_recompute_pass_rate(base, flips_by_prob)
    # ---- restore original run: .sv + .log + summary.txt ----
    for sv, orig in backups:
        with open(sv, "wb") as f:
            f.write(orig)
        name = os.path.basename(sv)[:-3]
        prob_dir = os.path.dirname(sv)
        binf = os.path.join(prob_dir, name)
        test_log = os.path.join(prob_dir, name + "-sv-iv-test.log")
        for stale in (binf, test_log):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass
        tgt = os.path.relpath(test_log, base)
        try:
            subprocess.run(["make", tgt], cwd=base,
                            capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass
    if backups:
        try:
            subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                            capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass
    out_path = os.path.join(base, "ve_cc.rescored.jsonl")
    meta = {"model": model, "tested": tested, "flips": flips,
            "old_pass_rate": round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2),
            "new_pass_rate": new_pr, "n_problems": n_prob,
            "flips_by_problem": {k: sorted(list(v)) for k, v in flips_by_prob.items()}}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[ve-cc {model}] tested={tested} flips(fail->pass)={flips}  pass_rate {meta['old_pass_rate']}->{new_pr}  -> {out_path}")
    return flips, tested, rows, new_pr


def _reverify_ve_v2_core(model, task, mode, jobs):
    """Shared re-extract+re-verify for VE spec-to-rtl (mode='spec', pick_verilog
    full module, no interface) and code-complete (mode='cc', pick_ve_cc
    interface+body). Targets ALL S-fail (syntax-fail) samples; re-verifies via
    `make -jN sv-iv-test` + `sv-iv-analyze`; counts S->. flips; restores."""
    base = os.path.join(RESULTS_DIR, model, task)
    if not os.path.isdir(base):
        print(f"[ve-{mode}-v2] BUILD dir not found: {base}")
        return 0, 0, None
    sm = _ve_read_summary_csv(base)
    DS = os.path.join(ROOT, "ext", "verilog-eval", "dataset_code-complete-iccad2023")
    targets = []
    skipped = 0
    for prob, (ns, pr, marks) in sm.items():
        for sidx, mk in enumerate(marks, 1):
            if mk != "S":
                continue
            sv = os.path.join(base, prob, f"{prob}_sample{sidx:02d}.sv")
            lg = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-generate.log")
            if not os.path.exists(sv) or not os.path.exists(lg):
                skipped += 1
                continue
            interface = ""
            if mode == "cc":
                ifc = os.path.join(DS, f"{prob}_ifc.txt")
                interface = open(ifc, errors="ignore").read().strip() if os.path.exists(ifc) else ""
            raw = ve_raw_from_log(lg)
            if mode == "spec":
                new, status = pick_verilog(raw)
                if status != "complete" or not new:
                    skipped += 1
                    continue
            else:  # cc
                new, status = pick_ve_cc(raw, interface)
                if status in ("empty", "no_endmodule") or not new:
                    skipped += 1
                    continue
            orig = open(sv, "rb").read()
            # skip if the re-extraction is identical to on-disk (already correct)
            if new.strip() == orig.decode(errors="ignore").strip():
                continue
            logp = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-iv-test.log")
            targets.append((sv, logp, os.path.relpath(logp, base), orig, new, prob, sidx))
    print(f"[ve-{mode}-v2 {model}] {len(targets)} S-fail samples to re-extract+re-verify "
          f"({skipped} skipped: no log/no module/truncated)")
    if not targets:
        return 0, 0, round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    for sv, logp, tgt, orig, new, prob, sidx in targets:
        open(sv, "w").write(new + "\n")
        binf = sv[:-3]
        for st in (binf, logp):
            if os.path.exists(st):
                try: os.remove(st)
                except OSError: pass
    try:
        subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                        capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        print("  [warn] make sv-iv-test timeout (partial)")
    try:
        subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                        capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print("  [warn] make sv-iv-analyze timeout")
    new_sm = _ve_read_summary_csv(base)
    flips = 0
    flip_rows = []
    for sv, logp, tgt, orig, new, prob, sidx in targets:
        new_marks = new_sm.get(prob, (0, 0.0, []))[2]
        new_mk = new_marks[sidx - 1] if sidx - 1 < len(new_marks) else "?"
        flipped = (new_mk == ".")
        if flipped:
            flips += 1
        flip_rows.append({"sample": f"{prob}_sample{sidx:02d}",
                          "old_marker": "S", "new_marker": new_mk,
                          "flipped": flipped, "new_len": len(new)})
    new_pr, n_prob = _ve_recompute_pass_rate(base, {})
    old_pr = round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    # restore
    for sv, logp, tgt, orig, new, prob, sidx in targets:
        with open(sv, "wb") as f:
            f.write(orig)
        binf = sv[:-3]
        for st in (binf, logp):
            if os.path.exists(st):
                try: os.remove(st)
                except OSError: pass
    try:
        subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                        capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        pass
    try:
        subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                        capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pass
    out_path = os.path.join(base, f"ve_{mode}_v2.rescored.jsonl")
    meta = {"model": model, "mode": mode, "targets": len(targets),
            "flips": flips, "old_pass_rate": old_pr, "new_pass_rate": new_pr,
            "n_problems": n_prob}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in flip_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ve-{mode}-v2 {model}] targets={len(targets)} flips(S->.)={flips}  "
          f"pass_rate {old_pr}->{new_pr}  -> {out_path}")
    return flips, len(targets), new_pr


def reverify_ve_cc_v2(model: str, task: str = "code-complete-iccad2023", jobs: int = 8):
    return _reverify_ve_v2_core(model, task, "cc", jobs)


def reverify_ve_spec_v2(model: str, task: str = "spec-to-rtl", jobs: int = 8):
    return _reverify_ve_v2_core(model, task, "spec", jobs)


def reverify_ve_spec_v3(model: str, task: str = "spec-to-rtl", jobs: int = 8):
    """Multi-candidate VE-spec re-verify. For EVERY non-pass sample (S-fail
    syntax AND R-fail functional), generate ALL complete ``module...endmodule``
    blocks the raw response contains and try each, flipping if any passes.

    This bounds VE-spec honestly: an R-fail means the on-disk module compiles
    but gives wrong output. If the model ALSO emitted a correct alternative
    module that production's first-block-grab dropped, this recovers it. If
    every module the model emitted is wrong, no flip (real functional gap)."""
    from _rescore_extract import pick_verilog_candidates
    base = os.path.join(RESULTS_DIR, model, task)
    if not os.path.isdir(base):
        print(f"[ve-spec-v3] BUILD dir not found: {base}")
        return 0, 0, None
    sm = _ve_read_summary_csv(base)
    samples = []
    skipped = 0
    for prob, (ns, pr, marks) in sm.items():
        for sidx, mk in enumerate(marks, 1):
            if mk == ".":
                continue  # already passing
            sv = os.path.join(base, prob, f"{prob}_sample{sidx:02d}.sv")
            lg = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-generate.log")
            if not os.path.exists(sv) or not os.path.exists(lg):
                skipped += 1
                continue
            raw = ve_raw_from_log(lg)
            cands = pick_verilog_candidates(raw)
            if not cands:
                skipped += 1
                continue
            orig = open(sv, "rb").read()
            # skip if on-disk already matches the best candidate
            if any(c.strip() == orig.decode(errors="ignore").strip() for c in cands):
                # on-disk IS the last complete module — still try EARLIER alts
                alts = [c for c in cands if c.strip() != orig.decode(errors="ignore").strip()]
                if not alts:
                    continue
                cands = alts
            logp = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-iv-test.log")
            samples.append((sv, logp, orig, cands, prob, sidx, mk))
    print(f"[ve-spec-v3 {model}] {len(samples)} non-pass samples (S+R) with "
          f">=1 alt module ({skipped} skipped)")
    if not samples:
        return 0, 0, round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    max_rounds = max(len(s[3]) for s in samples)
    passed = {}
    for r in range(max_rounds):
        active = [(k, s) for k, s in enumerate(samples)
                  if r < len(s[3]) and k not in passed]
        if not active:
            break
        for k, s in active:
            sv, logp, orig, cands, prob, sidx, mk = s
            open(sv, "w").write(cands[r] + "\n")
            binf = sv[:-3]
            for st in (binf, logp):
                if os.path.exists(st):
                    try: os.remove(st)
                    except OSError: pass
        try:
            subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                            capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            print(f"  [warn] round {r} make timeout")
        try:
            subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                            capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass
        new_sm = _ve_read_summary_csv(base)
        for k, s in enumerate(samples):
            if k in passed or r >= len(s[3]):
                continue
            sv, logp, orig, cands, prob, sidx, mk = s
            new_marks = new_sm.get(prob, (0, 0.0, []))[2]
            new_mk = new_marks[sidx - 1] if sidx - 1 < len(new_marks) else "?"
            if new_mk == ".":
                passed[k] = r
    flips = len(passed)
    # proper flips_by_prob
    flips_by_prob = {}
    for k, s in enumerate(samples):
        if k in passed:
            flips_by_prob.setdefault(s[4], set()).add(s[5])
    new_pr, n_prob = _ve_recompute_pass_rate(base, flips_by_prob)
    old_pr = round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    # restore
    for sv, logp, orig, cands, prob, sidx, mk in samples:
        with open(sv, "wb") as f:
            f.write(orig)
        binf = sv[:-3]
        for st in (binf, logp):
            if os.path.exists(st):
                try: os.remove(st)
                except OSError: pass
    try:
        subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                        capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        pass
    try:
        subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                        capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pass
    from collections import Counter
    round_dist = dict(Counter(passed.values()))
    out_path = os.path.join(base, "ve_spec_v3.rescored.jsonl")
    meta = {"model": model, "mode": "spec-v3", "targets": len(samples),
            "flips": flips, "old_pass_rate": old_pr, "new_pass_rate": new_pr,
            "n_problems": n_prob, "flips_by_round": round_dist}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for k, s in enumerate(samples):
            sv, logp, orig, cands, prob, sidx, mk = s
            f.write(json.dumps({"sample": f"{prob}_sample{sidx:02d}",
                                 "old_marker": mk,
                                 "flipped": k in passed,
                                 "round": passed.get(k, -1)}, ensure_ascii=False) + "\n")
    print(f"[ve-spec-v3 {model}] targets={len(samples)} flips(non-pass->.)={flips}  "
          f"pass_rate {old_pr}->{new_pr}  flips_by_round={round_dist}  -> {out_path}")
    return flips, len(samples), new_pr


def reverify_ve_cc_v3(model: str, task: str = "code-complete-iccad2023", jobs: int = 8):
    """Multi-candidate VE-cc re-verify. For each S-fail sample, generate ALL
    candidate .sv reconstructions (every code-looking fenced block + the
    walk-back from the last endmodule) and try them in round-robin order,
    counting a flip if ANY candidate passes. This is model-agnostic: some
    models' real answer is usually in a fence, others' is usually raw body — trying
    both per sample finds whichever the model actually got right.

    Legitimacy: the model DID emit a correct body somewhere in its response;
    the extractor's job is to locate it. The recovered score is the upper bound
    on 'what the model would score with a perfect extractor' (always finds the
    correct block the model emitted). Round-based: build candidate[r] for every
    still-failing sample, `make -jN sv-iv-test` (incremental — only changed .sv
    rebuild), mark passes, advance r. Restores originals at the end."""
    from _rescore_extract import pick_ve_cc_candidates
    base = os.path.join(RESULTS_DIR, model, task)
    if not os.path.isdir(base):
        print(f"[ve-cc-v3] BUILD dir not found: {base}")
        return 0, 0, None
    sm = _ve_read_summary_csv(base)
    DS = os.path.join(ROOT, "ext", "verilog-eval", "dataset_code-complete-iccad2023")
    # per-sample candidate lists
    samples = []  # (sv, logp, orig_bytes, [cand_sv...], prob, sidx)
    skipped = 0
    for prob, (ns, pr, marks) in sm.items():
        for sidx, mk in enumerate(marks, 1):
            if mk != "S":
                continue
            sv = os.path.join(base, prob, f"{prob}_sample{sidx:02d}.sv")
            lg = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-generate.log")
            if not os.path.exists(sv) or not os.path.exists(lg):
                skipped += 1
                continue
            ifc = os.path.join(DS, f"{prob}_ifc.txt")
            interface = open(ifc, errors="ignore").read().strip() if os.path.exists(ifc) else ""
            raw = ve_raw_from_log(lg)
            cands = pick_ve_cc_candidates(raw, interface)
            if not cands:
                skipped += 1
                continue
            orig = open(sv, "rb").read()
            # if the on-disk already matches the best candidate, nothing to recover
            if any(c.strip() == orig.decode(errors="ignore").strip() for c in cands):
                continue
            logp = os.path.join(base, prob, f"{prob}_sample{sidx:02d}-sv-iv-test.log")
            samples.append((sv, logp, orig, cands, prob, sidx))
    print(f"[ve-cc-v3 {model}] {len(samples)} S-fail samples with >=1 candidate "
          f"({skipped} skipped: no log/no candidates/already-matches)")
    if not samples:
        return 0, 0, round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    max_rounds = max(len(s[3]) for s in samples)
    passed = set()  # index into samples
    flip_rows = []
    for r in range(max_rounds):
        active = [s for s in samples if r < len(s[3]) and samples.index(s) not in passed]
        if not active:
            break
        # write candidate[r] for each active sample
        for s in active:
            sv, logp, orig, cands, prob, sidx = s
            open(sv, "w").write(cands[r] + "\n")
            binf = sv[:-3]
            for st in (binf, logp):
                if os.path.exists(st):
                    try: os.remove(st)
                    except OSError: pass
        try:
            subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                            capture_output=True, text=True, timeout=2400)
        except subprocess.TimeoutExpired:
            print(f"  [warn] round {r} make timeout (partial)")
        try:
            subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                            capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            pass
        new_sm = _ve_read_summary_csv(base)
        for k, s in enumerate(samples):
            if k in passed or r >= len(s[3]):
                continue
            sv, logp, orig, cands, prob, sidx = s
            new_marks = new_sm.get(prob, (0, 0.0, []))[2]
            new_mk = new_marks[sidx - 1] if sidx - 1 < len(new_marks) else "?"
            if new_mk == ".":
                passed.add(k)
                flip_rows.append({"sample": f"{prob}_sample{sidx:02d}",
                                  "round": r, "new_marker": ".",
                                  "flipped": True, "new_len": len(cands[r])})
    # record non-flips (final marker, round = last tried)
    for k, s in enumerate(samples):
        if k in passed:
            continue
        sv, logp, orig, cands, prob, sidx = s
        new_sm = _ve_read_summary_csv(base)
        new_marks = new_sm.get(prob, (0, 0.0, []))[2]
        new_mk = new_marks[sidx - 1] if sidx - 1 < len(new_marks) else "?"
        flip_rows.append({"sample": f"{prob}_sample{sidx:02d}",
                          "round": -1, "new_marker": new_mk,
                          "flipped": False, "new_len": len(cands[-1])})
    flips = len(passed)
    # recompute pass_rate: apply flips to the ORIGINAL summary.csv
    flips_by_prob = {}
    for s in samples:
        if samples.index(s) in passed:
            flips_by_prob.setdefault(s[4], set()).add(s[5])
    new_pr, n_prob = _ve_recompute_pass_rate(base, flips_by_prob)
    old_pr = round(100.0 * sum(v[1] for v in sm.values()) / max(1, len(sm)), 2)
    # restore originals
    for sv, logp, orig, cands, prob, sidx in samples:
        with open(sv, "wb") as f:
            f.write(orig)
        binf = sv[:-3]
        for st in (binf, logp):
            if os.path.exists(st):
                try: os.remove(st)
                except OSError: pass
    try:
        subprocess.run(["make", f"-j{jobs}", "sv-iv-test"], cwd=base,
                        capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        pass
    try:
        subprocess.run(["make", "sv-iv-analyze"], cwd=base,
                        capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        pass
    out_path = os.path.join(base, "ve_cc_v3.rescored.jsonl")
    meta = {"model": model, "mode": "cc-v3", "targets": len(samples),
            "flips": flips, "old_pass_rate": old_pr, "new_pass_rate": new_pr,
            "n_problems": n_prob, "max_rounds": max_rounds}
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for r in flip_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ve-cc-v3 {model}] targets={len(samples)} flips(S->.)={flips}  "
          f"pass_rate {old_pr}->{new_pr}  -> {out_path}")
    return flips, len(samples), new_pr


def cmd_reverify(args):
    print(f"\n=== Phase-2 re-verify: {args.model} / {args.bench} ===")
    if args.bench == "archx":
        reverify_archx(args.model)
    elif args.bench == "ve-cc":
        reverify_ve_cc(args.model, BENCH_TASK["ve-cc"])
    elif args.bench == "ve-cc-v2":
        reverify_ve_cc_v2(args.model, BENCH_TASK["ve-cc"], getattr(args, "jobs", 8))
    elif args.bench == "ve-cc-v3":
        reverify_ve_cc_v3(args.model, BENCH_TASK["ve-cc"], getattr(args, "jobs", 8))
    elif args.bench == "ve-spec-v2":
        reverify_ve_spec_v2(args.model, BENCH_TASK["ve-spec"], getattr(args, "jobs", 8))
    elif args.bench == "ve-spec-v3":
        reverify_ve_spec_v3(args.model, BENCH_TASK["ve-spec"], getattr(args, "jobs", 8))
    elif args.bench == "ve-spec":
        print("(VE-spec v1: use ve-spec-v2 for the S-fail re-extraction.)")
    else:
        print(f"(reverify for {args.bench} not implemented.)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--model", required=True)
    d.add_argument("--bench", required=True, choices=["ve-spec", "ve-cc", "archx"])
    d.set_defaults(func=cmd_diagnose)
    rv = sub.add_parser("reverify")
    rv.add_argument("--model", required=True)
    rv.add_argument("--bench", required=True, choices=["ve-spec", "ve-spec-v2", "ve-spec-v3", "ve-cc", "ve-cc-v2", "ve-cc-v3", "archx"])
    rv.add_argument("--jobs", type=int, default=8)
    rv.set_defaults(func=cmd_reverify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
