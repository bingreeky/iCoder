#!/usr/bin/env python3
"""verify/profiles/build_profiles.py — precompute RTLLM (and RealBench) verdict
profiles.

A verdict profile pins the EXACT golden stdout a correct solution must produce
under a pinned toolchain version, so judgement becomes a SHA256 match instead
of the fragile `re.search(r"\\b(pass|passed)\\b")` regex (which a model can
game by $display-ing "pass", and which flips on testbenches whose failure
message contains "passed"). This is the canonical RTL-benchmark oracle
(verifiers/icarus_benchmark_verifier.py:516-518).

RTLLM: each design dir has design_description.txt + testbench.v + a golden
reference .v (the reference solution). We compile golden_ref + testbench.v with
iverilog -g2012 (pinned v12 via IVERILOG12_BIN on PATH), run vvp, capture
stdout SHA256. Repeated runs (a couple samples) hedge against non-deterministic
$random testbenches by collecting a SET of acceptable hashes per design.

Output: a JSON map  {design_name: [stdout_sha256, ...]}  written to
RTLLM_VERDICT_PROFILE (default <RTLLM_ROOT>/rtllm_verdict_profile.json). The
RTLLM runner loads this and judges candidates via verify.icarus.judge_against_profile.

Usage:
    IVERILOG12_BIN=.../iverilog12/bin python verify/profiles/build_profiles.py \\
        --rtllm-root $RTLLM_ROOT --backend rtllm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verify.icarus import execute_icarus  # noqa: E402
from verify.realbench import (  # noqa: E402
    RealBenchProfile,
    execute_realbench,
    normalize_transcript,
    dependency_status as rb_dependency_status,
)


def _find_designs(rtllm_root: Path) -> list[Path]:
    """RTLLM design dirs: contain design_description.txt + testbench.v."""
    out = []
    for root, _dirs, files in os.walk(rtllm_root):
        if "design_description.txt" in files and "testbench.v" in files:
            out.append(Path(root))
    return sorted(out)


def _golden_ref(design_dir: Path) -> Path | None:
    """The golden reference .v in a design dir: any .v that is NOT testbench.v
    and NOT a generated sample (*_sN.v). If several, prefer one whose module
    name matches design_description's first module line (best-effort)."""
    cands = [p for p in design_dir.glob("*.v")
             if p.name != "testbench.v" and not p.stem.endswith(tuple(f"_s{i}" for i in range(10)))]
    if not cands:
        return None
    # smallest-named non-sample .v is usually the reference; fall back to first
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


def build_rtllm(rtllm_root: Path, output: Path, timeout: float,
                repeats: int) -> dict[str, list[str]]:
    designs = _find_designs(rtllm_root)
    print(f"[build-profiles] rtllm: {len(designs)} designs under {rtllm_root}",
          flush=True)
    profile: dict[str, list[str]] = {}
    # load existing (resume)
    if output.exists():
        try:
            profile = {k: list(v) for k, v in json.loads(output.read_text()).items()}
        except Exception:
            profile = {}

    for i, d in enumerate(designs, 1):
        name = d.name
        if name in profile and len(profile[name]) >= 1 and repeats <= 1:
            continue  # already profiled
        ref = _golden_ref(d)
        if ref is None:
            print(f"  [{i}/{len(designs)}] {name}: NO golden ref (.v) — skip",
                  flush=True)
            continue
        golden_src = ref.read_text(encoding="utf-8", errors="replace")
        tb = d / "testbench.v"
        hashes: set[str] = set()
        last_stage = ""
        for _ in range(max(1, repeats)):
            res = execute_icarus(golden_src, testbench_path=tb,
                                 timeout=timeout, eval_backend="rtllm")
            h = res.get("stdout_sha256")
            if h:
                hashes.add(h)
                last_stage = (res.get("verify_meta") or {}).get("stage", "")
            else:
                # golden failed to produce a stdout → trusted_reference_failure
                # infra (the harness itself is broken; no model can pass).
                last_stage = (res.get("verify_meta") or {}).get("stage", res.get("info", ""))
                print(f"  [{i}/{len(designs)}] {name}: golden FAIL "
                      f"({res.get('failure_origin')}/{last_stage})", flush=True)
                break
        if hashes:
            profile[name] = sorted(hashes)
            print(f"  [{i}/{len(designs)}] {name}: {len(hashes)} hash(es) "
                  f"[{','.join(h[:8] for h in sorted(hashes))}]", flush=True)
        elif name not in profile:
            profile[name] = []  # mark broken-harness so the runner treats as infra
        # incremental flush
        output.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"[build-profiles] wrote {len(profile)} entries -> {output}", flush=True)
    return profile


def build_realbench(args, stderr) -> dict[str, list[str]]:
    """Manifest-driven RealBench golden-hash precompute. Each manifest entry:
      {task, harness_dir, rtl_rel_path, top_module, binary_name,
       trusted_rtl_rel: [...], golden_rel_path}
    The golden reference is composed as the candidate and run through
    execute_realbench; the normalized stdout SHA256 is recorded. The profile
    PINS the runtime verilator_version (captured here, checked at judge time).
    Repeats collects a hash set to hedge non-deterministic $finish perf lines
    (normalize_transcript already strips them, but repeats are belt+braces).
    Run on the eval box where RB_ROOT + verilator live.
    """
    import json as _json
    if not args.realbench_manifest:
        print("ERROR: --backend realbench needs --realbench-manifest (JSON list of "
              "per-task wiring; see docstring)", file=stderr)
        raise SystemExit(2)
    entries = _json.loads(Path(args.realbench_manifest).read_text())
    runtime = rb_dependency_status()
    if not runtime["ready"]:
        print(f"ERROR: realbench runtime unavailable: {runtime}", file=stderr)
        raise SystemExit(1)
    out = Path(args.output) if args.output else Path("realbench_verdict_profile.json")
    profile: dict[str, list[str]] = {}
    if out.exists():
        try:
            profile = {k: list(v) for k, v in _json.loads(out.read_text()).items()}
        except Exception:
            profile = {}
    for i, e in enumerate(entries, 1):
        task = e["task"]
        golden = (Path(e["harness_dir"]) / e["golden_rel_path"]).read_text(
            encoding="utf-8", errors="replace")
        hashes: set[str] = set()
        for _ in range(max(1, args.repeats)):
            res = execute_realbench(
                golden, harness_dir=e["harness_dir"], rtl_rel_path=e["rtl_rel_path"],
                top_module=e["top_module"], binary_name=e["binary_name"],
                trusted_rtl_rel=e.get("trusted_rtl_rel", []),
                timeout=args.timeout, eval_backend="realbench")
            h = res.get("stdout_sha256")
            if h:
                hashes.add(h)
            else:
                print(f"  [{i}/{len(entries)}] {task}: golden FAIL "
                      f"({res.get('failure_origin')}/{(res.get('verify_meta') or {}).get('stage', res.get('info'))})",
                      flush=True)
                break
        if hashes:
            profile[task] = sorted(hashes)
        elif task not in profile:
            profile[task] = []
        out.write_text(_json.dumps(
            {"schema_version": 1, "kind": "realbench",
             "verilator_version": runtime["verilator_version"],
             "transcript_normalization": "realbench_runtime_metric_strip_v1",
             "dataset_revision": "", "tasks": profile},
            indent=2, ensure_ascii=False))
        print(f"  [{i}/{len(entries)}] {task}: {len(hashes)} hash(es)", flush=True)
    print(f"[build-profiles] realbench: {len(profile)} tasks -> {out} "
          f"(verilator={runtime['verilator_version']})", flush=True)
    return profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtllm-root", default=None, help="RTLLM dataset root")
    ap.add_argument("--backend", default="rtllm", choices=["rtllm", "realbench"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per design to collect a hash set (non-deterministic tbs)")
    ap.add_argument("--realbench-manifest", default=None,
                    help="JSON manifest list for realbench backend (see build_realbench)")
    args = ap.parse_args()

    if args.backend == "rtllm":
        rtllm_root = Path(args.rtllm_root or os.environ.get("RTLLM_ROOT", ""))
        if not rtllm_root.is_dir():
            print(f"ERROR: rtllm-root not a dir: {rtllm_root} (set --rtllm-root or RTLLM_ROOT)",
                  file=sys.stderr)
            sys.exit(1)
        out = Path(args.output) if args.output else rtllm_root / "rtllm_verdict_profile.json"
        build_rtllm(rtllm_root, out, args.timeout, args.repeats)
    else:
        build_realbench(args, sys.stderr)


if __name__ == "__main__":
    main()
