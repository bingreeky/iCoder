"""verify/realbench.py — hardened RealBench verilator+stdout-hash judgment (local).

Ports the on-disk equivalent of realbench_verifier.execute_realbench
(verifiers/realbench_verifier.py:472-626) for the eval_harness local edition:
the harness files live on disk (RealBench task dirs: RTL + testbench + Makefile),
the candidate Verilog is composed in, verilator --binary --assert --timing
--trace compiles, the binary runs, and the verdict is the SHA256 of the
NORMALIZED stdout (wallclock/CPU perf-report lines stripped — they vary per
run) matched against a precomputed verdict profile that PINS the verilator
version. Upstream :463-469 (normalize_transcript), :548-564 (compile args),
:593 (stdout_digest).

Why this instead of RealBench's run_verify.py: run_verify's estimator counts
"passed" via transcript heuristics that game the same way the RTLLM regex did,
and it doesn't pin the verilator build — a verilator upgrade silently rescores
every prior run. The hardened path pins toolchain + exact stdout, the canonical
RealBench oracle (upstream :648: stdout_digest in expected_stdout_sha256).

Anti-cheat: reuse verify.icarus.unsafe_runtime_construct ($display/$dumpvars/
DPI-C ban) on the candidate. Version pin: load_realbench_profile refuses to
judge when the runtime verilator version != profile version (infra, not model).

Usage: see verify/profiles/build_profiles.py (--backend realbench) for the
one-time golden-hash precompute; run_realbench.sh opt-in via
REALBENCH_VERDICT_PROFILE.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import (
    base_result,
    finalize_failure_classification,
    infra_failure,
    model_failure,
    run_procgroup,
)
from .icarus import extract_verilog, unsafe_runtime_construct

# Verilator's wall-clock/CPU perf report lines are non-deterministic across
# runs (depend on host load) — strip them before hashing, or every "correct"
# solution would produce a different hash. Upstream :63-65.
_RUNTIME_METRIC_LINE = re.compile(
    br"^- Verilator: (?:\$finish\b.*(?:walltime|speed)\b.*|cpu\b.*)$"
)

TRANSCRIPT_NORMALIZATION = "realbench_runtime_metric_strip_v1"


def normalize_transcript(stdout: bytes) -> bytes:
    """Drop only Verilator's wall-clock/CPU performance report lines."""
    return b"".join(
        line
        for line in stdout.splitlines(keepends=True)
        if not _RUNTIME_METRIC_LINE.fullmatch(line.rstrip(b"\r\n"))
    )


def _verilator_path() -> str:
    """Honor SETUP_ENV (config.sh sources it for verilator/yosys + CUDA libs)
    via PATH; fall back to `which verilator`."""
    return shutil.which("verilator") or ""


def dependency_status() -> dict[str, Any]:
    verilator = _verilator_path()
    make = shutil.which("make") or ""
    cxx = shutil.which(os.environ.get("CXX", "g++")) or ""
    version = ""
    if verilator:
        try:
            c = subprocess.run([verilator, "--version"], capture_output=True,
                               text=True, timeout=10)
            lines = (c.stdout or c.stderr).strip().splitlines()
            version = lines[0] if lines else ""
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return {"verilator": verilator, "make": make, "cxx": cxx,
            "verilator_version": version,
            "ready": bool(verilator and make and cxx and version)}


@dataclass(frozen=True)
class RealBenchProfile:
    """{task_name: [stdout_sha256, ...]} + the pinned verilator version."""
    expected_stdout_sha256: dict[str, frozenset[str]]
    verilator_version: str
    dataset_revision: str = ""
    transcript_normalization: str = TRANSCRIPT_NORMALIZATION


def load_realbench_profile(path: str | os.PathLike[str], *,
                           require_runtime_match: bool = True) -> RealBenchProfile:
    """Load + validate a realbench verdict profile. Refuses to judge (raises)
    when the runtime verilator version != profile version — a toolchain drift
    is infrastructure, not a model failure. Upstream :274-318."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, dict)
            or value.get("schema_version") != 1
            or value.get("kind") != "realbench"):
        raise ValueError("invalid RealBench verdict profile")
    tasks = value.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("RealBench profile has no tasks")
    expected: dict[str, frozenset[str]] = {}
    for task, hashes in tasks.items():
        if not isinstance(hashes, list) or not hashes:
            raise ValueError(f"RealBench task {task} has no stdout fingerprints")
        norm = frozenset(str(h).lower() for h in hashes)
        if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in norm):
            raise ValueError(f"invalid stdout SHA256 in task {task}")
        expected[str(task)] = norm
    version = str(value.get("verilator_version", ""))
    if not version:
        raise ValueError("RealBench profile lacks verilator_version")
    norm = str(value.get("transcript_normalization", ""))
    if norm != TRANSCRIPT_NORMALIZATION:
        raise ValueError("unsupported RealBench transcript normalization")
    if require_runtime_match:
        runtime = dependency_status()
        if not runtime["ready"]:
            raise RuntimeError(f"RealBench runtime unavailable: {runtime}")
        if runtime["verilator_version"] != version:
            raise RuntimeError(
                "RealBench profile/runtime mismatch: "
                f"profile={version!r}, runtime={runtime['verilator_version']!r}")
    return RealBenchProfile(expected_stdout_sha256=expected,
                           verilator_version=version,
                           dataset_revision=str(value.get("dataset_revision", "")),
                           transcript_normalization=norm)


def execute_realbench(
    candidate_source: str,
    *,
    harness_dir: str | os.PathLike,
    rtl_rel_path: str,            # where the candidate is written (rel to harness_dir)
    top_module: str,
    binary_name: str,            # V<name> (the obj_dir/ prefix is added here)
    trusted_rtl_rel: list[str],  # other .v/.sv in the harness (not the candidate)
    timeout: float = 120.0,
    eval_backend: str = "realbench",
) -> dict[str, Any]:
    """Compile candidate+trusted RTL with verilator --binary --assert --timing
    --trace, run the binary, return a verify.core result with stdout_sha256
    (normalized). NOT yet judged correct — caller matches against the profile.
    Upstream :548-617."""
    runtime = dependency_status()
    if not runtime["ready"]:
        return finalize_failure_classification(
            infra_failure(infra_code="trusted_runtime_dependency",
                          eval_backend=eval_backend,
                          info="realbench_runtime_unavailable",
                          error_type="missing_dependency",
                          verify_meta={"stage": "runtime_unavailable"})
        )
    source = extract_verilog(candidate_source) if "```" in candidate_source else candidate_source
    if len(source) < 10:
        return finalize_failure_classification(
            model_failure(reason="too_short", eval_backend=eval_backend,
                          info="skipped_too_short"))
    unsafe = unsafe_runtime_construct(source)
    if unsafe:
        return finalize_failure_classification(
            model_failure(reason=unsafe, eval_backend=eval_backend,
                          info=unsafe, error_type=unsafe,
                          verify_meta={"stage": "unsafe_runtime_construct"}))

    tmpdir = Path(os.environ.get("REALBENCH_TMPDIR", "/tmp"))
    with tempfile.TemporaryDirectory(prefix="realbench_verify_", dir=str(tmpdir)) as wd:
        root = Path(wd)
        # copy the on-disk harness (trusted RTL + tb + include dirs)
        shutil.copytree(harness_dir, root, dirs_exist_ok=True)
        (root / rtl_rel_path).write_text(source, encoding="utf-8")
        trusted = [str(root / p) for p in trusted_rtl_rel]
        compile_args = [
            runtime["verilator"], "--binary", "--assert", "--timing", "--trace",
            "--coverage-line", "-fno-table", "-Wno-fatal", "-j", "1",
            "--top", top_module, "-I.", *trusted, rtl_rel_path,
        ]
        # run_procgroup reaps a verilator-built binary that forks (DPI / $system)
        # on timeout, instead of leaking into the next sample. stdout comes back
        # decoded; re-encode for the SHA256 (byte-exact for verilator text output;
        # build_profiles precomputes the golden through this same path).
        comp_rc, _comp_out, comp_err, comp_to = run_procgroup(
            compile_args, timeout=timeout, cwd=root)
        if comp_to:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=False,
                            info="compile_timeout", error_type="task_timeout",
                            verify_meta={"stage": "compile_timeout"}))
        if comp_rc != 0:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=False,
                            info="compile_fail",
                            error_type=comp_err[-800:],
                            verify_meta={"stage": "compile_fail"}))
        binary = root / "obj_dir" / binary_name
        if not binary.is_file():
            return finalize_failure_classification(
                infra_failure(infra_code="backend_configuration",
                              eval_backend=eval_backend,
                              info="missing_simulator_binary",
                              error_type="verilator_build_inconsistency",
                              verify_meta={"stage": "missing_binary"}))
        run_rc, run_out, run_err, run_to = run_procgroup(
            [str(binary)], timeout=timeout, cwd=root)
        if run_to:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=True,
                            info="timeout", error_type="task_timeout",
                            verify_meta={"stage": "sim_timeout"}))
        stdout = (run_out or "").encode("utf-8", "replace")
        stderr = run_err or ""
        stdout_digest = hashlib.sha256(normalize_transcript(stdout)).hexdigest()
        if run_rc != 0:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=True, correct=False,
                            info="test_fail",
                            error_type=stderr[-800:],
                            stdout_sha256=stdout_digest,
                            verify_meta={"stage": "test_fail"}))
        return finalize_failure_classification(
            base_result(eval_backend=eval_backend, compiled=True, correct=False,
                        info="executed", stdout_sha256=stdout_digest,
                        verify_meta={"stage": "executed"}))


def judge_against_profile(result: dict[str, Any], profile: RealBenchProfile,
                          task_name: str) -> dict[str, Any]:
    """Promote an 'executed' result to correct/incorrect via normalized
    stdout SHA256 match. No profile entry for the task → model failure
    (caller should precompute first). Upstream :644-653."""
    if result.get("info") in ("compile_fail", "compile_timeout", "test_fail",
                              "timeout", "too_short") or \
       result.get("failure_origin") == "infrastructure":
        return result
    expected = profile.expected_stdout_sha256.get(task_name)
    digest = result.get("stdout_sha256")
    if not digest or expected is None:
        result["correct"] = False
        result["info"] = "executed_no_profile"
        vm = result.get("verify_meta") or {}
        vm["stage"] = "no_profile"
        result["verify_meta"] = vm
        return result
    correct = digest in expected
    result["correct"] = correct
    result["info"] = "pass" if correct else "stdout_hash_mismatch"
    vm = result.get("verify_meta") or {}
    vm["stage"] = "ok" if correct else "stdout_hash_mismatch"
    result["verify_meta"] = vm
    return result


def _self_test() -> None:
    # normalize strips the walltime line, keeps the rest
    raw = b"some output\n- Verilator: $finish walltime=1.23s speed=2x cpu=0.5s\nmore\n"
    n = normalize_transcript(raw)
    assert b"walltime" not in n and b"more\n" in n and b"some output\n" in n, n
    # judge mismatch / pass
    r = {"info": "executed", "stdout_sha256": "abc", "failure_origin": "model",
         "verify_meta": {}}
    prof = RealBenchProfile(expected_stdout_sha256={"t": frozenset({"def"})},
                           verilator_version="x")
    r2 = judge_against_profile(dict(r), prof, "t")
    assert r2["correct"] is False and r2["info"] == "stdout_hash_mismatch"
    prof2 = RealBenchProfile(expected_stdout_sha256={"t": frozenset({"abc"})},
                            verilator_version="x")
    r3 = judge_against_profile(dict(r), prof2, "t")
    assert r3["correct"] is True and r3["info"] == "pass"
    # no profile entry
    r4 = judge_against_profile(dict(r), prof, "missing")
    assert r4["correct"] is False and r4["info"] == "executed_no_profile"
    print("verify.realbench smoke: 5/5 passed")


if __name__ == "__main__":
    _self_test()
