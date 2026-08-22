"""verify/icarus.py — hardened iverilog compile+run+stdout-hash judgment (local).

Ports the on-disk equivalent of icarus_benchmark_verifier.execute_icarus_benchmark
(verifiers/icarus_benchmark_verifier.py:419-475) for the eval_harness local
edition: instead of a sealed base64+zlib+pickle payload, the harness files live
on disk (one directory per RTLLM design: testbench.v + golden reference). The
candidate Verilog is composed in, iverilog -g2012 compiles, vvp runs, and the
verdict is the SHA256 of stdout matched against a precomputed verdict profile
(run the golden reference through the same harness once, pin iverilog v12).

Why stdout-hash instead of the regex `re.search(r"\\b(pass|passed)\\b", out)`:
the regex is the single biggest RTLLM correctness hole. A candidate that
$display's the word "pass" (or anything matching) passes without doing the
work; a testbench whose failure message contains "passed through" also flips.
The hardened path pins the EXACT golden stdout under a pinned iverilog version,
so any behavioural divergence — including self-printing "pass" — fails the
hash. This is the canonical RTLLM/RealBench oracle (upstream :464,
:516-518: stdout_digest = sha256(stdout); correctness = digest in profile).

Anti-cheat: unsafe_runtime_construct bans $display/$write/$dumpvars/$system/
force/release/DPI-C in the CANDIDATE source (port of _UNSAFE_RUNTIME, :43-51),
so the model cannot print "pass" itself, read stimulus from disk, or call out.
This runs BEFORE compile, so a cheating candidate fails fast with a named stage.

VerilogEval v2 is intentionally NOT routed through here — it stays on the
existing eval_harness sv-iv-analyze path (per the alignment scope: VEval stays
as-is). This module serves RTLLM; RealBench (verilator) has verify/realbench.py.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .core import (
    base_result,
    finalize_failure_classification,
    infra_failure,
    model_failure,
    run_procgroup,
)

# --- anti-cheat: ban IO / control / DPI in candidate RTL (upstream :43-51) ---
_UNSAFE_RUNTIME = re.compile(
    r"`\s*(?:include|pragma)|"
    r"\$(?:system|fopen|fclose|fwrite|fdisplay|readmemh|readmemb|"
    r"writememh|writememb|dumpfile|dumpvars|vpi|stop|finish|fatal|"
    r"error|warning|info|display|write|strobe|monitor|root)\b|"
    r"\b(?:force|release|bind)\b|"
    r"\b(?:import|export)\s+\"DPI-C\"",
    re.IGNORECASE,
)
_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_FENCE = re.compile(r"```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL)


def extract_verilog(text: str) -> str:
    """Pull the largest ```fenced``` Verilog block, else the raw text."""
    blocks = _FENCE.findall(text)
    return max(blocks, key=len) if blocks else text


def unsafe_runtime_construct(source: str) -> str | None:
    """Return the first unsafe-construct match name in candidate RTL, or None.
    Strips comments first so a `$display` in a comment isn't a false positive.
    Upstream :435 / icarus_benchmark_verifier.unsafe_runtime_construct.
    """
    cleaned = _COMMENT.sub("", source)
    m = _UNSAFE_RUNTIME.search(cleaned)
    if not m:
        return None
    # name the matched construct for the error_type field
    tok = m.group(0).strip().lstrip("`").strip()
    return f"unsafe_runtime_construct:{tok[:40]}"


def dependency_status() -> dict[str, Any]:
    """Locate iverilog/vvp. Honors IVERILOG12_BIN (config.sh pins v12 to the
    front of PATH — v14-devel has a $dumpvars forward-ref bug → pass_rate=0)."""
    iverilog = shutil.which("iverilog") or ""
    vvp = shutil.which("vvp") or ""
    return {"ready": bool(iverilog and vvp), "iverilog": iverilog, "vvp": vvp}


def execute_icarus(
    candidate_source: str,
    *,
    testbench_path: str | os.PathLike,
    golden_ref_path: str | os.PathLike | None = None,
    design_module: str | None = None,
    extra_compile_files: list[str] | None = None,
    compile_flags: list[str] | None = None,
    timeout: float = 120.0,
    eval_backend: str = "rtllm",
) -> dict[str, Any]:
    """Compile candidate+testbench with iverilog -g2012, run vvp, return a
    verify.core result with stdout_sha256 attached (NOT yet judged correct —
    the caller matches stdout_sha256 against the verdict profile).

    golden_ref_path: when judging a CANDIDATE this is None (the candidate IS
      the DUT). When PRECOMPUTING the profile it is the golden reference .v and
      candidate_source is the golden too (or the golden ref is composed as the
      candidate). Kept symmetric so build_profiles reuses this exact path.
    design_module: optional module-name guard (the candidate must define
      exactly one module with this name — port of design_module_count :440).
    """
    runtime = dependency_status()
    if not runtime["ready"]:
        return finalize_failure_classification(
            infra_failure(infra_code="trusted_runtime_dependency",
                          eval_backend=eval_backend,
                          info="rtl_benchmark_runtime_unavailable",
                          error_type="missing_dependency",
                          verify_meta={"stage": "runtime_unavailable"})
        )

    source = extract_verilog(candidate_source) if "```" in candidate_source else candidate_source
    if len(source) < 3:
        return finalize_failure_classification(
            model_failure(reason="too_short", eval_backend=eval_backend,
                          info="skipped_too_short")
        )
    unsafe = unsafe_runtime_construct(source)
    if unsafe:
        return finalize_failure_classification(
            model_failure(reason=unsafe, eval_backend=eval_backend,
                          info=unsafe, error_type=unsafe,
                          verify_meta={"stage": "unsafe_runtime_construct"})
        )

    tb = Path(testbench_path)
    with tempfile.TemporaryDirectory(prefix=f"{eval_backend}_verify_") as wd:
        root = Path(wd)
        cand_path = root / "candidate.v"
        cand_path.write_text(source, encoding="utf-8")
        files = [str(cand_path), str(tb)]
        if extra_compile_files:
            files = [str(cand_path)] + extra_compile_files + [str(tb)]
        # Compile + run go through run_procgroup (start_new_session +
        # os.killpg) so a vvp that forks (DPI-C / $system / PLI) is reaped
        # on timeout instead of leaking file locks into the next sample.
        # stdout/stderr come back decoded; re-encode for the SHA256. This is
        # byte-exact for iverilog/vvp text output, and self-consistent because
        # build_profiles precomputes the golden hash through this same path.
        comp_rc, _comp_out, comp_err, comp_to = run_procgroup(
            ["iverilog", "-g2012", "-Wno-timescale",
             *(compile_flags or []), "-o", "simv", *files],
            timeout=timeout, cwd=root,
        )
        if comp_to:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=False,
                            info="compile_timeout", error_type="task_timeout",
                            verify_meta={"stage": "compile_timeout"})
            )
        if comp_rc != 0:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=False,
                            info="compile_fail",
                            error_type=comp_err[-800:],
                            verify_meta={"stage": "compile_fail"})
            )
        run_rc, run_out, run_err, run_to = run_procgroup(
            ["vvp", "simv"], timeout=timeout, cwd=root,
        )
        if run_to:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=True,
                            info="timeout", error_type="task_timeout",
                            verify_meta={"stage": "sim_timeout"})
            )
        stdout = (run_out or "").encode("utf-8", "replace")
        stderr = run_err or ""
        stdout_digest = hashlib.sha256(stdout).hexdigest()
        if run_rc != 0:
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=True, correct=False,
                            info="test_fail",
                            error_type=stderr[-800:],
                            stdout_sha256=stdout_digest,
                            verify_meta={"stage": "test_fail"})
            )
        return finalize_failure_classification(
            base_result(eval_backend=eval_backend, compiled=True, correct=False,
                        info="executed",
                        stdout_sha256=stdout_digest,
                        stdout_tail=stdout.decode("utf-8", "replace")[-400:],
                        verify_meta={"stage": "executed"})
        )


def judge_against_profile(
    candidate_result: dict[str, Any],
    expected_hashes: frozenset[str] | set[str] | None,
) -> dict[str, Any]:
    """Promote an execute_icarus 'executed' result to correct/incorrect by
    matching its stdout_sha256 against the verdict-profile allowlist. A None
    allowlist means "no profile for this task" → stays a model failure with
    info=executed_no_profile (the caller should precompute first). Upstream
    :516-518 (correctness = digest in expected_stdout_sha256).
    """
    if candidate_result.get("info") in ("compile_fail", "compile_timeout",
                                       "test_fail", "timeout",
                                       "too_short") or \
       candidate_result.get("failure_origin") == "infrastructure":
        return candidate_result  # already terminal
    digest = candidate_result.get("stdout_sha256")
    if not digest or expected_hashes is None:
        candidate_result["correct"] = False
        candidate_result["info"] = "executed_no_profile" if not digest else "executed_no_profile"
        candidate_result["verify_meta"] = {**(candidate_result.get("verify_meta") or {}),
                                           "stage": "no_profile"}
        return candidate_result
    correct = digest in set(expected_hashes)
    candidate_result["correct"] = correct
    candidate_result["info"] = "pass" if correct else "stdout_hash_mismatch"
    vm = candidate_result.get("verify_meta") or {}
    vm["stage"] = "ok" if correct else "stdout_hash_mismatch"
    candidate_result["verify_meta"] = vm
    return candidate_result


# --- smoke (no iverilog/dataset needed for the pure-string gates) ---
def _self_test() -> None:
    assert unsafe_runtime_construct("module m; initial $display(\"hi\"); endmodule")
    assert unsafe_runtime_construct("module m; initial $dumpvars(0, top); endmodule")
    assert unsafe_runtime_construct("module m; import \"DPI-C\"; endmodule")
    assert unsafe_runtime_construct("module m; // $display in comment\nendmodule") is None
    assert unsafe_runtime_construct("module m; wire a; endmodule") is None
    assert extract_verilog("noise\n```verilog\nmodule m;\nendmodule\n```\n") == "module m;\nendmodule\n"
    # judge: mismatch
    r = {"info": "executed", "stdout_sha256": "abc", "failure_origin": "model",
         "verify_meta": {}}
    r2 = judge_against_profile(dict(r), frozenset({"def"}))
    assert r2["correct"] is False and r2["info"] == "stdout_hash_mismatch"
    r3 = judge_against_profile(dict(r), frozenset({"abc"}))
    assert r3["correct"] is True and r3["info"] == "pass"
    print("verify.icarus smoke: 7/7 passed")


if __name__ == "__main__":
    _self_test()
