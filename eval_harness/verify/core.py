"""verify/core.py — shared failure classification + sandbox, local edition.

Ports the classification policy and subprocess sandbox from the upstream
verifier suite (verifiers_unzip/verifiers/verify_server_v2.py +
rtl_verifier.py) with the server plumbing stripped.

The single source of truth for "is this row an infrastructure failure that
must be excluded from the denominator?". Both the per-bench merge scripts
(e.g. tbg_merge_shards.py) and summarize.sh import is_infra()/is_skipped()
from here — collapsing the three divergent _skipped definitions that caused
the TBG 137/139 denominator drift.

References (upstream, verifiers_unzip/verifiers/):
  verify_server_v2.py:68-94   VERIFIER_VERSION / GPU/CPU_BACKENDS / infra codes
  verify_server_v2.py:634-696 _base_result / _infra_failure / _model_failure
  verify_server_v2.py:699-734 _trusted_backend_infra_code
  verify_server_v2.py:737-790 _finalize_failure_classification
  rtl_verifier.py:255-291     _limit_child / _run_command (sandbox)
"""

from __future__ import annotations

import hashlib
import math
import os
import resource
import signal
import subprocess
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Classification policy (upstream explicit_infra_blacklist_v1, verbatim)
# ---------------------------------------------------------------------------

VERIFIER_VERSION = "2.7.13-local"
CLASSIFICATION_POLICY = "explicit_infra_blacklist_v1"

#: Channels allowed to assert infra_error. Anything carrying
#: candidate-controlled text (stdout/stderr/traceback/exit codes) is NOT a
#: trusted channel — infra claims from such sources are demoted to model fail.
_TRUSTED_INFRA_CHANNELS = frozenset(
    {
        "trusted_control_plane",
        "trusted_cpu_wrapper",
    }
)

#: The closed set of high-confidence infrastructure codes. A result is only
#: an infrastructure failure if it carries one of these codes on a trusted
#: channel. Anything else is a model failure (zero reward, counts in the
#: denominator). Mirrors verify_server_v2.py:79-94.
INFRA_CODES = frozenset(
    {
        "backend_configuration",
        "scheduler_failure",
        "trusted_payload_integrity",
        "trusted_golden_failure",
        "trusted_reference_configuration",
        "trusted_reference_failure",
        "trusted_runtime_dependency",
        "trusted_source_integrity",
        "worker_bootstrap_timeout",
        "worker_capture_setup",
        "worker_construction",
        "worker_start",
    }
)

CPU_BACKENDS = (
    "rtl",
    "cvdp",
    "archxbench",
    "archxbench_tb",
    "realbench",
    "verilogeval",
    "rtllm",
)
GPU_BACKENDS = ("kernelbench", "tritonbench")
SUPPORTED_BACKENDS = (*GPU_BACKENDS, *CPU_BACKENDS)


# ---------------------------------------------------------------------------
# Result constructors
# ---------------------------------------------------------------------------

def base_result(eval_backend: str | None = None, **updates: Any) -> dict[str, Any]:
    """A neutral verify result. The neutral/zero state is a model failure:
    a result is presumed model-failure unless a trusted path explicitly marks
    it correct or infra. Mirrors verify_server_v2.py:634-655 (gpu field
    dropped — local drivers pin CUDA_VISIBLE_DEVICES themselves).
    """
    result: dict[str, Any] = {
        "correct": False,
        "info": "fail",
        "compiled": False,
        "triton_launched": 0,
        "identity_hack": False,
        "framework_delegation": False,
        "speedup": None,
        "infra_error": False,
        "error_type": None,
        "failure_origin": None,
        "infra_code": None,
        "classification_policy": CLASSIFICATION_POLICY,
        "classification_reason": None,
        "classification_channel": None,
        "original_info": None,
        "original_error_type": None,
        "verifier_version": VERIFIER_VERSION,
        "eval_backend": eval_backend,
    }
    result.update(updates)
    return result


def infra_failure(
    *,
    infra_code: str,
    eval_backend: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Create an infrastructure failure from the explicit trusted blacklist.

    Raises ValueError if the code is not in INFRA_CODES — this is intentional:
    infra codes are a closed enum, not free text. Use model_failure() for the
    default zero-reward class. Mirrors verify_server_v2.py:658-676.
    """
    if infra_code not in INFRA_CODES:
        raise ValueError(f"unknown high-confidence infra code: {infra_code}")
    updates.pop("infra_error", None)
    updates.setdefault("classification_channel", "trusted_control_plane")
    return base_result(
        eval_backend,
        infra_error=True,
        failure_origin="infrastructure",
        infra_code=infra_code,
        classification_reason=infra_code,
        **updates,
    )


def model_failure(
    *,
    reason: str,
    eval_backend: str | None = None,
    **updates: Any,
) -> dict[str, Any]:
    """Create a zero-reward model failure; the default failure class.

    Anything that is not correct and not a trusted infra failure lands here.
    Mirrors verify_server_v2.py:679-696.
    """
    updates.pop("infra_error", None)
    updates.setdefault("classification_channel", "request_failure")
    updates.setdefault("info", f"model_execution_error:{reason}")
    updates.setdefault("error_type", "model_execution_error")
    return base_result(
        eval_backend,
        infra_error=False,
        failure_origin="model",
        classification_reason=reason,
        **updates,
    )


# ---------------------------------------------------------------------------
# CPU-verifier outcome -> infra code mapping
# ---------------------------------------------------------------------------

def trusted_backend_infra_code(result: dict[str, Any]) -> str | None:
    """Map fixed CPU-verifier outcomes onto the explicit infra blacklist.

    A CPU verifier may report a failure that is genuinely the harness's fault
    (payload unrunnable, runtime dep missing, golden trace incomplete). This
    maps those known outcome strings onto trusted infra codes so they are
    excluded from the denominator. Mirrors verify_server_v2.py:699-734.
    """
    backend = str(result.get("eval_backend") or "")
    info = str(result.get("info") or "")
    error_type = str(result.get("error_type") or "")
    if backend not in CPU_BACKENDS:
        return None
    if info in {"payload_not_allowlisted", "invalid_verifier_payload"}:
        return "trusted_payload_integrity"
    if info in {"invalid_verify_mode", "profile_backend_mismatch"} and (
        error_type == "dataset_configuration_error"
    ):
        return "trusted_reference_configuration"
    if info in {
        "dependency_missing",
        "cvdp_runtime_unavailable",
        "archxbench_runtime_unavailable",
        "selfcheck_tb_runtime_unavailable",
        "realbench_runtime_unavailable",
        "rtl_benchmark_runtime_unavailable",
        "missing_simulator_binary",
    }:
        return "trusted_runtime_dependency"
    if backend == "rtl":
        if info in {"invalid_golden_or_ports", "testbench_generation_failed"}:
            return "trusted_reference_configuration"
        if info == "golden_trace_incomplete" or (
            info.startswith("golden_") and error_type == "golden_verifier_failure"
        ):
            return "trusted_golden_failure"
    return None


# ---------------------------------------------------------------------------
# Classification enforcement (the API-boundary gate)
# ---------------------------------------------------------------------------

def finalize_failure_classification(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Enforce the explicit-infra blacklist at the result boundary.

    Candidate-controlled text, traceback strings, exit codes, and backend
    diagnostics cannot authorize infra_error. Any result that sets
    infra_error without a trusted infra_code on a trusted channel is converted
    to a model failure. This is the single gate that prevents a model fail from
    masquerading as infra (which would deflate the denominator).

    Mirrors verify_server_v2.py:737-790.
    """
    normalized = dict(result)
    normalized["classification_policy"] = CLASSIFICATION_POLICY
    if normalized.get("infra_error"):
        channel = normalized.get("classification_channel")
        infra_code = (
            trusted_backend_infra_code(normalized)
            if channel == "trusted_cpu_wrapper"
            else normalized.get("infra_code")
        )
        if channel in _TRUSTED_INFRA_CHANNELS and infra_code in INFRA_CODES:
            normalized["correct"] = False
            normalized["compiled"] = False
            normalized["infra_code"] = infra_code
            normalized["failure_origin"] = "infrastructure"
            normalized["classification_reason"] = infra_code
            return normalized
        original_info = str(normalized.get("info") or "failure")
        original_error_type = str(normalized.get("error_type") or "unknown")
        normalized.update(
            correct=False,
            infra_error=False,
            failure_origin="model",
            infra_code=None,
            classification_reason="default_non_infra",
            original_info=original_info,
            original_error_type=original_error_type,
            info=f"model_execution_error:{original_error_type}",
            error_type="model_execution_error",
        )
        return normalized
    if normalized.get("correct"):
        normalized["infra_error"] = False
        normalized["infra_code"] = None
        normalized["failure_origin"] = None
        normalized["classification_reason"] = "success"
    else:
        normalized["infra_error"] = False
        normalized["infra_code"] = None
        normalized["failure_origin"] = "model"
        normalized["classification_reason"] = (
            normalized.get("classification_reason") or "model_output"
        )
    return normalized


# ---------------------------------------------------------------------------
# Denominator predicates — the single source of truth for skip/infra
# ---------------------------------------------------------------------------

def is_infra(result: dict[str, Any]) -> bool:
    """True iff this row is an infrastructure failure (excluded from the
    denominator). A row is infra only if finalize_failure_classification
    would accept it: infra_error set, trusted channel, trusted code.

    This replaces the three divergent _skipped() definitions in
    summarize.sh:105-108 / summarize.sh:246-255 / tbg_merge_shards.py:50-53.
    """
    if not result:
        return False
    # Accept either the post-finalize shape (failure_origin == "infrastructure")
    # or the legacy verify_skipped flag carried by old TBG rows.
    if result.get("failure_origin") == "infrastructure":
        return True
    if result.get("infra_error") and result.get("infra_code") in INFRA_CODES:
        return True
    # Legacy TBG rows written by the old verify_one carry verify_skipped=True
    # + verify_meta.stage == "ref_smoke_failed". That stage IS the canonical
    # "reference unrunnable" infra case; honour it so old runs still merge.
    meta = result.get("verify_meta") or {}
    stage = str(meta.get("stage") or "")
    if result.get("verify_skipped") or stage == "ref_smoke_failed":
        return True
    return False


def is_skipped(result: dict[str, Any]) -> bool:
    """Alias of is_infra for the merge/summarize call sites that used to call
    their own _skipped(). One definition, imported everywhere.
    """
    return is_infra(result)


# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sandboxed subprocess runner
# ---------------------------------------------------------------------------

def _limit_child(max_file_bytes: int, timeout: float) -> None:
    """preexec_fn applied in the child before exec. Bounds output file size,
    CPU seconds, and address space so a candidate cannot exhaust the host.
    Ported from rtl_verifier.py:255-261.
    """
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
    cpu = max(2, int(math.ceil(timeout)) + 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    address_space = int(
        os.environ.get("VERIFY_MAX_ADDRESS_SPACE", str(4 * 1024**3))
    )
    resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    # Cap open files + nproc too (upstream sets these per-verifier; we default
    # generously so heavy autotune/cocotb runs still breathe).
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    except (ValueError, OSError):
        pass


def run_sandboxed(
    args: list[str],
    cwd: str | Path,
    timeout: float,
    *,
    max_file_bytes: int = 64 * 1024 * 1024,
    env: dict[str, str] | None = None,
    stdin: bytes | None = None,
) -> tuple[int | None, str, str, bool]:
    """Run args in a new session with rlimits, kill the whole group on timeout.

    Returns (returncode, stdout, stderr, timed_out). Mirrors rtl_verifier.py
    _run_command but writes to pipes (not files) so callers don't have to
    manage a stdout.txt/stderr.txt pair per invocation.
    """
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: _limit_child(max_file_bytes, timeout),
        )
    except FileNotFoundError as exc:
        return -1, "", f"launch failed: {exc}", False
    timed_out = False
    try:
        out, err = process.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
    return (
        process.returncode,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        timed_out,
    )


def run_procgroup(
    args: list[str],
    *,
    timeout: float,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int | None, str, str, bool]:
    """Run args in a NEW session and SIGKILL the whole process group on
    timeout.

    Used by the GPU/CPU verify workers that run model-generated code in a
    throwaway temp script (kernelbench / tritonbench / cvdp). A plain
    ``subprocess.run(timeout=...)`` only kills the direct child; CUDA
    (torch multiprocessing workers) and triton (autotune compile subprocesses)
    and pytest (cocotb) spawn grandchildren that survive the timeout and leak
    GPU memory / file locks into the next sample. ``start_new_session=True``
    puts the child in its own session/process-group so ``os.killpg`` reaps the
    whole tree.

    Lighter than ``_run_command`` (no rlimit preexec) — torch/triton set their
    own resource needs and a 64MB file rlimit would break .so/.cubin emission.

    Returns ``(returncode, stdout, stderr, timed_out)``; stdout/stderr decoded
    utf-8 (replace). On timeout the group is killed and drained before return.
    """
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return -1, "", f"launch failed: {exc}", False
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
    return (
        process.returncode,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        timed_out,
    )


def _self_test() -> None:
    # INFRA_CODES is a closed enum — unknown codes rejected
    try:
        infra_failure(infra_code="not_a_real_code", eval_backend="x")
        raise AssertionError("infra_failure accepted unknown code")
    except ValueError:
        pass
    # trusted infra → is_infra True; model → False
    assert is_infra(finalize_failure_classification(
        infra_failure(infra_code="trusted_reference_failure", eval_backend="x")))
    assert not is_infra(finalize_failure_classification(
        model_failure(reason="x", eval_backend="y")))
    # finalize forces correct=False on an infra result even if correct was True
    r = finalize_failure_classification(base_result(eval_backend="x", correct=True,
        infra_error=True, infra_code="trusted_reference_failure"))
    assert r["correct"] is False
    print("verify.core smoke: 4/4 passed")


if __name__ == "__main__":
    _self_test()
