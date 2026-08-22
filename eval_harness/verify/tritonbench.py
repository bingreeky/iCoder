"""verify/tritonbench.py — hardened TritonBench-G/T correctness verify (local).

Ports the verify-worker + anti-cheat logic from the upstream RLVR verifier
(verify_server_v2.py:313-631, 1434-1565) into a local function called directly
by benches/tbg_verify_correctness.py. No FastAPI server, no sealed payload;
refs are read from disk and the caller is responsible for pinning them (the
driver SHA256-checks the task file before calling here).

What this fixes vs. the old eval_harness verify_one
(teacher_triton_rollout_tbg.py:307-382):
  * Determinism: torch.manual_seed + cuda.manual_seed_all +
    use_deterministic_algorithms(warn_only=True) per variant. The old path
    seeded nothing → autotune/non-deterministic kernels flipped pass/fail
    across runs, drifting the denominator (the 137/139 bug).
  * Timeout floor: per_variant_timeout = max(180.0, timeout/2). The old flat
    TBG_TIMEOUT=180 clipped the 38 @triton.autotune tasks (76-190s autotune
    alone) at the boundary → ref_smoke_failed set shifted per run.
  * Anti-cheat: triton completed-launch counter (only real CUDA launches
    count, not bracket lookups/warmups) + identity-hack detection
    (load→store kernels with +0.0/*1.0/.to()) + framework-delegation gate
    (AST ban on torch compute ops outside @triton.jit). The old path had none.
  * Comparison: _nested_allclose (recursive dict/list/tuple/tensor + shape
    precheck, atol/rtol from payload) vs. the old flat-dict torch.allclose.
  * Classification: results flow through verify.core.finalize, so ref failures
    are trusted_reference_failure infra (one definition), never a skip that
    drifts.

References (upstream, verifiers_unzip/verifiers/verify_server_v2.py):
  :313-317  _split_tritonbench_task
  :320-349  _nested_allclose
  :352-371  _TRITONBENCH_RESULT_TAIL
  :374-415  _triton_launch_counter_source + _TRITON_LAUNCH_COUNTER
  :418-491  _run_tritonbench_variant
  :494-502  _call_name
  :504-591  _detect_framework_delegation + _DISALLOWED_TORCH_OPS
  :594-631  _detect_identity_hack
  :1434-1565 _tritonbench_eval_worker (the shape this mirrors)
"""

from __future__ import annotations

import ast
import os
import pickle
import re
import subprocess
import sys
import tempfile
from typing import Any

from .core import (
    INFRA_CODES,
    base_result,
    finalize_failure_classification,
    infra_failure,
    model_failure,
    run_procgroup,
)

_TRITONBENCH_SEPARATOR = re.compile(r"#{120,}")

_DISALLOWED_TORCH_OPS = {
    "abs", "add", "argmax", "argmin", "bmm", "cat", "clamp", "conv1d",
    "conv2d", "conv3d", "cos", "cumsum", "einsum", "exp", "gelu",
    "layer_norm", "log", "log_softmax", "matmul", "max", "mean", "min",
    "mm", "mul", "norm", "relu", "sigmoid", "sin", "softmax", "sort",
    "sqrt", "stack", "sum", "tanh", "topk", "var",
}

_TRITON_LAUNCH_COUNTER = "triton_runtime_launch_exit_hook_v1"


def _launch_hook_available() -> bool:
    """True iff this triton exposes runtime.launch_exit_hook (with .add).

    Triton 2.3.x has neither triton.knobs nor the hook; triton 3.x+ does.
    Cached after first probe. When unavailable, we skip launch counting and
    the launch-based anti-cheat gates (launches<=0, identity_hack) — those
    need real launch telemetry. framework_delegation (pure AST) still applies.
    """
    cache = getattr(_launch_hook_available, "_cache", None)
    if cache is None:
        try:
            from triton import knobs as _k  # noqa: F401
            hook = getattr(getattr(_k, "runtime", None), "launch_exit_hook", None)
            cache = hasattr(hook, "add")
        except Exception:
            cache = False
        _launch_hook_available._cache = cache
    return cache


def split_tritonbench_task(source: str) -> tuple[str, str] | None:
    """Split a TBG ref .py into (kernel_code, test_block) on a run of 120+
    `#`. The last part must contain `def test_` or the split is rejected
    (returns None) — the test harness is mandatory. Upstream :313-317.
    """
    parts = _TRITONBENCH_SEPARATOR.split(source)
    if len(parts) < 2 or "def test_" not in parts[-1]:
        return None
    return parts[0], parts[-1]


def _nested_allclose(
    reference: Any,
    candidate: Any,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Recursive allclose over dict / list / tuple / torch tensor / scalar.
    Upstream :320-349.
    """
    import torch

    if isinstance(reference, dict) and isinstance(candidate, dict):
        return set(reference) == set(candidate) and all(
            _nested_allclose(reference[key], candidate[key], atol=atol, rtol=rtol)
            for key in reference
        )
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        return len(reference) == len(candidate) and all(
            _nested_allclose(left, right, atol=atol, rtol=rtol)
            for left, right in zip(reference, candidate)
        )
    if torch.is_tensor(reference) and torch.is_tensor(candidate):
        return reference.shape == candidate.shape and torch.allclose(
            reference, candidate, atol=atol, rtol=rtol, equal_nan=True,
        )
    # Fallback: leaves that escaped the dict/list/torch-tensor branches above
    # — most often NUMPY arrays (torch.is_tensor is False for numpy), but also
    # python/numpy scalars and mixed types. A bare `reference == candidate`
    # returns an element-wise numpy array whose truth value is then AMBIGUOUS
    # in the caller's `all(...)` / `and` => ValueError => the whole shard
    # crashes on row 1 => 0 rows verified (silently swallowed by
    # run_tritonbench's `wait ... || true`, which hid this for the whole TBG
    # resume batch). Use np.allclose (atol/rtol, matching the torch branch,
    # with an explicit shape guard against broadcast false-positives like
    # (3,) vs (1,)) so this always returns a plain bool.
    import numpy as np
    try:
        ra, ca = np.asarray(reference), np.asarray(candidate)
        if ra.shape != ca.shape:
            return False
        return bool(np.allclose(ra, ca, atol=atol, rtol=rtol, equal_nan=True))
    except (ValueError, TypeError):
        # Non-numeric leaves (e.g. str ids) / incompatible dtypes — exact
        # equality, coerced to bool (defends against any array result).
        res = reference == candidate
        if hasattr(res, "all"):
            return bool(res.all())
        return bool(res)


_TRITONBENCH_RESULT_TAIL = r'''
import pickle as _tb_pickle
import torch as _tb_torch

def _tb_to_cpu(value):
    if isinstance(value, dict):
        return {key: _tb_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tb_to_cpu(item) for item in value]
    if _tb_torch.is_tensor(value):
        return value.detach().float().cpu()
    return value

_tb_result = globals().get("result_gold", globals().get("test_results"))
_tb_payload = {
    "result": _tb_to_cpu(_tb_result),
}
with open(__TB_OUTPUT_PATH__, "wb") as _tb_handle:
    _tb_pickle.dump(_tb_payload, _tb_handle)
'''


# ---------------------------------------------------------------------------
# Performance (speedup) measurement for the `fast` metric.
#
# Upstream reward_kernel_v2.py:706-707 defines:
#   speedup = verify_result.get("speedup")
#   fast = speedup is not None and speedup > 1.05
# i.e. a candidate is `fast` iff it is correct AND >1.05x quicker than the
# reference. The local verify (above) only measures correctness (speedup=None).
# To produce the real `fast` tier we time the wrapper call: instrument every
# top-level plain function (the @triton.jit kernel is a JITFunction instance,
# NOT an inspect.isfunction, so it is skipped — only the human-facing wrapper
# is wrapped), run the test fn (which calls the wrapper once per test case),
# capture the wrapper's call args, then cuda-event-time it N times and take the
# median ms. ref and cand each timed on the SAME test_block (same case set) so
# the ratio is comparable; we use the LAST captured call (the largest test
# case, matching KernelBench's get_init_inputs largest-size convention).
#
# The instrument block is injected AFTER code_body (so the wrapper exists) and
# BEFORE test_body (so the test fn's calls go through the recorder). The perf
# tail REPLACES the result tail (it pickles BOTH result + runtime_ms).
# ---------------------------------------------------------------------------

_TRITONBENCH_PERF_INSTRUMENT = r'''
import inspect as _perf_inspect
_perf_captured = {}  # name -> [func, args, kwargs] (last call wins)
for _perf_nm, _perf_obj in list(globals().items()):
    if _perf_nm.startswith("_perf") or _perf_nm.startswith("_tb"):
        continue
    if not _perf_inspect.isfunction(_perf_obj):
        continue  # JITFunction instances (triton.jit kernels) are skipped
    try:
        def _perf_make(_f, _n):
            def _perf_wrap(*a, **k):
                _perf_captured[_n] = (_f, a, k)  # last call overwrites
                return _f(*a, **k)
            return _perf_wrap
        globals()[_perf_nm] = _perf_make(_perf_obj, _perf_nm)
    except Exception:
        pass
'''

_TRITONBENCH_PERF_TAIL = r'''
import pickle as _tb_pickle
import torch as _tb_torch

def _tb_to_cpu(value):
    if isinstance(value, dict):
        return {key: _tb_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tb_to_cpu(item) for item in value]
    if _tb_torch.is_tensor(value):
        return value.detach().float().cpu()
    return value

def _perf_time_ms(_fn, _args, _kwargs, _warmup=5, _trials=20):
    try:
        for _ in range(_warmup):
            _fn(*_args, **_kwargs)
        _tb_torch.cuda.synchronize()
    except Exception:
        return None
    _starts = [_tb_torch.cuda.Event(enable_timing=True) for _ in range(_trials)]
    _ends = [_tb_torch.cuda.Event(enable_timing=True) for _ in range(_trials)]
    for _s, _e in zip(_starts, _ends):
        _s.record()
        try:
            _fn(*_args, **_kwargs)
        except Exception:
            return None
        _e.record()
    _tb_torch.cuda.synchronize()
    _ms = sorted(_s.elapsed_time(_e) for _s, _e in zip(_starts, _ends))
    return float(_ms[len(_ms) // 2])

_tb_result = globals().get("result_gold", globals().get("test_results"))
_perf_runtime_ms = None
# wrapper = last-called plain (non-JIT) function; fall back to any captured call.
_perf_pick = None
for _perf_n in list(_perf_captured):
    if _perf_n.startswith("test_"):
        continue  # the test fn itself, called with no args — not the kernel
    _f = _perf_captured[_perf_n][0]
    if "JITFunction" in type(_f).__name__:
        continue
    _perf_pick = _perf_captured[_perf_n]  # last non-JIT call wins
if _perf_pick is None and _perf_captured:
    for _perf_n in list(_perf_captured):
        if _perf_n.startswith("test_"):
            continue
        _perf_pick = _perf_captured[_perf_n]
        break
if _perf_pick is not None:
    _perf_runtime_ms = _perf_time_ms(_perf_pick[0], _perf_pick[1], _perf_pick[2])

_tb_payload = {
    "result": _tb_to_cpu(_tb_result),
    "runtime_ms": _perf_runtime_ms,
}
with open(__TB_OUTPUT_PATH__, "wb") as _tb_handle:
    _tb_pickle.dump(_tb_payload, _tb_handle)
'''


def _triton_launch_counter_source(counter_path: str) -> str:
    """Install a counter at Triton's completed CUDA launch boundary.
    Upstream :377-415. Only counts completed launches (not bracket lookups or
    warmup compiles), so a candidate that never actually runs a Triton kernel
    gets triton_launched=0 and is scored 0 by the reward gate.
    """
    return f'''\
import os as _tb_counter_os
from triton import knobs as _tb_triton_knobs

_tb_launch_exit_chain = _tb_triton_knobs.runtime.launch_exit_hook
if not hasattr(_tb_launch_exit_chain, "add"):
    raise RuntimeError("Triton runtime launch-exit hook is unavailable")

def _tb_make_completed_launch_counter(_tb_counter_path):
    _tb_counter_fd = _tb_counter_os.open(
        _tb_counter_path,
        _tb_counter_os.O_WRONLY | _tb_counter_os.O_APPEND,
    )
    _tb_write = _tb_counter_os.write
    def _tb_record_completed_launch(_tb_launch_metadata):
        _tb_write(_tb_counter_fd, b"1\\n")
    return _tb_record_completed_launch

_tb_completed_launch_counter = _tb_make_completed_launch_counter({counter_path!r})
_tb_launch_exit_chain.add(_tb_completed_launch_counter)
del (
    _tb_completed_launch_counter,
    _tb_counter_os,
    _tb_launch_exit_chain,
    _tb_make_completed_launch_counter,
    _tb_triton_knobs,
)
'''


def _read_triton_completed_launch_count(counter_path: str) -> int:
    with open(counter_path, "rb") as handle:
        events = handle.read()
    if events.replace(b"1\n", b""):
        raise ValueError("invalid Triton completed-launch counter")
    return events.count(b"\n")


def _run_tritonbench_variant(
    code_body: str,
    test_body: str,
    output_path: str,
    timeout: float,
    seed: int = 0,
    *,
    measure_perf: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    """Execute one reference or candidate in a disposable Python file.

    Seeds torch (CPU + CUDA + deterministic-algorithms warn_only), installs
    the completed-launch counter (when the triton version exposes the
    launch_exit_hook), runs code_body + test_body + the result-tail that
    pickles result_gold/test_results, and returns the payload. Upstream
    :426-491; the counter injection is made version-conditional so triton
    2.3.x (no knobs) still runs — triton_launched is None then, and the
    caller skips the launch-based anti-cheat gates.

    When measure_perf=True the perf instrument is injected between code_body
    and test_body (wraps top-level plain functions, i.e. the wrapper) and the
    perf tail replaces the result tail — the payload gains ``runtime_ms``
    (median of 20 cuda-event-timed wrapper calls) alongside ``result``.
    """
    hook_ok = _launch_hook_available()
    counter_path: str | None = None
    if hook_ok:
        counter_descriptor, counter_path = tempfile.mkstemp(
            suffix=".triton_completed_launches"
        )
        os.close(counter_descriptor)
    counter_block = (
        _triton_launch_counter_source(counter_path) if hook_ok else ""
    )
    instrument = _TRITONBENCH_PERF_INSTRUMENT if measure_perf else ""
    tail = (
        _TRITONBENCH_PERF_TAIL if measure_perf else _TRITONBENCH_RESULT_TAIL
    ).replace("__TB_OUTPUT_PATH__", repr(output_path))
    source = f'''\
import torch as _tb_seed_torch
_tb_seed_torch.manual_seed({seed!r})
_tb_seed_torch.cuda.manual_seed_all({seed!r})
try:
    _tb_seed_torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

{counter_block}

{code_body}
{instrument}
{test_body}
{tail}
'''
    descriptor, script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(source)
        rc, _out, err, timed_out = run_procgroup(
            [sys.executable, script_path],
            timeout=timeout,
        )
        if timed_out:
            return None, "timeout"
        if rc != 0 or not os.path.isfile(output_path):
            stderr = (err or "").strip().splitlines()
            detail = stderr[-1][:240] if stderr else f"exitcode_{rc}"
            return None, f"exec_fail:{detail}"
        try:
            with open(output_path, "rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            return None, f"result_read_fail:{type(exc).__name__}"
        if not isinstance(payload, dict):
            return None, "result_read_fail:invalid_payload"
        if counter_path is not None:
            try:
                payload["triton_launched"] = _read_triton_completed_launch_count(counter_path)
            except (OSError, ValueError) as exc:
                return None, f"launch_counter_read_fail:{type(exc).__name__}"
            payload["triton_launch_counter"] = _TRITON_LAUNCH_COUNTER
        else:
            payload["triton_launched"] = None
            payload["triton_launch_counter"] = None
        return payload, ""
    finally:
        for path in (script_path, output_path, counter_path):
            if path is None:
                continue
            try:
                os.unlink(path)
            except OSError:
                pass


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def detect_framework_delegation(source: str) -> bool:
    """Conservative static gate for obvious PyTorch compute delegation.
    Output allocation and CUDA sync are allowed. Upstream :504-591.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    framework_nodes: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _call_name(decorator).endswith("triton.jit")
            or _call_name(decorator) == "triton.jit"
            for decorator in statement.decorator_list
        ):
            continue
        framework_nodes.extend(ast.walk(statement))

    torch_roots = {"torch"}
    functional_roots = {"F"}
    tensor_argument_names: set[str] = set()
    direct_compute_calls: set[str] = set()
    for node in framework_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tensor_argument_names.update(argument.arg for argument in node.args.args)
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                if alias.name == "torch":
                    torch_roots.add(local_name)
                elif alias.name == "torch.nn.functional":
                    functional_roots.add(local_name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                if module == "torch" and alias.name in _DISALLOWED_TORCH_OPS:
                    direct_compute_calls.add(local_name)
                elif module == "torch.nn" and alias.name == "functional":
                    functional_roots.add(local_name)
                elif (
                    module == "torch.nn.functional"
                    and alias.name in _DISALLOWED_TORCH_OPS
                ):
                    direct_compute_calls.add(local_name)

    for node in framework_nodes:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        if isinstance(node, ast.BinOp) and (
            (isinstance(node.left, ast.Name) and node.left.id in tensor_argument_names)
            or (isinstance(node.right, ast.Name) and node.right.id in tensor_argument_names)
        ):
            return True
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.operand, ast.Name)
            and node.operand.id in tensor_argument_names
        ):
            return True
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        parts = name.split(".")
        root = parts[0] if parts else ""
        leaf = parts[-1] if parts else ""
        if name in direct_compute_calls:
            return True
        if root in torch_roots and (
            "ops" in parts[1:]
            or leaf in _DISALLOWED_TORCH_OPS
            or ".".join(parts[1:]) in {"compile", "jit.script"}
        ):
            return True
        if root in functional_roots and leaf in _DISALLOWED_TORCH_OPS:
            return True
        if root not in {"tl", "triton", "math"} and leaf in _DISALLOWED_TORCH_OPS:
            return True
    return False


def detect_identity_hack(source: str, triton_launched: int) -> bool:
    """Detect load→store identity kernels (+0.0, *1.0, .to() casts) that
    trivially pass allclose without doing the computation. Upstream :594-631.
    """
    if triton_launched <= 0:
        return False
    jit_bodies = re.findall(
        r"@triton\.jit\s*\ndef\s+\w+\([^)]*\):\s*(.*?)(?=\ndef |\nclass |\n@|\Z)",
        source,
        re.DOTALL,
    )
    for body in jit_bodies:
        lines = [
            line.strip()
            for line in body.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        load_match = re.search(r"(\w+)\s*=\s*tl\.load\(", body)
        if not load_match:
            continue
        identity_vars = {load_match.group(1)}
        for line in lines:
            assignment = re.match(r"(\w+)\s*=\s*(.+)$", line)
            if not assignment:
                continue
            variable, rhs = assignment.group(1), assignment.group(2).strip()
            preserves_identity = (
                re.match(r"^tl\.load\(", rhs) is not None
                or rhs in identity_vars
                or any(re.match(rf"^{name}\s*\+\s*0\.?0?\s*$", rhs) for name in identity_vars)
                or any(re.match(rf"^{name}\s*\*\s*1\.?0?\s*$", rhs) for name in identity_vars)
                or any(re.match(rf"^{name}\.to\(", rhs) for name in identity_vars)
            )
            identity_vars.discard(variable)
            if preserves_identity:
                identity_vars.add(variable)
        store_match = re.search(r"tl\.store\([^,]+,\s*([^,)]+)", body)
        if store_match and store_match.group(1).strip() in identity_vars:
            return True
    return False


def verify_tbg(
    ref_src: str,
    teacher_code: str,
    test_block: str,
    *,
    timeout: int = 180,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    seed: int = 0,
    measure_perf: bool = False,
) -> dict[str, Any]:
    """Run ref + candidate side-by-side, compare result_gold (TBG) /
    test_results (TBT) via _nested_allclose, attach anti-cheat signals, and
    return a verify.core result dict (already finalize'd).

    Mirrors _tritonbench_eval_worker (verify_server_v2.py:1434-1565) minus the
    server/mp plumbing. The caller (tbg_verify_correctness.py) handles
    GPU pinning via CUDA_VISIBLE_DEVICES and the sharded merge.

    per_variant_timeout = max(180.0, timeout/2): 38 TBG tasks use
    @triton.autotune with up to 38 configs (76-190s autotune alone); a flat
    180s would clip them. This floor is the single biggest denominator-stab
    fix — without it the ref_smoke_failed set shifts per run.

    When measure_perf=True both ref and candidate are timed (cuda-event median
    of 20 wrapper calls). The result gains ``speedup`` (ref_ms/cand_ms) and
    ``fast`` (= correct AND speedup>1.05, per reward_kernel_v2.py:706-707).
    """
    per_variant_timeout = max(180.0, float(timeout) / 2.0)

    # Empty/garbage teacher code is a model failure, not infra.
    if not teacher_code or not teacher_code.strip():
        return finalize_failure_classification(
            model_failure(reason="empty_teacher_code", eval_backend="tritonbench",
                          info="skipped_no_teacher_code")
        )
    if not test_block or not test_block.strip():
        return finalize_failure_classification(
            model_failure(reason="no_test_block", eval_backend="tritonbench",
                          info="skipped_no_test_block")
        )

    ref_fd, ref_out = tempfile.mkstemp(suffix=".reference.pkl")
    cand_fd, cand_out = tempfile.mkstemp(suffix=".candidate.pkl")
    os.close(ref_fd)
    os.close(cand_fd)
    try:
        reference, ref_error = _run_tritonbench_variant(
            ref_src, test_block, ref_out, per_variant_timeout, seed,
            measure_perf=measure_perf,
        )
        if reference is None:
            # Reference unrunnable → unwinnable for ANY model → infra, excluded
            # from the denominator (the canonical "trusted_reference_failure").
            return finalize_failure_classification(
                infra_failure(
                    infra_code="trusted_reference_failure",
                    eval_backend="tritonbench",
                    info=f"reference_{ref_error}",
                    error_type="reference_harness_failure",
                    verify_meta={"stage": "ref_smoke_failed",
                                 "error": ref_error},
                )
            )

        candidate, cand_error = _run_tritonbench_variant(
            teacher_code, test_block, cand_out, per_variant_timeout, seed,
            measure_perf=measure_perf,
        )
        if candidate is None:
            info = "timeout" if cand_error == "timeout" else "compile_fail"
            error_type = ("task_timeout" if cand_error == "timeout"
                          else "model_execution_error")
            return finalize_failure_classification(
                base_result(
                    eval_backend="tritonbench",
                    info=info,
                    error_type=error_type,
                    compiled=False,
                    verify_meta={"stage": "candidate_run_failed",
                                 "error": cand_error},
                )
            )

        raw_launches = candidate.get("triton_launched")
        # Match upstream _tritonbench_eval_worker (verify_server_v2.py:1541-1553):
        # `correct` is decided by _nested_allclose ALONE. identity_hack and
        # framework_delegation are ANTI-CHEAT SIGNALS — recorded as peer fields
        # (the upstream VerifyResponse carries them alongside `correct`), NOT
        # hard reward gates. The upstream detector docstring calls it a
        # heuristic that "supplements, but does not replace, adversarial GPU
        # fixtures" — it false-positives on legitimate code (Python builtins
        # `min`/`max`/`sum`, tensor `.sum()`, scalar-arg binops), so it must
        # not zero the reward directly. The RL trainer consumes these signals
        # downstream; eval_harness has no reward layer, so verify_correct IS
        # the metric = allclose, exactly as upstream.
        launches = int(raw_launches) if raw_launches is not None else None
        correct = _nested_allclose(
            reference.get("result"), candidate.get("result"),
            atol=atol, rtol=rtol,
        )
        # identity_hack needs a real launch count (triton 3.x hook); on 2.3.x
        # (None) it is indeterminate → False, matching upstream's
        # `int(... or 0)` → _detect_identity_hack(launches=0) → False guard.
        identity_hack = (
            detect_identity_hack(teacher_code, launches) if launches is not None
            else False
        )
        framework_delegation = detect_framework_delegation(teacher_code)
        # `fast` metric (reward_kernel_v2.py:706-707): correct AND speedup>1.05.
        # speedup = ref_runtime / cand_runtime (both median ms of the wrapper).
        speedup = None
        if measure_perf:
            ref_ms = reference.get("runtime_ms")
            cand_ms = candidate.get("runtime_ms")
            if ref_ms and cand_ms and cand_ms > 0:
                speedup = float(ref_ms) / float(cand_ms)
        fast = bool(correct and speedup is not None and speedup > 1.05)
        result = finalize_failure_classification(
            base_result(
                eval_backend="tritonbench",
                correct=correct,
                compiled=True,
                info="pass" if correct else "compiled_but_wrong",
                triton_launched=launches,
                identity_hack=identity_hack,
                framework_delegation=framework_delegation,
                speedup=speedup,
                reward_style="binary",
                verify_meta={"stage": "ok" if correct else "allclose_failed"},
            )
        )
        if measure_perf:
            result["fast"] = fast
        return result
    finally:
        for path in (ref_out, cand_out):
            try:
                os.unlink(path)
            except OSError:
                pass



def _self_test() -> None:
    # _nested_allclose: scalar / tensor / dict
    assert _nested_allclose(1, 1) is True
    assert _nested_allclose({"a": 1}, {"a": 2}) is False
    # framework_delegation: torch matmul outside @triton.jit → True
    assert detect_framework_delegation("import torch\nc = torch.matmul(a, b)\n") is True
    assert detect_framework_delegation("import torch\nVAL = 5\n") is False
    # identity_hack needs launches > 0
    assert detect_identity_hack("import torch\n", 0) is False
    # split_tritonbench_task
    src = "def k(): pass\n" + "#" * 120 + "\ndef test_x(): pass\n"
    assert split_tritonbench_task(src) is not None
    print("verify.tritonbench smoke: 5/5 passed")


if __name__ == "__main__":
    _self_test()
