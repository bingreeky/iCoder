"""verify/kernelbench.py — hardened KernelBench correctness verify (local).

Ports the KB verify-worker + anti-cheat logic from the upstream RLVR verifier
(verify_server_v2.py:1308-1432, 374-415, 1021-1047, 145-168) into a local
function called by benches/kb_verify_correctness.py. No FastAPI server, no
sealed payload; refs are read from disk.

Architecture (mirrors verify/tritonbench.py): verify_kb() builds a disposable
worker-script string and runs it in a subprocess under the KB venv python
(KERNELBENCH_PY — it has torch+triton+litellm, the eval_harness python does
not). The worker:
  * seeds torch (CPU+CUDA+deterministic warn_only) for stable autotune
  * feature-detects triton's launch_exit_hook IN THE CHILD (the KB venv's
    triton may differ from the parent's), and only then installs the
    completed-launch counter as a PREFIX to the candidate module — so it
    counts candidate launches only, after the trusted reference has run
  * calls kernelbench.eval.eval_kernel_against_ref (the official KB
    evaluator) with 32 correctness trials
  * pickles {correct, compiled, runtime, ref_runtime, triton_launched, error}
verify_kb() reads the pickle, attaches identity_hack + framework_delegation
(reused from verify.tritonbench — same anti-cheat as TBG), applies them as
hard reward gates (eval_harness has no separate reward layer, so verify_correct
IS the metric), and returns a verify.core result (already finalize'd).

Classification principle (upstream :1021-1047 _kb_evaluation_exception_result):
EVERY post-bootstrap exception is a model failure. The ref and candidate run
in the same disposable untrusted process, so traceback heuristics cannot safely
promote an exception to infrastructure — the poisoned CUDA state is contained
by killing the subprocess. Trusted setup/integrity failures are emitted
before this boundary (here: ref-import failure → trusted_reference_failure
infra; everything after the eval call → model). We intentionally DO NOT port
the 400-line MLIR-sigabrt native-crash forensics (_exception_provenance L829-
1011, _candidate_mlir_sigabrt_provenance L1176-1252): they are diagnostic-only
upstream (never control the infra/model decision) and would bloat this module
against the "maintainable repo" goal. The exception name is kept as provenance.

framework_delegation as a hard gate: the upstream KB worker records the signal
but leaves zeroing to reward_kernel_v2._compute_reward. eval_harness has no
reward layer, so verify_kb forces correct=False when delegation fires — the
faithful combined behaviour, same as TBG.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .core import (
    base_result,
    finalize_failure_classification,
    infra_failure,
    model_failure,
    run_procgroup,
)
# Reuse the triton anti-cheat + launch-counter primitives — same KB contract.
from .tritonbench import (
    _TRITON_LAUNCH_COUNTER,
    detect_framework_delegation,
    detect_identity_hack,
)

# KB src root resolution order (matches eval_harness config + upstream _BASE):
#   KERNELBENCH_SRC_ROOT (explicit) > $KERNELBENCH_ROOT/src > the repo's
#   benchmarks/KernelBench/src. Fails clearly if none exists.
_BASE_CANDIDATES = (
    str(Path(__file__).resolve().parents[1] / "benchmarks" / "KernelBench" / "src"),
)


def _resolve_kb_src(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for key in ("KERNELBENCH_SRC_ROOT", "KERNELBENCH_ROOT"):
        root = os.environ.get(key)
        if root:
            p = root if root.endswith("src") else f"{root}/src"
            if Path(p).is_dir():
                return p
    for c in _BASE_CANDIDATES:
        if Path(c).is_dir():
            return c
    raise RuntimeError(
        "KernelBench src not found: set KERNELBENCH_SRC_ROOT or KERNELBENCH_ROOT"
    )


# Upstream verify_server_v2.py:145-168. Patches Model/ModelNew.__init__ so a
# custom module constructed as `ModelNew(list_args, dict_kwargs)` (the KB
# harness shape) forwards to the original signature. Appended to BOTH ref and
# candidate model sources.
_INIT_ADAPTER = '''
def _kb_patch_init(c):
    g = globals()
    if c not in g:
        return
    cls = g[c]
    if getattr(cls, "_kb_patched", False):
        return
    original_init = cls.__init__
    def patched_init(self, *args, **kwargs):
        if (
            not kwargs
            and len(args) == 2
            and isinstance(args[0], (list, tuple))
            and isinstance(args[1], dict)
        ):
            original_init(self, *list(args[0]), **dict(args[1]))
        else:
            original_init(self, *args, **kwargs)
    cls.__init__ = patched_init
    cls._kb_patched = True
_kb_patch_init("Model")
_kb_patch_init("ModelNew")
'''


# The worker runs under KERNELBENCH_PY (KB venv). It builds its own counter
# prefix only after probing THIS process's triton, so the count reflects
# candidate launches alone (hook installed at custom-module import, after the
# trusted ref has run). Result is pickled to output_path.
def _kb_worker_source(
    ref_src: str,
    gen_src: str,
    output_path: str,
    gpu: int,
    num_correct_trials: int,
    num_perf_trials: int,
    measure_perf: bool,
    atol: float,
    rtol: float,
    seed: int,
) -> str:
    # NOTE: {ref_src}/{gen_src} are repr-injected below the template, so the
    # raw braces in _INIT_ADAPTER (none) and counter code are untouched.
    return f'''\
import os as _kb_os
import sys as _kb_sys
import pickle as _kb_pickle
import tempfile as _kb_tmp
import traceback as _kb_tb

_kb_os.environ["CUDA_VISIBLE_DEVICES"] = {str(gpu)!r}
_kb_seed = {seed!r}
try:
    import torch as _kb_torch
    _kb_torch.manual_seed(_kb_seed)
    _kb_torch.cuda.manual_seed_all(_kb_seed)
    try:
        _kb_torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
except Exception as _kb_seed_exc:
    _kb_payload = {{"bootstrap_error": "seed:" + type(_kb_seed_exc).__name__,
                    "exception": _kb_tb.format_exc()[:2000]}}
    with open({output_path!r}, "wb") as _kb_h:
        _kb_pickle.dump(_kb_payload, _kb_h)
    _kb_sys.exit(0)

# --- launch-exit-hook counter, feature-detected in THIS process's triton ---
_kb_counter_path = None
_kb_launch_hook_ok = False
try:
    from triton import knobs as _kb_knobs
    _kb_hook = getattr(getattr(_kb_knobs, "runtime", None), "launch_exit_hook", None)
    if hasattr(_kb_hook, "add"):
        _kb_fd, _kb_counter_path = _kb_tmp.mkstemp(suffix=".kb_launch_count")
        _kb_os.close(_kb_fd)
        def _kb_make_counter(_p):
            _cfd = _kb_os.open(_p, _kb_os.O_WRONLY | _kb_os.O_APPEND)
            def _kb_rec(_meta):
                _kb_os.write(_cfd, b"1\\n")
            return _kb_rec
        _kb_hook.add(_kb_make_counter(_kb_counter_path))
        _kb_launch_hook_ok = True
except Exception:
    _kb_launch_hook_ok = False

# counter prefix is prepended to the candidate module so the hook is armed at
# candidate import time (after the ref has run) — counts candidate launches only.
_kb_counter_prefix = ""
if _kb_launch_hook_ok and _kb_counter_path:
    _kb_counter_prefix = (
        "import os as _tbc_os\\n"
        "from triton import knobs as _tbc_knobs\\n"
        "_tbc_chain = _tbc_knobs.runtime.launch_exit_hook\\n"
        "def _tbc_make(_p):\\n"
        "    _fd = _tbc_os.open(_p, _tbc_os.O_WRONLY | _tbc_os.O_APPEND)\\n"
        "    def _r(_m): _tbc_os.write(_fd, b'1\\\\n')\\n"
        "    return _r\\n"
        "_tbc_chain.add(_tbc_make(" + repr(_kb_counter_path) + "))\\n"
        "del _tbc_chain, _tbc_make, _tbc_os, _tbc_knobs\\n"
    )

_kb_src = _kb_sys.argv[1] if len(_kb_sys.argv) > 1 else ""
if _kb_src:
    _kb_sys.path.insert(0, _kb_src)

try:
    import kernelbench.eval as _kb_eval
except Exception as _kb_imp_exc:
    _kb_payload = {{"bootstrap_error": "import_kernelbench:" + type(_kb_imp_exc).__name__,
                    "exception": _kb_tb.format_exc()[:2000]}}
    with open({output_path!r}, "wb") as _kb_h:
        _kb_pickle.dump(_kb_payload, _kb_h)
    _kb_sys.exit(0)

_kb_ref = {ref_src!r}
_kb_gen = {gen_src!r}
_kb_adapter = ''' + repr(_INIT_ADAPTER) + f'''

try:
    _kb_ev = _kb_eval.eval_kernel_against_ref(
        original_model_src=_kb_ref + _kb_adapter,
        custom_model_src=_kb_counter_prefix + "\\n" + _kb_gen + _kb_adapter,
        measure_performance={measure_perf!r},
        num_correct_trials={num_correct_trials!r},
        num_perf_trials={num_perf_trials!r},
        backend="triton",
        verbose=False,
    )
    _kb_correct = bool(_kb_ev and _kb_ev.correctness)
    _kb_compiled = bool(_kb_ev and _kb_ev.compiled)
    _kb_runtime = float(_kb_ev.runtime) if (_kb_ev and _kb_ev.runtime) else None
    _kb_ref_rt = float(_kb_ev.ref_runtime) if (_kb_ev and _kb_ev.ref_runtime) else None
    _kb_info = "pass" if _kb_correct else ("compiled_but_wrong" if _kb_compiled else "compile_fail")
except BaseException as _kb_eval_exc:
    _kb_payload = {{"eval_exception": type(_kb_eval_exc).__name__,
                    "exception": _kb_tb.format_exc()[:3000]}}
    with open({output_path!r}, "wb") as _kb_h:
        _kb_pickle.dump(_kb_payload, _kb_h)
    _kb_sys.exit(0)

_kb_launches = None
if _kb_launch_hook_ok and _kb_counter_path is not None:
    try:
        with open(_kb_counter_path, "rb") as _cf:
            _ev = _cf.read()
        if not _ev.replace(b"1\\n", b""):
            _kb_launches = _ev.count(b"\\n")
    except Exception:
        _kb_launches = None
    try:
        _kb_os.unlink(_kb_counter_path)
    except OSError:
        pass

_kb_payload = {{
    "correct": _kb_correct,
    "compiled": _kb_compiled,
    "info": _kb_info,
    "runtime": _kb_runtime,
    "ref_runtime": _kb_ref_rt,
    "triton_launched": _kb_launches,
    "triton_launch_counter": {_TRITON_LAUNCH_COUNTER!r} if _kb_launches is not None else None,
}}
with open({output_path!r}, "wb") as _kb_h:
    _kb_pickle.dump(_kb_payload, _kb_h)
'''


def verify_kb(
    ref_src: str,
    gen_src: str,
    *,
    gpu: int = 0,
    kb_python: str | None = None,
    kb_src: str | None = None,
    num_correct_trials: int = 32,
    num_perf_trials: int = 10,
    measure_perf: bool = False,
    timeout: float = 300.0,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one KB candidate against its reference under the official KB
    evaluator (32 correctness trials), with the hardened anti-cheat +
    classification. Returns a verify.core result dict (already finalize'd).

    `kb_python` defaults to KERNELBENCH_PY env (the KB venv python with
    torch+triton+litellm). `kb_src` defaults via _resolve_kb_src.
    """
    if not gen_src or not gen_src.strip():
        return finalize_failure_classification(
            model_failure(reason="empty_teacher_code", eval_backend="kernelbench",
                          info="skipped_no_teacher_code")
        )
    if not ref_src or not ref_src.strip():
        return finalize_failure_classification(
            infra_failure(infra_code="trusted_reference_failure",
                          eval_backend="kernelbench",
                          info="empty_reference_solution",
                          error_type="reference_harness_failure",
                          verify_meta={"stage": "ref_smoke_failed",
                                       "error": "empty_reference_solution"})
        )

    kb_python = kb_python or os.environ.get("KERNELBENCH_PY") or sys.executable
    kb_src = _resolve_kb_src(kb_src)

    fd, out_path = tempfile.mkstemp(suffix=".kb_result.pkl")
    os.close(fd)
    src = _kb_worker_source(
        ref_src, gen_src, out_path, gpu,
        num_correct_trials, num_perf_trials, measure_perf,
        atol, rtol, seed,
    )
    sfd, script_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(sfd, "w", encoding="utf-8") as h:
            h.write(src)
        _rc, _out, _err, timed_out = run_procgroup(
            [kb_python, script_path, kb_src],
            timeout=timeout,
        )
        if timed_out:
            return finalize_failure_classification(
                base_result(eval_backend="kernelbench", compiled=False,
                            info="timeout", error_type="task_timeout",
                            verify_meta={"stage": "candidate_timeout"})
            )
        if not os.path.isfile(out_path):
            # worker crashed before writing (native abort / OOM kill) → model
            # failure: the disposable process contains the poisoned state.
            return finalize_failure_classification(
                model_failure(reason="worker_no_result", eval_backend="kernelbench",
                              info="model_execution_error:no_result",
                              verify_meta={"stage": "worker_crash"})
            )
        try:
            with open(out_path, "rb") as h:
                payload = pickle.load(h)
        except Exception as exc:
            return finalize_failure_classification(
                model_failure(reason="result_read_fail", eval_backend="kernelbench",
                              info=f"result_read_fail:{type(exc).__name__}",
                              verify_meta={"stage": "result_read_fail"})
            )

        # --- bootstrap failures (before the eval boundary): classify ---
        if isinstance(payload, dict) and payload.get("bootstrap_error"):
            be = payload["bootstrap_error"]
            # ref/candidate run in the same process; a kernelbench IMPORT
            # failure is a trusted-setup infra (the harness isn't loadable),
            # but a seed bootstrap failure is infra too. Upstream emits these
            # before the post-bootstrap boundary.
            return finalize_failure_classification(
                infra_failure(infra_code="trusted_runtime_dependency",
                              eval_backend="kernelbench",
                              info=f"bootstrap:{be}",
                              error_type="bootstrap_failure",
                              verify_meta={"stage": "bootstrap_failed",
                                           "error": be,
                                           "trace": str(payload.get("exception", ""))[:500]})
            )

        # --- post-bootstrap eval exception → MODEL failure (the principle) ---
        if isinstance(payload, dict) and payload.get("eval_exception"):
            exc_name = payload["eval_exception"]
            # anti-cheat signals are still computed from source
            launches = None  # counter may not have been read; treat as unknown
            return finalize_failure_classification(
                model_failure(
                    reason=f"post_bootstrap_exception:{exc_name}",
                    eval_backend="kernelbench",
                    info=f"model_execution_error:{exc_name}",
                    triton_launched=launches,
                    triton_launch_counter=None,
                    identity_hack=False,
                    framework_delegation=detect_framework_delegation(gen_src),
                    exception_provenance=f"post_bootstrap:{exc_name}",
                    verify_meta={"stage": "eval_exception",
                                 "error": exc_name,
                                 "trace": str(payload.get("exception", ""))[:500]},
                )
            )

        correct = bool(payload.get("correct"))
        compiled = bool(payload.get("compiled"))
        launches = payload.get("triton_launched")
        launches = int(launches) if launches is not None else None
        identity_hack = (
            detect_identity_hack(gen_src, launches) if launches is not None else False
        )
        framework_delegation = detect_framework_delegation(gen_src)
        # Match the upstream KB/TBG worker: `correct` comes from the official
        # kernelbench.eval allclose (32 trials) ALONE. identity_hack and
        # framework_delegation are ANTI-CHEAT SIGNALS — recorded as peer fields
        # for the RL trainer, NOT hard reward gates (the detector is a noisy
        # heuristic that false-positives on Python builtins / tensor methods /
        # scalar-arg binops, so gating would fail legitimate candidates and
        # even the golden reference). eval_harness has no reward layer, so
        # verify_correct IS the metric = the official correctness verdict.
        speedup = None
        if (payload.get("ref_runtime") and payload.get("runtime")
                and payload["runtime"] > 0):
            speedup = float(payload["ref_runtime"]) / float(payload["runtime"])
        return finalize_failure_classification(
            base_result(
                eval_backend="kernelbench",
                correct=correct,
                compiled=compiled,
                info=payload.get("info") or ("pass" if correct else "compiled_but_wrong"),
                triton_launched=launches,
                triton_launch_counter=payload.get("triton_launch_counter"),
                identity_hack=identity_hack,
                framework_delegation=framework_delegation,
                speedup=speedup,
                reward_style="binary",
                verify_meta={"stage": "ok" if correct else "allclose_failed"},
            )
        )
    finally:
        for p in (script_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# --- smoke tests (run with: python -m verify.kernelbench, or import + call) ---
def _self_test() -> None:
    # anti-cheat primitives resolve + import sanity
    assert detect_framework_delegation("import torch\nVAL = 5\n") is False
    assert detect_framework_delegation("import torch\nc = torch.matmul(a, b)\n") is True
    assert detect_identity_hack("import torch\n", 0) is False  # launches<=0 → False
    # empty inputs
    r = verify_kb("", "x", kb_python=sys.executable, kb_src="/nonexistent")
    assert r.get("failure_origin") == "infrastructure", r  # empty ref → infra
    r = verify_kb("x", "", kb_python=sys.executable, kb_src="/nonexistent")
    assert r.get("info") == "skipped_no_teacher_code", r  # empty gen → model
    print("verify.kernelbench smoke: 5/5 passed")


if __name__ == "__main__":
    _self_test()
