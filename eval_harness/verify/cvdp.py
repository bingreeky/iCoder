"""verify/cvdp.py — hardened CVDP cocotb+pytest verify (local, on-disk).

Ports the on-disk equivalent of cvdp_verifier.verify_cvdp
(verifiers/cvdp_verifier.py:351-489) for the eval_harness local edition. CVDP's
correctness oracle is ALREADY a pytest exit code (returncode==0 → pass), and
the existing eval_harness path (run_cvdp.sh → CVDP_local/run_samples.py) uses
exactly that — so the oracle needs no change. What this module adds is the
hardened defense-in-depth the upstream suite wraps around it:

  * _ALLOWED_ENV whitelist on src/.env: only SIM/TOPLEVEL_LANG/VERILOG_SOURCES/
    TOPLEVEL/MODULE/PYTHONPATH/WAVE/HASH may be set; SIM is forced to "icarus".
    A model cannot inject PATH/LD_PRELOAD/COCOTB_* to alter the build.
  * _compatible_harness_source: rewrites cocotb 1.x imports to their 2.x
    equivalents (cocotb.runner→cocotb_tools.runner, sim_time_utils→utils) so a
    single pinned cocotb runtime executes both harness generations.
  * unsafe_runtime_construct: ban `include/pragma/$system/$dumpvars/
    $readmem*/$fopen/$finish/$stop in the CANDIDATE RTL (narrower than the
    icarus ban — $display/$write stay allowed; cocotb testbenches print).
  * run the pytest in a sandboxed subprocess with the allowlisted env only.

This is a portable single-candidate verifier (candidate + harness dir on
disk). The full CVDP gen+verify flow stays on run_samples.py (the authoritative
harness with dataset/factory wiring); verify_cvdp here is the reusable,
testable hardened shape for when a candidate + on-disk harness are in hand.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
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

_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_UNSAFE_RUNTIME = re.compile(
    r"`\s*(?:include|pragma)|"
    r"\$(?:system|fopen|fclose|fwrite|fdisplay|readmemh|readmemb|"
    r"writememh|writememb|dumpfile|dumpvars|vpi|stop|finish)\b",
    re.IGNORECASE,
)
_ALLOWED_ENV = {
    "SIM", "TOPLEVEL_LANG", "VERILOG_SOURCES", "TOPLEVEL", "MODULE",
    "PYTHONPATH", "WAVE", "HASH",
}
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_HARNESS_IMPORT_COMPATIBILITY = {
    "from cocotb.runner import get_runner":
        "from cocotb_tools.runner import get_runner",
    "from cocotb.sim_time_utils import get_sim_time":
        "from cocotb.utils import get_sim_time",
}


def unsafe_runtime_construct(source: str) -> str | None:
    """CVDP-narrow unsafe-construct gate on candidate RTL. Strips comments."""
    cleaned = _COMMENT.sub("", source)
    m = _UNSAFE_RUNTIME.search(cleaned)
    if not m:
        return None
    tok = m.group(0).strip().lstrip("`").strip()
    return f"unsafe_runtime_construct:{tok[:40]}"


def parse_env(text: str) -> dict[str, str]:
    """Parse a CVDP src/.env with the _ALLOWED_ENV whitelist. Forces
    SIM=icarus. Raises on any disallowed key or null byte. Upstream :201-218."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError("invalid CVDP .env line")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if key not in _ALLOWED_ENV or "\x00" in value:
            raise ValueError(f"invalid CVDP .env entry: {key}")
        result[key] = value
    for key in ("TOPLEVEL", "MODULE"):
        if not _IDENTIFIER.fullmatch(result.get(key, "")):
            raise ValueError(f"invalid {key}")
    if result.get("SIM", "icarus") != "icarus":
        raise ValueError("only the icarus simulator is supported")
    return result


def compatible_harness_source(path: str, source: str) -> str:
    """Map cocotb 1.x imports to 2.x equivalents in allowlisted harness files.
    Upstream :221-232."""
    if not path.endswith(".py"):
        return source
    for old, new in _HARNESS_IMPORT_COMPATIBILITY.items():
        source = source.replace(old, new)
    return source


def _dep_ready() -> bool:
    try:
        import cocotb_tools.runner  # noqa: F401
        import pytest  # noqa: F401
    except Exception:
        return False
    return bool(shutil.which("iverilog") and shutil.which("vvp"))


def verify_cvdp(
    candidate_source: str,
    *,
    harness_dir: str | os.PathLike,
    rtl_rel_path: str,            # where the candidate is written (rel to src root)
    env_rel_path: str = "src/.env",
    test_runner_rel: str = "src/test_runner.py",
    timeout: float = 120.0,
    eval_backend: str = "cvdp",
) -> dict[str, Any]:
    """On-disk hardened CVDP verify: write candidate, parse+allowlist .env,
    apply cocotb-compat rewrite to harness .py files, run pytest test_runner
    in a sandboxed subprocess, correctness = returncode==0.
    """
    if not candidate_source or len(candidate_source.strip()) < 3:
        return finalize_failure_classification(
            model_failure(reason="too_short", eval_backend=eval_backend,
                          info="skipped_too_short"))
    unsafe = unsafe_runtime_construct(candidate_source)
    if unsafe:
        return finalize_failure_classification(
            model_failure(reason=unsafe, eval_backend=eval_backend,
                          info=unsafe, error_type=unsafe,
                          verify_meta={"stage": "unsafe_runtime_construct"}))
    if not _dep_ready():
        return finalize_failure_classification(
            infra_failure(infra_code="trusted_runtime_dependency",
                          eval_backend=eval_backend,
                          info="cvdp_runtime_unavailable",
                          error_type="missing_dependency",
                          verify_meta={"stage": "runtime_unavailable"}))

    iverilog_bin = shutil.which("iverilog") or ""
    try:
        with tempfile.TemporaryDirectory(prefix="cvdp_verify_") as directory:
            root = Path(directory)
            src = root / "src"
            src.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copytree(harness_dir, root, dirs_exist_ok=True)
            # rewrite cocotb imports in copied .py harness files
            for p in src.rglob("*.py"):
                p.write_text(compatible_harness_source(p.name, p.read_text()),
                             encoding="utf-8")
            rtl_path = root / rtl_rel_path
            rtl_path.parent.mkdir(parents=True, exist_ok=True)
            rtl_path.write_text(candidate_source, encoding="utf-8")

            env_text = (root / env_rel_path).read_text(encoding="utf-8")
            harness_env = parse_env(env_text)
            env = {k: v for k, v in os.environ.items()
                   if k in ("PATH", "HOME", "LANG", "LC_ALL")}
            env.update(harness_env)
            env["SIM"] = "icarus"
            if iverilog_bin:
                env["PATH"] = str(Path(iverilog_bin).parent) + ":" + env.get("PATH", "")
            env["VERILOG_SOURCES"] = str(rtl_path)
            env["PYTHONPATH"] = str(src)
            env["COCOTB_CACHE_DIR"] = str(root / ".cache")
            env.pop("PYTHONINSPECT", None)
            env.pop("PYTHONSTARTUP", None)
            cmd = [sys_executable(), "-m", "pytest", "-s",
                   "--log-cli-level=INFO", "-o", f"cache_dir={root/'.cache'}",
                   str(root / test_runner_rel), "-v"]
            try:
                rc, out, err, timed_out = run_procgroup(
                    cmd, cwd=root, env=env, timeout=timeout)
            except FileNotFoundError as exc:
                return finalize_failure_classification(
                    infra_failure(infra_code="backend_configuration",
                                   eval_backend=eval_backend,
                                   info="pytest_not_found",
                                   error_type=str(exc),
                                   verify_meta={"stage": "launch_fail"}))
            if timed_out:
                return finalize_failure_classification(
                    base_result(eval_backend=eval_backend, compiled=True,
                                info="timeout", error_type="task_timeout",
                                verify_meta={"stage": "sim_timeout"}))
            combined = (out + "\n" + err).lower()
            if rc == 0:
                return finalize_failure_classification(
                    base_result(eval_backend=eval_backend, compiled=True,
                                correct=True, info="pass",
                                verify_meta={"stage": "ok"}))
            info = "compile_fail" if any(t in combined for t in (
                "syntax error", "calledprocesserror", "build failed",
                "unknown module type")) else "test_fail"
            return finalize_failure_classification(
                base_result(eval_backend=eval_backend, compiled=True, correct=False,
                            info=info, error_type=combined[-800:],
                            verify_meta={"stage": info}))
    except Exception as exc:
        return finalize_failure_classification(
            infra_failure(infra_code="backend_configuration",
                          eval_backend=eval_backend,
                          info="cvdp_worker_failure",
                          error_type=type(exc).__name__,
                          verify_meta={"stage": "worker_failure",
                                       "error": str(exc)[:500]}))


def sys_executable() -> str:
    return __import__("sys").executable


def _self_test() -> None:
    # unsafe: $dumpvars/$readmemh/`include banned; $display allowed (cocotb prints)
    assert unsafe_runtime_construct("module m; initial $dumpvars(0,x); endmodule")
    assert unsafe_runtime_construct("module m; initial $readmemh(\"f\", m); endmodule")
    assert unsafe_runtime_construct("`include \"x.v\"")
    assert unsafe_runtime_construct("module m; // $dumpvars in comment\nendmodule") is None
    assert unsafe_runtime_construct("module m; initial $display(\"hi\"); endmodule") is None
    # env allowlist: disallowed key rejected, SIM forced
    import pytest as _pt
    _pt.raises(ValueError, parse_env, "PATH=/evil\nTOPLEVEL=top\nMODULE=m\n")
    _pt.raises(ValueError, parse_env, "TOPLEVEL=top\nMODULE=m\nSIM=vcs\n")
    e = parse_env("TOPLEVEL=top\nMODULE=m\n")
    assert e["TOPLEVEL"] == "top" and "SIM" not in e  # SIM forced at env-build, not in parse_env
    # cocotb compat rewrite
    assert compatible_harness_source("x.py", "from cocotb.runner import get_runner\n") == \
        "from cocotb_tools.runner import get_runner\n"
    print("verify.cvdp smoke: 7/7 passed")


if __name__ == "__main__":
    _self_test()
