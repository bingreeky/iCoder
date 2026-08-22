"""Fail-closed static and simulation gates for generated RTL triplets."""

from __future__ import annotations

import asyncio
import shutil
import re
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

from .ir import GateResult


async def _run(gate: str, command: Sequence[str], timeout_s: float) -> GateResult:
    started = time.monotonic()
    binary = command[0] if Path(command[0]).is_file() else shutil.which(command[0])
    if binary is None:
        return GateResult(gate=gate, passed=False, reason=f"required tool not found: {command[0]}")
    try:
        process = await asyncio.create_subprocess_exec(
            binary, *command[1:], stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return GateResult(gate=gate, passed=False, reason=f"timeout after {timeout_s}s", duration_s=time.monotonic() - started)
    text = (stdout + stderr).decode("utf-8", errors="replace")[-4000:]
    return GateResult(
        gate=gate, passed=process.returncode == 0,
        reason="" if process.returncode == 0 else text,
        metrics={"returncode": process.returncode, "output_tail": text},
        duration_s=time.monotonic() - started,
    )


KEYWORDS = {"input", "output", "inout", "wire", "reg", "logic", "module", "endmodule"}


def declared_ports(golden: str) -> List[str]:
    match = re.search(r"\bmodule\s+\w+\s*\((.*?)\)\s*;", golden, re.DOTALL)
    if not match:
        return []
    names = []
    for part in match.group(1).split(","):
        tokens = re.findall(r"[A-Za-z_]\w*", re.sub(r"\[[^\]]+\]", " ", part))
        tokens = [token for token in tokens if token.lower() not in KEYWORDS]
        if tokens:
            names.append(tokens[-1])
    return names


async def schema_interface_gate(module_name: str, golden: str, tests: str, expected_ports: Sequence[str] | None = None) -> GateResult:
    if not golden.strip() or not tests.strip():
        return GateResult("schema_interface", False, "empty golden or test")
    marker = f"module {module_name}"
    if marker not in golden:
        return GateResult("schema_interface", False, f"golden does not declare {marker}")
    actual = declared_ports(golden)
    if expected_ports is not None and actual != list(expected_ports):
        return GateResult("schema_interface", False, f"interface mismatch: expected {list(expected_ports)!r}, got {actual!r}")
    return GateResult("schema_interface", True)


async def static_tool_gates(
    module_name: str, golden: str, tests: str, timeout_s: float = 30.0,
    expected_ports: Sequence[str] | None = None,
    verilator: str = "verilator", yosys: str = "yosys",
) -> List[GateResult]:
    """Run schema, Icarus compile/sim, Verilator lint and Yosys check."""
    schema = await schema_interface_gate(module_name, golden, tests, expected_ports)
    if not schema.passed:
        return [schema]
    with tempfile.TemporaryDirectory(prefix="ve-hard-gate-") as temp:
        root = Path(temp)
        rtl = root / "golden.sv"
        tb = root / "test.sv"
        sim = root / "sim.out"
        rtl.write_text(golden, encoding="utf-8")
        tb.write_text(tests, encoding="utf-8")
        compile_result = await _run(
            "iverilog_compile", ("iverilog", "-g2012", "-o", str(sim), str(rtl), str(tb)), timeout_s
        )
        results = [schema, compile_result]
        if not compile_result.passed:
            return results
        simulation = await _run("iverilog_sim", ("vvp", str(sim)), timeout_s)
        output = str(simulation.metrics.get("output_tail", ""))
        if simulation.passed and (
            "Your Design Passed" not in output or "MISMATCH" in output
        ):
            simulation = GateResult(
                "iverilog_sim", False,
                "simulation did not emit an unambiguous success marker",
                metrics=simulation.metrics, duration_s=simulation.duration_s,
            )
        results.append(simulation)
        if not simulation.passed:
            return results
        results.append(await _run(
            "verilator_lint", (verilator, "--lint-only", "--language", "1800-2012", str(rtl)), timeout_s
        ))
        if not results[-1].passed:
            return results
        yosys_script = f"read_verilog -sv {rtl}; hierarchy -check -top {module_name}; proc; check"
        results.append(await _run("yosys_check", (yosys, "-q", "-p", yosys_script), timeout_s))
        return results
