"""Fresh-solver difficulty comparison for parent and child tasks."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from ..llm import LLMRouter
from ..methods.benchevolver import extract_verilog, force_module_name
from .ir import CandidateRecord, GateResult


SOLVER_SYSTEM = (
    "Solve the RTL design specification from scratch. Output exactly one full "
    "iverilog -g2012 module in a verilog code fence, with no testbench."
)


async def verify_solution(rtl: str, tests: str, timeout_s: float = 60.0, oracle_rtl: str = "") -> Tuple[bool, str]:
    if not rtl.strip() or not tests.strip():
        return False, "empty RTL or test"
    with tempfile.TemporaryDirectory(prefix="ve-hard-solver-") as temp:
        root = Path(temp); solution = root / "solution.sv"; tb = root / "test.sv"; sim = root / "sim.out"; oracle = root / "oracle.sv"
        solution.write_text(rtl); tb.write_text(tests); oracle.write_text(oracle_rtl)
        compile_proc = await asyncio.create_subprocess_exec(
            "iverilog", "-g2012", "-o", str(sim), str(solution), str(oracle), str(tb),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(compile_proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            compile_proc.kill(); await compile_proc.wait(); return False, "compile_timeout"
        if compile_proc.returncode != 0:
            return False, "compile_failed:" + (stdout + stderr).decode(errors="replace")[-1000:]
        run = await asyncio.create_subprocess_exec("vvp", str(sim), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(run.communicate(), timeout_s)
        except asyncio.TimeoutError:
            run.kill(); await run.wait(); return False, "simulation_timeout"
        output = (stdout + stderr).decode(errors="replace")
        passed = run.returncode == 0 and (
            "Your Design Passed" in output or "Mismatches: 0" in output
        ) and "MISMATCH" not in output and not _positive_mismatches(output)
        return passed, output[-1000:]


def _positive_mismatches(output: str) -> bool:
    import re
    return any(int(value) > 0 for value in re.findall(r"Mismatches:\s*(\d+)", output, re.IGNORECASE))


async def solve_once(router: LLMRouter, prompt: str, module_name: str, tests: str, role: str, attempt: int, oracle_rtl: str = "") -> Dict[str, object]:
    raw = await router.chat_traj(SOLVER_SYSTEM, prompt)
    rtl = force_module_name(extract_verilog(raw), module_name)
    passed, detail = await verify_solution(rtl, tests, oracle_rtl=oracle_rtl)
    return {
        "role": role, "attempt": attempt,
        "model": getattr(router.traj_llm, "model", "?"),
        "passed": passed, "detail": detail, "rtl": rtl,
    }


async def difficulty_gate(record: CandidateRecord, solver: LLMRouter) -> Tuple[GateResult, List[Dict[str, object]]]:
    attempts: List[Dict[str, object]] = []
    for index in range(2):
        attempts.append(await solve_once(
            solver, record.prompt, record.task.module_name, record.tests,
            "teacher_child", index,
        ))
    teacher_passes = sum(bool(item["passed"]) for item in attempts)
    parent_attempt = await solve_once(
        solver, record.task.parent.prompt, record.task.parent.module_name,
        record.task.parent.tests, "panel_parent", 0,
        oracle_rtl=record.task.parent.golden,
    )
    child_attempt = await solve_once(
        solver, record.prompt, record.task.module_name, record.tests,
        "panel_child", 0,
    )
    attempts.extend((parent_attempt, child_attempt))
    panel_parent = int(bool(parent_attempt["passed"]))
    panel_child = int(bool(child_attempt["passed"]))
    passed = teacher_passes >= 1 and panel_child <= panel_parent - 1
    reason = "" if passed else (
        f"teacher_child_passes={teacher_passes}/2, panel_parent={panel_parent}, "
        f"panel_child={panel_child}; require teacher>=1 and child<=parent-1"
    )
    return GateResult(
        "difficulty", passed, reason,
        metrics={
            "teacher_child_passes": teacher_passes,
            "panel_parent_passes": panel_parent,
            "panel_child_passes": panel_child,
            "solver_delta": panel_parent - panel_child,
            "same_model_panel": True,
        },
    ), attempts
