"""Operator-specific mutant generation and testbench kill scoring."""

from __future__ import annotations

import asyncio
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .ir import GateResult


@dataclass(frozen=True)
class Mutant:
    name: str
    core: bool
    rtl: str


def _replace_once(rtl: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, rtl, count=1, flags=re.IGNORECASE)


def generate_mutants(operator: str, rtl: str) -> List[Mutant]:
    """Create explicit likely faults; unchanged transformations are omitted."""
    specs: Dict[str, List[Tuple[str, str, str, bool]]] = {
        "controllerize_datapath": [
            ("done_stuck_high", r"assign\s+done\s*=\s*[^;]+;", "assign done = 1'b1;", True),
            ("done_stuck_low", r"assign\s+done\s*=\s*[^;]+;", "assign done = 1'b0;", True),
            ("latency_short", r"(counter\s*==\s*)\d+", r"\g<1>0", True),
            ("ignore_start", r"\bstart\b", "1'b1", True),
            ("busy_stuck_low", r"assign\s+busy\s*=\s*[^;]+;", "assign busy = 1'b0;", False),
        ],
        "add_handshake_backpressure": [
            ("ignore_output_stall", r"\bout_ready\b", "1'b1", True),
            ("never_ready", r"assign\s+in_ready\s*=\s*[^;]+;", "assign in_ready = 1'b0;", True),
            ("always_valid", r"assign\s+out_valid\s*=\s*[^;]+;", "assign out_valid = 1'b1;", True),
            ("drop_input_valid", r"\bin_valid\b", "1'b0", True),
            ("reset_disabled", r"\breset\b", "1'b0", False),
        ],
        "add_buffer_fifo": [
            ("full_stuck_low", r"assign\s+full\s*=\s*[^;]+;", "assign full = 1'b0;", True),
            ("empty_stuck_low", r"assign\s+empty\s*=\s*[^;]+;", "assign empty = 1'b0;", True),
            ("ignore_push", r"\bpush\b", "1'b0", True),
            ("ignore_pop", r"\bpop\b", "1'b0", True),
            ("reset_disabled", r"\breset\b", "1'b0", False),
        ],
        "add_timeout_retry_recovery": [
            ("timeout_never", r"assign\s+timeout_error\s*=\s*[^;]+;", "assign timeout_error = 1'b0;", True),
            ("retry_never", r"assign\s+retry\s*=\s*[^;]+;", "assign retry = 1'b0;", True),
            ("response_ignored", r"\bresponse_valid\b", "1'b0", True),
            ("request_forced", r"\brequest\b", "1'b1", True),
            ("reset_disabled", r"\breset\b", "1'b0", False),
        ],
        "add_arbitration_or_burst": [
            ("grant_stuck_zero", r"assign\s+grant\s*=\s*[^;]+;", "assign grant = 4'b0;", True),
            ("grant_stuck_one", r"assign\s+grant\s*=\s*[^;]+;", "assign grant = 4'b1;", True),
            ("requests_ignored", r"\brequest_valid\b", "4'b0", True),
            ("reset_disabled", r"\breset\b", "1'b0", False),
            ("priority_reversed", r"\[0\]", "[3]", True),
        ],
    }
    mutants = []
    for name, pattern, replacement, core in specs[operator]:
        changed = _replace_once(rtl, pattern, replacement)
        if changed != rtl:
            mutants.append(Mutant(name, core, changed))
    # Structure-preserving fallbacks mutate expressions/statements without
    # touching module headers.
    fallbacks = [
        ("invert_first_if", r"\bif\s*\(([^\n()]+)\)", r"if (!(\1))", True),
        ("force_first_if_false", r"\bif\s*\(([^\n()]+)\)", "if (1'b0)", True),
        ("force_first_if_true", r"\bif\s*\(([^\n()]+)\)", "if (1'b1)", True),
        ("invert_first_assign_rhs", r"(assign\s+\w+\s*=\s*)([^;]+);", r"\1~(\2);", True),
        ("zero_first_assign_rhs", r"(assign\s+\w+\s*=\s*)([^;]+);", r"\g<1>1'b0;", True),
        ("one_first_assign_rhs", r"(assign\s+\w+\s*=\s*)([^;]+);", r"\g<1>1'b1;", True),
        ("invert_first_nonblocking_rhs", r"(\w+\s*<=\s*)([^;]+);", r"\1~(\2);", True),
        ("zero_first_nonblocking_rhs", r"(\w+\s*<=\s*)([^;]+);", r"\g<1>1'b0;", True),
        ("one_first_nonblocking_rhs", r"(\w+\s*<=\s*)([^;]+);", r"\g<1>1'b1;", True),
    ]
    existing = {mutant.rtl for mutant in mutants}
    for name, pattern, replacement, core in fallbacks:
        changed = _replace_once(rtl, pattern, replacement)
        if changed != rtl and changed not in existing:
            mutants.append(Mutant(name, core, changed))
            existing.add(changed)
    return mutants


async def _mutant_survives(mutant: Mutant, tests: str, timeout_s: float) -> Tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="ve-hard-mutant-") as temp:
        root = Path(temp); rtl = root / "mutant.sv"; tb = root / "test.sv"; sim = root / "sim.out"
        rtl.write_text(mutant.rtl); tb.write_text(tests)
        try:
            compile_proc = await asyncio.create_subprocess_exec(
                "iverilog", "-g2012", "-o", str(sim), str(rtl), str(tb),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, compile_err = await asyncio.wait_for(compile_proc.communicate(), timeout_s)
        except asyncio.TimeoutError:
            return "invalid", "compile_timeout"
        if compile_proc.returncode != 0:
            return "invalid", "compile_failed:" + compile_err.decode(errors="replace")[-500:]
        run = await asyncio.create_subprocess_exec("vvp", str(sim), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(run.communicate(), timeout_s)
        except asyncio.TimeoutError:
            run.kill(); await run.wait(); return "killed", "simulation_timeout"
        output = (stdout + stderr).decode(errors="replace")
        survived = run.returncode == 0 and "Your Design Passed" in output and "MISMATCH" not in output
        return ("survived" if survived else "killed"), output[-500:]


async def mutation_gate(operator: str, rtl: str, tests: str, minimum: float = 0.8, timeout_s: float = 30.0) -> Tuple[GateResult, List[Dict[str, object]]]:
    mutants = generate_mutants(operator, rtl)
    if len(mutants) < 5:
        return GateResult("mutation", False, f"only {len(mutants)} effective mutants; require 5"), []
    outcomes = []
    for mutant in mutants:
        status, detail = await _mutant_survives(mutant, tests, timeout_s)
        outcomes.append({"name": mutant.name, "core": mutant.core, "status": status, "killed": status == "killed", "detail": detail})
    valid = [item for item in outcomes if item["status"] != "invalid"]
    killed = sum(bool(item["killed"]) for item in valid)
    score = killed / len(valid) if valid else 0.0
    core_survivors = [item["name"] for item in valid if item["core"] and not item["killed"]]
    passed = len(valid) >= 5 and score >= minimum and not core_survivors
    return GateResult(
        "mutation", passed,
        "" if passed else f"valid={len(valid)}, score={score:.3f}, core_survivors={core_survivors}",
        metrics={"score": score, "killed": killed, "valid": len(valid), "generated": len(outcomes), "core_survivors": core_survivors},
    ), outcomes
