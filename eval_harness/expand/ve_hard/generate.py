"""TaskIR-first generation using separate golden, test and prompt roles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..llm import LLMRouter
from ..methods.benchevolver import (
    extract_verilog, force_module_name, iverilog_compile_pair,
    iverilog_syntax_check,
)
from .ir import CandidateRecord, ChildTaskIR


GOLDEN_SYSTEM = (
    "You are the Golden RTL author. Implement only the supplied ChildTaskIR. "
    "The ChildTaskIR module_name and ports are an exact closed interface: use "
    "every listed port with the exact direction, name, and width; do not add, "
    "remove, or rename ports. Never use a Verilog keyword as an identifier. "
    "Output one complete iverilog -g2012 module in a verilog fence."
)
TEST_SYSTEM = (
    "You are an independent adversarial RTL test writer. Treat ChildTaskIR as "
    "the source of truth and emit one deterministic self-checking testbench in "
    "a verilog fence. Instantiate every port in the exact closed child interface "
    "by name and do not invent ports. Print MISMATCH on "
    "failure and Your Design Passed on success. Instantiate exactly one DUT. "
    "Do not instantiate RefModule, a second DUT, or any undefined helper module; "
    "implement the independent behavioral oracle inside the testbench. For "
    "synchronous state, assert reset for at least two active clock edges and do "
    "not compare state-derived outputs before the first clock edge after reset."
)
PROMPT_SYSTEM = (
    "You write VerilogEval-style specifications from ChildTaskIR. State the exact "
    "module name, every port and width, reset/clock rules, functional behavior, "
    "and cycle-level temporal contract. Output plain text only."
)


@dataclass(frozen=True)
class RoleModels:
    golden: str
    test: str
    prompt: str

    def validate(self) -> None:
        if not all((self.golden, self.test, self.prompt)):
            raise ValueError("all generation role model names are required")


def task_payload(task: ChildTaskIR) -> str:
    return json.dumps(task.to_record(), ensure_ascii=False, sort_keys=True, indent=2)


class CandidateGenerator:
    def __init__(self, golden_router: LLMRouter, test_router: LLMRouter, prompt_router: LLMRouter):
        self.golden_router = golden_router
        self.test_router = test_router
        self.prompt_router = prompt_router

    async def generate(self, task: ChildTaskIR) -> CandidateRecord:
        payload = task_payload(task)
        golden = ""
        golden_error = ""
        for _ in range(3):
            request = payload
            if golden_error:
                request += ("\n\nPrevious RTL failed iverilog:\n" + golden_error +
                            "\nFix and emit the full module.")
            golden_raw = await self.golden_router.chat_traj(GOLDEN_SYSTEM, request)
            golden = force_module_name(extract_verilog(golden_raw), task.module_name)
            if not golden:
                golden_error = "no parseable Verilog module"
                continue
            ok, golden_error = await iverilog_syntax_check(golden, task.module_name)
            if ok:
                break
        if not golden:
            return CandidateRecord(task.id, task, state="rejected", rejection_reason="golden_parse_failed")
        if golden_error:
            return CandidateRecord(task.id, task, state="rejected", golden=golden, rejection_reason=f"golden_compile_failed:{golden_error}")

        seed_hash = int(hashlib.sha256(task.id.encode()).hexdigest()[:8], 16)
        tests = ""
        test_error = ""
        base_test_request = (
            f"ChildTaskIR:\n{payload}\n\nGolden RTL:\n```verilog\n{golden}\n```"
        )
        for attempt in range(3):
            request = base_test_request
            if test_error:
                request += ("\n\nPrevious testbench failed when compiled with the Golden:\n" +
                            test_error +
                            "\nFix only the testbench and emit it in full.")
            test_raw = await self.test_router.chat_prompt(
                TEST_SYSTEM, request, seed_hash=seed_hash, variant_idx=attempt,
            )
            tests = extract_verilog(test_raw)
            if not tests:
                test_error = "no parseable Verilog testbench"
                continue
            ok, test_error = await iverilog_compile_pair(
                tests, golden, task.module_name
            )
            if ok:
                break
        if not tests:
            return CandidateRecord(task.id, task, state="rejected", golden=golden, rejection_reason="test_parse_failed")
        if test_error:
            return CandidateRecord(task.id, task, state="rejected", golden=golden, tests=tests, rejection_reason=f"test_compile_failed:{test_error}")

        prompt = (await self.prompt_router.chat_prompt(
            PROMPT_SYSTEM, payload, seed_hash=seed_hash
        )).strip()
        if not prompt:
            return CandidateRecord(task.id, task, state="rejected", golden=golden, tests=tests, rejection_reason="prompt_parse_failed")
        return CandidateRecord(task.id, task, state="test_ready", prompt=prompt, golden=golden, tests=tests)
