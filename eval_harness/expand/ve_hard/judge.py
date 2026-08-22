"""Structured contract-consistency judge."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

from ..llm import LLMRouter
from .ir import CandidateRecord, GateResult


JUDGE_SYSTEM = (
    "You are a strict RTL contract auditor. Compare ChildTaskIR, generated "
    "Prompt, Golden RTL, and Testbench. Reject any interface mismatch, omitted "
    "functional or temporal obligation, Golden behavior contradicting the IR, "
    "or Testbench that does not check every listed test obligation. Return only "
    "JSON: {\"passed\": boolean, \"issues\": [string], "
    "\"covered_obligations\": [string]}. Never repair the artifacts."
)


def parse_judge_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        raise ValueError("judge did not return a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value.get("passed"), bool):
        raise ValueError("judge JSON missing boolean passed")
    if not isinstance(value.get("issues"), list):
        raise ValueError("judge JSON missing issues list")
    if not isinstance(value.get("covered_obligations"), list):
        raise ValueError("judge JSON missing covered_obligations list")
    return value


async def contract_gate(record: CandidateRecord, judge: LLMRouter) -> GateResult:
    request = json.dumps({
        "child_task_ir": record.task.to_record(),
        "prompt": record.prompt,
        "golden": record.golden,
        "testbench": record.tests,
    }, ensure_ascii=False, sort_keys=True)
    raw = await judge.chat_prompt(JUDGE_SYSTEM, request)
    try:
        result = parse_judge_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return GateResult("contract", False, f"invalid judge response: {exc}")
    expected = set(record.task.test_obligations)
    covered = set(str(item) for item in result["covered_obligations"])
    missing = sorted(expected - covered)
    passed = bool(result["passed"]) and not result["issues"] and not missing
    reason = "" if passed else json.dumps({"issues": result["issues"], "missing_obligations": missing}, ensure_ascii=False)
    return GateResult("contract", passed, reason, metrics={"judge": result})
