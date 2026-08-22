"""VerilogEval source audit and RootTaskIR construction."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ..base import Seed
from .ir import RootTaskIR


MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_]\w*)", re.IGNORECASE)
PORT_RE = re.compile(
    r"\b(input|output|inout)\b\s*(?:wire|reg|logic)?\s*"
    r"(?:\[\s*([^:\]]+)\s*:\s*([^\]]+)\s*\])?\s*([A-Za-z_]\w*)",
    re.IGNORECASE,
)
MODULE_HEADER_RE = re.compile(
    r"\bmodule\s+[A-Za-z_]\w*\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL
)
VERILOG_KEYWORDS = {
    "always", "and", "assign", "begin", "case", "else", "end", "endmodule",
    "for", "if", "inout", "input", "module", "not", "or", "output", "reg",
    "wire", "xor",
}


@dataclass(frozen=True)
class SourceAudit:
    seed_id: str
    accepted: bool
    issues: List[str]
    task_class: str
    module_name: str
    port_count: int
    features: Dict[str, Any]

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


def _width(msb: str | None, lsb: str | None) -> str:
    return "1" if msb is None else f"({msb})-({lsb})+1"


def extract_ports(text: str) -> List[Dict[str, Any]]:
    header = MODULE_HEADER_RE.search(text)
    if header:
        text = header.group(1)
    ports: List[Dict[str, Any]] = []
    seen = set()
    for match in PORT_RE.finditer(text):
        name = match.group(4)
        if name.lower() in VERILOG_KEYWORDS:
            continue
        if name in seen:
            continue
        seen.add(name)
        ports.append({
            "direction": match.group(1).lower(),
            "name": name,
            "width": _width(match.group(2), match.group(3)),
        })
    return ports


def static_features(seed: Seed) -> Dict[str, Any]:
    text = seed.reference_solution
    lower = text.lower()
    return {
        "always_blocks": len(re.findall(r"\balways\b", text, re.IGNORECASE)),
        "case_blocks": len(re.findall(r"\bcase[xz]?\b", text, re.IGNORECASE)),
        "has_clock": bool(re.search(r"\b(clk|clock)\b", lower)),
        "has_reset": bool(re.search(r"\b(rst|reset)\b", lower)),
        "has_state": bool(re.search(r"\b(state|fsm)\b", lower)),
        "has_counter": bool(re.search(r"\b(count|counter|timer|lfsr)\b", lower)),
        "has_stream": bool(re.search(r"\b(valid|ready|serial|data_in|data_out)\b", lower)),
        "has_memory": bool(re.search(r"\b(mem|memory|ram|fifo|address|addr)\b", lower)),
        "has_protocol": bool(re.search(r"\b(handshake|uart|ps2|hdlc|frame|packet|protocol)\b", lower)),
    }


def classify(features: Dict[str, Any]) -> str:
    if features["has_memory"]:
        return "memory_shared_resource"
    if features["has_protocol"] or features["has_state"]:
        return "fsm_protocol"
    if features["has_stream"] and features["has_clock"]:
        return "stream"
    if features["has_counter"] or features["has_clock"]:
        return "sequential_counter"
    return "combinational_arithmetic"


def audit_seed(seed: Seed) -> SourceAudit:
    text = seed.reference_solution
    features = static_features(seed)
    ports = extract_ports(text)
    module_match = MODULE_RE.search(text)
    module_name = str(seed.evaluator_info.get("top_module") or (
        module_match.group(1) if module_match else ""
    ))
    issues: List[str] = []
    if not seed.original_prompt.strip():
        issues.append("empty_prompt")
    if not seed.reference_solution.strip():
        issues.append("missing_golden")
    if not seed.tests.strip():
        issues.append("missing_test")
    if not module_name:
        issues.append("missing_module_name")
    if not ports:
        issues.append("missing_ports")
    return SourceAudit(
        seed_id=seed.id,
        accepted=not issues,
        issues=issues,
        task_class=classify(features),
        module_name=module_name,
        port_count=len(ports),
        features=features,
    )


def build_root(seed: Seed, license_name: str = "unknown") -> RootTaskIR:
    audit = audit_seed(seed)
    if not audit.accepted:
        raise ValueError(f"source {seed.id} rejected: {', '.join(audit.issues)}")
    text = seed.reference_solution
    ports = extract_ports(text)
    clock = next((p["name"] for p in ports if p["name"].lower() in {"clk", "clock"}), None)
    reset_port = next((p["name"] for p in ports if "reset" in p["name"].lower() or p["name"].lower() == "rst"), None)
    return RootTaskIR(
        id=seed.id,
        source=seed.source_dataset,
        prompt=seed.original_prompt,
        module_name=audit.module_name,
        ports=ports,
        functional_contract=["Implement every behavior stated in the parent VerilogEval specification."],
        temporal_contract=["Preserve the parent clock and reset semantics unless the child contract explicitly strengthens them."],
        golden=seed.reference_solution,
        tests=seed.tests,
        task_class=audit.task_class,
        clock=clock,
        reset={"port": reset_port, "polarity": "from_spec"} if reset_port else None,
        license=license_name,
        contamination_group=f"direct-ve:{seed.metadata.get('prob_id', seed.id)}",
        difficulty_tags=[audit.task_class],
        static_features=audit.features,
        lineage={"source_seed_id": seed.id, "source_metadata": seed.metadata},
    )


def build_roots(seeds: Iterable[Seed], license_name: str = "unknown") -> Tuple[List[RootTaskIR], List[SourceAudit]]:
    roots: List[RootTaskIR] = []
    audits: List[SourceAudit] = []
    for seed in seeds:
        audit = audit_seed(seed)
        audits.append(audit)
        if audit.accepted:
            roots.append(build_root(seed, license_name=license_name))
    return roots, audits


def audit_summary(audits: Iterable[SourceAudit]) -> Dict[str, Any]:
    values = list(audits)
    return {
        "total": len(values),
        "accepted": sum(a.accepted for a in values),
        "rejected": sum(not a.accepted for a in values),
        "issues": dict(sorted(Counter(i for a in values for i in a.issues).items())),
        "classes": dict(sorted(Counter(a.task_class for a in values if a.accepted).items())),
    }
