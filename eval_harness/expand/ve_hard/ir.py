"""Versioned task and result records for the VE-hard pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION = "ve-hard.v2.1"
STATES = (
    "pending", "ir_ready", "golden_ready", "test_ready", "static_passed",
    "mutation_passed", "contract_passed", "dedup_passed",
    "difficulty_passed", "accepted", "rejected",
)


def stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RootTaskIR:
    id: str
    source: str
    prompt: str
    module_name: str
    ports: List[Dict[str, Any]]
    functional_contract: List[str]
    temporal_contract: List[str]
    golden: str
    tests: str
    task_class: str
    clock: Optional[str] = None
    reset: Optional[Dict[str, Any]] = None
    license: str = "unknown"
    contamination_group: str = ""
    difficulty_tags: List[str] = field(default_factory=list)
    static_features: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RootTaskIR":
        obj = cls(**dict(record))
        if obj.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported RootTaskIR schema {obj.schema_version!r}")
        return obj


@dataclass(frozen=True)
class ChildTaskIR:
    id: str
    root_id: str
    source: str
    operator: str
    parameter_signature: str
    parameters: Dict[str, Any]
    functional_contract: List[str]
    temporal_contract: List[str]
    test_obligations: List[str]
    forbidden_mutations: List[str]
    module_name: str
    ports: List[Dict[str, Any]]
    parent: RootTaskIR
    schema_version: str = SCHEMA_VERSION

    def to_record(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task_hash"] = self.task_hash
        return data

    @property
    def task_hash(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ChildTaskIR":
        data = dict(record)
        expected = data.pop("task_hash", None)
        data["parent"] = RootTaskIR.from_record(data["parent"])
        obj = cls(**data)
        if obj.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported ChildTaskIR schema {obj.schema_version!r}")
        if expected and expected != obj.task_hash:
            raise ValueError(f"task hash mismatch for {obj.id}")
        return obj


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class CandidateRecord:
    id: str
    task: ChildTaskIR
    state: str = "pending"
    prompt: str = ""
    golden: str = ""
    tests: str = ""
    mutants: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[GateResult] = field(default_factory=list)
    solver_attempts: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"invalid candidate state {self.state!r}")

    def to_record(self) -> Dict[str, Any]:
        data = asdict(self)
        data["record_hash"] = stable_hash(data)
        return data

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CandidateRecord":
        data = dict(record)
        expected = data.pop("record_hash", None)
        data["task"] = ChildTaskIR.from_record(data["task"])
        data["gates"] = [GateResult(**gate) for gate in data.get("gates", [])]
        obj = cls(**data)
        actual = obj.to_record()["record_hash"]
        if expected and expected != actual:
            raise ValueError(f"candidate record hash mismatch for {obj.id}")
        return obj
