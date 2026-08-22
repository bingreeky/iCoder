"""Shardable, resumable per-candidate stage execution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import socket
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .gates import static_tool_gates
from .judge import contract_gate
from .dedup import fingerprints, jaccard
from .difficulty import difficulty_gate
from .mutation import mutation_gate
from .generate import CandidateGenerator
from .ir import CandidateRecord, ChildTaskIR, GateResult
from .store import Lease, atomic_write_json, read_jsonl_tolerant


def task_key(task_id: str) -> str:
    digest = hashlib.sha256(task_id.encode()).hexdigest()[:16]
    readable = task_id.replace("/", "_").replace("::", "__").replace(",", "_").replace("=", "-")
    return f"{readable[:96]}-{digest}"


def in_shard(task_id: str, num_shards: int, shard_index: int) -> bool:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard configuration")
    value = int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16)
    return value % num_shards == shard_index


class ArtifactStore:
    def __init__(self, work_dir: Path):
        self.root = Path(work_dir) / "candidates"

    def directory(self, task_id: str) -> Path:
        return self.root / task_key(task_id)

    def record_path(self, task_id: str) -> Path:
        return self.directory(task_id) / "candidate.json"

    def lease(self, task_id: str, ttl_s: float) -> Lease:
        owner = f"{socket.gethostname()}:{os.getpid()}"
        return Lease(self.directory(task_id) / "claim.lease", owner, ttl_s)

    def load(self, task_id: str) -> Optional[CandidateRecord]:
        path = self.record_path(task_id)
        if not path.exists():
            return None
        import json
        return CandidateRecord.from_record(json.loads(path.read_text(encoding="utf-8")))

    def save(self, record: CandidateRecord) -> None:
        atomic_write_json(self.record_path(record.id), record.to_record())


def load_plan(path: Path) -> List[ChildTaskIR]:
    return [ChildTaskIR.from_record(row) for row in read_jsonl_tolerant(path)]


class StageRunner:
    def __init__(self, store: ArtifactStore, concurrency: int = 4, lease_ttl_s: float = 3600.0, tools: Optional[Dict[str, str]] = None):
        self.store = store
        self.semaphore = asyncio.Semaphore(concurrency)
        self.lease_ttl_s = lease_ttl_s
        self.tools = tools or {}

    async def generate_one(self, task: ChildTaskIR, generator: CandidateGenerator, resume: bool) -> str:
        existing = self.store.load(task.id)
        if resume and existing and existing.state not in {"pending", "ir_ready", "rejected"}:
            return "skipped"
        lease = self.store.lease(task.id, self.lease_ttl_s)
        if not lease.acquire():
            return "claimed"
        try:
            async with self.semaphore:
                record = await generator.generate(task)
                self.store.save(record)
                return record.state
        except Exception as exc:
            record = CandidateRecord(task.id, task, state="rejected", rejection_reason=f"generate_exception:{type(exc).__name__}:{exc}")
            self.store.save(record)
            return "rejected"
        finally:
            lease.release()

    async def gate_one(self, task: ChildTaskIR, retry: bool = False) -> str:
        record = self.store.load(task.id)
        if record is None:
            return "missing_candidate"
        if record.state == "accepted" or (record.gates and not retry):
            return "skipped"
        if retry and record.state == "rejected" and record.golden and record.tests:
            record.state = "test_ready"
        if record.state not in {"test_ready", "static_passed"}:
            return "wrong_state"
        lease = self.store.lease(task.id, self.lease_ttl_s)
        if not lease.acquire():
            return "claimed"
        try:
            async with self.semaphore:
                gates = await static_tool_gates(task.module_name, record.golden, record.tests, expected_ports=[str(port["name"]) for port in task.ports], verilator=self.tools.get("verilator", "verilator"), yosys=self.tools.get("yosys", "yosys"))
                record.gates = gates
                failed = next((gate for gate in gates if not gate.passed), None)
                if failed:
                    record.state = "rejected"
                    record.rejection_reason = f"{failed.gate}:{failed.reason}"
                else:
                    record.state = "static_passed"
                    record.rejection_reason = ""
                self.store.save(record)
                return record.state
        finally:
            lease.release()

    async def run_generate(self, tasks: Iterable[ChildTaskIR], generator: CandidateGenerator, resume: bool) -> Dict[str, int]:
        results = await asyncio.gather(*(self.generate_one(task, generator, resume) for task in tasks))
        return dict(sorted(Counter(results).items()))

    async def run_gate(self, tasks: Iterable[ChildTaskIR], retry: bool) -> Dict[str, int]:
        results = await asyncio.gather(*(self.gate_one(task, retry) for task in tasks))
        return dict(sorted(Counter(results).items()))
    async def mutation_one(self, task: ChildTaskIR, minimum: float, retry: bool) -> str:
        record = self.store.load(task.id)
        if record is None:
            return "missing_candidate"
        if record.state == "mutation_passed" and not retry:
            return "skipped"
        if retry and record.state == "mutation_passed":
            record.state = "static_passed"
        if retry and record.state == "rejected" and any(gate.gate == "mutation" for gate in record.gates):
            record.state = "static_passed"
        if record.state != "static_passed":
            return "wrong_state"
        result, outcomes = await mutation_gate(
            task.operator, record.golden, record.tests, minimum
        )
        record.gates = [gate for gate in record.gates if gate.gate != "mutation"] + [result]
        record.mutants = outcomes
        record.state = "mutation_passed" if result.passed else "rejected"
        record.rejection_reason = "" if result.passed else f"mutation:{result.reason}"
        self.store.save(record)
        return record.state

    async def contract_one(self, task: ChildTaskIR, judge, retry: bool) -> str:
        record = self.store.load(task.id)
        if record is None:
            return "missing_candidate"
        if record.state == "contract_passed" and not retry:
            return "skipped"
        if retry and record.state == "contract_passed":
            record.state = "mutation_passed"
        if retry and record.state == "rejected" and any(gate.gate == "contract" for gate in record.gates):
            record.state = "mutation_passed"
        if record.state != "mutation_passed":
            return "wrong_state"
        result = await contract_gate(record, judge)
        record.gates = [gate for gate in record.gates if gate.gate != "contract"] + [result]
        record.state = "contract_passed" if result.passed else "rejected"
        record.rejection_reason = "" if result.passed else f"contract:{result.reason}"
        self.store.save(record)
        return record.state

    def dedup_one(self, task: ChildTaskIR, accepted: List[CandidateRecord], prompt_threshold: float, rtl_threshold: float) -> str:
        record = self.store.load(task.id)
        if record is None:
            return "missing_candidate"
        if record.state != "contract_passed":
            return "wrong_state"
        current = fingerprints(record.prompt, record.golden, task.ports)
        duplicate = ""
        near = ""
        for other in accepted:
            other_fp = fingerprints(other.prompt, other.golden, other.task.ports)
            if current.prompt_hash == other_fp.prompt_hash or current.rtl_hash == other_fp.rtl_hash:
                duplicate = other.id
                break
            same_interface = current.interface_hash == other_fp.interface_hash
            if same_interface and (
                jaccard(record.prompt, other.prompt) >= prompt_threshold or
                jaccard(record.golden, other.golden) >= rtl_threshold
            ):
                near = other.id
                break
        if duplicate or near:
            result = GateResult(
                "dedup", False,
                f"{'exact' if duplicate else 'near'} duplicate of {duplicate or near}",
            )
            record.state = "rejected"
            record.rejection_reason = f"dedup:{result.reason}"
        else:
            result = GateResult(
                "dedup", True, metrics={"fingerprints": asdict(current)}
            )
            record.state = "dedup_passed"
            record.rejection_reason = ""
            accepted.append(record)
        record.gates = [gate for gate in record.gates if gate.gate != "dedup"] + [result]
        self.store.save(record)
        return record.state


    async def difficulty_one(self, task: ChildTaskIR, solver, retry: bool) -> str:
        record = self.store.load(task.id)
        if record is None:
            return "missing_candidate"
        if record.state == "difficulty_passed" and not retry:
            return "skipped"
        if retry and record.state == "difficulty_passed":
            record.state = "dedup_passed"
        if retry and record.state == "rejected" and any(
            gate.gate == "difficulty" for gate in record.gates
        ):
            record.state = "dedup_passed"
        if record.state != "dedup_passed":
            return "wrong_state"
        result, attempts = await difficulty_gate(record, solver)
        record.gates = [
            gate for gate in record.gates if gate.gate != "difficulty"
        ] + [result]
        record.solver_attempts = attempts
        record.state = "difficulty_passed" if result.passed else "rejected"
        record.rejection_reason = (
            "" if result.passed else f"difficulty:{result.reason}"
        )
        self.store.save(record)
        return record.state
