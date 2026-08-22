"""Coverage-first final selection with hard acceptance invariants."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .ir import CandidateRecord


def _difficulty_score(record: CandidateRecord) -> Tuple[float, float, str]:
    mutation = next((gate for gate in record.gates if gate.gate == "mutation"), None)
    difficulty = next((gate for gate in record.gates if gate.gate == "difficulty"), None)
    mutation_score = float(mutation.metrics.get("score", 0.0)) if mutation else 0.0
    solver_delta = float(difficulty.metrics.get("solver_delta", 0.0)) if difficulty else 0.0
    return solver_delta, mutation_score, record.id


def select_direct_ve(
    records: Iterable[CandidateRecord], target: int = 1000,
    min_roots: int = 140, per_root_cap: int = 8,
    per_root_operator_cap: int = 3, operator_cap: float = 0.40,
) -> List[CandidateRecord]:
    pool = [record for record in records if record.state == "difficulty_passed"]
    roots = {record.task.root_id for record in pool}
    if len(pool) < target:
        raise ValueError(f"only {len(pool)} difficulty-passed Direct VE candidates; require {target}")
    if len(roots) < min_roots:
        raise ValueError(f"only {len(roots)} covered roots; require {min_roots}")
    by_root: Dict[str, List[CandidateRecord]] = defaultdict(list)
    for record in pool:
        by_root[record.task.root_id].append(record)
    for values in by_root.values():
        values.sort(key=_difficulty_score, reverse=True)

    selected: List[CandidateRecord] = []
    root_counts = Counter(); root_op_counts = Counter(); op_counts = Counter()
    ordered_roots = sorted(by_root)
    while len(selected) < target:
        progress = False
        for root_id in ordered_roots:
            for record in by_root[root_id]:
                key = (root_id, record.task.operator)
                if record in selected:
                    continue
                if root_counts[root_id] >= per_root_cap:
                    break
                if root_op_counts[key] >= per_root_operator_cap:
                    continue
                if op_counts[record.task.operator] + 1 > int(target * operator_cap):
                    continue
                selected.append(record)
                root_counts[root_id] += 1; root_op_counts[key] += 1
                op_counts[record.task.operator] += 1; progress = True
                break
            if len(selected) == target:
                break
        if not progress:
            raise ValueError(
                f"selection constraints exhausted at {len(selected)}/{target}; "
                f"operators={dict(op_counts)} roots={len(root_counts)}"
            )
    if len(root_counts) < min_roots:
        raise ValueError(f"selection covers {len(root_counts)} roots; require {min_roots}")
    return selected


def accepted_record(record: CandidateRecord) -> Dict[str, object]:
    return {
        "id": record.id,
        "source": record.task.source,
        "root_id": record.task.root_id,
        "operator": record.task.operator,
        "parameters": record.task.parameters,
        "prompt": record.prompt,
        "reference_solution": record.golden,
        "tests": record.tests,
        "messages": [
            {"role": "user", "content": record.prompt},
            {"role": "assistant", "content": record.golden},
        ],
        "lineage": record.task.parent.lineage,
        "contamination_group": record.task.parent.contamination_group,
        "gates": [asdict(gate) for gate in record.gates],
        "solver_attempts": record.solver_attempts,
    }
