"""Immutable candidate planning with Direct-VE wave expansion."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from .contracts import CONTRACTS, MutationContract
from .ir import ChildTaskIR, RootTaskIR


WAVE_SIZE = {"A": 8, "B": 12, "C": 16}


def parameter_signature(parameters: Dict[str, object]) -> str:
    return ",".join(f"{key}={parameters[key]}" for key in sorted(parameters))


def child_ports(root: RootTaskIR, contract: MutationContract) -> List[Dict[str, object]]:
    ports = [dict(port) for port in root.ports]
    names = {str(port["name"]) for port in ports}
    for direction, name, width in contract.added_ports:
        if name not in names:
            ports.append({"direction": direction, "name": name, "width": width})
            names.add(name)
    return ports


class CandidatePlanner:
    """Plan reproducible siblings while respecting contract applicability."""

    def __init__(self, contracts: Dict[str, MutationContract] = CONTRACTS):
        self.contracts = contracts

    def plan_root(self, root: RootTaskIR, wave: str = "A") -> List[ChildTaskIR]:
        if wave not in WAVE_SIZE:
            raise ValueError(f"unknown wave {wave!r}; expected one of {sorted(WAVE_SIZE)}")
        applicable = [contract for contract in self.contracts.values() if contract.applicable(root)]
        if not applicable:
            raise ValueError(f"no mutation contract applies to {root.id}")
        seed = int(hashlib.sha256(root.id.encode()).hexdigest()[:16], 16)
        rotation = seed % len(applicable)
        applicable = applicable[rotation:] + applicable[:rotation]
        tasks: List[ChildTaskIR] = []
        seen_signatures = set()
        per_operator = Counter()
        ordinal = 0
        while len(tasks) < WAVE_SIZE[wave]:
            contract = applicable[ordinal % len(applicable)]
            op_ordinal = per_operator[contract.name]
            parameters = contract.parameters(op_ordinal)
            signature = parameter_signature(parameters)
            pair = (contract.name, signature)
            per_operator[contract.name] += 1
            ordinal += 1
            if pair in seen_signatures:
                if ordinal > 256:
                    raise ValueError(f"insufficient unique parameter combinations for {root.id}")
                continue
            seen_signatures.add(pair)
            task_id = f"{root.source}/{root.id.split('/', 1)[-1]}::{contract.name}::{signature}"
            tasks.append(ChildTaskIR(
                id=task_id,
                root_id=root.id,
                source=root.source,
                operator=contract.name,
                parameter_signature=signature,
                parameters=parameters,
                functional_contract=list(root.functional_contract) + list(contract.functional_delta),
                temporal_contract=list(root.temporal_contract) + list(contract.temporal_delta),
                test_obligations=list(contract.test_obligations),
                forbidden_mutations=list(contract.forbidden_mutations),
                module_name=root.module_name,
                ports=child_ports(root, contract),
                parent=root,
            ))
        return tasks

    def plan(self, roots: Iterable[RootTaskIR], wave: str = "A") -> List[ChildTaskIR]:
        tasks = [task for root in roots for task in self.plan_root(root, wave)]
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate plan contains duplicate task ids")
        return tasks

    @staticmethod
    def summary(tasks: Sequence[ChildTaskIR]) -> Dict[str, object]:
        return {
            "tasks": len(tasks),
            "roots": len({task.root_id for task in tasks}),
            "operators": dict(sorted(Counter(task.operator for task in tasks).items())),
            "plan_hash": hashlib.sha256(
                json.dumps([task.to_record() for task in tasks], sort_keys=True).encode()
            ).hexdigest(),
        }
