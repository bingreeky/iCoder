"""Shared types and IO helpers for expansion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class Seed:
    """Normalised seed sample produced by a dataset adapter."""

    id: str
    source_dataset: str
    original_prompt: str
    reference_solution: str = ""
    expected_output: Optional[str] = None
    tests: str = ""
    evaluator_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Expanded:
    """Output schema for one expanded sample (jsonl row)."""

    id: str
    source_dataset: str
    expansion_method: str
    original_prompt: str
    expanded_prompt: str
    reference_solution: str = ""
    expected_output: Optional[str] = None
    tests: str = ""
    evaluator_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


REQUIRED_EXPANDED_FIELDS = (
    "id",
    "source_dataset",
    "expansion_method",
    "original_prompt",
    "expanded_prompt",
)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def take(iterable: Iterable, limit: Optional[int]) -> List:
    if limit is None:
        return list(iterable)
    out = []
    for i, item in enumerate(iterable):
        if i >= limit:
            break
        out.append(item)
    return out
