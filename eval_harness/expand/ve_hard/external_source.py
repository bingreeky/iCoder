"""Fail-closed adapters for externally supplied verified RTL roots."""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

from ..base import Seed
from .source import SourceAudit, build_roots


REQUIRED_MAPPING = ("id", "prompt", "golden", "tests")


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet sources") from exc
        yield from pq.read_table(path).to_pylist()
        return
    raise ValueError(f"unsupported source format: {path}")


def mapped_value(record: Mapping[str, Any], field: str, mapping: Mapping[str, str]) -> str:
    source_field = mapping[field]
    value = record.get(source_field)
    if value is None:
        return ""
    return str(value)


def load_external_seeds(
    source: str, path: Path, mapping: Mapping[str, str], license_name: str,
    verified_field: str, limit: int | None = None,
) -> Tuple[List[Seed], Dict[str, int]]:
    missing_mapping = [field for field in REQUIRED_MAPPING if not mapping.get(field)]
    if missing_mapping:
        raise ValueError(f"{source}: missing field mappings {missing_mapping}")
    if not license_name or license_name == "unknown":
        raise ValueError(f"{source}: explicit license is required")
    if not verified_field:
        raise ValueError(f"{source}: verified_field is required")
    seeds: List[Seed] = []
    rejected = Counter()
    for record in iter_records(path):
        if limit is not None and len(seeds) >= limit:
            break
        if record.get(verified_field) is not True:
            rejected["not_verified"] += 1
            continue
        values = {field: mapped_value(record, field, mapping) for field in REQUIRED_MAPPING}
        missing = [field for field, value in values.items() if not value.strip()]
        if missing:
            rejected["missing_" + "_".join(missing)] += 1
            continue
        seeds.append(Seed(
            id=f"{source}/{values['id']}", source_dataset=source,
            original_prompt=values["prompt"],
            reference_solution=values["golden"], tests=values["tests"],
            evaluator_info={"top_module": mapping.get("module_name", "TopModule"), "verified": True},
            metadata={"license": license_name, "source_record_id": values["id"]},
        ))
    return seeds, dict(sorted(rejected.items()))
