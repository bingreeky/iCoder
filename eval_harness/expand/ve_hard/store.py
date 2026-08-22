"""Shared-storage-friendly atomic artifacts and expiring task leases."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional


def read_jsonl_tolerant(path: Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                return
            raise


def atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return count


class Lease:
    def __init__(self, path: Path, owner: str, ttl_s: float = 3600.0):
        self.path = Path(path)
        self.owner = owner
        self.ttl_s = ttl_s

    def acquire(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"owner": self.owner, "created_at": now, "expires_at": now + self.ttl_s})
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if float(existing.get("expires_at", 0)) > now:
                return False
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return self.acquire(now=now)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def release(self) -> None:
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if existing.get("owner") == self.owner:
            self.path.unlink(missing_ok=True)
