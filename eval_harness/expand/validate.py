"""Basic validation for expanded JSONL and SFT JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from .base import REQUIRED_EXPANDED_FIELDS


def validate_expanded(path: Path) -> Tuple[int, int, list]:
    """Return (n_total, n_ok, errors)."""
    n = 0
    ok = 0
    errs: list = []
    with Path(path).open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append((lineno, f"json: {e}"))
                continue
            missing = [k for k in REQUIRED_EXPANDED_FIELDS
                       if not rec.get(k)]
            if missing:
                errs.append((lineno,
                             f"missing/empty fields: {missing}"))
                continue
            ok += 1
    return n, ok, errs


def validate_sft(path: Path) -> Tuple[int, int, list]:
    """Validate the ms-swift `messages` SFT format produced by
    scripts/convert_to_sft.py."""
    n = 0
    ok = 0
    errs: list = []
    with Path(path).open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append((lineno, f"json: {e}"))
                continue
            msgs = rec.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                errs.append((lineno, "messages: need ≥2 entries"))
                continue
            roles = [m.get("role") for m in msgs]
            if "user" not in roles or "assistant" not in roles:
                errs.append((lineno, f"messages: roles={roles}"))
                continue
            user_idx = next(i for i, m in enumerate(msgs)
                            if m.get("role") == "user")
            asst_idx = next(i for i, m in enumerate(msgs)
                            if m.get("role") == "assistant")
            if not (msgs[user_idx].get("content") or "").strip():
                errs.append((lineno, "empty user content"))
                continue
            if not (msgs[asst_idx].get("content") or "").strip():
                errs.append((lineno, "empty assistant content"))
                continue
            ok += 1
    return n, ok, errs
