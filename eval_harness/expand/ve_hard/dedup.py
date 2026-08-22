"""Deterministic fingerprints and near-duplicate screening."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def normalize_text(text: str) -> str:
    text = re.sub(r"//.*?$|/\*.*?\*/", " ", text, flags=re.MULTILINE | re.DOTALL)
    return " ".join(re.findall(r"[A-Za-z_]\w*|\d+'[bdhoBDHO][0-9a-fA-F_xXzZ]+|\d+|\S", text.lower()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def ngrams(text: str, n: int = 5) -> Set[Tuple[str, ...]]:
    tokens = normalize_text(text).split()
    return {tuple(tokens[index:index + n]) for index in range(max(0, len(tokens) - n + 1))}


def jaccard(left: str, right: str, n: int = 5) -> float:
    a, b = ngrams(left, n), ngrams(right, n)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def interface_hash(ports: Sequence[Dict[str, object]]) -> str:
    value = "|".join(
        f"{port['direction']}:{port['name']}:{port['width']}" for port in ports
    )
    return hashlib.sha256(value.encode()).hexdigest()


def control_signature(rtl: str) -> str:
    features = {
        "always": len(re.findall(r"\balways\b", rtl, re.IGNORECASE)),
        "case": len(re.findall(r"\bcase[xz]?\b", rtl, re.IGNORECASE)),
        "if": len(re.findall(r"\bif\s*\(", rtl, re.IGNORECASE)),
        "nonblocking": rtl.count("<="),
        "registers": len(re.findall(r"\b(reg|logic)\b", rtl, re.IGNORECASE)),
    }
    return hashlib.sha256(repr(sorted(features.items())).encode()).hexdigest()


@dataclass(frozen=True)
class Fingerprints:
    prompt_hash: str
    rtl_hash: str
    interface_hash: str
    control_signature: str


def fingerprints(prompt: str, rtl: str, ports: Sequence[Dict[str, object]]) -> Fingerprints:
    return Fingerprints(
        prompt_hash=sha256_text(prompt), rtl_hash=sha256_text(rtl),
        interface_hash=interface_hash(ports), control_signature=control_signature(rtl),
    )
