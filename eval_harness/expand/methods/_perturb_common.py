"""Shared helpers for KB/TBG × {BenchEvolver, Evol-Instruct} minor-perturbation
pipelines.

Purpose: take a KernelBench seed (PyTorch ref) and produce a *minor functional
perturbation* — small enough that the variant ref is recognisable as a
descendant of the original, large enough that v4-pro can write a Triton kernel
matching the new behaviour and the KB harness can verify it.

This module owns:

  * The operator catalogue ``EVOL_OPERATORS_KB`` (7 mutation contracts,
    sharing the BenchEvolver shape).
  * The 5 validation gates (g1 format → g2 forward-body diff → g3 smoke →
    g5 Triton lint → g4 behavioural diff), run in FAIL-FAST order.
  * A behavioural-diff driver that runs original + variant refs in ONE
    subprocess (single torch import) over N seeded fuzz trials.
  * A few small AST helpers (extract forward-body source, lint forward body
    for Triton-incompatible Python idioms).

Reused (not re-implemented):

  * ``_extract_pytorch_interface``, ``_ref_smoke_test`` from
    :mod:`expand.methods.inversecoder` — interface extraction and the cold
    subprocess smoke driver.
  * ``diff_ratio`` from :mod:`expand.methods.benchevolver` — char-level
    SequenceMatcher fraction, used on the forward body (NOT the whole file)
    via :func:`forward_body_diff_ratio`.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import os
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchevolver import diff_ratio
from .inversecoder import (
    PyTorchInterface,
    _extract_pytorch_interface,
    _ref_smoke_test,
)


# ---------- global CUDA-cold-bound gate semaphore + GPU round-robin --------
#
# Both g3 smoke AND g4 behavioural-diff drivers spawn fresh subprocesses
# that pay full CUDA cold-init + lazy-load cost (25-50s per subprocess).
# Running many of these concurrently on the SAME GPU makes each one
# slower (NVIDIA driver context creation contention at the OS level),
# driving individual gates past their timeout.
#
# Two-pronged fix:
#   1. Global semaphore caps concurrent CUDA-cold-bound subprocesses.
#   2. GPU round-robin: each subprocess is pinned to a different physical
#      GPU via CUDA_VISIBLE_DEVICES. On an 8-GPU host with sem=8, 8 cold
#      inits proceed in true parallel because each hits a different
#      driver context.
#
# Env vars:
#   KB_G3_SMOKE_CONCURRENCY  shared semaphore size (default = NUM_GPUS or 1)
#   KB_NUM_GPUS              # physical GPUs available; subprocess i is
#                            pinned to GPU (i % NUM_GPUS). Default
#                            auto-detected via nvidia-smi; falls back to 1.

import subprocess as _sp
import itertools as _itertools


def _detect_num_gpus() -> int:
    """Probe nvidia-smi -L | wc -l once. Returns 1 on any failure (CPU host)."""
    env_override = os.environ.get("KB_NUM_GPUS")
    if env_override:
        return max(1, int(env_override))
    try:
        out = _sp.check_output(
            ["nvidia-smi", "-L"], stderr=_sp.DEVNULL, timeout=5
        ).decode()
        n = len([ln for ln in out.splitlines() if ln.strip()])
        return max(1, n)
    except Exception:
        return 1


_KB_NUM_GPUS = _detect_num_gpus()
_KB_G3_SMOKE_CONCURRENCY = int(
    os.environ.get("KB_G3_SMOKE_CONCURRENCY", str(_KB_NUM_GPUS)))
_g3_smoke_sem: Optional[asyncio.Semaphore] = None

# Atomic round-robin counter for CUDA_VISIBLE_DEVICES assignment.
_gpu_counter = _itertools.count()


def _get_g3_sem() -> asyncio.Semaphore:
    """Shared semaphore for g3 smoke + g4 behavioural-diff subprocess gates.
    Must be created inside the running event loop (lazy init)."""
    global _g3_smoke_sem
    if _g3_smoke_sem is None:
        _g3_smoke_sem = asyncio.Semaphore(_KB_G3_SMOKE_CONCURRENCY)
    return _g3_smoke_sem


def _next_gpu_env() -> Dict[str, str]:
    """Return an env dict that pins the next subprocess to a specific GPU
    via CUDA_VISIBLE_DEVICES round-robin. Merges with current os.environ
    so the subprocess inherits all other vars (LLM keys, PYTHONPATH, etc.).

    On a 1-GPU host this returns env with CUDA_VISIBLE_DEVICES=0 — same
    as not setting it. Safe no-op fallback."""
    idx = next(_gpu_counter)
    gpu = idx % _KB_NUM_GPUS
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


# ============================================================================
# Persistent gate worker pool
# ============================================================================
#
# Replaces per-gate subprocess fork (torch import + cuda.init = 5-10s) with
# a long-lived pool of workers (init once at startup, then process tasks
# from a queue). Same protocol for both kb_smoke / kb_behavioral / tbg_*
# gate types (see expand/methods/_gate_worker.py).
#
# Enabled by env var KB_GATE_WORKERS=N (or auto = num_gpus). Set to 0 to
# disable and fall back to subprocess-per-call.

_KB_GATE_WORKERS = int(os.environ.get("KB_GATE_WORKERS",
                                      str(_KB_NUM_GPUS)))
_gate_pool: Optional["_GateWorkerPool"] = None
_gate_pool_lock = asyncio.Lock()


class _GateWorkerPool:
    """N persistent worker subprocesses + per-worker lock for serialized
    stdin/stdout (the JSON protocol is request-response on a single pipe).
    Round-robin task dispatch; per-call timeout restarts the worker.

    Workers are created lazily on first call. They pin to GPU
    (idx % num_gpus) via CUDA_VISIBLE_DEVICES set in the env at spawn."""

    def __init__(self, n_workers: int, num_gpus: int):
        self.n_workers = n_workers
        self.num_gpus = num_gpus
        self.workers: List[asyncio.subprocess.Process] = []
        self.locks: List[asyncio.Lock] = []
        self._rr = 0

    async def start(self) -> None:
        if self.workers:
            return
        worker_script = str(Path(__file__).parent / "_gate_worker.py")
        for i in range(self.n_workers):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(i % self.num_gpus)
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, worker_script,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            except FileNotFoundError as e:
                raise RuntimeError(f"gate worker launch failed: {e}")
            # Block until worker prints READY (torch + cuda.init done).
            try:
                ready = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError(f"gate worker {i} READY timeout 120s")
            if ready.strip() != b"READY":
                raise RuntimeError(
                    f"gate worker {i} bad ready line: {ready!r}")
            self.workers.append(proc)
            self.locks.append(asyncio.Lock())

    async def _restart_worker(self, idx: int) -> None:
        """Kill + respawn a single worker (e.g. after timeout)."""
        proc = self.workers[idx]
        try:
            proc.kill()
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass
        worker_script = str(Path(__file__).parent / "_gate_worker.py")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(idx % self.num_gpus)
        new_proc = await asyncio.create_subprocess_exec(
            sys.executable, worker_script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        await asyncio.wait_for(new_proc.stdout.readline(), timeout=120)
        self.workers[idx] = new_proc

    async def call(self, op: str, timeout: float, **kwargs) -> Dict[str, Any]:
        """Send a task to the next round-robin worker. Returns the parsed
        response dict (with 'ok' / 'result' / 'err' keys)."""
        await self.start()
        idx = self._rr
        self._rr = (self._rr + 1) % self.n_workers
        proc = self.workers[idx]
        lock = self.locks[idx]
        async with lock:
            task = {"task_id": f"t{idx}_{id(kwargs)}", "op": op, **kwargs}
            try:
                proc.stdin.write((json.dumps(task) + "\n").encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # Worker died. Restart and bail with error (caller retries
                # at variant level via MAX_PERTURB_ATTEMPTS).
                await self._restart_worker(idx)
                return {"ok": False, "err": f"worker {idx} pipe broken"}
            try:
                resp_line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                # Worker stuck (infinite loop / CUDA hang). Restart it.
                await self._restart_worker(idx)
                return {"ok": False, "err": f"worker {idx} timeout {timeout}s"}
            if not resp_line:
                # Worker died mid-task without writing response.
                await self._restart_worker(idx)
                return {"ok": False, "err": f"worker {idx} EOF"}
            try:
                return json.loads(resp_line)
            except json.JSONDecodeError as e:
                return {"ok": False, "err": f"bad JSON from worker: {e}"}


async def _get_gate_pool() -> Optional["_GateWorkerPool"]:
    """Lazy singleton. Returns None if disabled via env (= 0 workers)."""
    global _gate_pool
    if _KB_GATE_WORKERS <= 0:
        return None
    if _gate_pool is None:
        async with _gate_pool_lock:
            if _gate_pool is None:
                _gate_pool = _GateWorkerPool(_KB_GATE_WORKERS, _KB_NUM_GPUS)
                await _gate_pool.start()
    return _gate_pool


# Ensure json is imported (we use it in the pool above).
import json  # noqa: E402,F811


# ---------- operator catalogue (mutation contracts) -------------------------
#
# Each operator is a contract fed verbatim into the LLM rewrite prompt. Shape
# matches BenchEvolver's: name / definition / scope / target_hint /
# do_not_change / old_property / new_property — minus ``test_must_check``
# because KB verify is allclose-based and doesn't need a separate testbench.
#
# All operators preserve ``forward()`` signature and ``get_inputs()``: the
# input contract is stable. Most operators insert ONE statement near the end
# of forward (or swap ONE literal) so the source-code diff stays small and
# the change is expressible in Triton.

EVOL_OPERATORS_KB: List[Dict[str, Any]] = [
    {
        "name": "range_offset",
        "definition": (
            "After computing the original output `y`, ADD a small constant "
            "offset `c` to elements where the FIRST tensor input value "
            "(elementwise) falls in a fixed interval [a, b]. Pick `c` "
            "small but non-trivial (0.05 to 0.2). Pick [a, b] so that "
            "approximately 10-20% of values from torch.rand-style fuzz "
            "inputs land in range — for a uniform [0,1] input, [a, b] = "
            "[0.3, 0.45] is a reasonable default; adjust based on the "
            "obvious input distribution."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return in "
            "forward(). You may name [a, b, c] as local constants. The "
            "mask must be elementwise (matching the FIRST input tensor's "
            "shape after any broadcasting needed) — do NOT use a Python "
            "if/while or a data-dependent loop."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body and any nn.Linear / Parameter / submodule allocations",
            "the original math producing y",
        ],
        "old_property": "forward returns y as in the reference implementation",
        "new_property": (
            "forward returns y + c * mask, where mask is 1 where the first "
            "input is in [a, b] and 0 elsewhere"),
    },
    {
        "name": "partial_activation",
        "definition": (
            "After computing the original output `y`, apply ONE additional "
            "cheap activation (choose ONE of `.clamp_min(0)`, `.abs()`, "
            "`.tanh()`, `.relu()`) to the LAST K% of channels along the "
            "LAST dimension of y. Pick K = 10 (i.e. last 1/10 of the last "
            "dimension)."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return. Use "
            "a slice assignment (e.g. `y[..., -y.shape[-1]//10:] = "
            "y[..., -y.shape[-1]//10:].clamp_min(0)`) or its functional "
            "equivalent. Do NOT use a Python for-loop or list slicing."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body and any submodule allocations",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with its last 10% of channels along dim=-1 "
            "passed through one additional activation"),
    },
    {
        "name": "boundary_clamp",
        "definition": (
            "After computing the original output `y`, apply a clamp with "
            "ONE one-sided threshold (either `.clamp(max=T)` OR "
            "`.clamp(min=T)`, choose ONE). T must be near the empirical p90 "
            "(or p10 if clamping min) of the original output distribution. "
            "For tasks with bounded-range outputs (e.g. sigmoid, softmax), "
            "pick T accordingly; for unbounded outputs, T = 2.0 (max) is a "
            "reasonable default."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return: "
            "`y = y.clamp(max=T)` (or `min=T`). No Python control flow."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unbounded as in the reference",
        "new_property": (
            "forward returns y clamped at T on one side"),
    },
    {
        "name": "scale_subregion",
        "definition": (
            "After computing the original output `y`, scale a fixed sub-slice "
            "by a constant `s` close to 1 (one of 1.05, 1.1, or 0.9). The "
            "sub-slice should cover roughly 10-15% of y — for a 4-D tensor "
            "this is `y[:, :, :, :y.shape[-1]//8]`; for a 2-D tensor this "
            "is `y[:, :y.shape[-1]//8]`; adapt to y's actual rank."),
        "scope": (
            "Insert ONE additional in-place multiply statement BEFORE the "
            "final return: `y[<slice>] = y[<slice>] * s` (or the equivalent "
            "out-of-place form). No Python for-loops, no data-dependent "
            "slicing."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with a fixed corner/sub-slice scaled by s"),
    },
    {
        "name": "region_invert",
        "definition": (
            "After computing the original output `y`, NEGATE a fixed "
            "sub-slice along the LAST dimension (the first 1/8 of the "
            "last dim). The slice is independent of input values."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return: "
            "`y[..., :y.shape[-1]//8] = -y[..., :y.shape[-1]//8]` "
            "(or its out-of-place form). No Python for-loops."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with its first 1/8 of the last dim negated"),
    },
    {
        "name": "region_shift",
        "definition": (
            "After computing the original output `y`, add a small constant "
            "`b` (e.g. 0.1) to a fixed sub-slice along the LAST dimension "
            "(the last 1/6 of the last dim). NOT input-conditional — just "
            "an additive shift on a fixed region."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return: "
            "`y[..., -y.shape[-1]//6:] = y[..., -y.shape[-1]//6:] + 0.1` "
            "(or its in-place form). No Python for-loops."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with a constant shift added to a fixed "
            "region of the last dim"),
    },
    {
        "name": "region_square",
        "definition": (
            "After computing the original output `y`, square a fixed "
            "sub-slice along the LAST dimension (the last 1/8 of the last "
            "dim). Squaring small magnitudes preserves them; squaring "
            "magnitudes ≥1 amplifies. The change is gentle for outputs "
            "near 0 and larger for outputs near 1."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return: "
            "`y[..., -y.shape[-1]//8:] = y[..., -y.shape[-1]//8:].pow(2)` "
            "(or equivalent `* y[...]`). No Python for-loops."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with a fixed region squared"),
    },
    # ----------------------------------------------------------------
    # The following operators require seed-specific structure such as named
    # channels, numerical literals, or threshold operations. Keep their
    # definitions available for a future preflight-aware rotation, but exclude
    # them from the active pool.
    # ----------------------------------------------------------------
]


# Disabled / archived operators (require seed-specific structure; preflight
# logic needed before re-enabling).
_DISABLED_OPERATORS_KB: List[Dict[str, Any]] = [
    {
        "name": "bias_inject",
        "definition": (
            "After computing the original output `y`, add a small constant "
            "bias `b` (e.g. 0.05) to exactly ONE channel index along ONE "
            "axis. For a tensor of rank R, pick the rightmost non-batch "
            "axis (typically dim=1 for (N, C) outputs, dim=-1 for "
            "feature-last shapes) and target index 0."),
        "scope": (
            "Insert ONE additional statement BEFORE the final return that "
            "indexes into y and adds the bias: e.g. `y[:, 0] = y[:, 0] + b`. "
            "Index must be a literal integer."),
        "target_hint": "forward() return path; final statement",
        "do_not_change": [
            "forward() signature",
            "get_inputs() / get_init_inputs() body",
            "__init__ body",
            "the original math producing y",
        ],
        "old_property": "forward returns y unchanged",
        "new_property": (
            "forward returns y with a constant bias added to one channel"),
    },
    {
        "name": "eps_perturb",
        "definition": (
            "Find ONE existing small numerical literal in forward() — "
            "typically a normalisation epsilon (e.g. 1e-5, 1e-6, 1e-8). "
            "Replace it with a value 100× larger (e.g. 1e-5 → 1e-3). If no "
            "such literal exists, abandon this operator and return the "
            "original forward unchanged so the gate layer rejects it."),
        "scope": (
            "EXACTLY ONE numerical literal in the forward body is changed. "
            "All other literals, all op names, and all structure stay byte-"
            "for-byte identical."),
        "target_hint": "small numerical literal inside an existing op",
        "do_not_change": [
            "forward() signature",
            "every numerical literal other than the chosen one",
            "get_inputs() / get_init_inputs()",
            "__init__ body",
        ],
        "old_property": "forward uses the original eps/literal value",
        "new_property": (
            "forward uses the perturbed value at the same call site"),
    },
    {
        "name": "threshold_swap",
        "definition": (
            "Find ONE comparison or threshold constant inside an existing "
            "op in forward() (e.g. the `0.0` in `F.threshold(x, 0.0, 0.0)`, "
            "or the `1.0` in `x.clamp(max=1.0)`). Replace it with a small "
            "shifted value (delta ≈ 0.05). If no such literal exists, "
            "abandon this operator unchanged."),
        "scope": (
            "EXACTLY ONE comparison/threshold constant is changed. The op "
            "identity (function name, argument order) stays the same."),
        "target_hint": "comparison or threshold constant in an existing op",
        "do_not_change": [
            "forward() signature",
            "all other literals and ops in forward",
            "get_inputs() / get_init_inputs()",
            "__init__ body",
        ],
        "old_property": "forward uses the original threshold value",
        "new_property": (
            "forward uses the shifted threshold at the same call site"),
    },
]

OP_BY_NAME_KB: Dict[str, Dict[str, Any]] = {op["name"]: op for op in EVOL_OPERATORS_KB}


def operator_for_seed(seed_id: str, variant_idx: int = 0,
                      pool: Optional[List[Dict[str, Any]]] = None
                      ) -> Dict[str, Any]:
    """Hash-rotate operator picker. Mirrors benchevolver.py:606-609 so seeds
    don't all land on the same operator when num_variants=1."""
    p = pool if pool is not None else EVOL_OPERATORS_KB
    h = int(hashlib.md5(seed_id.encode()).hexdigest()[:8], 16)
    return p[(h + variant_idx) % len(p)]


def fmt_invariants(items: List[str]) -> str:
    return "; ".join(items)


# ---------- python fence + AST helpers --------------------------------------

import re

_PY_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(raw: str) -> str:
    """Pull a Python block out of an LLM response. Returns the body of the
    first ```python ... ``` fence, or "" if no fence found."""
    if not raw:
        return ""
    m = _PY_FENCE_RE.search(raw)
    return m.group(1).strip() if m else ""


def _find_model_class(tree: ast.AST) -> Optional[ast.ClassDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if isinstance(b, ast.Attribute) and b.attr == "Module":
                    return node
                if isinstance(b, ast.Name) and b.id == "Module":
                    return node
    return None


def extract_forward_body(ref_src: str) -> str:
    """AST-unparse just the body of ``Model.forward``. Used for forward-body
    diff (not whole-file diff — too forgiving of unrelated edits to imports
    or get_inputs)."""
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return ""
    cls = _find_model_class(tree)
    if cls is None:
        return ""
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            try:
                return "\n".join(ast.unparse(s) for s in node.body)
            except Exception:
                return ""
    return ""


def forward_body_diff_ratio(orig_src: str, variant_src: str) -> float:
    """Char-level SequenceMatcher diff over the forward body only."""
    a = extract_forward_body(orig_src)
    b = extract_forward_body(variant_src)
    return diff_ratio(a, b)


# ---------- Triton-feasibility lint ----------------------------------------

_FORBIDDEN_ATTR_CALLS = {"tolist", "item", "numpy", "cpu", "tobytes"}


class _TritonLintVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.failures: List[str] = []

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.failures.append("list comprehension in forward body")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.failures.append("dict comprehension in forward body")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.failures.append("set comprehension in forward body")

    def visit_For(self, node: ast.For) -> None:
        # Allow ``for ... in range(<int literal>)``; reject anything else.
        ok = False
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and all(isinstance(a, ast.Constant) for a in node.iter.args)):
            ok = True
        if not ok:
            self.failures.append("for-loop over non-constant range in forward")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.failures.append("while-loop in forward body")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _FORBIDDEN_ATTR_CALLS:
                self.failures.append(
                    f"forbidden tensor->python call: .{node.func.attr}()")
        self.generic_visit(node)


def triton_feasibility_lint(ref_src: str) -> Tuple[bool, str]:
    """AST walk over Model.forward body. Reject patterns that v4-pro cannot
    sensibly translate into Triton (data-dependent Python control flow,
    tensor→python materialisation).

    Returns (ok, reason). reason is empty on success."""
    try:
        tree = ast.parse(ref_src)
    except SyntaxError as e:
        return False, f"syntax error in variant: {e}"
    cls = _find_model_class(tree)
    if cls is None:
        return False, "no Model(nn.Module) class found"
    fwd = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            fwd = node
            break
    if fwd is None:
        return False, "no forward() in Model class"
    v = _TritonLintVisitor()
    for stmt in fwd.body:
        v.visit(stmt)
    if v.failures:
        return False, "; ".join(sorted(set(v.failures)))
    return True, ""


# ---------- behavioural diff (twin-subprocess) ------------------------------
#
# IMPORTANT: g4 driver runs in a FRESH subprocess. Cold torch import + CUDA
# init + lazy module loading add 25-50s of overhead. Without amortising
# that into a single warmup, the first fuzz trial pays the full cold cost
# and the 180s default budget gets blown on simple-activation seeds where
# the actual forward is <1ms.
#
# Fix: explicit warmup at driver start — import torch, init CUDA, run a
# tiny dummy forward, sync. Subsequent fuzz trials see only "warm" timing.

_BEHAVIORAL_DIFF_DRIVER = textwrap.dedent("""
    import importlib.util, sys, os, traceback, torch
    # Warmup: amortise CUDA cold init + lazy module loading BEFORE any
    # per-trial timing. Without this, the first trial pays 20-40s of
    # one-off cost (PyTorch's lazy CUDA module loading per
    # CUDA_MODULE_LOADING=LAZY guidance), which on simple-activation refs
    # is 1000x the actual forward time and blows the driver's wall budget.
    if torch.cuda.is_available():
        torch.cuda.init()
        _dummy = torch.randn(1, device='cuda')
        _dummy = _dummy + 1.0
        torch.cuda.synchronize()
        del _dummy

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _diff_count(a, b, atol=1e-4, rtol=1e-4):
        \"\"\"Return (n_diff_elems, n_total_elems). Recurses into list/tuple/
        dict; falls back to scalar comparison for non-tensor leaves.\"\"\"
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor):
                return (1, 1)
            if a.shape != b.shape:
                # shape mismatch — count every element as differing
                return (a.numel() + b.numel(), a.numel() + b.numel())
            if a.dtype != b.dtype:
                a = a.float(); b = b.float()
            diff = ~torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            return (int(diff.sum().item()), a.numel())
        if isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                return (1, 1)
            d = 0; t = 0
            for x, y in zip(a, b):
                dd, tt = _diff_count(x, y, atol, rtol)
                d += dd; t += tt
            return (d, t)
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a.keys()) != set(b.keys()):
                return (1, 1)
            d = 0; t = 0
            for k in a:
                dd, tt = _diff_count(a[k], b[k], atol, rtol)
                d += dd; t += tt
            return (d, t)
        # scalar / str / None — treat as 1 element
        return (0 if a == b else 1, 1)

    orig_path = sys.argv[1]
    variant_path = sys.argv[2]
    n_trials = int(sys.argv[3])
    # argv[4]: cache file path (empty string = caching disabled). Cache
    # stores up to CACHE_TARGET trials of (inputs, orig_outputs) so a lite
    # n=4 call and a full n=16 call share one cache slot.
    cache_path = sys.argv[4] if len(sys.argv) > 4 else ""
    # CACHE_TARGET=n_trials disables the lite→full cache pre-fill that
    # hardcoded 16-trial forward warmups even when only 4 were needed.
    # Big-input KB seeds (e.g. 24_LogSoftmax: 4096×393216 fp32 = 6.4GB
    # per tensor) made the pre-fill alone take 8-15 min CPU per (variant,
    # model), blowing the 2700s per-seed cap. Cost of the change: lite-
    # passing variants now re-fetch 12 extra trials in full mode (the
    # ~10% path), but the 90% lite-reject path runs ~4x faster.
    CACHE_TARGET = n_trials

    try:
        orig = _load(orig_path, "kb_orig")
        var = _load(variant_path, "kb_variant")
    except Exception:
        traceback.print_exc()
        sys.exit(2)

    init_args = orig.get_init_inputs() if hasattr(orig, "get_init_inputs") else []

    # P2: try cache hit first — skip orig forward when cached.
    # Cache stores ONLY orig outputs (not inputs); inputs are
    # regenerated deterministically each call via torch.manual_seed(1000+i)
    # — saves ~50% cache size on input-heavy seeds (e.g. matmul where
    # the 2D-tensor A dominates).
    orig_outs_list = None
    if cache_path and os.path.exists(cache_path):
        try:
            # No map_location: restore tensors to whichever device they
            # were saved on (matches the device that orig forward + variant
            # forward both run on — typically CPU for KB seeds since their
            # get_inputs() returns CPU tensors).
            cache = torch.load(cache_path, weights_only=False)
            if isinstance(cache, dict) and len(cache.get('outputs', [])) >= n_trials:
                orig_outs_list = cache['outputs'][:n_trials]
                print(f"CACHE_HIT n={n_trials} path={cache_path}",
                      file=sys.stderr)
        except Exception:
            # Cache corrupted (race / partial write / dtype change). Recompute.
            orig_outs_list = None

    if orig_outs_list is None:
        # Cold path: build cache fresh. Target the MAX of {n_trials,
        # CACHE_TARGET} so a lite call (n=4) seeds the cache for a later
        # full call (n=16) on the same orig.
        try:
            m_orig = orig.Model(*init_args)
        except Exception:
            traceback.print_exc()
            sys.exit(3)
        target = max(n_trials, CACHE_TARGET)
        all_outs = []
        for i in range(target):
            torch.manual_seed(1000 + i)
            try:
                inps = orig.get_inputs()
                with torch.no_grad():
                    y = m_orig(*inps)
            except Exception:
                traceback.print_exc()
                sys.exit(4)
            all_outs.append(y)
        if cache_path:
            try:
                tmp_path = cache_path + ".tmp"
                torch.save({'outputs': all_outs}, tmp_path)
                # Atomic rename: race-safe even if a concurrent variant
                # writes the same cache at the same moment.
                os.replace(tmp_path, cache_path)
                print(f"CACHE_WROTE n={target} path={cache_path}",
                      file=sys.stderr)
            except Exception:
                # Cache write failed — fine, just skip persisting.
                pass
        orig_outs_list = all_outs[:n_trials]

    try:
        m_var = var.Model(*init_args)
    except Exception:
        traceback.print_exc()
        sys.exit(3)

    # Regenerate inputs deterministically (NOT cached — keeps cache size
    # bounded by orig output volume only). g1 already verified that
    # var.get_inputs == orig.get_inputs source, so seeded RNG yields
    # identical inputs for both modules.
    total_diff = 0
    total_elem = 0
    for i in range(n_trials):
        torch.manual_seed(1000 + i)
        try:
            inps = orig.get_inputs()
            with torch.no_grad():
                y_var = m_var(*inps)
        except Exception:
            traceback.print_exc()
            sys.exit(4)
        d, t = _diff_count(orig_outs_list[i], y_var)
        total_diff += d
        total_elem += t
    if total_elem == 0:
        print("DIFF_RATIO=0.0")
    else:
        print(f"DIFF_RATIO={total_diff/total_elem:.4f}")
""").strip()


async def behavioral_diff(
    orig_src: str, variant_src: str,
    n_trials: int = 16, timeout: int = 120,
    cache_dir: Optional[str] = None,
) -> Tuple[float, str]:
    """Run original ref + variant ref side-by-side over ``n_trials``
    seeded fuzz inputs. Returns (diff_ratio, err). err is empty on
    success. diff_ratio = fraction of trials where outputs are NOT
    allclose.

    Routes through the persistent gate worker pool when KB_GATE_WORKERS>0
    (default = num_gpus); else falls back to the legacy per-call
    subprocess driver.

    ``cache_dir``: orig outputs keyed by sha1(orig_src) so subsequent
    variants of the same seed skip the orig forward pass."""
    if not orig_src.strip() or not variant_src.strip():
        return -1.0, "empty source"
    cache_path = ""
    if cache_dir:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            orig_hash = hashlib.sha1(orig_src.encode()).hexdigest()[:16]
            cache_path = str(Path(cache_dir) / f"orig_{orig_hash}.pt")
        except OSError:
            cache_path = ""

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        orig_p = Path(tmp) / "orig.py"
        var_p = Path(tmp) / "variant.py"
        orig_p.write_text(orig_src)
        var_p.write_text(variant_src)

        # --- Try persistent worker pool first ---
        pool = await _get_gate_pool()
        if pool is not None:
            res = await pool.call(
                "kb_behavioral", timeout=timeout,
                orig_path=str(orig_p), variant_path=str(var_p),
                n_trials=n_trials, cache_path=cache_path,
            )
            if res.get("ok"):
                return float(res["result"]["diff_ratio"]), ""
            return -1.0, res.get("err", "worker failed")[-1024:]

        # --- Fallback: legacy subprocess driver ---
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _BEHAVIORAL_DIFF_DRIVER,
                str(orig_p), str(var_p), str(n_trials), cache_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_next_gpu_env(),
            )
        except FileNotFoundError as e:
            return -1.0, f"python launch failed: {e}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return -1.0, f"behavioural diff timeout ({timeout}s)"
        if proc.returncode != 0:
            return -1.0, stderr.decode("utf-8", errors="ignore")[-1024:]
        out = stdout.decode("utf-8", errors="ignore")
        m = re.search(r"DIFF_RATIO=([\d.]+)", out)
        if not m:
            return -1.0, f"no DIFF_RATIO in stdout: {out[:200]}"
        return float(m.group(1)), ""


# ---------- top-level gate runner ------------------------------------------

@dataclass
class GateConfig:
    diff_lo: float = 0.02
    diff_hi: float = 0.40
    diff_lo_final: float = 0.001
    diff_hi_final: float = 0.50
    behavioral_lo: float = 0.05
    behavioral_hi: float = 0.30
    behavioral_n_trials: int = 16
    behavioral_timeout: int = 300   # bumped from 120 — driver now warms CUDA
                                    # before timing trials, but cold subprocess
                                    # import torch + CUDA init still ~30-40s
    # P1: g4 two-stage gate. Run a lite version with few trials + loose
    # bands first to reject the 90% of candidates that obviously zero-diff
    # or all-diff before paying for the full 16-trial fuzz. Setting
    # `behavioral_n_trials_lite=0` disables the lite stage (legacy path).
    behavioral_n_trials_lite: int = 4
    # Lite lower bound > 0 is critical: 0.0 inclusive would let exact
    # zero-diff variants survive lite and pay for full (worst-of-both).
    # 0.005 rejects clear zero-diff but lets tiny-perturb (1-element diff
    # among millions) survive to full where the tight band rules.
    behavioral_lo_lite: float = 0.005
    behavioral_hi_lite: float = 0.60     # lite only kills clear all-diff
    behavioral_timeout_lite: int = 120   # lite is shorter than full
    # P2: orig forward output cache. When set, behavioral_diff caches the
    # orig model's outputs per (sha1(orig_src), max_n_trials) inside this
    # directory so subsequent variants of the same seed skip the orig
    # forward pass. Empty/None = disabled. Recommended: per-job tmpdir
    # like /tmp/g4_cache_<pid>.
    behavioral_cache_dir: Optional[str] = None
    smoke_timeout: int = 60


@dataclass
class GateResult:
    ok: bool
    reason: str             # "" on success; otherwise short stage label
    forward_body_diff: float
    behavioral_diff: float
    # P1: separate column for the lite-stage diff so audits can tell
    # whether a candidate failed at lite vs full. -1.0 = skipped/error.
    behavioral_diff_lite: float = -1.0


async def run_gates(orig_src: str, variant_src: str, cfg: GateConfig,
                    is_final_attempt: bool = False,
                    orig_interface: Optional[PyTorchInterface] = None
                    ) -> GateResult:
    """g1..g5 in FAIL-FAST order:

      g1 format          — variant parses + class/forward args identical
      g2 forward-body    — char-level diff on the forward body ∈ [LO, HI]
      g3 smoke           — variant imports + runs + finite (~3s subprocess)
      g5 Triton lint     — AST walk for non-Triton-friendly idioms (cheap)
      g4 behavioural     — twin-subprocess fuzz; diff ratio ∈ [LO, HI]

    Returns GateResult; on first failure, populates ``reason`` and skips the
    remaining gates."""
    out = GateResult(ok=False, reason="", forward_body_diff=-1.0,
                     behavioral_diff=-1.0)

    # g1: format
    iface_var = _extract_pytorch_interface(variant_src)
    if iface_var is None:
        out.reason = "g1: variant interface extraction failed"
        return out
    if orig_interface is not None:
        if iface_var.class_name != orig_interface.class_name:
            out.reason = (f"g1: class renamed "
                          f"{orig_interface.class_name!r} → {iface_var.class_name!r}")
            return out
        if iface_var.forward_args != orig_interface.forward_args:
            out.reason = (f"g1: forward args changed "
                          f"{orig_interface.forward_args} → {iface_var.forward_args}")
            return out
        if iface_var.init_args != orig_interface.init_args:
            out.reason = (f"g1: __init__ args changed "
                          f"{orig_interface.init_args} → {iface_var.init_args}")
            return out
        if iface_var.get_inputs_src != orig_interface.get_inputs_src:
            out.reason = "g1: get_inputs() body changed"
            return out
        if iface_var.get_init_inputs_src != orig_interface.get_init_inputs_src:
            out.reason = "g1: get_init_inputs() body changed"
            return out

    # g2: forward-body diff
    body_ratio = forward_body_diff_ratio(orig_src, variant_src)
    out.forward_body_diff = body_ratio
    lo = cfg.diff_lo_final if is_final_attempt else cfg.diff_lo
    hi = cfg.diff_hi_final if is_final_attempt else cfg.diff_hi
    if not (lo <= body_ratio <= hi):
        out.reason = (f"g2: forward-body diff {body_ratio:.3f} outside "
                      f"[{lo:.3f}, {hi:.3f}]")
        return out

    # g3: smoke (gated by global semaphore — only N concurrent CUDA-bound
    # smokes at a time, prevents the cold-init contention that nuked the
    # earlier kb_perturb_be full run).
    async with _get_g3_sem():
        smoke_ok, smoke_err = await _ref_smoke_test(
            variant_src, timeout=cfg.smoke_timeout)
    if not smoke_ok:
        # Larger window (200 → 800) — the pynvml deprecation warning
        # eats the first ~200 chars of stderr; the real error comes after.
        out.reason = f"g3: smoke fail: {smoke_err[-800:]}"
        return out

    # g5: triton-feasibility lint
    lint_ok, lint_reason = triton_feasibility_lint(variant_src)
    if not lint_ok:
        out.reason = f"g5: triton-infeasible: {lint_reason}"
        return out

    # g4: behavioural diff (gated by same global semaphore as g3 — both
    # spawn fresh CUDA subprocesses and fight for cold init at the OS level)
    # P1 two-stage: lite (n_trials=4, loose band) early-rejects clear no-diff
    # / all-diff candidates. Full (n_trials=16, tight band) only runs if
    # lite survives. P2 cache: lite writes the orig-outputs cache, full
    # reads it — net effect = orig forward runs once per seed.
    if cfg.behavioral_n_trials_lite > 0:
        async with _get_g3_sem():
            bd_lite, lite_err = await behavioral_diff(
                orig_src, variant_src,
                n_trials=cfg.behavioral_n_trials_lite,
                timeout=cfg.behavioral_timeout_lite,
                cache_dir=cfg.behavioral_cache_dir)
        out.behavioral_diff_lite = bd_lite
        if bd_lite < 0.0:
            out.reason = f"g4-lite: driver failed: {lite_err[:200]}"
            return out
        if not (cfg.behavioral_lo_lite <= bd_lite <= cfg.behavioral_hi_lite):
            out.reason = (
                f"g4-lite: behavioural diff {bd_lite:.3f} outside lite "
                f"[{cfg.behavioral_lo_lite:.3f}, {cfg.behavioral_hi_lite:.3f}]")
            return out

    async with _get_g3_sem():
        bd, bd_err = await behavioral_diff(
            orig_src, variant_src,
            n_trials=cfg.behavioral_n_trials,
            timeout=cfg.behavioral_timeout,
            cache_dir=cfg.behavioral_cache_dir)
    out.behavioral_diff = bd
    if bd < 0.0:
        out.reason = f"g4: behavioural diff driver failed: {bd_err[:200]}"
        return out
    if not (cfg.behavioral_lo <= bd <= cfg.behavioral_hi):
        out.reason = (f"g4: behavioural diff {bd:.3f} outside "
                      f"[{cfg.behavioral_lo:.3f}, {cfg.behavioral_hi:.3f}]")
        return out

    out.ok = True
    return out


# ---------- TBT smoke (the existing TBG smoke checks `result_gold`; TBT uses
# `test_results`; add a variant that accepts the test-var name) -------------

_TBT_REF_SMOKE_DRIVER = textwrap.dedent("""
    import importlib.util, sys, traceback
    p = sys.argv[1]
    test_var = sys.argv[2]
    spec = importlib.util.spec_from_file_location("tbt_ref_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    rg = getattr(mod, test_var, None)
    if rg is None:
        print(f'{test_var} attribute missing', file=sys.stderr)
        sys.exit(3)
    print('OK')
""").strip()


async def _ref_smoke_test_with_var(
    ref_src: str, test_var_name: str = "result_gold", timeout: int = 120,
) -> Tuple[bool, str]:
    """Generic smoke test for TBG (`result_gold`) / TBT (`test_results`)
    refs. Imports the file, runs its body, checks the named attribute."""
    if not ref_src.strip():
        return False, "empty source"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        ref_p = Path(tmp) / "ref_under_test.py"
        ref_p.write_text(ref_src)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _TBT_REF_SMOKE_DRIVER,
                str(ref_p), test_var_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_next_gpu_env(),
            )
        except FileNotFoundError as e:
            return False, f"python launch failed: {e}"
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return False, f"smoke timeout ({timeout}s)"
        if proc.returncode != 0:
            return False, stderr.decode("utf-8", errors="ignore")[-1024:]
        return True, ""


# ===========================================================================
#                          TBG / TBT extensions
# ===========================================================================
#
# TBG perturbs `@triton.jit` kernels directly; TBT perturbs `def foo(...)`
# PyTorch functions. Both verify by running `test_xxx()` and counting how
# many keys in the result dict differ between original and variant — a
# different metric from KB's element-level fuzz.


# ---------- TBG operator catalogue -----------------------------------------
#
# Operators target the kernel body or wrapper interior of TritonBench-G refs.
# Always preserves wrapper signature + test_xxx() byte-for-byte.

EVOL_OPERATORS_TBG: List[Dict[str, Any]] = [
    {
        "name": "kernel_const_shift",
        "definition": (
            "Inside ONE `@triton.jit` kernel, BEFORE the final `tl.store`, "
            "add a small constant (e.g. 0.1) to the value being stored. "
            "Apply only where the existing store mask is true (so this "
            "doesn't store out-of-bounds). Pick a single kernel — if "
            "there are multiple kernels, choose the one whose output is "
            "consumed by the wrapper's final return."),
        "scope": (
            "Insert ONE additional expression between the last compute "
            "step and the final `tl.store`: e.g. `out = out + 0.1` "
            "right before `tl.store(out_ptr + offs, out, mask=mask)`."),
        "target_hint": "@triton.jit kernel, final tl.store",
        "do_not_change": [
            "wrapper function signature",
            "test_xxx() function body (verbatim)",
            "kernel name(s)",
            "kernel arg names / types",
            "the original math producing the stored value",
        ],
        "old_property": "kernel stores the original computed value",
        "new_property": (
            "kernel stores the original value plus a small constant"),
    },
    {
        "name": "kernel_threshold_clamp",
        "definition": (
            "Inside the final `@triton.jit` kernel, BEFORE `tl.store`, "
            "apply a one-sided clamp using `tl.minimum(out, T)` or "
            "`tl.maximum(out, T)`. Pick T such that ~10-30% of typical "
            "output values are clipped. T = 2.0 (with `tl.minimum`) is "
            "a reasonable default for unbounded outputs."),
        "scope": (
            "Insert ONE `tl.minimum(...)` or `tl.maximum(...)` line "
            "between final compute and `tl.store`. No control flow."),
        "target_hint": "@triton.jit kernel, before final tl.store",
        "do_not_change": [
            "wrapper function signature",
            "test_xxx() function body",
            "kernel name(s)",
            "kernel arg names / types",
        ],
        "old_property": "kernel stores unbounded value",
        "new_property": "kernel stores value clamped at T on one side",
    },
    {
        "name": "kernel_eps_swap",
        "definition": (
            "Find ONE small numerical literal inside a kernel body "
            "(typically a normalisation eps like 1e-5, 1e-6, 1e-8). "
            "Replace it with a value 100× larger. If no such literal "
            "exists in any kernel, abandon this operator unchanged so "
            "the gate rejects the no-op."),
        "scope": (
            "EXACTLY ONE numerical literal in the kernel body changed. "
            "All other literals, op names, structure stay identical."),
        "target_hint": "numerical literal inside a kernel",
        "do_not_change": [
            "wrapper signature",
            "test_xxx() body",
            "every other numerical literal in the kernel",
            "kernel name(s)",
        ],
        "old_property": "kernel uses the original eps/literal value",
        "new_property": "kernel uses the perturbed value at the same site",
    },
    {
        "name": "kernel_pid_partition",
        "definition": (
            "Inside the final `@triton.jit` kernel, add ONE conditional "
            "perturbation gated by `pid % K == 0` for a small K (e.g. 8). "
            "Programs that satisfy the condition compute output + small "
            "constant; others are unchanged. K chosen so ~10-15% of "
            "program tiles trigger."),
        "scope": (
            "Insert ONE `if pid % K == 0:` block (or its tl.where "
            "equivalent) before the final tl.store. K = 8 is the "
            "default."),
        "target_hint": "@triton.jit kernel, program-id partition",
        "do_not_change": [
            "wrapper signature",
            "test_xxx() body",
            "kernel name(s)",
            "the program_id setup",
        ],
        "old_property": "every program tile stores the original value",
        "new_property": (
            "programs with pid % K == 0 store original + delta"),
    },
    {
        "name": "kernel_block_offset",
        "definition": (
            "Inside the final `@triton.jit` kernel, on program_id 0 only, "
            "add a small bias (e.g. 0.05) to the stored value. Other "
            "program_ids unchanged. This affects exactly one output tile."),
        "scope": (
            "Insert ONE `if pid == 0:` block (or `tl.where(pid == 0, ...)` "
            "equivalent) before the final tl.store."),
        "target_hint": "@triton.jit kernel, first program tile",
        "do_not_change": [
            "wrapper signature",
            "test_xxx() body",
            "kernel name(s)",
        ],
        "old_property": "program_id 0's tile holds the original value",
        "new_property": "program_id 0's tile holds original + bias",
    },
    {
        "name": "kernel_dtype_widen",
        "definition": (
            "Inside the final `@triton.jit` kernel, widen the dtype of "
            "ONE accumulator from `tl.float16`/`tl.bfloat16` to "
            "`tl.float32` (or insert an explicit `.to(tl.float32)` cast "
            "on the accumulator). The output dtype stays the same — only "
            "the internal accumulation precision changes. If the kernel "
            "already accumulates in fp32, abandon this operator."),
        "scope": (
            "EXACTLY ONE accumulator dtype changed. Output cast back to "
            "original dtype at tl.store."),
        "target_hint": "kernel accumulator dtype",
        "do_not_change": [
            "wrapper signature",
            "test_xxx() body",
            "output dtype (cast back if needed)",
            "kernel name(s)",
        ],
        "old_property": "kernel accumulates in original (narrow) dtype",
        "new_property": "kernel accumulates in fp32, then casts to output dtype",
    },
    {
        "name": "wrapper_post_offset",
        "definition": (
            "After the kernel launch in the wrapper function, add a "
            "small constant (e.g. 0.05) to the output tensor in place "
            "via PyTorch (`out.add_(0.05)` or `out = out + 0.05`). The "
            "kernel itself is unchanged — only the wrapper's post-call "
            "step adds the offset. Simplest operator, works on any TBG "
            "ref but only changes outputs by a global shift."),
        "scope": (
            "Insert ONE statement in the wrapper between the kernel "
            "launch and the `return` statement."),
        "target_hint": "wrapper function, post-kernel-launch",
        "do_not_change": [
            "wrapper function signature",
            "test_xxx() body",
            "kernel body (entirely)",
            "kernel name(s)",
        ],
        "old_property": "wrapper returns the kernel's output unchanged",
        "new_property": (
            "wrapper returns the kernel's output plus a constant"),
    },
]

OP_BY_NAME_TBG: Dict[str, Dict[str, Any]] = {
    op["name"]: op for op in EVOL_OPERATORS_TBG
}


# ---------- TBG kernel-body extraction (g2 input) --------------------------

def extract_triton_kernels(ref_src: str) -> str:
    """Concatenate all `@triton.jit` function bodies in ref_src into a single
    string (newline-separated). Used for kernel-body diff measurement.

    Returns "" if no `@triton.jit` kernel found or syntax error.
    """
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return ""
    bodies: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        is_jit = False
        for dec in node.decorator_list:
            d = dec
            # Unwrap `@triton.jit()` call form.
            if isinstance(d, ast.Call):
                d = d.func
            if isinstance(d, ast.Attribute) and d.attr == "jit":
                v = d.value
                if isinstance(v, ast.Name) and v.id == "triton":
                    is_jit = True
                    break
            elif isinstance(d, ast.Name) and d.id == "jit":
                is_jit = True
                break
        if not is_jit:
            continue
        try:
            bodies.append("\n".join(ast.unparse(s) for s in node.body))
        except Exception:
            pass
    return "\n\n".join(bodies)


def kernel_body_diff_ratio(orig_src: str, variant_src: str) -> float:
    """Char-level SequenceMatcher diff over all kernel bodies concatenated."""
    return diff_ratio(extract_triton_kernels(orig_src),
                      extract_triton_kernels(variant_src))


def extract_tbt_func_body(ref_src: str) -> str:
    """Concatenate the FIRST top-level `def` body's source (TBT shape) as
    a string. Used for TBT g2 diff measurement."""
    try:
        tree = ast.parse(ref_src)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            try:
                return "\n".join(ast.unparse(s) for s in node.body)
            except Exception:
                return ""
    return ""


def tbt_func_body_diff_ratio(orig_src: str, variant_src: str) -> float:
    """Char-level diff over the first top-level def body (TBT)."""
    return diff_ratio(extract_tbt_func_body(orig_src),
                      extract_tbt_func_body(variant_src))


# ---------- TBG/TBT behavioral diff: count differing test_xxx keys --------

_BEHAVIORAL_DIFF_TBG_DRIVER = textwrap.dedent("""
    import importlib.util, sys, os, traceback, torch
    # Warmup CUDA before timing trials (see _BEHAVIORAL_DIFF_DRIVER above).
    if torch.cuda.is_available():
        torch.cuda.init()
        _dummy = torch.randn(1, device='cuda')
        _dummy = _dummy + 1.0
        torch.cuda.synchronize()
        del _dummy

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _tensor_close(a, b, atol=1e-2, rtol=1e-2):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor):
                return False
            if a.shape != b.shape:
                return False
            if a.dtype != b.dtype:
                a = a.float(); b = b.float()
            return bool(torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
        if isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)) or len(a) != len(b):
                return False
            return all(_tensor_close(x, y, atol, rtol) for x, y in zip(a, b))
        return a == b

    orig_path = sys.argv[1]
    variant_path = sys.argv[2]
    test_var = sys.argv[3]    # 'result_gold' (G) or 'test_results' (T)
    try:
        orig = _load(orig_path, "tbg_orig")
        variant = _load(variant_path, "tbg_variant")
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    try:
        orig_dict = getattr(orig, test_var)
        variant_dict = getattr(variant, test_var)
    except AttributeError as e:
        print(f"missing {test_var}: {e}", file=sys.stderr)
        sys.exit(3)
    if not isinstance(orig_dict, dict) or not isinstance(variant_dict, dict):
        print(f"{test_var} is not a dict", file=sys.stderr)
        sys.exit(4)
    orig_keys = set(orig_dict.keys())
    variant_keys = set(variant_dict.keys())
    if orig_keys != variant_keys:
        print(f"key mismatch orig={orig_keys} variant={variant_keys}",
              file=sys.stderr)
        sys.exit(5)
    n_diff = 0
    for k in sorted(orig_keys):
        if not _tensor_close(orig_dict[k], variant_dict[k]):
            n_diff += 1
    n_keys = len(orig_keys)
    print(f"N_KEYS={n_keys}")
    print(f"N_DIFF={n_diff}")
""").strip()


async def behavioral_diff_tbg(
    orig_src: str, variant_src: str, test_var_name: str,
    timeout: int = 240,
) -> Tuple[int, int, str]:
    """Run both refs' `test_xxx()`, compare the result dict key-by-key with
    `torch.allclose`. Returns (n_keys, n_diff, err). err == "" on success.
    `test_var_name` = 'result_gold' (TBG) or 'test_results' (TBT)."""
    if not orig_src.strip() or not variant_src.strip():
        return -1, -1, "empty source"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        orig_p = Path(tmp) / "orig.py"
        var_p = Path(tmp) / "variant.py"
        orig_p.write_text(orig_src)
        var_p.write_text(variant_src)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _BEHAVIORAL_DIFF_TBG_DRIVER,
                str(orig_p), str(var_p), test_var_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_next_gpu_env(),
            )
        except FileNotFoundError as e:
            return -1, -1, f"python launch failed: {e}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except (ProcessLookupError, OSError):
                pass
            return -1, -1, f"behavioural diff timeout ({timeout}s)"
        if proc.returncode != 0:
            return -1, -1, stderr.decode("utf-8", errors="ignore")[-1024:]
        out = stdout.decode("utf-8", errors="ignore")
        nk = re.search(r"N_KEYS=(\d+)", out)
        nd = re.search(r"N_DIFF=(\d+)", out)
        if not nk or not nd:
            return -1, -1, f"unparseable output: {out[:200]}"
        return int(nk.group(1)), int(nd.group(1)), ""


# ---------- TBG/TBT gate config + runner -----------------------------------

@dataclass
class GateConfigTBG:
    """Gate thresholds for TBG/TBT pipelines (dict-compare verify)."""
    diff_lo: float = 0.005     # kernel-body or func-body source diff
    diff_hi: float = 0.50
    diff_lo_final: float = 0.001
    diff_hi_final: float = 0.70
    behavioral_n_low: int = 1   # at least 1 test_xxx key must differ
    behavioral_ratio_hi: float = 0.40   # at most 40% of keys differ
    behavioral_timeout: int = 240
    smoke_timeout: int = 120


@dataclass
class GateResultTBG:
    ok: bool
    reason: str
    body_diff: float
    n_keys: int
    n_diff: int


def _format_check_tbg(orig_src: str, variant_src: str,
                      orig_interface) -> Tuple[bool, str]:
    """g1 for TBG: variant must keep wrapper signature, kernel names,
    test_func_name identical. `orig_interface` is a TritonInterface."""
    # Lazy import to avoid circular dep
    from .inversecoder import _extract_triton_interface  # type: ignore

    iface_var = _extract_triton_interface(variant_src)
    if iface_var is None:
        return False, "variant has no @triton.jit kernel"
    if set(iface_var.kernel_names) != set(orig_interface.kernel_names):
        return False, (f"kernel names changed "
                       f"{orig_interface.kernel_names} → {iface_var.kernel_names}")
    if iface_var.primary_wrapper != orig_interface.primary_wrapper:
        return False, (f"primary wrapper changed "
                       f"{orig_interface.primary_wrapper!r} → "
                       f"{iface_var.primary_wrapper!r}")
    if iface_var.test_func_name != orig_interface.test_func_name:
        return False, "test_func_name changed"
    if iface_var.primary_arg_names != orig_interface.primary_arg_names:
        return False, (f"wrapper args changed "
                       f"{orig_interface.primary_arg_names} → "
                       f"{iface_var.primary_arg_names}")
    return True, ""


async def run_gates_tbg(orig_src: str, variant_src: str,
                        cfg: GateConfigTBG,
                        is_final_attempt: bool = False,
                        orig_interface=None,
                        test_var_name: str = "result_gold",
                        body_kind: str = "triton_kernels",
                        ) -> GateResultTBG:
    """5-gate FAIL-FAST for TBG/TBT:
      g1 format        — keep kernel names + wrapper signature + test_func name
      g2 body diff     — depends on body_kind:
                          'triton_kernels' (TBG): concatenated @triton.jit bodies
                          'tbt_func'        (TBT): first top-level def body
      g3 smoke         — variant imports + test_xxx() runs to set the named attr
                         (`result_gold` for TBG, `test_results` for TBT)
      g5 triton lint   — (no-op for TBG/TBT; rely on g3 to catch broken Python)
      g4 behavioural   — count test_xxx keys that differ; 1 ≤ n_diff ≤ ceil(hi·N)
    """
    out = GateResultTBG(ok=False, reason="", body_diff=-1.0,
                        n_keys=-1, n_diff=-1)

    # g1 — only run if we have an interface to compare against. TBT format
    # checks live in the TBT method (different shape than TritonInterface).
    if orig_interface is not None and body_kind == "triton_kernels":
        ok, reason = _format_check_tbg(orig_src, variant_src, orig_interface)
        if not ok:
            out.reason = f"g1: {reason}"
            return out

    # g2 — body diff (kind-specific)
    if body_kind == "triton_kernels":
        body_ratio = kernel_body_diff_ratio(orig_src, variant_src)
    elif body_kind == "tbt_func":
        body_ratio = tbt_func_body_diff_ratio(orig_src, variant_src)
    else:
        out.reason = f"g2: unknown body_kind={body_kind!r}"
        return out
    out.body_diff = body_ratio
    lo = cfg.diff_lo_final if is_final_attempt else cfg.diff_lo
    hi = cfg.diff_hi_final if is_final_attempt else cfg.diff_hi
    if not (lo <= body_ratio <= hi):
        kind_label = "kernel-body" if body_kind == "triton_kernels" else "func-body"
        out.reason = (f"g2: {kind_label} diff {body_ratio:.3f} outside "
                      f"[{lo:.3f}, {hi:.3f}]")
        return out

    # g3 — smoke (gated by global semaphore). For TBG we reuse the existing
    # _tbg_ref_smoke_test (checks `result_gold`); for TBT we use the var-aware
    # smoke (`test_results`).
    async with _get_g3_sem():
        if test_var_name == "result_gold":
            from .inversecoder import _tbg_ref_smoke_test  # type: ignore
            smoke_ok, smoke_err = await _tbg_ref_smoke_test(
                variant_src, timeout=cfg.smoke_timeout)
        else:
            smoke_ok, smoke_err = await _ref_smoke_test_with_var(
                variant_src, test_var_name=test_var_name,
                timeout=cfg.smoke_timeout)
    if not smoke_ok:
        # Larger window (200 → 800) — the pynvml deprecation warning
        # eats the first ~200 chars of stderr; the real error comes after.
        out.reason = f"g3: smoke fail: {smoke_err[-800:]}"
        return out

    # g5 — light triton-feasibility lint on the variant Python.
    # We DON'T parse the kernel body itself (it's all tl.*) — just check the
    # wrapper / file doesn't introduce data-dependent Python control flow.
    # Reuse the same lint helper as KB, which targets the FIRST `Model.forward`
    # body. For TBG/TBT there's no Model class so the lint is a no-op — fine.

    # g4 — behavioural dict-key diff count (gated by same global semaphore
    # as g3 — both spawn CUDA cold-init subprocesses).
    async with _get_g3_sem():
        n_keys, n_diff, bd_err = await behavioral_diff_tbg(
            orig_src, variant_src, test_var_name,
            timeout=cfg.behavioral_timeout)
    out.n_keys = n_keys
    out.n_diff = n_diff
    if n_keys < 0:
        out.reason = f"g4: behavioural driver failed: {bd_err[:200]}"
        return out
    if n_keys == 0:
        out.reason = "g4: test_xxx() returned empty dict"
        return out
    upper = max(1, int(cfg.behavioral_ratio_hi * n_keys + 0.5))  # round
    if not (cfg.behavioral_n_low <= n_diff <= upper):
        out.reason = (f"g4: behavioural diff {n_diff}/{n_keys} outside "
                      f"[{cfg.behavioral_n_low}, {upper}]")
        return out

    out.ok = True
    return out


# ---------- TBT format check (PyTorch function) ----------------------------

def _extract_tbt_func_interface(func_src: str) -> Optional[Dict[str, Any]]:
    """Parse a TBT function ref: top-level `def name(args)`.

    Returns a dict with the interface keys consumed by
    teacher_triton_rollout_tbg.py: primary_wrapper, primary_arg_names, and
    primary_signature.
    """
    try:
        tree = ast.parse(func_src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            arg_names = [a.arg for a in node.args.args]
            sig = f"{node.name}({', '.join(arg_names)})"
            return {
                "primary_wrapper": node.name,
                "primary_arg_names": arg_names,
                "primary_signature": sig,
                "kernel_names": [node.name],
            }
    return None


__all__ = [
    "EVOL_OPERATORS_KB",
    "OP_BY_NAME_KB",
    "operator_for_seed",
    "fmt_invariants",
    "extract_python",
    "extract_forward_body",
    "forward_body_diff_ratio",
    "triton_feasibility_lint",
    "behavioral_diff",
    "GateConfig",
    "GateResult",
    "run_gates",
    # TBG/TBT additions:
    "EVOL_OPERATORS_TBG",
    "OP_BY_NAME_TBG",
    "extract_triton_kernels",
    "kernel_body_diff_ratio",
    "extract_tbt_func_body",
    "tbt_func_body_diff_ratio",
    "behavioral_diff_tbg",
    "GateConfigTBG",
    "GateResultTBG",
    "run_gates_tbg",
    "_ref_smoke_test_with_var",
    "_extract_tbt_func_interface",
]
