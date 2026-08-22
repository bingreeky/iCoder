"""Persistent gate worker — replaces per-call subprocess fork+import_torch+
cuda.init overhead with a long-lived worker process.

Protocol (JSON over stdin/stdout, one task per line):

  Request:  {"task_id": "...", "op": "...", "<task-specific kwargs>"}
  Response: {"task_id": "...", "ok": true,  "result": {...}}
       or:  {"task_id": "...", "ok": false, "err": "..."}

Ops:
  kb_smoke
    args: ref_path
    result: {} on success
  kb_behavioral
    args: orig_path, variant_path, n_trials, cache_path (str or "")
    result: {diff_ratio: float}
  tbg_smoke / tbg_smoke_with_var
    args: ref_path, test_var_name (default "result_gold")
    result: {} on success
  tbg_behavioral
    args: orig_path, variant_path, test_var_name, test_func_name
    result: {n_keys: int, n_diff: int}

The worker emits a single "READY\n" line on stdout once warmup is done.
The master MUST read that line before sending the first task.
"""

import importlib.util
import json
import os
import sys
import traceback

# Don't write __pycache__/*.pyc — the master process puts orig.py / variant.py
# in a tempfile.TemporaryDirectory and the rmtree on context exit fails with
# "Directory not empty" if we leave .pyc files behind.
sys.dont_write_bytecode = True

import torch

# ---- Warmup: amortize CUDA cold init + lazy module loading -----------
if torch.cuda.is_available():
    torch.cuda.init()
    _dummy = torch.randn(1, device='cuda')
    _dummy = _dummy + 1.0
    torch.cuda.synchronize()
    del _dummy


def _load(path: str, name: str):
    """Import a Python file at `path` as a fresh module. Caller is
    responsible for purging from sys.modules afterwards (avoid leaks)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _purge_module(name: str) -> None:
    """Remove module + its child entries from sys.modules so the next
    load is fresh and we don't leak."""
    for k in list(sys.modules):
        if k == name or k.startswith(name + "."):
            del sys.modules[k]


def _diff_count(a, b, atol=1e-4, rtol=1e-4):
    """Recursive elementwise diff count. Returns (n_diff_elems, n_total_elems)."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            return (1, 1)
        if a.shape != b.shape:
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
    return (0 if a == b else 1, 1)


# ---- ops -----------------------------------------------------------------

def op_kb_smoke(args):
    """KB-side smoke: import ref.py, instantiate Model, run one forward."""
    ref_path = args["ref_path"]
    try:
        mod = _load(ref_path, "kb_ref_under_test")
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    try:
        init = mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []
        m = mod.Model(*init)
        inputs = mod.get_inputs()
        with torch.no_grad():
            _y = m(*inputs)
        return {"ok": True, "result": {}}
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    finally:
        _purge_module("kb_ref_under_test")


def op_kb_behavioral(args):
    """KB-side behavioral diff. Optional cache: if cache_path file exists,
    load orig outputs (skip orig forward); else compute & save."""
    orig_path = args["orig_path"]
    variant_path = args["variant_path"]
    n_trials = int(args["n_trials"])
    cache_path = args.get("cache_path", "") or ""
    # CACHE_TARGET=n_trials: see rationale in _perturb_common.py
    # _BEHAVIORAL_DIFF_DRIVER. Same fix applied here for the persistent
    # worker pool path (currently disabled via KB_GATE_WORKERS=0).
    CACHE_TARGET = n_trials

    try:
        orig = _load(orig_path, "kb_orig")
        var = _load(variant_path, "kb_variant")
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    try:
        init = orig.get_init_inputs() if hasattr(orig, "get_init_inputs") else []

        # Cache hit path
        orig_outs_list = None
        if cache_path and os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path, weights_only=False)
                if isinstance(cache, dict) and len(cache.get("outputs", [])) >= n_trials:
                    orig_outs_list = cache["outputs"][:n_trials]
            except Exception:
                orig_outs_list = None

        if orig_outs_list is None:
            m_orig = orig.Model(*init)
            target = max(n_trials, CACHE_TARGET)
            all_outs = []
            for i in range(target):
                torch.manual_seed(1000 + i)
                inps = orig.get_inputs()
                with torch.no_grad():
                    y = m_orig(*inps)
                all_outs.append(y)
            if cache_path:
                try:
                    tmp_path = cache_path + ".tmp"
                    torch.save({"outputs": all_outs}, tmp_path)
                    os.replace(tmp_path, cache_path)
                except Exception:
                    pass
            orig_outs_list = all_outs[:n_trials]
            del m_orig

        m_var = var.Model(*init)
        total_diff = total_elem = 0
        for i in range(n_trials):
            torch.manual_seed(1000 + i)
            inps = orig.get_inputs()
            with torch.no_grad():
                y_var = m_var(*inps)
            d, t = _diff_count(orig_outs_list[i], y_var)
            total_diff += d
            total_elem += t
        ratio = 0.0 if total_elem == 0 else total_diff / total_elem
        return {"ok": True, "result": {"diff_ratio": ratio}}
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    finally:
        _purge_module("kb_orig")
        _purge_module("kb_variant")


def op_tbg_smoke(args):
    """TBG ref smoke: import file (triggers `result_gold = test_xxx()`)
    and check the named test var is set."""
    ref_path = args["ref_path"]
    test_var = args.get("test_var_name", "result_gold")
    try:
        mod = _load(ref_path, "tbg_ref_under_test")
        rg = getattr(mod, test_var, None)
        if rg is None:
            return {"ok": False, "err": f"{test_var} attribute missing"}
        if isinstance(rg, dict) and not rg:
            return {"ok": False, "err": f"{test_var} is empty dict"}
        return {"ok": True, "result": {}}
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    finally:
        _purge_module("tbg_ref_under_test")


def op_tbt_smoke(args):
    """TBT ref smoke: like TBG but test_var_name defaults to test_results."""
    args = dict(args)
    args.setdefault("test_var_name", "test_results")
    return op_tbg_smoke(args)


def op_tbg_behavioral(args):
    """TBG/TBT behavioral diff: compare orig.test_xxx() vs variant.test_xxx()
    by dict-key allclose."""
    orig_path = args["orig_path"]
    variant_path = args["variant_path"]
    test_var = args.get("test_var_name", "result_gold")
    try:
        orig = _load(orig_path, "tbg_orig")
        var = _load(variant_path, "tbg_variant")
        o = getattr(orig, test_var, None)
        v = getattr(var, test_var, None)
        if not isinstance(o, dict):
            return {"ok": False, "err": f"orig.{test_var} not dict"}
        if not isinstance(v, dict):
            return {"ok": False, "err": f"variant.{test_var} not dict"}
        keys = set(o.keys()) | set(v.keys())
        n_keys = len(keys)
        n_diff = 0
        for k in keys:
            if k not in o or k not in v:
                n_diff += 1
                continue
            d, t = _diff_count(o[k], v[k])
            if d > 0:
                n_diff += 1
        return {"ok": True, "result": {"n_keys": n_keys, "n_diff": n_diff}}
    except Exception:
        return {"ok": False, "err": traceback.format_exc()[-2000:]}
    finally:
        _purge_module("tbg_orig")
        _purge_module("tbg_variant")


OPS = {
    "kb_smoke": op_kb_smoke,
    "kb_behavioral": op_kb_behavioral,
    "tbg_smoke": op_tbg_smoke,
    "tbt_smoke": op_tbt_smoke,
    "tbg_behavioral": op_tbg_behavioral,
}


def main():
    # Signal readiness once warmup is done.
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except Exception:
            sys.stdout.write(json.dumps({
                "task_id": "?", "ok": False, "err": "invalid JSON"}) + "\n")
            sys.stdout.flush()
            continue
        task_id = task.get("task_id", "?")
        op_name = task.get("op")
        fn = OPS.get(op_name)
        if fn is None:
            result = {"task_id": task_id, "ok": False,
                      "err": f"unknown op: {op_name}"}
        else:
            try:
                result = fn(task)
                result["task_id"] = task_id
            except Exception:
                result = {"task_id": task_id, "ok": False,
                          "err": traceback.format_exc()[-2000:]}
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
