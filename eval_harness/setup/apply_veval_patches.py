#!/usr/bin/env python3
"""apply_veval_patches.py — make a verilog-eval checkout harness-compatible.

sv-generate has a ``local_vllm_models`` whitelist; a model not in it is
rejected with "Unknown model" -> 0 VEval pass_rate. This appends the served
model name (the ``--model`` the harness drives sv-generate with) so the
proxy (which rewrites the model field + injects auth) can serve it.

Usage: python apply_veval_patches.py <verilog_eval_root> [served_name]
  where <verilog_eval_root> contains scripts/sv-generate.
  served_name defaults to $API_SERVED_NAME (must be set, or pass as arg).
"""
import os
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if not args:
        print("ERROR: usage: apply_veval_patches.py <verilog_eval_root> [served_name]",
              file=sys.stderr)
        sys.exit(1)
    ve = Path(args[0]).resolve()
    served = args[1] if len(args) > 1 else os.environ.get("API_SERVED_NAME")
    if not served:
        print("ERROR: served_name required (pass as arg or set API_SERVED_NAME)",
              file=sys.stderr)
        sys.exit(1)
    svg = ve / "scripts/sv-generate"
    if not svg.exists():
        print(f"ERROR: {svg} not found", file=sys.stderr)
        sys.exit(1)
    txt = svg.read_text()
    marker = "# harness patch: accept the proxy-served model"
    if marker in txt:
        print("[veval-patch] sv-generate whitelist: already patched, skip")
        return
    anchor = "local_vllm_models = ["
    if anchor not in txt:
        print("[veval-patch] local_vllm_models anchor not found (verilog-eval version mismatch?), skip")
        return
    patch = (
        f'    "{served}",  # harness patch: accept the proxy-served model (the proxy rewrites\n'
        "                 # the model field + injects auth, so any gateway-served name is valid).\n"
    )
    svg.write_text(txt.replace(anchor, anchor + "\n" + patch, 1))
    print(f"[veval-patch] sv-generate whitelist: added '{served}' on {ve}")


if __name__ == "__main__":
    main()
