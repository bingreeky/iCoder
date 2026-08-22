#!/usr/bin/env python3
"""apply_cvdp_patches.py — make a CVDP checkout harness-compatible.

eval_5models_factory.py has a fixed ALIASES list; a model not in it is
rejected with "Unsupported model type" -> CVDP 0. The proxy works for any
OpenAI-compatible model, so this appends the served model name to ALIASES.

Usage: python apply_cvdp_patches.py <cvdp_local_root> [served_name]
  where <cvdp_local_root> contains local_extensions/eval_5models_factory.py.
  served_name defaults to $API_SERVED_NAME (must be set, or pass as arg).
"""
import os
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if not args:
        print("ERROR: usage: apply_cvdp_patches.py <cvdp_local_root> [served_name]",
              file=sys.stderr)
        sys.exit(1)
    cvdp = Path(args[0]).resolve()
    served = args[1] if len(args) > 1 else os.environ.get("API_SERVED_NAME")
    if not served:
        print("ERROR: served_name required (pass as arg or set API_SERVED_NAME)",
              file=sys.stderr)
        sys.exit(1)
    f = cvdp / "local_extensions/eval_5models_factory.py"
    if not f.exists():
        print(f"ERROR: {f} not found", file=sys.stderr)
        sys.exit(1)
    txt = f.read_text()
    marker = "# harness patch: accept the proxy-served model"
    if marker in txt:
        print("[cvdp-patch] factory ALIASES: already patched, skip")
        return
    anchor = "ALIASES = ["
    if anchor not in txt:
        print("[cvdp-patch] ALIASES anchor not found (CVDP version mismatch?), skip")
        return
    patch = (
        f'    "{served}",  # harness patch: accept the proxy-served model (the proxy works\n'
        "                 # for any OpenAI-compatible model via the gateway).\n"
    )
    f.write_text(txt.replace(anchor, anchor + "\n" + patch, 1))
    print(f"[cvdp-patch] factory ALIASES: added '{served}' on {cvdp}")


if __name__ == "__main__":
    main()
