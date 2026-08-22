#!/usr/bin/env python3
"""apply_kb_patches.py — make a KernelBench checkout eval-harness-compatible.

Applies 3 patches to a KernelBench source tree (idempotent — skips if already
patched). The harness's run_kernelbench.sh needs these because upstream KB:
  - uses legacy /v1/completions (gateways only support /v1/chat/completions)
  - hardcodes server_port=10210 in the "local" preset (no way to point at the
    shared :8000 proxy via CLI)
  - hardcodes the eval output path (no per-shard files for sharded eval)

Usage: python apply_kb_patches.py <kernelbench_root>
"""
import os, sys
from pathlib import Path

def patch_file(path: Path, find: str, replace: str, marker: str, label: str):
    txt = path.read_text()
    if marker in txt:
        print(f"[kb-patch] {label}: already patched, skip"); return
    if find not in txt:
        print(f"[kb-patch] {label}: anchor not found (KB version mismatch?), skip"); return
    path.write_text(txt.replace(find, replace, 1))
    print(f"[kb-patch] {label}: applied")

def main():
    kb = Path(sys.argv[1]).resolve()
    utils = kb / "src/kernelbench/utils.py"
    evalgen = kb / "scripts/eval_from_generations.py"
    for p in (utils, evalgen):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr); sys.exit(1)

    # 1) local preset: read server_port/address from env (default 10210/localhost)
    patch_file(utils,
        '''    "local": {  # this is for running locally (SGLang, vLLM, Tokasaurus), mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": 10210,
        "server_address": "localhost",
        "max_tokens": 8192,
    },''',
        '''    "local": {  # this is for running locally (SGLang, vLLM, Tokasaurus), mostly for Llama
        "temperature": 0.8, # human eval pass@N temperature
        "server_port": int(os.environ.get("KB_SERVER_PORT", "10210")),
        "server_address": os.environ.get("KB_SERVER_ADDRESS", "localhost"),
        "max_tokens": 8192,
    },''',
        "KB_SERVER_PORT", "utils.SERVER_PRESETS.local env")

    # 2) query_server: string prompt -> chat (gateways lack /v1/completions)
    patch_file(utils,
        '''        if isinstance(prompt, str):
            response = client.completions.create(
                model="default",
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]''',
        '''        if isinstance(prompt, str):
            # harness patch: route string prompts through /v1/chat/completions
            response = client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.message.content for choice in response.choices]''',
        "harness patch: route string prompts", "utils.query_server string->chat")

    # 3) eval output path: read KB_EVAL_FILE_SUFFIX env (per-shard files)
    patch_file(evalgen,
        '''    run_dir = os.path.join(config.runs_dir, config.run_name)
    eval_file_path = os.path.join(run_dir, f"eval_results.json")''',
        '''    run_dir = os.path.join(config.runs_dir, config.run_name)
    # harness patch: allow sharded eval to write per-shard files via env
    _ev_suffix = os.environ.get("KB_EVAL_FILE_SUFFIX", "")
    eval_file_path = os.path.join(run_dir, f"eval_results{_ev_suffix}.json")''',
        "KB_EVAL_FILE_SUFFIX", "evalgen.eval_file_path env")

    print(f"[kb-patch] done on {kb}")

if __name__ == "__main__":
    main()
