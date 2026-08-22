#!/usr/bin/env bash
# sync_expand.sh — re-copy SFT/expand -> eval_harness/expand (vendored copy).
#
# eval_harness/expand is a synced copy, NOT a fork. Run this after editing
# SFT/expand to keep the two in lockstep (see eval_harness/expand/README.md).
#
# Usage:  bash setup/sync_expand.sh [SFT_ROOT]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"            # eval_harness/
# expand/ is vendored in this repo; by default sync from the repo itself. Pass
# SFT_ROOT (or export it) to sync from a separate checkout of the teacher pkg.
SFT_ROOT="${1:-${SFT_ROOT:-$HERE}}"
SRC="$SFT_ROOT/expand"
DST="$HERE/expand"

if [ ! -d "$SRC" ]; then
  echo "ERROR: $SRC not found (set SFT_ROOT or pass it as \$1)" >&2
  exit 1
fi

echo "[sync_expand] $SRC -> $DST"
# Atomic-ish: stage into a temp dir, then swap. Preserves the live copy if cp
# fails midway. __pycache__ is excluded (rebuilt on first import).
TMP="$(mktemp -d -p "$HERE")"
# cp -a preserves attrs; find prunes bytecode (rebuilt on first import).
cp -a "$SRC/." "$TMP/expand/"
find "$TMP/expand" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$DST"
mv "$TMP/expand" "$DST"
rmdir "$TMP" 2>/dev/null || true

echo "[sync_expand] done. Smoke-checking imports..."
cd "$HERE"
python3 - <<'PY' || { echo "[sync_expand] WARN: import smoke failed" >&2; exit 2; }
import sys; sys.path.insert(0, ".")
from expand.llm import LLM, make_llm, collect_api_keys
from expand.datasets.tritonbench_g import TritonBenchGAdapter
from expand.methods.inversecoder import _extract_triton_interface
from expand.methods._common import check_v4pro_format, extract_v4pro_answer, synthesize_v4pro_wrap
from expand.methods._perturb_common import _extract_tbt_func_interface
print("[sync_expand] import smoke OK")
PY
