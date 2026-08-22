#!/bin/bash
# ============================================================
# run_cvdp.sh — CVDP cid003 eval (SystemVerilog, cocotb + iverilog).
#
# CVDP is CPU-parallel (cocotb harnesses), NOT GPU-parallel: -t sets the
# simultaneous harness count. The model is served by the shared engine proxy;
# this runner just points the CVDP factory at :$PROXY_PORT and shells out to
# the external CVDP_local/run_samples.py (the authoritative harness).
#
# Engine must be ONLINE (queries the API throughout). No DOWN phase.
#
# The served name ($KEY) MUST be a CVDP-registered alias — see
# $CVDP_ROOT/local_extensions/eval_5models_factory.py:ALIASES (add yours via
# setup/apply_cvdp_patches.py <cvdp_root> <served_name>).
#
# Usage:  SAMPLES=1 bash run_cvdp.sh <served_name>
# ============================================================
set -uo pipefail
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"

KEY=${1:?usage: run_cvdp.sh <served_name>}
SAMPLES="${SAMPLES:-1}"
HARNESS_THREADS="${HARNESS_THREADS:-8}"
OUT="$RESULTS_DIR/$KEY/cvdp"
mkdir -p "$OUT"

require_engine
[ -d "$CVDP_ROOT" ] || { echo "ERROR: CVDP_ROOT=$CVDP_ROOT not found"; exit 1; }

# common.sh loads CVDP_LOCAL_EDA_ROOT (iverilog/verilator/yosys on PATH) + caches.
# It also sets CVDP_VENV to the venv DIR (convention: $CVDP_VENV/bin/python).
# shellcheck disable=SC1091
source "$CVDP_ROOT/scripts/common.sh"

# Resolve a working python. CVDP_VENV may be a dir (common.sh convention), a
# python binary (config.local.sh), or a broken symlink (.venv-cvdp/bin/python
# -> a python3.11 the base image lacks). Fall back to
# $SYS_PY (image python, has cocotb pip-installed by the image setup).
if [ -d "$CVDP_VENV" ] && [ -x "$CVDP_VENV/bin/python" ]; then
  CVDP_PY="$CVDP_VENV/bin/python"
elif [ -n "${CVDP_VENV:-}" ] && [ -x "$CVDP_VENV" ] && [ ! -d "$CVDP_VENV" ]; then
  CVDP_PY="$CVDP_VENV"
else
  CVDP_PY="$SYS_PY"
fi
log "CVDP python: $CVDP_PY"

# Point the CVDP factory's DeepSeekOpenAICompatible client at our shared proxy.
export DEEPSEEK_OPENAI_API_BASE="$OPENAI_BASE_URL"
export DEEPSEEK_OPENAI_API_KEY="$OPENAI_API_KEY"
export DEEPSEEK_SERVED_NAME="$KEY"
# DeepSeekOpenAICompatible.prompt() defaults timeout to 60s then reads
# MODEL_TIMEOUT. A local 27B-class vLLM model has slow TTFT + long codegen;
# 60s fires mid-generation -> "Request timed out" retry loop. Per the directive
# "不要设置时间限制", set to 86400 (24h) = effectively no practical limit so
# slow generations complete instead of being killed (the old 600s ceiling
# dropped the model CVDP rows under cocotb concurrency).
export MODEL_TIMEOUT=86400  # FORCE 86400 (not ${MODEL_TIMEOUT:-86400}) — local.env sets MODEL_TIMEOUT=180 which the :- default doesn't override (180 already in env). 180s < CVDP gen+queue time (8 cocotb threads → queue) → "Request timed out".

log "CVDP $KEY (dataset=$(basename "$CVDP_DATASET") samples=$SAMPLES threads=$HARNESS_THREADS)"
cd "$CVDP_ROOT"
# Prefix is the run subdir under CVDP_root/runs/; we also mirror results into
# $OUT for summarize.sh. --execution-backend host = run iverilog locally (not docker).
# --llm is REQUIRED: without it run_benchmark sets golden=(not args.llm)=True and
# runs in Golden Mode (uses the golden reference, never calls the model -> empty
# .sv -> 0/78). --llm flips to model-eval mode.
"$CVDP_PY" run_samples.py \
  -f "$CVDP_DATASET" \
  -m "$KEY" \
  -c local_extensions/eval_5models_factory.py \
  --execution-backend host \
  --n-samples "$SAMPLES" \
  --prefix "$KEY" \
  -t "$HARNESS_THREADS" \
  --llm \
  2>&1 | tee "$OUT/run.log" | tail -30

# Mirror raw_result.json files into the harness results dir for summarize.
RUN_DIR="$CVDP_ROOT/runs/$KEY"
if [ -d "$RUN_DIR" ]; then
  find "$RUN_DIR" -name raw_result.json -path '*/sample_*' \
    -exec cp -f --parents {} "$OUT/" 2>/dev/null \; || true
  log "mirrored $(find "$OUT" -name raw_result.json | wc -l) raw_result.json -> $OUT"
fi
log "DONE CVDP $KEY -> $OUT"
