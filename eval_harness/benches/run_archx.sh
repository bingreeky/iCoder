#!/bin/bash
# ============================================================
# run_archx.sh — ArchXBench (71 RTL designs, level-0..6). Drives run_archx.py:
# query proxy → iverilog -g2012 compile + vvp → self-check tb / golden compare.
#
# Metric: SAMPLES=1 => syntax/func pass@1. SAMPLES>1 TEMP>0 => n/t:
#   n = avg #syntactically-valid candidates (0..SAMPLES)
#   t = avg best-candidate testbench assertion-pass % (0..100)
#
# NOTE: max_tokens must be < engine max-model-len minus the longest prompt, or
# vLLM returns HTTP 500 (context overflow). Default MAXTOK=65536 needs the engine
# started with MAX_MODEL_LEN=81920 (80K) — config.sh's default. On a 32K-window
# debug box override MAXTOK down (e.g. MAXTOK=49152).
#
# Assumes the engine is already up on :$PROXY_PORT.
# Usage:  SAMPLES=5 TEMP=0.8 MAXTOK=49152 bash run_archx.sh <served_name> [limit]
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
load_verilog_toolchain

KEY=$1; LIMIT="${2:-${LIMIT:-0}}"
[ -z "$KEY" ] && { echo "usage: [SAMPLES=n TEMP=t MAXTOK=m] run_archx.sh <served_name> [limit]"; exit 1; }
SAMPLES="${SAMPLES:-1}"; GEN_TEMP="${TEMP:-0}"; WORKERS="${WORKERS:-64}"; MAXTOK="${MAXTOK:-49152}"
LIM_ARG=""; [ "$LIMIT" != "0" ] && LIM_ARG="--limit $LIMIT"
# iverilog/vvp read TEMP/TMP as their scratch dir; the sampling temperature
# is a float (e.g. 0.8) -> iverilog would write temp files into dir "0.8" ->
# every compile fails -> pass_rate=0. Capture into GEN_TEMP then scrub TEMP/TMP
# and pin TMPDIR. (Upstream A1 scrub.)
scrub_temp_for_iverilog

require_engine
log "ArchXBench $KEY (samples=$SAMPLES temp=$GEN_TEMP workers=$WORKERS maxtok=$MAXTOK)"
mkdir -p "$RESULTS_DIR/$KEY/archxbench"
"$SYS_PY" "$BENCHINFRA_ROOT/benches/run_archx.py" --base-url "$OPENAI_BASE_URL" --model "$KEY" \
  --out "$RESULTS_DIR/$KEY/archxbench" --workers $WORKERS --num-samples $SAMPLES --temperature $GEN_TEMP \
  --max-tokens $MAXTOK $LIM_ARG > "$RESULTS_DIR/$KEY/archx.log" 2>&1 || log "archx nonzero"
grep '\[archx\]' "$RESULTS_DIR/$KEY/archx.log" | tail -1
log "DONE $KEY"
