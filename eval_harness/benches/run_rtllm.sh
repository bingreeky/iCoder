#!/bin/bash
# ============================================================
# run_rtllm.sh — RTLLM v2 (50 designs). Drives run_rtllm.py: query proxy →
# extract module → iverilog -g2012 compile + vvp run → pass/fail.
#
# Metric: SAMPLES=1 => syntax/func pass@1. SAMPLES>1 TEMP>0 => average@SAMPLES
#   (mean over designs of #passing candidates / SAMPLES) + legacy pass@1.
#
# Assumes the engine is already up on :$PROXY_PORT.
# Usage:  SAMPLES=4 TEMP=0.8 bash run_rtllm.sh <served_name>
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
load_verilog_toolchain

KEY=$1
[ -z "$KEY" ] && { echo "usage: [SAMPLES=n TEMP=t] run_rtllm.sh <served_name>"; exit 1; }
SAMPLES="${SAMPLES:-1}"; GEN_TEMP="${TEMP:-0}"; WORKERS="${WORKERS:-64}"; MAXTOK="${MAXTOK:-49152}"
# iverilog/vvp read TEMP/TMP as their scratch dir; the sampling temperature
# is a float (e.g. 0.8) -> iverilog would write temp files into dir "0.8" ->
# every compile fails -> pass_rate=0. Capture into GEN_TEMP then scrub TEMP/TMP
# and pin TMPDIR. (Upstream A1 scrub.)
scrub_temp_for_iverilog

require_engine
log "RTLLM $KEY (samples=$SAMPLES temp=$GEN_TEMP workers=$WORKERS)"
mkdir -p "$RESULTS_DIR/$KEY/rtllm"
"$SYS_PY" "$BENCHINFRA_ROOT/benches/run_rtllm.py" --base-url "$OPENAI_BASE_URL" --model "$KEY" \
  --out "$RESULTS_DIR/$KEY/rtllm" --workers $WORKERS --num-samples $SAMPLES --temperature $GEN_TEMP --max-tokens $MAXTOK \
  > "$RESULTS_DIR/$KEY/rtllm.log" 2>&1 || log "rtllm nonzero"
grep '\[rtllm\]' "$RESULTS_DIR/$KEY/rtllm.log" | tail -1
log "DONE $KEY"
