#!/bin/bash
# ============================================================
# run_realbench.sh — RealBench (60 module-level RTL tasks from real IP:
# SDC 14 / AES 6 / E203 40). Two stages:
#   1. generate: gen_realbench.py queries the proxy → samples/<key>/<system>.jsonl
#   2. verify:   RealBench/run_verify.py compiles+runs each verilator testbench.
#
# Metric: SAMPLES=1 => syntax/func pass@1. SAMPLES=5 => Syn@1/@5 + Func@1/@5
#   (run_verify's estimate_pass_at_k). Module-level only; formal is skipped.
#
# NOTE: E203's longest prompt is ~10K tokens; with a 65536 window use MAXTOK<=49152
# to leave headroom (else HTTP 500 on context overflow).
#
# Assumes the engine is already up on :$PROXY_PORT. Needs verilator+yosys (SETUP_ENV).
# Usage:  SAMPLES=5 TEMP=0.8 MAXTOK=49152 bash run_realbench.sh <served_name>
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
load_verilog_toolchain

RB="${RB_ROOT:-$CODERBENCH_ROOT/RealBench}"
RB_PY="${REALBENCH_PY:-$RB/.venv/bin/python}"
KEY=$1
[ -z "$KEY" ] && { echo "usage: [SAMPLES=n TEMP=t MAXTOK=m] run_realbench.sh <served_name>"; exit 1; }
SAMPLES="${SAMPLES:-1}"; GEN_TEMP="${TEMP:-0.8}"; WORKERS="${WORKERS:-60}"; MAXTOK="${MAXTOK:-49152}"
# verilator/iverilog in the verify step read TEMP/TMP as scratch dir; the
# sampling temperature is a float passed to gen via --temperature, not the env.
# Without scrubbing, iverilog writes temp files into dir "0.8" -> compile fails.
# Capture into GEN_TEMP then scrub TEMP/TMP and pin TMPDIR. (Upstream A1 scrub.)
scrub_temp_for_iverilog

require_engine
mkdir -p "/run/user/$(id -u)"   # run_verify uses it for tempfiles
# decrypt + generate problems.jsonl if the bench hasn't been prepped yet
[ -f "$RB/aes/aes_sbox/aes_sbox.md" ] || ( cd "$RB" && make decrypt >/tmp/rb_decrypt.log 2>&1 ) || true
[ -f "$RB/problems/aes/problems.jsonl" ] || ( cd "$RB" && "$RB_PY" generate_problem.py --task_level module >/tmp/rb_problems.log 2>&1 )

log "generate RealBench $KEY (samples=$SAMPLES temp=$GEN_TEMP workers=$WORKERS)"
mkdir -p "$RESULTS_DIR/$KEY"
RB_ROOT="$RB" "$SYS_PY" "$BENCHINFRA_ROOT/benches/gen_realbench.py" --base-url "$OPENAI_BASE_URL" \
  --model "$KEY" --workers $WORKERS --num-samples $SAMPLES --temperature $GEN_TEMP --max-tokens $MAXTOK \
  > "$RESULTS_DIR/$KEY/realbench_gen.log" 2>&1 || log "gen nonzero"
tail -3 "$RESULTS_DIR/$KEY/realbench_gen.log" 2>/dev/null || true

log "verify with run_verify.py (verilator, $SAMPLES samples)"
mkdir -p "$RESULTS_DIR/$KEY/realbench"
( cd "$RB" && "$RB_PY" run_verify.py --solution_name "$KEY" --task_level module --num_samples $SAMPLES ) \
  > "$RESULTS_DIR/$KEY/realbench/verify.log" 2>&1 || log "verify nonzero"
tail -3 "$RESULTS_DIR/$KEY/realbench/verify.log"
cp -rf "$RB/samples_after_verilator/$KEY" "$RESULTS_DIR/$KEY/realbench/" 2>/dev/null || true
log "DONE $KEY"
