#!/bin/bash
# ============================================================
# run_verilogeval.sh — VerilogEval v2 (spec-to-rtl + code-complete-iccad2023,
# 156 problems each). Drives the official verilog-eval Makefile → sv-generate →
# proxy, then compiles+runs each testbench with iverilog v12.
#
# Metric: sv-iv-analyze reports pass_rate = mean over problems of npass/nsamples,
#   i.e. average@SAMPLES. SAMPLES=1 TEMP=0 => pass@1; SAMPLES=4 TEMP=0.8 => average@4.
#
# Assumes the engine is already up on :$PROXY_PORT.
# NOTE: <served_name> must be registered in verilog-eval/scripts/sv-generate.
#
# Usage:  SAMPLES=4 TEMP=0.8 MAXTOK=49152 bash run_verilogeval.sh <served_name>
#   (MAXTOK env is honored — canonical VE max_tokens=49152; a positional $2
#    still overrides for ad-hoc runs.)
# ============================================================
set -e
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"
load_verilog_toolchain
export OPENAI_BASE_URL OPENAI_API_KEY

VE_ROOT="${VERILOGEVAL_ROOT:-$CODERBENCH_ROOT/verilog-eval}"
KEY=$1
# MAXTOK env wins (canonical eval-config: VE max_tokens=49152), then the
# positional $2 override, then the canonical default 49152. Was ${2:-32768}
# which silently ignored the MAXTOK env that every other runner reads — so
# registry profile MAXTOK=49152 never reached the Makefile.
MAX_TOKENS="${MAXTOK:-${2:-49152}}"
JOBS="${JOBS:-128}"
SAMPLES="${SAMPLES:-1}"
GEN_TEMP="${TEMP:-0}"
TOPP="${TOPP:-0.01}"
[ -z "$KEY" ] && { echo "usage: [SAMPLES=n TEMP=t] run_verilogeval.sh <served_name> [max_tokens]"; exit 1; }
# iverilog/vvp use TEMP and TMP as scratch-directory variables. Preserve the
# model sampling temperature in GEN_TEMP, then provide the simulator with a
# valid scratch directory.
scrub_temp_for_iverilog

require_engine
for TASK in ${TASKS:-spec-to-rtl code-complete-iccad2023}; do
  BUILD="$RESULTS_DIR/$KEY/$TASK"
  # Wipe before build because verilog-eval's incremental Makefile reuses any
  # generated .sv that already exists. A clean directory guarantees that the
  # current endpoint, parser, and configuration are applied to every sample.
  rm -rf "$BUILD"
  mkdir -p "$BUILD"; cd "$BUILD"
  "$VE_ROOT/configure" --with-model="$KEY" --with-task="$TASK" \
    --with-examples=0 --with-samples=$SAMPLES --with-temperature=$GEN_TEMP --with-top-p=$TOPP > configure.log 2>&1
  # verilog-eval's configure formats samples.mk with the optional `column`
  # utility. Regenerate the declarations directly when that utility is absent;
  # the formatting step is cosmetic and does not change Makefile semantics.
  if ! grep -q '_num_samples' "$BUILD/samples.mk" 2>/dev/null; then
    sed -e '/^\s*$/d' -e "s/^\(.*\)$/\1_num_samples = ${SAMPLES}/" \
      "$VE_ROOT/dataset_${TASK}/problems.txt" >> "$BUILD/samples.mk"
    echo "patched samples.mk: $(grep -c _num_samples "$BUILD/samples.mk") num_samples lines (SAMPLES=${SAMPLES})" >> configure.log
  fi
  make -j$JOBS MAX_TOKENS=$MAX_TOKENS > make.log 2>&1 || true
  echo "[$KEY/$TASK] $(grep pass_rate summary.txt 2>/dev/null | tail -1)"
done
log "DONE $KEY (SAMPLES=$SAMPLES GEN_TEMP=$GEN_TEMP)"
