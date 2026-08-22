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
# iverilog/vvp read TEMP/TMP as their scratch dir; the sampling temperature
# passed to sv-generate is a float (e.g. 0.8) -> iverilog would try to write
# temp files into a dir named "0.8" -> every compile fails -> pass_rate=0.
# Capture the model temperature into GEN_TEMP, then scrub TEMP/TMP and pin
# TMPDIR so iverilog uses a real scratch dir. (Upstream A1 scrub.)
scrub_temp_for_iverilog

require_engine
for TASK in ${TASKS:-spec-to-rtl code-complete-iccad2023}; do
  BUILD="$RESULTS_DIR/$KEY/$TASK"
  # Wipe before build: verilog-eval uses `make` incrementally and SKIPS any
  # problem whose generated .sv already exists, so a rerun without cleaning
  # silently reuses STALE .sv (pre-fence-fix, possibly fence-leaked -> fake
  # low). rm -rf forces full regeneration so the A4 [BEGIN]/[DONE] fence-strip
  # patch applies to every problem. (INFRA_LOG A7.)
  rm -rf "$BUILD"
  mkdir -p "$BUILD"; cd "$BUILD"
  "$VE_ROOT/configure" --with-model="$KEY" --with-task="$TASK" \
    --with-examples=0 --with-samples=$SAMPLES --with-temperature=$GEN_TEMP --with-top-p=$TOPP > configure.log 2>&1
  # verilog-eval's configure builds samples.mk via `sed ... | column -t`. If the
  # `column` binary is absent (stripped util-linux on this host), that pipeline
  # emits nothing -> samples.mk has no `Prob*_num_samples` lines -> the Makefile
  # defaults to 1 sample per problem -> a "SAMPLES=4" run silently becomes @1.
  # `column -t` is cosmetic (column alignment) only, so regenerate the lines
  # directly. No-op when configure already wrote them (column present).
  if ! grep -q '_num_samples' "$BUILD/samples.mk" 2>/dev/null; then
    sed -e '/^\s*$/d' -e "s/^\(.*\)$/\1_num_samples = ${SAMPLES}/" \
      "$VE_ROOT/dataset_${TASK}/problems.txt" >> "$BUILD/samples.mk"
    echo "patched samples.mk: $(grep -c _num_samples "$BUILD/samples.mk") num_samples lines (SAMPLES=${SAMPLES})" >> configure.log
  fi
  make -j$JOBS MAX_TOKENS=$MAX_TOKENS > make.log 2>&1 || true
  echo "[$KEY/$TASK] $(grep pass_rate summary.txt 2>/dev/null | tail -1)"
done
log "DONE $KEY (SAMPLES=$SAMPLES GEN_TEMP=$GEN_TEMP)"
