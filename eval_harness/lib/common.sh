# ============================================================
# lib/common.sh — shared helpers sourced by every bench runner.
# Assumes config.sh has already been sourced.
# ============================================================

log() { echo "[$(basename "${0%.sh}") $(date +%H:%M:%S)] $*"; }

# Fail early if the engine isn't serving on :$PROXY_PORT.
require_engine() {
  curl -s --max-time 3 "http://localhost:$PROXY_PORT/" | grep -q ok || {
    echo "ERROR: no engine on :$PROXY_PORT. Start it first:"
    echo "  bash $BENCHINFRA_ROOT/engine/serve_vllm.sh <model_path> <served_name>"
    exit 1
  }
}

# Put iverilog v12 on PATH front + source setup_env for verilator/yosys/CUDA compat.
load_verilog_toolchain() {
  [ -f "$SETUP_ENV" ] && source "$SETUP_ENV" >/dev/null 2>&1
  [ -d "$IVERILOG12_BIN" ] && export PATH="$IVERILOG12_BIN:$PATH"
  # verilog-eval's configure pipes through `column -t` to build samples.mk. If
  # util-linux's column is missing the pipe silently yields an EMPTY samples.mk,
  # so --with-samples=K is dropped and average@K degrades to single-sample
  # (pass_rate looks fine but is really average@1). Warn loudly so it's caught.
  command -v column >/dev/null 2>&1 || \
    echo "[warn] 'column' not found (util-linux); VerilogEval average@K will silently fall back to 1 sample. Install column or add a shim to PATH." >&2
}

# Scrub TEMP/TMP so iverilog/vvp/verilator don't treat a *sampling temperature*
# (a float like 0.8) as their scratch dir. iverilog reads TEMP/TMP as a path;
# if the sampling temp leaks in, every compile writes into dir "0.8" and fails
# → pass_rate=0. Call once near the top of every RTL bench runner, after
# config.sh. Export TMPDIR back to a sane default so the tools still get a
# writable scratch location.
scrub_temp_for_iverilog() {
  unset TEMP TMP
  export TMPDIR="${TMPDIR:-/tmp}"
}
