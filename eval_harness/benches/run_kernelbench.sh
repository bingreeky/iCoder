#!/bin/bash
# ============================================================
# run_kernelbench.sh — KernelBench (CUDA/Triton kernel generation, level 1..4).
#
# Two stages (VRAM-aware, matches run_tritonbench.sh):
#   gen  : engine ONLINE. generate_samples.py → runs/<key>_level<N>/...kernel.py
#   eval : engine STOPPED. sharded eval_from_generations.py (one card per shard
#          via kb_sharded_eval.sh) → eval_results.json + fast_0. Eval needs the
#          full VRAM, so the engine MUST be stopped first.
#   all  : gen then eval in one call (legacy; eval contends with a live engine
#          unless eval_gpus is small — prefer the registry UP/DOWN split).
#
# Metric: compiled count + fast_0 = correctness (functional pass@1).
# Backend: cuda (default — canonical eval-config) / triton / cute / tilelang.
#
# Sharded eval (N_GPU processes, one per card) avoids the mp.Pool crash under
# shared-FS + CUDA init contention — see kb_sharded_eval.sh / kb_merge_shards.py.
#
# Usage:
#   bash run_kernelbench.sh gen  <served_name> [levels] [backend]
#   bash run_kernelbench.sh eval <served_name> [levels] [backend]
#   bash run_kernelbench.sh all  <served_name> [levels] [eval_gpus] [backend]
# ============================================================
set -uo pipefail
source "$(dirname "$0")/../config.sh"
source "$BENCHINFRA_ROOT/lib/common.sh"

MODE=${1:-all}
KEY=${2:?usage: run_kernelbench.sh <gen|eval|all> <served_name> [levels] [backend]}
LEVELS=${3:-1}; BACKEND=${4:-cuda}
# Backend default = cuda (canonical eval-config: "kernelbench要用cuda backend").
# Scored profiles also pass cuda explicitly via the registry extra field, so an
# unset $4 still lands on cuda. Supported: triton / cuda / cute / tilelang.
# only 'all' legacy mode honours eval_gpus (in-engine eval). Sharded eval uses N_GPU.
EVAL_GPUS=${5:-$N_GPU}
LEVELS=${LEVELS//,/ }
MAX_TOKENS="${MAXTOK:-58000}"; WORKERS="${WORKERS:-128}"; TEMPERATURE="${TEMP:-0}"
GPU_ARCH="${GPU_ARCH:-Hopper}"; GPU_LABEL="${GPU_LABEL:-H100}"

KB="${KERNELBENCH_ROOT:-$CODERBENCH_ROOT/KernelBench}"
KB_PY="${KERNELBENCH_PY:-$KB/.venv/bin/python}"
[ -d "$KB" ] || { echo "ERROR: KERNELBENCH_ROOT=$KB not found"; exit 1; }
[ -x "$KB_PY" ] || { echo "ERROR: KERNELBENCH_PY=$KB_PY not executable (install torch+triton+litellm)"; exit 1; }

# activate_kb.sh puts nvcc on PATH + reused-torch LD paths. Optional if absent.
[ -f "$KB/activate_kb.sh" ] && source "$KB/activate_kb.sh" >/dev/null 2>&1
export OPENAI_API_KEY SGLANG_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL
# eval_from_generations.py / generate_samples.py both `from kernelbench... import`,
# and the KB package isn't pip-installed on the host — needs KB/src on PYTHONPATH.
# Set at top level so BOTH gen() and eval() (and the kbeval-only profile, which
# skips gen) have it.
export PYTHONPATH="$KB/src:${PYTHONPATH:-}"
cd "$KB"

gen() {
  require_engine
  # KB's local query_server reads server_port/address from the "local" preset in
  # utils.py, which we patched to honor KB_SERVER_PORT/KB_SERVER_ADDRESS env.
  export KB_SERVER_PORT="$PROXY_PORT" KB_SERVER_ADDRESS=localhost PYTHONPATH="$KB/src:${PYTHONPATH:-}"
  for LV in $LEVELS; do
    RUN="${KEY}_level${LV}"
    log "KB level $LV gen (backend=$BACKEND, workers=$WORKERS)"
    "$KB_PY" scripts/generate_samples.py dataset_src=local level=$LV run_name="$RUN" \
      server_type=local model_name="$KEY" \
      temperature=$TEMPERATURE max_tokens=$MAX_TOKENS num_workers=$WORKERS backend=$BACKEND check_kernel=False \
      > /tmp/kb_gen_${RUN}.log 2>&1 || log "generate nonzero (see /tmp/kb_gen_${RUN}.log)"
  done
  log "KB gen done -> $KB/runs/${KEY}_level*"
}

eval() {
  # engine should be STOPPED for full-VRAM eval. Do NOT require_engine here.
  for LV in $LEVELS; do
    RUN="${KEY}_level${LV}"
    DEST="$RESULTS_DIR/$KEY/kernelbench/level${LV}"; mkdir -p "$DEST"
    if [ "$N_GPU" -gt 1 ]; then
      log "KB level $LV sharded eval ($N_GPU shards, backend=$BACKEND)"
      # TOTAL = #problems in this level (count .py files in KernelBench/level<N>).
      # Overshoot is harmless (empty ranges skipped), undershoot drops problems.
      # KB_LEVEL_TOTAL env overrides if the layout differs.
      LDIR="$KB/KernelBench/level${LV}"
      if [ -z "${KB_LEVEL_TOTAL:-}" ] && [ -d "$LDIR" ]; then
        TOTAL=$(find "$LDIR" -maxdepth 1 -name '*.py' -type f 2>/dev/null | wc -l)
      else
        TOTAL="${KB_LEVEL_TOTAL:-100}"
      fi
      [ "$TOTAL" -lt 1 ] && TOTAL=100
      KB_SHARD_WAIT=1 bash "$BENCHINFRA_ROOT/benches/kb_sharded_eval.sh" "$RUN" "$BACKEND" "$TOTAL" \
        > /tmp/kb_shard_${RUN}.log 2>&1 || log "sharded eval nonzero"
      "$KB_PY" "$BENCHINFRA_ROOT/benches/kb_merge_shards.py" "$RUN" "$KB" \
        > /tmp/kb_merge_${RUN}.log 2>&1 || log "merge nonzero"
    else
      log "KB level $LV eval (single GPU, backend=$BACKEND)"
      "$KB_PY" scripts/eval_from_generations.py dataset_src=local level=$LV run_name="$RUN" \
        gpu="$GPU_LABEL" gpu_arch="['$GPU_ARCH']" num_gpu_devices=1 backend="$BACKEND" \
        > /tmp/kb_eval_${RUN}.log 2>&1 || log "eval nonzero"
    fi
    cp -f "$KB/runs/$RUN/eval_results.json" "$DEST/" 2>/dev/null && log "saved $DEST" || log "no eval_results.json for level $LV"
  done
  log "KB eval done (levels: $LEVELS)"
}

case "$MODE" in
  gen)  gen ;;
  eval) eval ;;
  all)  gen; EVAL_GPUS=${EVAL_GPUS:-$N_GPU}; eval ;;
  *)    echo "usage: run_kernelbench.sh <gen|eval|all> <served_name> [levels] [backend]"; exit 1 ;;
esac
