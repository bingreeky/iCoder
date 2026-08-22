#!/usr/bin/env bash
# ============================================================
# kb_sharded_eval.sh — KernelBench eval split across N GPUs, one process per card
# (num_gpu_devices=1 each). Avoids the BrokenPipe crash that an 8-worker mp.Pool
# hits under shared-FS + CUDA init contention. Each process evals a problem-id
# range and writes its own eval_results_gpuN.json; merge with kb_merge_shards.py.
#
# Run this ONLY after the engine is stopped (eval needs the full VRAM).
# Usage:  bash kb_sharded_eval.sh <run_name> [backend=triton] [total=100] [timeout=300]
# ============================================================
source "$(dirname "$0")/../config.sh"
set -u
KB="${KERNELBENCH_ROOT:-$CODERBENCH_ROOT/KernelBench}"
KB_PY="${KERNELBENCH_PY:-$KB/.venv/bin/python}"
RUN=${1:?usage: kb_sharded_eval.sh <run_name> [backend] [total] [timeout]}; BACKEND=${2:-triton}; TOTAL=${3:-100}; TIMEOUT=${4:-300}
GPU_ARCH="${GPU_ARCH:-Hopper}"
# Derive level from run_name suffix _levelN (default 1) — eval_from_generations
# needs the explicit level, and the original hardcode only handled level 1.
case "$RUN" in
  *_level[0-9]) LV=${RUN##*_level} ;;
  *) LV=1 ;;
esac
source "$KB/activate_kb.sh" >/dev/null 2>&1
cd "$KB"
CHUNK=$(( (TOTAL + N_GPU - 1) / N_GPU ))
rm -f runs/$RUN/eval_results_gpu*.json 2>/dev/null
mkdir -p /tmp/logs/kb_shard
echo "=== KB sharded eval [$RUN] level=$LV backend=$BACKEND $N_GPU shards start $(date +%H:%M:%S) ==="
PIDS=""
for g in $(seq 0 $((N_GPU-1))); do
  start=$(( g * CHUNK + 1 )); end=$(( start + CHUNK - 1 ))
  [ $start -gt $TOTAL ] && continue
  [ $end -gt $TOTAL ] && end=$TOTAL
  if [ "${KB_HARDENED:-0}" = "1" ]; then
    # Hardened RLVR path (verify.kernelbench.verify_kb): 32 correctness trials,
    # deterministic seeding, triton launch counter + identity_hack +
    # framework_delegation anti-cheat, principled exception→model classify.
    # One disposable KB-venv subprocess per problem → per-problem CUDA
    # isolation. Output eval_results_hardened_gpuN.json (merged by
    # kb_merge_shards.py, which walks for dicts containing "compiled").
    CUDA_VISIBLE_DEVICES=$g KERNELBENCH_PY="$KB_PY" \
      "$SYS_PY" "$BENCHINFRA_ROOT/benches/kb_verify_correctness.py" \
      --run-name "$RUN" --level $LV --gpu $g \
      --subset-start $start --subset-end $end \
      --num-correct-trials "${KB_NUM_CORRECT_TRIALS:-32}" \
      --timeout $TIMEOUT \
      > "/tmp/logs/kb_shard/${RUN}_gpu${g}.log" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES=$g KB_EVAL_FILE_SUFFIX="_gpu${g}" "$KB_PY" scripts/eval_from_generations.py \
      run_name="$RUN" dataset_src=local level=$LV \
      num_gpu_devices=1 timeout=$TIMEOUT gpu_arch="['$GPU_ARCH']" backend="$BACKEND" \
      subset="($start,$end)" \
      > "/tmp/logs/kb_shard/${RUN}_gpu${g}.log" 2>&1 < /dev/null &
  fi
  PIDS="$PIDS $!"
  # NOTE: no setsid/disown — setsid makes the eval subprocess exit silently
  # without writing eval_results, and disown breaks `wait`.
  echo "  GPU$g -> [$start,$end]"
done
echo "=== $N_GPU shards launched $(date +%H:%M:%S) — merge with kb_merge_shards.py when done ==="
# When invoked from run_kernelbench.sh (automated), wait for all shards so the
# caller can merge immediately. Set KB_SHARD_WAIT=1 (or run interactively w/o it).
if [ "${KB_SHARD_WAIT:-0}" = "1" ] && [ -n "$PIDS" ]; then
  echo "  (waiting for shards to finish...)"
  for p in $PIDS; do wait "$p" 2>/dev/null || true; done
  echo "=== shards complete $(date +%H:%M:%S) ==="
fi
