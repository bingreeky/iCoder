#!/bin/bash
# ============================================================
# run_tritonbench.sh — TritonBench G/T eval-the-model runner.
#
# Two-stage (mirrors run_kernelbench.sh's VRAM-aware split):
#   gen  : engine ONLINE. Prompt the model with each seed's spec via the SFT
#          teacher client (teacher_triton_rollout_tbg.py, NO --verify) and
#          write rollout.jsonl. Generation only hits the proxy — no GPU.
#   eval : engine STOPPED. Shard rollout.jsonl across N_GPU, run
#          tbg_verify_correctness.py (one shard per card, --gpu $g) which
#          rebuilds candidate.py = teacher_code + ref test_block and diffs
#          result_gold (TBG) / test_results (TBT) via torch.allclose.
#          Merge shards -> verified.jsonl + fast_0.
#   all  : gen then eval in one call. Only safe if the engine is stopped
#          between (use the registry UP/DOWN split instead for full VRAM eval).
#
# Reuses SFT primitives in place (single source of truth):
#   $BENCHINFRA_ROOT/benches/teacher_triton_rollout_tbg.py (vendored)   (generate)
#   $BENCHINFRA_ROOT/benches/tbg_verify_correctness.py (vendored)       (sharded verify)
#   benches/tbg_dump_seeds.py  +  benches/tbg_merge_shards.py  (this repo)
#
# Usage:
#   bash run_tritonbench.sh gen  <served_name> [split=g|t] [samples]
#   bash run_tritonbench.sh eval <served_name> [split=g|t]
#   bash run_tritonbench.sh all  <served_name> [split=g|t] [samples]
# ============================================================
set -uo pipefail
# Preserve caller-provided interpreter overrides across config.sh (mirrors the
# _PRESET_* guard in run_all.sh). config.sh sources config.local.sh, which on
# every host unconditionally `export SYS_PY=/usr/local/bin/python3` — the image
# python (torch 2.3.1 / triton 2.3.1). That clobbers a caller's SYS_PY: e.g.
# tbg_eval_only.sh exports SYS_PY=.venv-vllm/bin/python (torch 2.11 / triton 3.6)
# so the TBG verify subprocess and the per-variant ref exec ([sys.executable,
# script]) run on triton 3.6, where the reference kernels' APIs that only
# exist in triton >=3.1 (tl.interleave / tl.cast / tl.rsqrt /
# triton.language.extra.cuda.libdevice / torch._inductor.runtime /
# triton.jit(launch_metadata=)) are present. Without this guard, run_tritonbench
# re-sources config.sh and re-clobbers SYS_PY back to the image python, so ~28
# of ~29 "skip" rows are spurious trusted_reference_failure from version-mismatch
# AttributeError — under-counting every model's TBG correct.
_PRESET_SYS_PY="${SYS_PY:-}"
_PRESET_VLLM_PY="${VLLM_PY:-}"
_PRESET_KB_PY="${KERNELBENCH_PY:-}"
_PRESET_RESULTS_DIR="${RESULTS_DIR:-}"
source "$(dirname "$0")/../config.sh"
[ -n "$_PRESET_SYS_PY" ] && export SYS_PY="$_PRESET_SYS_PY"
[ -n "$_PRESET_VLLM_PY" ] && export VLLM_PY="$_PRESET_VLLM_PY"
[ -n "$_PRESET_KB_PY" ] && export KERNELBENCH_PY="$_PRESET_KB_PY"
[ -n "$_PRESET_RESULTS_DIR" ] && export RESULTS_DIR="$_PRESET_RESULTS_DIR"
source "$BENCHINFRA_ROOT/lib/common.sh"

MODE=${1:-all}
KEY=${2:?usage: run_tritonbench.sh <gen|eval|all> <served_name> [split=g|t] [samples]}
SPLIT=${3:-g}
SAMPLES=${4:-1}

[ "$SPLIT" != "g" ] && [ "$SPLIT" != "t" ] && { echo "split must be g|t"; exit 1; }

OUT="$RESULTS_DIR/$KEY/tritonbench_${SPLIT}"
mkdir -p "$OUT"
SEEDS="$OUT/seeds.jsonl"
ROLLOUT="$OUT/rollout.jsonl"

TEMP="${TEMP:-0}"
MAXTOK="${MAXTOK:-58000}"
WORKERS="${WORKERS:-16}"
TBG_TIMEOUT="${TBG_TIMEOUT:-180}"

TEACHER="$BENCHINFRA_ROOT/benches/teacher_triton_rollout_tbg.py"
VERIFY="$BENCHINFRA_ROOT/benches/tbg_verify_correctness.py"
DUMP="$BENCHINFRA_ROOT/benches/tbg_dump_seeds.py"
MERGE="$BENCHINFRA_ROOT/benches/tbg_merge_shards.py"

# --- Stage A: generate rollout (engine online) ---
gen() {
  require_engine
  # teacher uses the SFT expand/llm.py client, which reads EXPAND_LLM_* env.
  export EXPAND_LLM_BASE_URL="$OPENAI_BASE_URL"
  export EXPAND_LLM_API_KEY="$OPENAI_API_KEY"
  export EXPAND_LLM_MODEL="$KEY"

  # Emit original bench seeds if not already present (resume-friendly).
  if [ ! -s "$SEEDS" ]; then
    log "dumping $SPLIT seeds -> $SEEDS"
    PYTHONPATH="$BENCHINFRA_ROOT" "$SYS_PY" "$DUMP" "$SPLIT" "$SEEDS" "$TRITONBENCH_ROOT" || {
      echo "ERROR: seed dump failed (is $TRITONBENCH_ROOT present?)"; exit 1; }
  fi

  log "TBG-${SPLIT} gen $KEY (temp=$TEMP maxtok=$MAXTOK workers=$WORKERS samples=$SAMPLES)"
  # No --verify => pure generate; rollout rows are always written (resume-safe).
  # NOTE: teacher samples 1/seed; average@K (SAMPLES>1) needs --num-samples
  # support (TODO) — for now SAMPLES>1 re-runs gen K times with temp>0 and
  # appends (ids dedup on merge by last-wins, so use distinct run dirs for K).
  PYTHONPATH="$BENCHINFRA_ROOT:$BENCHINFRA_ROOT/benches" "$SYS_PY" "$TEACHER" \
    --input "$SEEDS" --output "$ROLLOUT" \
    --temperature "$TEMP" --max-tokens "$MAXTOK" --concurrency "$WORKERS" \
    2>&1 | tee "$OUT/gen.log" | tail -20
  log "gen done -> $ROLLOUT"
}

# --- Stage B: sharded verify (engine STOPPED for full VRAM) ---
eval() {
  [ -s "$ROLLOUT" ] || { echo "ERROR: no $ROLLOUT — run 'gen' first (engine up)"; exit 1; }
  # engine should be stopped; do NOT require_engine here.
  # TBG_RESUME=1: KEEP existing shard_verified_*.jsonl so tbg_verify skips ids
  # already verified (re-run recovers rows lost to a shard worker being killed
  # mid-eval — e.g. 3/4 shards cut at 4/24/13 of 46, job still "Succeeded" via
  # partial-shard merge). Fresh run (default): rm all shards first.
  if [ "${TBG_RESUME:-0}" != 1 ]; then
    rm -f "$OUT"/shard_input_*.jsonl "$OUT"/shard_verified_*.jsonl 2>/dev/null
  fi

  # Round-robin split rollout -> N_GPU shard inputs (deterministic by line#).
  awk -v n="$N_GPU" '{print > ("'"$OUT"'/shard_input_" (NR-1)%n ".jsonl")}' "$ROLLOUT"
  log "TBG-${SPLIT} eval $KEY ($N_GPU shards, timeout=${TBG_TIMEOUT}s)"
  # Per-shard verify logs go to shared storage ($OUT), NOT /tmp — /tmp is
  # host-local and LOST on job exit, so a verify subprocess that crashes at
  # import / on row 1 (silently swallowed by `wait ... || true` below) left
  # the job "Succeeded" with unchanged shard_verified (0 new rows verified).
  PIDS=""
  for g in $(seq 0 $((N_GPU-1))); do
    [ -s "$OUT/shard_input_$g.jsonl" ] || continue
    # --gpu $g pins via the script's internal CUDA_VISIBLE_DEVICES=$g
    # (do NOT set outer CUDA_VISIBLE_DEVICES — it would be overwritten).
    # NOTE: no setsid/disown — setsid makes the verify subprocess (python -c
    # in verify_one) exit silently without writing shard_verified_*.jsonl,
    # and disown breaks `wait`. Plain `&` + `wait $pid` works.
    PYTHONPATH="$BENCHINFRA_ROOT:$BENCHINFRA_ROOT/benches" "$SYS_PY" "$VERIFY" \
      --input "$OUT/shard_input_$g.jsonl" \
      --output "$OUT/shard_verified_$g.jsonl" \
      --gpu "$g" --timeout "$TBG_TIMEOUT" \
      > "$OUT/shard_verify_${g}.log" 2>&1 < /dev/null &
    PIDS="$PIDS $!"
    echo "  GPU$g launched (pid $!)"
  done
  for p in $PIDS; do wait "$p"; rc=$?; [ "$rc" != 0 ] && log "  !! shard verify pid $p exited rc=$rc (see $OUT/shard_verify_*.log)"; done
  log "shards done — merging"
  PYTHONPATH="$BENCHINFRA_ROOT" "$SYS_PY" "$MERGE" "$OUT"
  log "DONE TBG-${SPLIT} $KEY -> $OUT/verified.jsonl"
}

case "$MODE" in
  gen)  gen ;;
  eval) eval ;;
  all)  gen; eval ;;
  *)    echo "usage: run_tritonbench.sh <gen|eval|all> <served_name> [g|t] [samples]"; exit 1 ;;
esac
