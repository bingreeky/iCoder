#!/bin/bash
# ============================================================
# run_all.sh — one-click eval across every registered benchmark.
#
# Two-stage engine flow (so the GPU-VRAM-heavy verify stages get full VRAM):
#   1. start the engine ONCE on :$PROXY_PORT
#   2. BENCHES_UP  — run while the engine is online: RTL benches + CVDP +
#      KB/TBG/TBT *generate* (model query only, no GPU)
#   3. stop the engine (frees VRAM)
#   4. BENCHES_DOWN — KB/TBG/TBT *verify*: sharded across N_GPU, one card per
#      shard, then merged
#   5. summarize
#
# The bench lists live in benches/registry.sh (BENCHES_UP / BENCHES_DOWN).
# Add a line there and it runs here automatically.
#
# Usage:
#   bash run_all.sh <model_path> <served_name> [max_model_len]
#   bash run_all.sh --no-engine <served_name>     # engine already up; UP group only
#   bash run_all.sh --down-only <served_name>     # engine already stopped; DOWN only
#   bash run_all.sh --group up   <served_name>    # just the UP group (engine up)
#   bash run_all.sh --group down <served_name>    # just the DOWN group (engine down)
#
# NOTE: <served_name> must be a CVDP-registered alias (whatever you registered
# in $CVDP_ROOT/local_extensions/eval_5models_factory.py:ALIASES via
# setup/apply_cvdp_patches.py) AND whatever the verilog-eval/RTLLM runners
# expect — pick one that satisfies all.
# ============================================================
set -u
# Preserve caller-provided interpreter overrides across config.sh. config.sh
# sources config.local.sh, which may hardcode VLLM_PY=/usr/local/bin/python3
# (image python: torch2.3, NO vllm 0.25). A caller that wants the isolated
# .venv-vllm python sets VLLM_PY BEFORE invoking run_all — but this source
# clobbers it, so serve_vllm tries
# the image python, fails to import vllm, the engine never starts, and every
# bench silently gets connection-refused -> all-empty results + job "Succeeded".
# Caller exports (set by the host setup that knows the venv) win over config.
_PRESET_VLLM_PY="${VLLM_PY:-}"
_PRESET_SYS_PY="${SYS_PY:-}"
_PRESET_KB_PY="${KERNELBENCH_PY:-}"
_PRESET_RESULTS_DIR="${RESULTS_DIR:-}"
source "$(dirname "$0")/config.sh"
[ -n "$_PRESET_VLLM_PY" ] && export VLLM_PY="$_PRESET_VLLM_PY"
[ -n "$_PRESET_SYS_PY" ] && export SYS_PY="$_PRESET_SYS_PY"
[ -n "$_PRESET_KB_PY" ] && export KERNELBENCH_PY="$_PRESET_KB_PY"
[ -n "$_PRESET_RESULTS_DIR" ] && export RESULTS_DIR="$_PRESET_RESULTS_DIR"
source "$BENCHINFRA_ROOT/lib/common.sh"
source "$BENCHINFRA_ROOT/benches/registry.sh"

# --- flag parse ---
API_MODE=0            # 1 = start proxy to external API (no local vLLM, no VRAM)
PROFILE="${PROFILE:-}"   # "" = full registry; "table" = table-column subset.
                         # Honors the PROFILE env var; --profile <name> overrides it.
START_ENGINE=1        # 1 = start engine/proxy here; 0 = already up / group-only
GROUP="all"           # all | up | down
while [ $# -gt 0 ]; do
  case "$1" in
    --api)        API_MODE=1; shift ;;
    --profile)    PROFILE="$2"; shift 2 ;;
    --no-engine)  START_ENGINE=0; GROUP=up; shift ;;
    --down-only)  START_ENGINE=0; GROUP=down; shift ;;
    --group)      START_ENGINE=0; GROUP="$2"; shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

USAGE="usage: run_all.sh [--api] [--profile table] <model_path|api_base> <served_name> [max_model_len]
       | --no-engine <key> | --down-only <key> | --group up|down <key>"

# Pick the registry arrays per profile. Profile name -> BENCHES_UP_<NAME> /
# BENCHES_DOWN_<NAME> (uppercased). "table" / "retry" / any in registry.sh.
if [ -z "$PROFILE" ]; then
  UP_ARR=BENCHES_UP; DN_ARR=BENCHES_DOWN
else
  # PROFILE becomes a bash variable-name suffix, so reject anything but
  # [A-Za-z0-9_] up front rather than relying on declare -p to incidentally
  # reject metacharacters.
  case "$PROFILE" in
    *[!A-Za-z0-9_]*) echo "profile='$PROFILE' must be [A-Za-z0-9_] only"; exit 1;;
  esac
  SUFFIX=$(echo "$PROFILE" | tr '[:lower:]' '[:upper:]')
  UP_ARR=BENCHES_UP_${SUFFIX}; DN_ARR=BENCHES_DOWN_${SUFFIX}
  # Verify both arrays are DECLARED. UP may be intentionally empty (e.g. kbeval
  # profile that only re-runs DOWN), so don't `:?`-fail on empty — just check
  # the variable exists. DN must be non-empty (else the profile does nothing).
  declare -p "$UP_ARR" >/dev/null 2>&1 || { echo "profile=$PROFILE needs $UP_ARR in registry.sh"; exit 1; }
  declare -p "$DN_ARR" >/dev/null 2>&1 || { echo "profile=$PROFILE needs $DN_ARR in registry.sh"; exit 1; }
fi

if [ "$API_MODE" = 1 ]; then
  # --api [api_base] <served_name> [max_model_len]   (api_base optional via env)
  API_BASE=${1:-${EXTERNAL_API_URL:-}}
  KEY=${2:?--api needs <served_name>}
  MLEN=${3:-$MAX_MODEL_LEN}
  [ -z "$API_BASE" ] && { echo "$USAGE"; exit 1; }
elif [ "$START_ENGINE" = 1 ]; then
  MODEL_PATH=${1:?}; KEY=${2:?}; MLEN=${3:-$MAX_MODEL_LEN}
else
  KEY=${1:?}
fi
[ -z "${KEY:-}" ] && { echo "$USAGE"; exit 1; }

run_group() {
  local arr_name="$1"
  local entries=()
  eval "entries=(\"\${${arr_name}[@]}\")"
  for entry in "${entries[@]}"; do
    name=$(echo "$entry"   | cut -d'|' -f1 | xargs)
    runner=$(echo "$entry" | cut -d'|' -f2 | xargs)
    mode=$(echo "$entry"   | cut -d'|' -f3 | xargs)
    extra=$(echo "$entry"  | cut -d'|' -f4 | xargs)
    denv=$(echo "$entry"   | cut -d'|' -f5 | xargs)
    log "############ $name ############"
    # mode/extra are intentionally unquoted so empty collapses and multi-token
    # extras (e.g. "g 1") split into positional args.
    env $denv bash "$BENCHINFRA_ROOT/benches/$runner" $mode "$KEY" $extra \
      || log "$name nonzero (continuing)"
  done
}

# --- Stage A: engine/proxy online, run UP group ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "up" ]; then
  if [ "$START_ENGINE" = 1 ] && [ "$API_MODE" = 1 ]; then
    log "starting API proxy for $KEY -> $API_BASE"
    bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" api "$API_BASE" "$KEY"
  elif [ "$START_ENGINE" = 1 ]; then
    log "starting engine for $KEY ($MODEL_PATH) mlen=${MLEN:-$MAX_MODEL_LEN}"
    bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" "$MODEL_PATH" "$KEY" "$MAX_NUM_SEQS" "${MLEN:-$MAX_MODEL_LEN}"
  else
    require_engine
  fi
  log "================ STAGE A (engine online): UP group ================"
  run_group "$UP_ARR"
fi

# --- Stage B: stop engine (local vLLM only — frees VRAM), run DOWN group ---
if [ "$GROUP" = "all" ] || [ "$GROUP" = "down" ]; then
  if [ "$START_ENGINE" = 1 ] && [ "$API_MODE" != 1 ]; then
    log "stopping engine (free VRAM for verify stages)"
    bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" stop
  fi
  # API mode: no local vLLM, no VRAM to free — verify has full GPU already.
  log "================ STAGE B (engine stopped): DOWN group ================"
  run_group "$DN_ARR"
fi

if [ "$START_ENGINE" = 1 ]; then
  if [ "$API_MODE" = 1 ]; then
    log "stopping API proxy"; bash "$BENCHINFRA_ROOT/engine/serve_vllm.sh" stop
  fi
fi
log "ALL DONE $KEY — results in $RESULTS_DIR/$KEY/"
echo
echo "Summarize with:  bash $BENCHINFRA_ROOT/summarize.sh $KEY"

