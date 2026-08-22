#!/bin/bash
# ============================================================
# serve_vllm.sh — N single-GPU vLLM OpenAI servers + a round-robin proxy on
# :$PROXY_PORT. All benches hit the one proxy; the engine starts once.
#
# Why vLLM (not a hand-rolled HF serve): continuous batching does per-seq early
# stop + dynamic scheduling, so short requests don't idle-wait behind long ones.
#
# Verified config (Qwen3.x GDN/Mamba-hybrid 27B; tune per model):
#   - gdn_prefill_backend=flashinfer  (triton/fla numerics are broken for GDN)
#   - VLLM_USE_DEEP_GEMM=0            (else KV-cache init forces FP8 deep_gemm -> crash)
#   - max_num_seqs default 256        (GDN: 1 Mamba cache block per decode seq)
#   - TP=1 per card                   (27B bf16 ~54GB fits; GDN risky at TP>1)
#   - dense (e.g. IQuestCoder) models: works too; drop --additional-config if unused
#
# Usage:  bash serve_vllm.sh <model_path> [served_name] [max_num_seqs] [max_model_len]
#         bash serve_vllm.sh stop
# ============================================================
# Preserve caller-provided VLLM_PY across config.sh (same reason as run_all.sh):
# config.local.sh may hardcode VLLM_PY=/usr/local/bin/python3 (image python, no
# vllm 0.25) for the dev box, but a caller using the isolated .venv-vllm sets
# VLLM_PY to that venv python. serve_vllm is a child process of
# run_all; it re-sources config.sh here, which would clobber the venv python
# back to the image one -> `vllm.entrypoints` import fails -> engine never
# starts -> all-empty results. Caller wins.
_PRESET_VLLM_PY="${VLLM_PY:-}"
_PRESET_RESULTS_DIR="${RESULTS_DIR:-}"
source "$(dirname "$0")/../config.sh"
[ -n "$_PRESET_VLLM_PY" ] && export VLLM_PY="$_PRESET_VLLM_PY"
[ -n "$_PRESET_RESULTS_DIR" ] && export RESULTS_DIR="$_PRESET_RESULTS_DIR"

log(){ echo "[serve_vllm $(date +%H:%M:%S)] $*"; }

# Kill vLLM + proxy processes owned by THIS user only, so a shared host's
# other users are never touched. vLLM subprocs show as VLLM::EngineCore /
# VLLM::Worker_TPn, NOT lowercase 'vllm' — plain `pkill -f vllm` misses them
# and leaks ~90GB/GPU, so the ladder targets each shape explicitly.
_MUID="$(id -u)"
_kill_engine() {
  pkill -u "$_MUID" -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  pkill -u "$_MUID" -f "proxy_rr.py" 2>/dev/null
  pkill -u "$_MUID" -f "anthro_proxy.py" 2>/dev/null
  pkill -9 -u "$_MUID" -f "VLLM::" 2>/dev/null
  pkill -9 -u "$_MUID" -f "EngineCore" 2>/dev/null
}

if [ "$1" = "stop" ]; then
  _kill_engine
  echo "stopped vllm servers / proxy"
  exit 0
fi

# ---- api mode: no local vLLM, just the proxy forwarding to an external API ----
# Uses NO GPU (the model is served by the gateway). PROXY_API_KEY +
# PROXY_SERVED_NAME are picked up by proxy_rr.py to inject Authorization and
# rewrite the request `model` field. Verify stages thus get the full VRAM.
# Usage: serve_vllm.sh api [api_base] [served_name]   (defaults from env)
if [ "$1" = "api" ]; then
  API_BASE=${2:-${EXTERNAL_API_URL:-}}
  SERVED_NAME=${3:-${API_SERVED_NAME:-model}}
  [ -z "$API_BASE" ] && { echo "usage: serve_vllm.sh api <api_base> <served_name>  (or set EXTERNAL_API_URL/API_SERVED_NAME)"; exit 1; }
  [ -z "$EXTERNAL_API_KEY" ] && { echo "WARN: EXTERNAL_API_KEY empty — gateway will 401"; }
  # Strip a trailing /v1 (and any trailing slash): proxy_rr concatenates
  # backend + self.path, and self.path already starts with /v1, so the
  # backend must be the bare origin (e.g. http://<your-gateway>:<port>).
  API_BASE_NOV1="${API_BASE%/}"
  API_BASE_NOV1="${API_BASE_NOV1%/v1}"
  API_BASE_NOV1="${API_BASE_NOV1%/}"
  pkill -u "$_MUID" -f "proxy_rr.py" 2>/dev/null; pkill -u "$_MUID" -f "anthro_proxy.py" 2>/dev/null; sleep 2
  export PROXY_API_KEY="${EXTERNAL_API_KEY:-}"
  export PROXY_SERVED_NAME="$SERVED_NAME"
  # PROXY_MODE=anthro: gateway speaks Anthropic Messages format ONLY (e.g.
  # some Anthropic-native gateways). Launch the translation proxy
  # (OpenAI chat/completions <-> Anthropic /v1/messages) instead of the dumb
  # passthrough proxy_rr.py — otherwise the OpenAI client sees no `choices`
  # -> content=None -> empty .sv / no kernel on EVERY row. PROXY_UPSTREAM_URL
  # is read by anthro_proxy.py (argv[2] also works; serve_vllm passes it below).
  PROXY_SCRIPT="proxy_rr.py"
  [ "${PROXY_MODE:-}" = "anthro" ] && PROXY_SCRIPT="anthro_proxy.py"
  log "api mode (proxy=$PROXY_SCRIPT): proxy :$PROXY_PORT -> $API_BASE_NOV1 (served=$SERVED_NAME, key=${EXTERNAL_API_KEY:+set})"
  nohup "$SYS_PY" "$BENCHINFRA_ROOT/engine/$PROXY_SCRIPT" "$PROXY_PORT" "$API_BASE_NOV1" \
    > /tmp/proxy_api_${SERVED_NAME//\//_}.log 2>&1 &
  sleep 3
  PROXY_UP=0
  for _ in $(seq 1 60); do
    if curl -s --max-time 2 "http://localhost:$PROXY_PORT/" >/dev/null 2>&1; then
      log "proxy up on :$PROXY_PORT"; PROXY_UP=1; break
    fi
    sleep 2
  done
  if [ "$PROXY_UP" != 1 ]; then
    log "proxy FAILED to start"; tail -8 /tmp/proxy_api_${SERVED_NAME//\//_}.log; exit 1
  fi
  # Model-availability probe through the proxy (which now passes through the
  # real upstream status). A served model returns 200; an unserved model
  # returns 503 persistently. Useful as a direct preflight before a long
  # run: on unserved, exit 1 so callers see the failure early.
  pfbody=$("$SYS_PY" -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"hi"}],"max_tokens":1}))' "$SERVED_NAME")
  for a in 1 2 3; do
    pfcode=$(curl -sS -m 30 -o /tmp/srvpf.body -w '%{http_code}' \
      -H "Content-Type: application/json" -d "$pfbody" \
      "http://localhost:$PROXY_PORT/v1/chat/completions" 2>/dev/null) || pfcode="000"
    case "$pfcode" in
      200|429) log "model $SERVED_NAME served via gateway (http $pfcode)"; exit 0;;
      503) log "model $SERVED_NAME: 503 (probe $a/3)"; sleep 10;;
      000|500|502|504) log "model $SERVED_NAME: http $pfcode (transient) — proceeding"; exit 0;;
      *) break;;
    esac
  done
  if [ "$pfcode" = "503" ]; then
    log "model $SERVED_NAME UNSERVED (503 persisted) — aborting api mode; b=$(head -c 100 /tmp/srvpf.body 2>/dev/null)"
    exit 1
  fi
  log "model $SERVED_NAME: http $pfcode — proceeding (b=$(head -c 100 /tmp/srvpf.body 2>/dev/null))"
  exit 0
fi

[ -f "$SETUP_ENV" ] && source "$SETUP_ENV" >/dev/null 2>&1  # CUDA compat libs on LD path

MODEL_PATH=$1
SERVED_NAME=${2:-model}
NSEQ=${3:-$MAX_NUM_SEQS}
MLEN=${4:-$MAX_MODEL_LEN}
[ -z "$MODEL_PATH" ] && { echo "usage: serve_vllm.sh <model_path> [name] [max_num_seqs] [max_model_len] | stop"; exit 1; }

export VLLM_USE_V1=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_WORKER_MULTIPROC_METHOD=spawn

_kill_engine; sleep 3
# TP_SIZE: tensor-parallel degree per backend (default 1 = pure data-parallel,
# N_GPU independent replicas each on one card). Set TP_SIZE>1 for dense models
# that can't fit a useful max_model_len on one card's KV budget — e.g. 32B dense
# @bf16 = 64GB weights -> only ~8GB KV on an 80GB card @TP1 (256KB/token =>
# max_model_len<=24576, which rejects the benches' 32K-65K max_tokens with 400).
# TP_SIZE=2 splits weights across 2 cards -> 32GB/card -> ~40GB KV -> 160K tokens
# => full 81920 context fits, all benches pass. N_GPU must be divisible by TP_SIZE;
# backend count = N_GPU/TP_SIZE (4 backends for 8 GPUs @TP2). Backends still
# round-robined by the proxy (data-parallel across TP groups).
TP="${TP_SIZE:-1}"
NBACKENDS=$(( N_GPU / TP ))
BACKENDS=""
for b in $(seq 0 $((NBACKENDS-1))); do
  port=$((BASE_PORT+b))
  # GPUs for this backend: b*TP .. b*TP+TP-1 (comma-sep for CUDA_VISIBLE_DEVICES)
  devs=$(seq -s, $((b*TP)) $((b*TP+TP-1)))
  # cuda-graph ON by default now (VLLM_ENFORCE_EAGER=0): with the GDN packages
  # (fla/causal_conv1d/flashinfer) installed in .venv-vllm, the model uses the
  # correct GDN backend (flashinfer) instead of the broken-numerics triton
  # fallback, AND cuda-graph capture gives ~1000 tok/s decode (vs 16 tok/s
  # eager). The 16 tok/s broken-slow decode was the real root cause of
  # KB/TBT/TBG drop (long Triton gens >600s client timeout → truncation/no
  # kernel → "Kernel not found" / APITimeoutError), NOT model OOD. Startup cap
  # is 40min (STARTUP_CAP=480) — enough for cuda-graph capture with GDN.
  # Set VLLM_ENFORCE_EAGER=1 to force eager (debug / if capture fails).
  EAGER=""
  [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ] && EAGER="--enforce-eager"
  CUDA_VISIBLE_DEVICES=$devs nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --port $port --host 0.0.0.0 \
    --tensor-parallel-size $TP --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MLEN --max-num-seqs $NSEQ \
    --trust-remote-code --dtype bfloat16 \
    --additional-config '{"gdn_prefill_backend":"flashinfer"}' \
    $EAGER $VLLM_EXTRA_ARGS \
    > "$RESULTS_DIR/_vllm_${SERVED_NAME}_g${b}.log" 2>&1 &
  BACKENDS="$BACKENDS http://localhost:$port"
done
log "launched $NBACKENDS vLLM servers (TP=$TP) for $SERVED_NAME ($MODEL_PATH) mlen=$MLEN nseq=$NSEQ eager=${EAGER:-no}"

# startup is slow (weight load + torch.compile + cuda graph + flashinfer JIT).
# Cap raised to 40min (480x5s, VLLM_STARTUP_CAP overrides) for big checkpoints;
# --enforce-eager skips the compile that used to blow the old 20min cap (the model).
STARTUP_CAP="${VLLM_STARTUP_CAP:-480}"
for g in $(seq 0 $((NBACKENDS-1))); do
  port=$((BASE_PORT+g)); ok=0
  for _ in $(seq 1 "$STARTUP_CAP"); do
    curl -s --max-time 2 http://localhost:$port/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] && log "vLLM g$g ready" || { log "vLLM g$g FAILED (cap=${STARTUP_CAP}x5s) — cleaning already-launched backends"; _kill_engine; tail -8 "$RESULTS_DIR/_vllm_${SERVED_NAME}_g${g}.log"; exit 1; }
done

# KB's query_server hardcodes model="default"; local vLLM serves $SERVED_NAME
# (e.g. the model) and rejects "default" with 404 -> all KB gen fail, 0 kernels,
# KB drop=100. Set PROXY_SERVED_NAME so the proxy rewrites the request `model`
# field to the real served name in LOCAL mode too (not just --api mode).
PROXY_SERVED_NAME="$SERVED_NAME" THINK=$THINK nohup "$SYS_PY" "$BENCHINFRA_ROOT/engine/proxy_rr.py" $PROXY_PORT $BACKENDS \
  > /tmp/proxy_${SERVED_NAME}.log 2>&1 &
sleep 3
curl -s --max-time 2 http://localhost:$PROXY_PORT/ | grep -q ok && log "proxy up on :$PROXY_PORT — ready"
