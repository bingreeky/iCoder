# ============================================================
# config.sh — central configuration for eval_harness.
#
# Every path and tunable lives here so the scripts stay host-agnostic. Override
# any value by exporting it before sourcing (or edit this file for a fixed host).
#   export CODERBENCH_ROOT=/path/to/your/benchmarks
#   source config.sh
# ============================================================

# --- This infra repo (auto-detected; the dir containing this config.sh) ---
# Resolved first so the CODERBENCH_ROOT default below can reference it.
BENCHINFRA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Root of the benchmark datasets + venvs (NOT this infra repo) ---
# This is where the actual benchmark repos live: VerilogEval/, RTLLM/,
# ArchXBench/, RealBench/, KernelBench/. The infra scripts drive them in place.
# Default points at a local `benchmarks/` checkout created by
# `setup/setup_datasets.sh` (next to this repo). Override per-host in
# config.local.sh (e.g. CODERBENCH_ROOT=/path/to/your/benchmarks).
: "${CODERBENCH_ROOT:=$BENCHINFRA_ROOT/benchmarks}"

# --- Per-host overrides (paths, venvs, toolchain, N_GPU). ---
# Drop a config.local.sh next to this file rather than editing defaults, so the
# same clone runs on a 1-GPU debug box and an N-GPU host unchanged.
if [ -f "$BENCHINFRA_ROOT/config.local.sh" ]; then
  # shellcheck disable=SC1091
  source "$BENCHINFRA_ROOT/config.local.sh"
fi

# --- Where per-model results are written ---
: "${RESULTS_DIR:=$CODERBENCH_ROOT/results}"

# --- Engine / proxy ---
: "${PROXY_PORT:=8000}"          # OpenAI-compatible endpoint all benches hit
# N_GPU is the SINGLE knob for card count. Auto-detect from nvidia-smi so the
# same scripts scale 1-GPU debug <-> N-GPU without edits. Override by export.
if [ -z "${N_GPU:-}" ]; then
  # Count GPU lines only — some drivers emit a warning header under MIG/
  # partitioning that would otherwise inflate the count.
  _NG=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  [ "$_NG" -lt 1 ] 2>/dev/null && _NG=1
  N_GPU=$_NG
fi
: "${BASE_PORT:=8101}"
# Engine context window. 81920 (80K) so a 65536-token generation + a ~10K prompt
# fits with headroom (the upstream default). Override smaller in
# config.local.sh for VRAM-constrained 1-GPU debug boxes (some managed hosts
# cap at 32768).
: "${MAX_MODEL_LEN:=81920}"
: "${MAX_NUM_SEQS:=256}"
: "${GPU_MEM_UTIL:=0.9}"
: "${THINK:=1}"                  # 1 = keep model reasoning, strip <think> from output

# --- Toolchain (Verilog benches need iverilog v12 on PATH front) ---
# oss-cad-suite's iverilog v14-devel has a $dumpvars forward-ref bug -> pass_rate=0.
# Default matches the quickstart (`setup/setup_toolchain.sh /opt/eval-toolchain`).
: "${IVERILOG12_BIN:=/opt/eval-toolchain/iverilog12/bin}"
# Sourced for verilator/yosys (RealBench) + CUDA compat libs. Optional.
: "${SETUP_ENV:=/opt/eval-toolchain/setup_env.sh}"

# --- Python interpreters ---
: "${VLLM_PY:=python3}"           # vLLM-serving interpreter (override in config.local.sh)
: "${SYS_PY:=python3}"           # stdlib-only runners (RTLLM/ArchX/RealBench gen)

# --- OpenAI-compatible client env (local vLLM ignores the key) ---
: "${OPENAI_BASE_URL:=http://localhost:$PROXY_PORT/v1}"
: "${OPENAI_API_KEY:=dummy-local-key}"

# --- External-API mode (serve_vllm.sh api + run_all.sh --api) ---
# When set, the proxy forwards to EXTERNAL_API_URL and injects
# EXTERNAL_API_KEY as Authorization. API_SERVED_NAME rewrites the request
# `model` field (KB sends "default" → this). Fill in config.local.sh.
: "${EXTERNAL_API_URL:=}"
: "${EXTERNAL_API_KEY:=}"
: "${API_SERVED_NAME:=}"

# --- Bench-specific roots (override in config.local.sh per host) ---
# SFT repo: the `expand/` teacher package now lives in this repo, so the default
# is the repo root (PYTHONPATH=$SFT_ROOT finds it). Override only if you keep a
# separate checkout.
: "${SFT_ROOT:=$BENCHINFRA_ROOT}"
# KernelBench official harness (generate_samples.py / eval_from_generations.py).
: "${KERNELBENCH_ROOT:=$CODERBENCH_ROOT/KernelBench}"
: "${KERNELBENCH_PY:=$KERNELBENCH_ROOT/.venv/bin/python}"
# TritonBench dataset (original seeds; the expand/datasets adapters point here).
: "${TRITONBENCH_ROOT:=$CODERBENCH_ROOT/TritonBench}"
# CVDP cid003 harness (run_samples.py + local_extensions factory + .venv-cvdp).
: "${CVDP_ROOT:=$CODERBENCH_ROOT/CVDP}"
: "${CVDP_VENV:=$CVDP_ROOT/.venv-cvdp/bin/python}"
: "${CVDP_DATASET:=$CVDP_ROOT/datasets/cvdp_v1.0.4_cid003_only.nonjudge.jsonl}"
# CVDP cocotb harness parallelism (CPU-bound, not GPU). Auto from nproc.
: "${HARNESS_THREADS:=$(nproc)}"
# TBG/TBT verify per-shard timeout (sec) and sharded verify shard count (=N_GPU).
: "${TBG_TIMEOUT:=180}"
# EXPAND_LLM_* : the SFT teacher client (expand/llm.py) reads these to talk to
# the proxy. Set by run_tritonbench.sh from OPENAI_BASE_URL / KEY / served name.

export CODERBENCH_ROOT BENCHINFRA_ROOT RESULTS_DIR PROXY_PORT N_GPU BASE_PORT \
       MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL THINK IVERILOG12_BIN SETUP_ENV \
       VLLM_PY SYS_PY OPENAI_BASE_URL OPENAI_API_KEY SFT_ROOT KERNELBENCH_ROOT \
       KERNELBENCH_PY TRITONBENCH_ROOT CVDP_ROOT CVDP_VENV CVDP_DATASET \
       HARNESS_THREADS TBG_TIMEOUT EXTERNAL_API_URL EXTERNAL_API_KEY API_SERVED_NAME \
       VERILOGEVAL_ROOT RTLLM_ROOT ARCHX_ROOT RB_ROOT
