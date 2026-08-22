# ============================================================
# benches/registry.sh — the single place that lists every benchmark.
#
# Two groups so run_all.sh can place the GPU-VRAM-heavy verify stages after
# the engine is stopped:
#   BENCHES_UP   : run while the engine is ONLINE (proxy on :PROXY_PORT).
#                  Includes the KB/TBG/TBT *generate* stages (query the model,
#                  no GPU) and the full RTL/CVDP benches.
#   BENCHES_DOWN : run after the engine is STOPPED (eval needs full VRAM).
#                  KB/TBG/TBT *verify* stages: sharded across N_GPU, one card
#                  per shard, then merged.
#
# Format (pipe-separated):  name | runner | mode | extra | default_env
#   name         short id, results/<key>/<name> hint
#   runner       script under benches/
#   mode         "" for single-phase benches | "gen"/"eval" for two-phase
#   extra        extra positional args appended AFTER the served name
#                (e.g. split "g"/"t" for tritonbench, levels "1" for kernelbench)
#   default_env  space-separated VAR=VAL applied when this bench runs
#
# run_all.sh invokes:  env $default_env bash $runner $mode $KEY $extra
# (empty mode/extra collapse, so single-phase benches get just `bash $runner $KEY`.)
#
# To add a bench: drop benches/run_<name>.sh and append a line here.
# ============================================================

BENCHES_UP=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "rtllm             | run_rtllm.sh        |       |       | SAMPLES=4 TEMP=0.8 WORKERS=64 MAXTOK=49152"
  "archx             | run_archx.sh        |       |       | SAMPLES=5 TEMP=0.8 WORKERS=64 MAXTOK=49152"
  "realbench         | run_realbench.sh    |       |       | SAMPLES=5 TEMP=0.8 WORKERS=60 MAXTOK=49152"
  # two-phase generate stages (engine online, model query only, no GPU).
  # KB = L1/L2/L3, cuda backend (canonical eval-config; NOT the triton default).
  "kernelbench_gen   | run_kernelbench.sh  | gen   | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
  "tritonbench_t_gen | run_tritonbench.sh  | gen   | t 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)

BENCHES_DOWN=(
  # two-phase verify stages (engine stopped, full VRAM, sharded across N_GPU):
  "kernelbench_eval   | run_kernelbench.sh  | eval  | 1,2,3 cuda | "
  "tritonbench_g_eval | run_tritonbench.sh  | eval  | g     | "
  "tritonbench_t_eval | run_tritonbench.sh  | eval  | t     | "
)

# ============================================================
# Table profile — the standard cross-domain comparison subset.
# Keep benchmark membership and runner settings here so every invocation uses
# the same configuration contract.
# Use: bash run_all.sh --api --profile table <api_base> <served_name>
# ============================================================
BENCHES_UP_TABLE=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "rtllm             | run_rtllm.sh        |       |       | SAMPLES=4 TEMP=0.8 MAXTOK=49152"
  "archx             | run_archx.sh        |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "realbench         | run_realbench.sh    |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "kernelbench_gen   | run_kernelbench.sh  | gen   | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)

BENCHES_DOWN_TABLE=(
  "kernelbench_eval   | run_kernelbench.sh  | eval  | 1,2,3 cuda | "
  "tritonbench_g_eval | run_tritonbench.sh  | eval  | g     | "
)

# ============================================================
# Retry profile — re-run only the benches that failed in a prior table run
# (CVDP + KB gen/eval + TBG gen/eval). VEval/RTLLM/Archx/RealBench results
# already on disk are left untouched; summarize --table reads everything.
# Use: bash run_all.sh --api --profile retry <api_base> <served_name>
# ============================================================
BENCHES_UP_RETRY=(
  "kernelbench_gen   | run_kernelbench.sh  | gen   | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)

BENCHES_DOWN_RETRY=(
  "kernelbench_eval   | run_kernelbench.sh  | eval  | 1,2,3 cuda | "
  "tritonbench_g_eval | run_tritonbench.sh  | eval  | g     | "
)

# ============================================================
# kbeval profile — re-run ONLY KB eval (gen + TBG already done).
# No proxy needed (eval is GPU-only), but --api starts a harmless one.
# Use: bash run_all.sh --api --profile kbeval (or --down-only locally)
# ============================================================
BENCHES_UP_KBEVAL=()
BENCHES_DOWN_KBEVAL=(
  "kernelbench_eval | run_kernelbench.sh | eval | 1,2,3 cuda | "
)

# ============================================================
# cvdp profile — re-run ONLY CVDP (model-eval mode, --llm). CPU-parallel
# (cocotb+iverilog), needs the proxy (model endpoint) but no GPU for eval.
# Use: bash run_all.sh --api --profile cvdp <key>
# ============================================================
BENCHES_UP_CVDP=(
  "cvdp | run_cvdp.sh | | | SAMPLES=1 HARNESS_THREADS=8"
)
BENCHES_DOWN_CVDP=()

# ============================================================
# vevalcvdp profile — re-run VEval (spec+code) + CVDP only.
# For filling the VEval/CVDP columns after the sv-generate whitelist fix.
# Use: bash run_all.sh --api --profile vevalcvdp <key>
# ============================================================
BENCHES_UP_VEVALCVDP=(
  "verilogeval | run_verilogeval.sh | | | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "cvdp        | run_cvdp.sh        | | | SAMPLES=1 HARNESS_THREADS=8"
)
BENCHES_DOWN_VEVALCVDP=()

# ============================================================
# veval profile — re-run ONLY VerilogEval (spec-to-rtl + code-complete) at
# average@4 (SAMPLES=4 TEMP=0.8 TOPP=0.95). No DOWN phase (VE is CPU+iverilog,
# queries the engine, no GPU verify). Use to upgrade a model's pass@1 VE
# columns to avg@4 without re-running KB/TBG/CVDP.
# gateway: bash run_all.sh --api --profile veval <key>
# local: bash run_all.sh --profile veval <path> <name> <mlen>
# ============================================================
BENCHES_UP_VEVAL=(
  "verilogeval | run_verilogeval.sh | | | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
)
BENCHES_DOWN_VEVAL=()

# ============================================================
# kbveval profile — run VerilogEval and KernelBench in one engine lifecycle.
# Useful for focused regression checks without running the full registry.
# bash run_all.sh --profile kbveval /path/to/your-model your-model 65536
# ============================================================
BENCHES_UP_KBVEVAL=(
  "verilogeval     | run_verilogeval.sh | | | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "kernelbench_gen | run_kernelbench.sh | gen | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
)
BENCHES_DOWN_KBVEVAL=(
  "kernelbench_eval | run_kernelbench.sh | eval | 1,2,3 cuda | "
)

# ============================================================
# coderrtl profile — RTL and Triton-focused evaluation for a local checkpoint.
# Generation runs with the engine online; Triton verification runs after the
# engine stops. CVDP requires the served name to be registered upstream.
# bash run_all.sh --profile coderrtl /path/to/your-model your-model 32768
# ============================================================
BENCHES_UP_CODERRTL=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "cvdp              | run_cvdp.sh         |       |       | SAMPLES=1 HARNESS_THREADS=8"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
  "tritonbench_t_gen | run_tritonbench.sh  | gen   | t 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_CODERRTL=(
  "tritonbench_g_eval | run_tritonbench.sh | eval | g | "
  "tritonbench_t_eval | run_tritonbench.sh | eval | t | "
)

# ============================================================
# arcb profile — run the CPU-backed ArchXBench, RealBench, and CVDP subset.
# Use: bash run_all.sh --api --profile arcb <key>
# ============================================================
BENCHES_UP_ARCB=(
  "archx     | run_archx.sh     | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "realbench | run_realbench.sh | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "cvdp      | run_cvdp.sh      | | | SAMPLES=1 HARNESS_THREADS=8"
)
BENCHES_DOWN_ARCB=()

# ============================================================
# kbonly profile — run KernelBench generation and sharded verification only.
# bash run_all.sh --profile kbonly /path/to/your-model your-model 65536
# ============================================================
BENCHES_UP_KBONLY=(
  "kernelbench_gen | run_kernelbench.sh | gen | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
)
BENCHES_DOWN_KBONLY=(
  "kernelbench_eval | run_kernelbench.sh | eval | 1,2,3 cuda | "
)

# ============================================================
# kbl3 profile — run the KernelBench L3 subset with the CUDA backend.
#   bash run_all.sh --profile kbl3 <model> <served_name> 81920
# ============================================================
BENCHES_UP_KBL3=(
  "kernelbench_gen | run_kernelbench.sh | gen | 3 cuda | TEMP=0 MAXTOK=58000 WORKERS=20"
)
BENCHES_DOWN_KBL3=(
  "kernelbench_eval | run_kernelbench.sh | eval | 3 cuda | "
)

# ============================================================
# tritonbench profile — run TritonBench-G and TritonBench-T only.
# bash run_all.sh --profile tritonbench <model> <served_name> 65536
# ============================================================
BENCHES_UP_TRITONBENCH=(
  "tritonbench_g_gen | run_tritonbench.sh | gen | g 1 | TEMP=0 MAXTOK=58000 WORKERS=16"
  "tritonbench_t_gen | run_tritonbench.sh | gen | t 1 | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_TRITONBENCH=(
  "tritonbench_g_eval | run_tritonbench.sh | eval | g | "
  "tritonbench_t_eval | run_tritonbench.sh | eval | t | "
)

# archx-only profile — run ArchXBench generation and verification only.
# bash run_all.sh --profile archx <model> <served_name> 65536
BENCHES_UP_ARCHXONLY=(
  "archx | run_archx.sh | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
)
BENCHES_DOWN_ARCHXONLY=()

# tbt-only profile — run TritonBench-T generation and verification only.
# bash run_all.sh --profile tbt <model> <served_name> 65536
BENCHES_UP_TBT=(
  "tritonbench_t_gen | run_tritonbench.sh | gen | t 1 | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_TBT=(
  "tritonbench_t_eval | run_tritonbench.sh | eval | t | "
)

# ============================================================
# smoke profile — minimal end-to-end coverage of generation, verification, and
# summarization after a configuration or verifier change. This is a diagnostic
# profile, not a scored evaluation.
# LIMIT=5 bash run_all.sh --profile smoke <model> <name> 81920
# LIMIT=5 bash run_all.sh --api --profile smoke <gateway_model>
# ============================================================
BENCHES_UP_SMOKE=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=1 TEMP=0 TOPP=0.95 MAXTOK=49152"
  "rtllm             | run_rtllm.sh        |       |       | SAMPLES=1 TEMP=0 WORKERS=64 MAXTOK=49152"
  "archx             | run_archx.sh        |       |       | SAMPLES=1 TEMP=0 WORKERS=64 MAXTOK=49152"
  "realbench         | run_realbench.sh    |       |       | SAMPLES=1 TEMP=0 MAXTOK=49152"
  "cvdp              | run_cvdp.sh         |       |       | SAMPLES=1 HARNESS_THREADS=8"
  "kernelbench_gen   | run_kernelbench.sh  | gen   | 1 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_SMOKE=(
  "kernelbench_eval   | run_kernelbench.sh  | eval  | 1 cuda | "
  "tritonbench_g_eval | run_tritonbench.sh  | eval  | g     | "
)

# ============================================================
# tracka compatibility profile — retained for existing local-vLLM automation.
# New integrations should prefer a descriptive profile assembled from the
# public benchmark runners above.
# MAX_NUM_SEQS=64 bash run_all.sh --profile tracka <model> <name> 81920
# ============================================================
BENCHES_UP_TRACKA=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "cvdp              | run_cvdp.sh         |       |       | SAMPLES=5 HARNESS_THREADS=8"
  "rtllm             | run_rtllm.sh        |       |       | SAMPLES=4 TEMP=0.8 WORKERS=64 MAXTOK=49152"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_TRACKA=(
  "tritonbench_g_eval | run_tritonbench.sh | eval | g | "
)

# ============================================================
# incoder2 compatibility profile — retained for existing orchestration that
# combines VerilogEval, ArchXBench, RealBench, and KernelBench. New campaigns
# should define a descriptive profile with an explicit configuration record.
# MAX_NUM_SEQS=64 TP_SIZE=2 bash run_all.sh --profile incoder2 <model> <name> 81920
# ============================================================
BENCHES_UP_INCODER2=(
  "verilogeval      | run_verilogeval.sh |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "archx            | run_archx.sh        |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152 WORKERS=64"
  "realbench        | run_realbench.sh   |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152 WORKERS=60"
  "kernelbench_gen  | run_kernelbench.sh | gen   | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
)
BENCHES_DOWN_INCODER2=(
  "kernelbench_eval | run_kernelbench.sh | eval  | 1,2,3 cuda | "
)

# ============================================================
# rtllm profile — RTLLM@4 ONLY (API mode, CPU node, no GPU: gen queries the
# gateway, verify is iverilog on CPU). WORKERS=20 = gateway concurrency cap.
#   EXTERNAL_API_URL=.. EXTERNAL_API_KEY=.. WORKERS=20 \
#     bash run_all.sh --api --profile rtllm <served_name>
# ============================================================
BENCHES_UP_RTLLM=(
  "rtllm | run_rtllm.sh | | | SAMPLES=4 TEMP=0.8 WORKERS=20 MAXTOK=49152"
)
BENCHES_DOWN_RTLLM=()

# ============================================================
# tbg profile — TritonBench-G ONLY (gen+eval), API mode. Gen queries the
# gateway at WORKERS=20; eval verifies on GPU (torch+triton).
#   EXTERNAL_API_URL=.. EXTERNAL_API_KEY=.. WORKERS=20 \
#     bash run_all.sh --api --profile tbg <served_name>
# ============================================================
BENCHES_UP_TBG=(
  "tritonbench_g_gen | run_tritonbench.sh | gen | g 1 | TEMP=0 MAXTOK=58000 WORKERS=20"
)
BENCHES_DOWN_TBG=(
  "tritonbench_g_eval | run_tritonbench.sh | eval | g | "
)

# ============================================================
# tbgresume profile — re-run ONLY the TBG-G eval with TBG_RESUME=1 so the
# shard workers SKIP already-verified ids and only attempt the ones a prior
# eval lost (shard killed mid-eval → partial shard_verified, job "Succeeded"
# via partial-shard merge). No gen (rollout unchanged), no engine needed (eval
# is GPU torch+triton, reads rollout.jsonl). N_GPU auto-detected = host GPU
# count, MUST match the prior eval's shard count for resume alignment.
# TBG_RESUME=1 bash run_all.sh --api --profile tbgresume <model>
# ============================================================
BENCHES_UP_TBGRESUME=()
BENCHES_DOWN_TBGRESUME=(
  "tritonbench_g_eval | run_tritonbench.sh | eval | g | TBG_RESUME=1"
)

# ============================================================
# dpskinfra compatibility profile — infrastructure-validation generation for
# VerilogEval, ArchXBench, CVDP, and TritonBench-G. Run Triton verification
# separately with the matching verifier environment.
# ============================================================
BENCHES_UP_DPSKINFRA=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "archx             | run_archx.sh        |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152 WORKERS=20"
  "cvdp              | run_cvdp.sh         |       |       | SAMPLES=5 HARNESS_THREADS=8"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=20"
)
BENCHES_DOWN_DPSKINFRA=()
