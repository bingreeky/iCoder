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
# Table profile — the 11-column deliverable table's subset.
# Rows = models, columns = [VEval-spec, VEval-code, RTLLM, CVDP cid003,
# RealBench Syn@5, RealBench Func@5, ArchXBench t, KB L1, KB L2, KB L3,
# TritonBench-G]. Archx/RealBench datasets are not on this host's mount → their
# runners are omitted here (columns stay blank). TBT is not a table column.
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
# kbveval profile — re-run ONLY VerilogEval + KernelBench (gen+eval) in one
# engine load. For local checkpoints where the full-table rerun would waste
# hours re-doing already-good ArchX/RTLLM/RealBench/TBG: restore those from a
# backup, then this fills the two broken columns (KB after the default-404 fix;
# VEval after cleaning stale .sv so the fence patch regenerates). No slow ArchX.
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
# coderrtl profile — full re-test of a local HF checkpoint (your-model):
# VerilogEval@4 + CVDP + TritonBench-G + TritonBench-T. UP = generate/RTL/CVDP
# (engine online); DOWN = TBG/TBT verify (engine stopped, sharded). No KB/RTLLM/
# archx/realbench (not in scope for this model). CVDP needs the served_name
# registered in CVDP ALIASES (local_extensions/eval_5models_factory.py).
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
# arcb profile — re-run ONLY the CPU benches (ArchXBench + RealBench + CVDP)
# to fill those columns after a prior run where they failed/were skipped, WITHOUT
# re-running the expensive GPU benches (KB/TBG) whose good results are kept.
# All three are CPU (iverilog/verilator/cocotb) + engine queries; DOWN empty.
# Use: bash run_all.sh --api --profile arcb <key>
# ============================================================
BENCHES_UP_ARCB=(
  "archx     | run_archx.sh     | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "realbench | run_realbench.sh | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
  "cvdp      | run_cvdp.sh      | | | SAMPLES=1 HARNESS_THREADS=8"
)
BENCHES_DOWN_ARCB=()

# ============================================================
# kbonly profile — re-run ONLY KernelBench (gen+eval, L1/L2/L3) to verify a
# vLLM-serve fix (e.g. cuda-graph ON / GDN packages) reduces KB drop, WITHOUT
# re-running VE/RTLLM/ArchX/RealBench/TBG (already good). Engine online for gen,
# stopped for sharded eval. Force fresh gen by rm-ing the old run-dir kernels.
# bash run_all.sh --profile kbonly /path/to/your-model your-model 65536
# ============================================================
BENCHES_UP_KBONLY=(
  "kernelbench_gen | run_kernelbench.sh | gen | 1,2,3 cuda | TEMP=0 MAXTOK=58000 WORKERS=32"
)
BENCHES_DOWN_KBONLY=(
  "kernelbench_eval | run_kernelbench.sh | eval | 1,2,3 cuda | "
)

# ============================================================
# kbl3 profile — KernelBench L3 ONLY (50 hard problems), cuda backend,
# SAMPLES=1 TEMP=0 pass@1. Single-runtime (image py; CUDA kernels need nvcc +
# torch CUDA exec, NOT triton 3.6). gen (UP, proxy online) + eval (DOWN,
# sharded, engine stopped).
#   bash run_all.sh --profile kbl3 <model> <served_name> 81920
# ============================================================
BENCHES_UP_KBL3=(
  "kernelbench_gen | run_kernelbench.sh | gen | 3 cuda | TEMP=0 MAXTOK=58000 WORKERS=20"
)
BENCHES_DOWN_KBL3=(
  "kernelbench_eval | run_kernelbench.sh | eval | 3 cuda | "
)

# ============================================================
# tritonbench profile — re-run ONLY TritonBench-G + TritonBench-T (gen+eval)
# with a vLLM-serve fix (cuda-graph ON), WITHOUT KB (done in kbonly) or the
# Verilog benches. Force fresh gen by rm-ing old rollout.jsonl.
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

# archx-only profile — re-run ONLY ArchXBench (single-phase gen+test, cuda-graph)
# to recover the 11 GEN_CONNECTION_RESET (eager+nseq256 worker crash).
# bash run_all.sh --profile archx <model> <served_name> 65536
BENCHES_UP_ARCHXONLY=(
  "archx | run_archx.sh | | | SAMPLES=5 TEMP=0.8 MAXTOK=49152"
)
BENCHES_DOWN_ARCHXONLY=()

# tbt-only profile — re-run ONLY TritonBench-T (gen+eval) after fixing the
# tbg_dump_seeds.py triton_interface bug (was None → teacher prompt had no
# primary_wrapper → model output kernel-only → 0/166). Now seeds have
# triton_interface, the teacher prompt specifies the def-wrapper → format_ok>0.
# bash run_all.sh --profile tbt <model> <served_name> 65536
BENCHES_UP_TBT=(
  "tritonbench_t_gen | run_tritonbench.sh | gen | t 1 | TEMP=0 MAXTOK=58000 WORKERS=16"
)
BENCHES_DOWN_TBT=(
  "tritonbench_t_eval | run_tritonbench.sh | eval | t | "
)

# ============================================================
# smoke profile — ONE pass@1 sample of EVERY benchmark, to confirm the whole
# pipeline (gen + verify + summarize) runs through end-to-end after a config/
# verifier change. NOT a scored run (pass@1 only). Lightest feasible "run each
# benchmark through" without per-runner problem-count limits (only run_archx.sh
# has a LIMIT knob; the others run their full problem set at SAMPLES=1). ArchX is
# capped via the LIMIT env (inject LIMIT=5 on the submit). KB is L1 only, TBG is
# g only — the GPU-verify benches kept minimal. Use to smoke both local-vLLM and
# API modes after touching config.sh / registry / verify/.
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
# tracka profile — local-vLLM scored run for 32B base models:
# VerilogEval@4 + CVDP n=5 pass@1 + RTLLM@4 + TritonBench-G (gen+eval). All
# UP benches hit the local engine (proxy :8000); DOWN = TBG verify (engine
# stopped, sharded). The 32B is KV-tight @TP1, so submit with MAX_NUM_SEQS=64
# and MLEN=81920 — set VLLM_EXTRA_ARGS=--swap-space 16 so
# vLLM accepts the 80K window AND TBG's 58K-token gens fit (KV spills to RAM).
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
# incoder2 profile — the scored benches for a 2×8-card campaign,
# each at the CANONICAL eval-config settings (see README "覆盖的 benchmark"). RTLLM@4 (65536 ≥ 49152, superset) and
# CVDP cid003 (n=5, not in the canonical table) are ALREADY DONE in
# results/<key>/ and NOT re-run here. TBG is skipped for this campaign.
# This profile (re-)runs:
#   VerilogEval spec+cc: @4 t0.8, MAXTOK=49152 — RE-RUN (prior was 32768;
#     run_verilogeval.sh rm -rf's the build so the 32768 results are cleanly
#     overwritten with the canonical-token run).
#   ArchX v1.5: 5 samples, func/syntax pass@1, MAXTOK=49152 (SAMPLES=5 TEMP=0).
#   RealBench-Module: 5 samples, syntax/func pass@1/@5, MAXTOK=49152.
#   KernelBench L1/L2/L3: 1 sample, compiled%/correct% pass@1, MAXTOK=58000,
#     **cuda backend** (extra="1,2,3 cuda" — model writes CUDA, eval runs
#     nvcc/CUDA, NOT the triton default).
# DOWN = KB eval (engine stopped, GPU-sharded verify, cuda backend).
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
# dpskinfra profile — the 5-bench infra-validation pass (VEval spec+cc @4,
# ArchX v1.5 func@1, CVDP cid003 n=5, TritonBench-G corr@1). This profile is
# the UP (engine-online) half — VEval + ArchX + CVDP(n=5) + TBG-G *generate*.
# The DOWN half (TBG-G *eval*) is run separately via the `tbgresume` profile
# AFTER switching SYS_PY to the triton-3.6 interpreter (the image's triton
# 2.3.1 makes ~28/29 TBG "skip" rows spurious version-mismatches).
#   KEY MUST equal the gateway model id (the proxy rewrites model -> KEY), so
# to keep prior results back up results/<KEY>/ -> results/<KEY>.prev_* first.
#   EXTERNAL_API_URL=.. EXTERNAL_API_KEY=.. \
#     bash run_all.sh --api --profile dpskinfra <served_name>
# (the model's served name must be CVDP-registered + VE sv-generate whitelisted).
# ============================================================
BENCHES_UP_DPSKINFRA=(
  "verilogeval       | run_verilogeval.sh  |       |       | SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152"
  "archx             | run_archx.sh        |       |       | SAMPLES=5 TEMP=0.8 MAXTOK=49152 WORKERS=20"
  "cvdp              | run_cvdp.sh         |       |       | SAMPLES=5 HARNESS_THREADS=8"
  "tritonbench_g_gen | run_tritonbench.sh  | gen   | g 1   | TEMP=0 MAXTOK=58000 WORKERS=20"
)
BENCHES_DOWN_DPSKINFRA=()
