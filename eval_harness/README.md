# eval_harness

A one-click eval suite for code LLMs (Verilog + Triton/CUDA). A single proxy
routes every benchmark to one model endpoint (local vLLM **or** an external
OpenAI-compatible API), using a two-stage **generate → stop engine → shard-verify**
flow. `summarize.sh --table` prints one row of the comparison table.

Covers **7 benchmark families (8 eval tasks)** — VerilogEval (which spans
spec-to-rtl + code-complete), RTLLM 2.0, ArchXBench v1.5, RealBench-Module,
KernelBench L1/L2/L3, TritonBench-G, and CVDP cid003. Six families form the
canonical scored table; CVDP runs on demand.

> **Canonical config** (problem counts / max_tokens / sampling / metrics /
> verification) is in the benchmark table below and in `benches/registry.sh` —
> the single source of truth. All scored runs must align to it.

---

## ⚠️ Security warning — the harness executes model-generated code

This harness **compiles and runs code the model produces**. Concretely:

- **KernelBench** — `nvcc` compiles and executes the model's CUDA kernel, with
  numeric comparison against a torch reference.
- **TritonBench** — executes the model's Triton kernel in-process.
- **VerilogEval / RTLLM / ArchXBench / RealBench** — `iverilog`/`verilator`
  compile and simulate the model's RTL.
- **CVDP** — `cocotb` + `iverilog` run the model's RTL against a testbench.

Model output is **untrusted, adversarial input**. A model can emit code that
`$system`s a shell, reads files, or burns resources. Therefore:

- **Do not run on a personal laptop, a host that holds credentials or secrets,
  or a host that mounts sensitive directories.** Use a disposable container or
  an isolated machine you are willing to throw away.
- **Using Docker is not, by itself, sufficient.** Run with hardening:
  - **read-only** repo + toolchain mounts (`:ro`) where the runner doesn't need
    to write;
  - **no `--privileged`**; drop capabilities (`--cap-drop=ALL`, add back only
    what the verifiers need, e.g. `--cap-add=SYS_PTRACE` is usually unnecessary);
  - resource caps — `--memory`, `--cpus`, `--pids-limit`, and an explicit
    `--network` (the eval box only needs to reach the model endpoint; consider
    `--network=none` for the offline verify stages);
  - run as a **non-root user**;
  - the verify workers already wrap every model-code subprocess in a **timeout
    + process-group kill + throwaway temp dir** (`verify/core.py:run_procgroup`),
    so a runaway kernel/triton grandchild is reaped, not leaked — but this is
    defense-in-depth, not a sandbox boundary.

A hardened verify-only invocation looks like:

```bash
docker run --rm -it --gpus all \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --memory=64g --pids-limit=2000 --network=none \
  -e CODERBENCH_ROOT=/benchmarks -e RESULTS_DIR=/harness/results \
  -v "$PWD":/harness:ro -v "$PWD/benchmarks":/benchmarks:ro \
  -v "$PWD/results":/harness/results \
  -w /harness --user 1000:1000 eval-harness \
  bash run_all.sh --down-only <served_name>
```

For the generate stage (needs the network to reach the model endpoint) drop
`--network=none` and scope it to that endpoint with a custom Docker network or
firewall rules. See [config.local.sh.example](config.local.sh.example) for where
gateway credentials are kept (never on the command line, never in logs).

---

## Benchmarks covered (canonical config)

| Benchmark | #problems | max_tokens | sampling / metric | verification |
|---|---|---|---|---|
| VerilogEval spec-to-rtl | 156 | 49152 | 4 samples, pass_rate (avg@4) | iverilog sim |
| VerilogEval code-complete | 156 | 49152 | 4 samples, pass_rate (avg@4) | iverilog sim |
| RTLLM 2.0 | 50 | 49152 | 4 samples, func/syntax avg@4 + pass@1 | iverilog sim |
| ArchXBench v1.5 | 71 | 49152 | 5 samples, func/syntax pass@1 | numeric compare |
| RealBench-Module | 60 | 49152 | 5 samples, syntax/func pass@1/@5 | verilator formal |
| KernelBench L1+L2+L3 | 250 (100/100/50) | 58000 | 1 sample, compiled% / correct% (pass@1) | exec numeric compare, **triton+cuda** |
| TritonBench-G | 184 | 58000 | 1 sample, correct% (pass@1) | compare vs reference solution |
| CVDP cid003 | 78 | — | n=5, pass@1 | cocotb + iverilog |

> KernelBench **must use the cuda backend** (not the default triton): the model
> writes a CUDA kernel, and the eval compiles/executes it with nvcc/CUDA and
> numerically compares against a torch reference. See [benchmark details](#per-benchmark-details).
> CVDP is not in the canonical scored set — run it on demand (cid003, n=5, pass@1).
> The CVDP dataset is **not** cloned by `setup/setup_datasets.sh`; obtain it
> separately and point `CVDP_ROOT` / `CVDP_DATASET` (see `config.sh`) at it.

---

## Two eval modes

Every benchmark hits the same OpenAI-compatible endpoint at `:PROXY_PORT`
(default 8000). The only difference is what sits behind that proxy — local vLLM,
or an external API gateway.

### 1. External API mode (`--api`)

The proxy forwards to an external gateway, injecting `EXTERNAL_API_KEY` as the
Authorization header and rewriting the request's `model` field to
`API_SERVED_NAME` (this is how benches like KB that send `"default"` get routed).
**No local vLLM, no VRAM consumed** — the verify stage gets the full GPU. To
switch models, change only `API_SERVED_NAME`.

```bash
# run locally (proxy + all benches + summarize)
bash run_all.sh --api "$EXTERNAL_API_URL" <served_name>
bash summarize.sh --table <served_name>
```

### 2. Local vLLM mode

`run_all.sh` first starts N vLLM backends (tensor-parallel `TP_SIZE` / the number
of data-parallel replicas follows `N_GPU` and `TP_SIZE`) + a `:8000` round-robin
proxy, runs the UP generate group; then **stops the engine to free VRAM**, then
runs the DOWN verify group.

```bash
# run locally
bash run_all.sh <model_path> <served_name> [max_model_len]
bash summarize.sh <served_name>
```

For a KV-tight 32B-class dense model: `TP_SIZE=2`, `MAX_NUM_SEQS=64`,
`MLEN=81920` (vLLM KV can spill to RAM, so TBG's 58K-token generations fit).

### Two-stage flow (why stop the engine first)

```
Stage A engine online → UP group: VerilogEval / RTLLM / ArchX / RealBench / CVDP +
                          KB/TBG/TBT *generate* (only queries the model, no GPU)
stop engine           → free VRAM
Stage B engine off    → DOWN group: KB/TBG/TBT *verify* (torch+triton/cuda, needs full VRAM),
                          sharded across N_GPU, one shard per card, then merged
```

KB/TBG verify is VRAM-heavy and must be staggered with generation; RTL/CVDP
benches are CPU (iverilog/cocotb) and only query the engine. API mode has no
local vLLM, so there's nothing to stop (VRAM is already free).

---

## Per-benchmark details

Each row = one `registry.sh` entry: `name | runner | mode | extra | default_env`.
`run_all.sh` calls `env $default_env bash benches/$runner $mode $KEY $extra`
(mode/extra are unquoted, empty values collapse, multi-token extras like
`1,2,3 cuda` split into positionals). The `MAXTOK` env var is honored by every runner.

### VerilogEval (spec-to-rtl + code-complete-iccad2023)
- **runner** `benches/run_verilogeval.sh`, single-stage (engine online).
- **config** `SAMPLES=4 TEMP=0.8 TOPP=0.95 MAXTOK=49152` (avg@4).
- **flow** drives the official verilog-eval Makefile → sv-generate (writes an
  OpenAI client that hits the proxy) → `make -j$JOBS MAX_TOKENS=$MAXTOK` generate
  → iverilog v12 compile + run testbench.
- **metric** sv-iv-analyze `pass_rate` = mean of npass/nsamples per problem = average@4.
- **infra carve-out (`verify/verilogeval.py`)**: when sv-generate's gateway call
  fails it retries 10×; if all fail `resp` is unbound → the script crashes leaving
  a 2-byte empty `.sv`, which sv-iv-analyze scores 0 and keeps in the denominator
  → gateway jitter depresses VE scores. `summarize.sh:ve()` now recomputes
  `pass_rate_clean` at table time: only samples with ≥10 `LLM query failed` lines
  in sv-generate.log count as infra (this is a count the harness itself produced,
  a trustworthy control-plane signal → `scheduler_failure`), gated out of the
  denominator via `verify.core`'s closed-blocklist; an empty `.sv` alone is NOT
  infra (candidate-controllable, still counts as a model failure). On a clean run
  (n_infra=0) clean==raw (verified 1:1 on a clean reference run 69.87/79.81).
  **Does not touch `run_verilogeval.sh`** — pure table-time recompute, zero impact
  on queued jobs.
- **gotcha** iverilog/vvp treat `TEMP`/`TMP` as a scratch dir; sampling temp 0.8
  makes it write into a dir named "0.8" → every compile fails; the runner captures
  `GEN_TEMP` first, then `scrub_temp_for_iverilog`. `run_verilogeval.sh` **honors
  the `MAXTOK` env var** (it used to read only positional `$2`, so the registry's
  MAXTOK never got in → it always ran 32768).

### RTLLM 2.0
- **runner** `benches/run_rtllm.sh` → `run_rtllm.py`, single-stage.
- **config** `SAMPLES=4 TEMP=0.8 WORKERS=64 MAXTOK=49152`.
- **flow** os.walk finds 50 designs (`design_description.txt` + `testbench.v`,
  skipping `_chatgpt`), samples 4 per design → iverilog sim.
- **metric** `results/<key>/rtllm/summary.json`: `func_avg@4` / `syntax_avg@4` /
  `func_pass@1`.

### ArchXBench v1.5
- **runner** `benches/run_archx.sh` → `run_archx.py`, single-stage.
- **config** `SAMPLES=5 TEMP=0.8 WORKERS=64 MAXTOK=49152` (5-sample pass@1).
- **flow** 71 designs, numeric compare (compile + run, diff output).
- **metric** `summary.json`: `func_pass@1` / `syntax_pass@1` / `t` (assertion
  pass %, preferring `t_clean`).
- **classification** sample-level infra (no testbench = `trusted_reference_configuration`;
  gateway query failure = `scheduler_failure`) goes through `verify/archxbench.py`
  → `verify.core`'s closed-blocklist gate; `r["infra"]` is derived by the gate;
  `scored = K - infra` denominator is unchanged. **resume**: re-runs skip already
  generated designs (cheap).

### RealBench-Module
- **runner** `benches/run_realbench.sh` → `gen_realbench.py` (generate) +
  `run_verify.py` (verilator formal verification), single-stage.
- **config** `SAMPLES=5 TEMP=0.8 MAXTOK=49152 WORKERS=60`, `--task_level module`
  hardcoded (60 problems).
- **flow** 3 systems (sdc/aes/e203_hbirdv2) × K samples (codeid 1 greedy, 2..K
  sampled) → verilator formal equivalence → Syn@5 + Func@5.
- **metric** Syn/Func pass@1/@5 in `verify.log`.
- **gotcha** `gen_realbench.py`'s `urlopen(timeout=...)` must be ≤900s (was 86400=24h;
  one hung request deadlocked the whole threadpool join → UP group stalled 24h).
  Keep `MAXTOK` ≤ `max_model_len - longest prompt` (E203 longest prompt ~10K),
  else vLLM returns 400 context overflow.

### KernelBench (L1/L2/L3)
- **runner** `benches/run_kernelbench.sh`, **two-stage**: `gen` (engine online) +
  `eval` (engine off, sharded).
- **config** `TEMP=0 MAXTOK=58000 WORKERS=32`, 1-sample pass@1, **backend=cuda**.
- **flow**
  - `gen`: `generate_samples.py dataset_src=local level=N run_name=… backend=cuda
    check_kernel=False` → the model writes a CUDA kernel (not the default triton).
  - `eval`: `eval_from_generations.py backend=cuda` or sharded (when N_GPU>1,
    `kb_sharded_eval.sh` one shard per card + `kb_merge_shards.py` merge) →
    nvcc compile + CUDA exec, numeric compare vs torch reference.
- **metric** `eval_results.json` per problem: `compiled`(bool) / `correctness`(bool)
  / `runtime`. comp/corr = compiled/correctness counts; **fast** = correct kernels
  whose speedup>1 over the A100 eager reference (requires
  `_kb_ref_timing_A100.json`, produced by `scripts/kb_ref_timing_A100.py`).
  `summarize.sh` reports fast inline (`fast=X/Y`) when ref timings are present;
  omitted when the ref file is absent, `n/a` when no correct kernel was timed.
  See `scripts/kb_fast_metric.py --selftest` for the regression test.
- **backend** `run_kernelbench.sh` defaults `BACKEND=${4:-cuda}`; the registry extra
  appends `cuda` (e.g. `1,2,3 cuda`) → `$3=1,2,3 $4=cuda`. Supports triton / cuda /
  cute / tilelang.
- **venv** when the KB root has no `.venv`, uses `KERNELBENCH_PY` (image python;
  triton 2.3.1 is enough for KB gen).

### TritonBench-G / -T
- **runner** `benches/run_tritonbench.sh`, two-stage gen+eval.
- **config** G: `TEMP=0 MAXTOK=58000 WORKERS=16`; T same. 1-sample correct%@1.
- **flow** dump seeds (`tbg_dump_seeds.py`, including `triton_interface`) → teacher
  generate → `tbg_verify_correctness.py` compares results vs reference. eval is
  sharded across N_GPU.
- **metric** correct/total. **gotcha** verify needs triton 3.x (`.venv-vllm`); the
  image's triton 2.3.1 turns ~28/29 "skip" lines into false version mismatches
  (the ref uses `tl.interleave/cast/rsqrt/libdevice`).

### CVDP cid003
- **runner** `benches/run_cvdp.sh`, single-stage (engine online, CPU cocotb+iverilog).
- **config** `SAMPLES=5 HARNESS_THREADS=8` (n=5 pass@1). **Not in the canonical
  scored set** — run on demand.
- **flow** `CVDP_VENV` (.venv-cvdp with pytest + cocotb 2.0.1) runs
  `local_extensions/eval_5models_factory.py --n-samples K --llm …`.
- **metric** per-sample pass/78 in `run.log`; pass@1 = mean of 5-sample pass rates
  (unbiased estimate).
- **gotcha** `--llm` must be passed (else it enters Golden Mode using the reference
  solution, never calling the model → empty `.sv` → 0/78). The served_name must be
  registered in `local_extensions/eval_5models_factory.py:ALIASES`.

---

## Profiles (running a subset)

`--profile <name>` selects `BENCHES_UP_<NAME>` / `BENCHES_DOWN_<NAME>` (registry.sh):
- `table` (default): the canonical 6 benches (VEval/RTLLM/Archx/RealBench/KB/TBG).
- `incoder2`: 32B campaign — VE@49152 rerun + ArchX@5 + RealBench@49152 + KB-cuda
  (no TBG).
- `retry` / `kbeval` / `kbonly` / `kbveval` / `veval` / `arcb` / `archxonly` /
  `tracka` / `rtllm` / `tbg` / `tbt` / `smoke` / `coderrtl` / `cvdp` / `vevalcvdp`:
  rerun subsets that fill in particular columns. Each bench's `MAXTOK`/samples is
  aligned to canonical.

To add a new bench: write `benches/run_<name>.sh` + add a line in registry.sh.

## Card count: single knob `N_GPU`

Auto-detected from `nvidia-smi -L`. All sharding (KB/TBG sharded eval, vLLM backend
count) follows `N_GPU`. Force-override with `export N_GPU=8`. Local vLLM replica
count = `N_GPU / TP_SIZE`.

---

## Prerequisites

You do **not** need to install everything to use a subset of benches. Install
the core, then add only what the benchmarks you plan to run require.

### Core (always)

| Requirement | Version / note |
|---|---|
| OS | Linux (CUDA + iverilog from source; not tested on macOS/Windows) |
| Python | **3.11** (Dockerfile uses the `python:3.11-slim` tag; 3.10+ likely works) |
| bash | 4+ (uses `mapfile`/associative arrays in places) |
| git, curl | for `setup/setup_datasets.sh` + toolchain download |
| Disk | ~3 GB for the dataset clones; + model weights (local vLLM) which can be 50–100 GB for a 27–32B bf16 checkpoint; + `results/` (a few hundred MB per run) |
| Python pkgs | the proxy + orchestration need `openai`, `litellm`, `python-dotenv` (see [requirements.txt](requirements.txt)) |

### Local vLLM mode (skip for `--api` external-API mode)

| Requirement | Note |
|---|---|
| NVIDIA GPU | ≥1 card; **80 GB** recommended for 27–32B bf16 @ full 81920 context. A 1-GPU box works for debug with `MAX_MODEL_LEN=32768`. |
| CUDA | **12.1** (matches the `build_venv_cu121.sh` helper + torch cu121 wheels) |
| `vllm` + `torch` + `triton` | NOT in requirements.txt — install the build matching **your** driver/CUDA into `VLLM_PY`'s venv. vLLM must support your model's architecture. |

### Per-benchmark (add only what you run)

| Bench | Needs | Python deps |
|---|---|---|
| VerilogEval | **iverilog v12** (v14-devel in oss-cad-suite has a `$dumpvars` bug → pass_rate=0; see [Security warning](#security-warning--the-harness-executes-model-generated-code)) | `langchain-openai`, `langchain-community` |
| RTLLM / ArchXBench | iverilog v12 (CPU sim) | none beyond core (`SYS_PY`) |
| RealBench | verilator + yosys (oss-cad-suite), iverilog | its own `run_verify.py` venv (`REALBENCH_PY`) |
| KernelBench | **CUDA GPU + nvcc + torch + triton 3.x** | `einops`, `tabulate`, `tomli`, `pydra-config`, `ninja`, `litellm` (`KERNELBENCH_PY` venv) |
| TritonBench | CUDA GPU + **triton 3.x** (the image's 2.3.1 mis-flags refs) | torch+triton in `.venv-vllm` |
| CVDP | iverilog v12 (+vpi, for cocotb), CPU cocotb+pytest | `cocotb`, `pytest` (`CVDP_VENV`) |

> `setup/setup_toolchain.sh` builds iverilog v12 + verilator + yosys into one
> dir; `setup/setup_datasets.sh` clones the public benches. The
> [Dockerfile](Dockerfile) preinstalls the toolchain + torch/triton/KB deps, so
> `docker build` is the fastest path to a working environment.

---

## Quickstart (from scratch)

```bash
git clone https://github.com/Magicpjl/eval_harness.git
cd eval_harness

# 1) toolchain (iverilog v12 + verilator + yosys) — binaries
bash setup/setup_toolchain.sh /opt/eval-toolchain

# 2) datasets (clone public benches + apply KB/VEval/CVDP patches)
bash setup/setup_datasets.sh ./benchmarks

# 3) fill in config (sanitized template)
cp config.local.sh.example config.local.sh
#   IVERILOG12_BIN=/opt/eval-toolchain/iverilog12/bin
#   SETUP_ENV=/opt/eval-toolchain/setup_env.sh
#   CODERBENCH_ROOT=./benchmarks (and the various *_ROOT, see setup output)
#   VLLM_PY / SYS_PY / KERNELBENCH_PY (per-interpreter)
#   EXTERNAL_API_URL / EXTERNAL_API_KEY / API_SERVED_NAME (API mode)
#   GPU_ARCH=Ampere (or Hopper), N_GPU, MAX_MODEL_LEN

# 4a) API mode, one click
bash run_all.sh --api "$EXTERNAL_API_URL" <served_name>
# 4b) local vLLM, one click
bash run_all.sh <model_path> <served_name>

# 5) table
bash summarize.sh --table <served_name>
```

Or Docker (toolchain preinstalled):

```bash
docker build -t eval-harness .
# mount the repo + datasets read-only and run hardened (see ⚠️ Security warning)
docker run --rm -it --gpus all --cap-drop=ALL --security-opt=no-new-privileges \
  --memory=64g --pids-limit=2000 \
  -e CODERBENCH_ROOT=/benchmarks -e RESULTS_DIR=/harness/results \
  -v "$PWD":/harness:ro -v "$PWD/benchmarks":/benchmarks:ro \
  -v "$PWD/results":/harness/results \
  -w /harness --user 1000:1000 eval-harness \
  bash run_all.sh --api "$EXTERNAL_API_URL" <served_name>
```

See [Dockerfile](Dockerfile).

---

## summarize table

```bash
bash summarize.sh <key>          # single-model detail
bash summarize.sh --table <key>  # one row of the 11-column comparison
```
Columns: VEval-spec / VEval-code / RTLLM(func_avg@4) / CVDP / RB-Syn@5 / RB-Func@5 /
ArchX-t / KB-L1 / KB-L2 / KB-L3 / TBG. KB column format `comp/n corr/n` plus
`fast=X/Y` (share of correct kernels beating A100 eager, speedup>1) when the
A100 reference timing file (`results/_kb_ref_timing_A100.json` from
`scripts/kb_ref_timing_A100.py`) is present — omitted otherwise, `n/a` if no
correct kernel was timed. ArchX takes `t_clean`.

---

## Directory structure

```
eval_harness/
├── run_all.sh           two-stage orchestration (UP engine online → stop → DOWN shard verify)
├── summarize.sh         aggregate to table (--table = one row)
├── config.sh            central config (paths/venv/N_GPU/MAX_MODEL_LEN/proxy/env)
├── config.local.sh(.example)  per-host overrides (config.local.sh takes precedence)
├── LICENSE / NOTICE / CITATION.cff / CHANGELOG.md / VERSION / requirements.txt
├── engine/
│   ├── serve_vllm.sh    three modes: local (N-card vLLM) / api (forward to gateway) / stop
│   └── proxy_rr.py      :8000 round-robin proxy, injects key + rewrites model field
├── expand/              **optional** teacher/SFT data-generation layer (llm client /
│   │                     datasets path resolution / perturb methods). NOT part of the
│   │                     standard eval flow — `run_all.sh`/`summarize.sh` never import
│   │                     it; it exists for generating training data. Safe to ignore /
│   │                     delete for evaluation-only use.
├── benches/
│   ├── registry.sh      all bench/profile lists (canonical alignment lives here)
│   ├── run_verilogeval.sh / run_rtllm.sh / run_archx.sh / run_realbench.sh
│   │   / run_kernelbench.sh / run_tritonbench.sh / run_cvdp.sh
│   ├── *.py             gen_realbench / run_rtllm / run_archx / kb_*/tbg_* sharding+merge and verify
│   └── kb_sharded_eval.sh / kb_merge_shards.py
├── verify/              hardened verify suite (RLVR alignment + unified classification)
│   ├── core.py          is_infra + finalize_failure_classification (closed blocklist)
│   ├── tritonbench.py / kernelbench.py / icarus.py / realbench.py / cvdp.py
│   ├── archxbench.py    ArchX sample-level classification → core gate (verifier unchanged)
│   ├── verilogeval.py   VE table-time pass_rate_clean recompute (sv-generate ≥10-retry = infra)
│   └── profiles/build_profiles.py
├── scripts/
│   └── README.md        local helper scripts (rescore / metric extraction / venv build / probes)
├── setup/               setup_toolchain.sh / setup_datasets.sh / apply_*_patches.py
├── ext/verilog-eval/    vendored upstream (scripts/sv-generate has the gateway model whitelist patch)
└── lib/common.sh        log / require_engine / load_verilog_toolchain
```

---

## Hardened verify suite (2026-08-12 / classification unified 2026-08-14)

The **failure classification** of all seven benchmark families is unified through
`verify.core`'s closed blocklist. Five of them (TBG / KernelBench / RTLLM /
RealBench / CVDP) also port the *verifier* (compile + diff against an oracle) from
the upstream RLVR suite (`verifiers.zip` → `verify_server_v2.py`); VerilogEval and
ArchXBench keep their own verifiers (sv-iv-analyze / `verify_candidate` iverilog
diff), but their **classification decisions** now also go through `verify.core`:

- **ArchXBench** (`verify/archxbench.py`): sample-level infra (no testbench / gateway
  query failure) is marked with a trusted trigger `_infra_trigger` → `infra_failure`
  + `finalize_failure_classification` gate; `r["infra"]` is derived by the gate.
  Denominator `scored = K - infra` is unchanged.
- **VerilogEval** (`verify/verilogeval.py`): recomputes `pass_rate_clean` at table
  time, counting only samples where sv-generate exhausted retries (≥10
  `LLM query failed`) as infra (a harness count, trustworthy control-plane →
  `scheduler_failure`), gating them out of the denominator. Does not touch
  `run_verilogeval.sh`.

**Core principle: infrastructure is a closed blocklist** — candidate-controllable
text (stdout/traceback/exit code) can never authorize an "infra" classification
that drops a row from the denominator. Not correct AND not a trusted infra failure
= **model failure** (zero score, stays in the denominator). A single
`verify.core.is_infra` + a single `finalize_failure_classification` is enforced at
each result boundary. Anti-cheat signals (`identity_hack`/`framework_delegation`)
are recorded only as peer fields, **not** hard-gating the reward (the detectors are
noisy heuristics that would false-positive on `min()/sum()/.exp()`/scalar binop, even
mis-flagging golden refs). `correct` is decided solely by the oracle (allclose /
pytest exit / stdout-SHA256).

Code in `verify/{core,tritonbench,kernelbench,icarus,realbench,cvdp,archxbench,verilogeval}.py`
+ `verify/profiles/build_profiles.py`. See [verify/README.md](verify/README.md) for
the per-module reference and the closed-blocklist policy.

### End-to-end real verification (real tasks + real toolchain, not smoke)

| Bench | positive path (correct) | negative path (¬correct) | bug fixed |
|---|---|---|---|
| TBG | golden self-match = True | known-wrong candidate = False | removed `if framework_delegation: correct=False` hard-gate (golden false-positive) |
| KernelBench | known-correct (pid3 sid0)=pass | pid1 wrong=compiled_but_wrong | worker f-string reopened as plain string, `CUDA_VISIBLE_DEVICES` int→str |
| RealBench | golden self-match=True (stdout SHA256 c761cae0…) | — | `binary_name="Vtb"` (code already `root/obj_dir/`) |
| CVDP | hand-written correct parametrized barrel_shifter=pass | wrong candidate=compile_fail | `.venv-cvdp` installs pytest; `rtl_rel_path` aligned to slot filename |
| RTLLM | golden self-match adder_bcd=True | — | `build_profiles` depth-3 glob; lowercase probe path first |
| ArchXBench | 7/7 classification smoke (infra gate accepts trusted trigger, rejects candidate text) | — | `classify_archx_sample` idempotent; resume of old results doesn't re-gate |
| VerilogEval | reference run 69.87/79.81 recomputed 1:1 (n_infra=0) | synthetic 10-retry log → infra gated out | empty standalone `.sv` is not infra (candidate-controllable) |

Plus 8 offline verify self-tests = **45/45** (core 4 / tritonbench 5 / kernelbench 5 /
icarus 7 / realbench 5 / cvdp 7 / archxbench 7 / verilogeval 5).

---

## Example results (reference run, 2026-07-06)

> The following is one real run of an internal reference model, included only for
> end-to-end validation and illustration; the numbers themselves do not constitute
> a public evaluation claim about that model.

VEval 67.95/67.95 · RTLLM func@4=51.0 · CVDP 28/78 · RealBench Syn@5=0.367/Func@5=0.167 ·
Archx t=29.53 · KB L1 93/100 32/100 · KB L2 91/100 40/100 · KB L3 43/50 2/50 · TBG 118/184 27/184.
**All 11 columns present.**

---

## License

MIT — see [LICENSE](LICENSE).

This harness drives and vendors material from several upstream projects
(VerilogEval, KernelBench, TritonBench, ArchXBench, RealBench, RTLLM, CVDP,
oss-cad-suite) whose own licenses apply. The root MIT does **not** automatically
cover those. See [NOTICE](NOTICE) for the layered licensing — notably
TritonBench is Apache-2.0 and `ext/verilog-eval/` preserves its upstream MIT.

## Citation

If you use eval_harness in your work, cite this repository. See
[CITATION.cff](CITATION.cff) (a stable, versioned reference to this repo;
no separate paper claim is implied).
