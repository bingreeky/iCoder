# scripts/

Local helper scripts — utilities for rescoring existing rollouts, extracting
metrics, building the verify venv, and running smoke probes. None of these
submit jobs or talk to any cloud; they're host-local conveniences. Each derives
the repo root itself and runs from anywhere.

| script | what it does |
|---|---|
| `build_venv_cu121.sh` | Build the CUDA 12.1 `.venv-vllm` (triton 3.x verify needs it; the image's triton 2.3.1 mis-flags TBG refs). |
| `kb_fast_metric.py` | Single source of truth for the KernelBench column + `fast` (speedup>1 over A100 eager) metric. `summarize.sh` imports `compute_kb_level` from it; run standalone for a multi-model comparison table. `--selftest` runs the regression test. |
| `kb_ref_timing_A100.py` | Measure reference-kernel runtimes on an A100 → `results/_kb_ref_timing_A100.json` (the speedup denominator for `fast`; required for the `fast` cell to appear in the table). |
| `_rescore_extract.py` | Shared extractor for re-deriving metrics from existing rollouts without regenerating. |
| `rescore_tbg.py` | Re-derive TritonBench-G correctness from existing rollouts. |
| `rescore_verilog.py` | Re-derive VerilogEval pass_rate (and `pass_rate_clean`) from existing sv-generate + sv-iv-analyze artifacts. |
| `_rtllm_selfmatch_smoke.py` | RTLLM golden self-match smoke check (positive path = True). |

> Conventions:
> - `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` derives the
>   repo root, so scripts work regardless of where they're invoked from.
> - Gateway credentials are never hardcoded: source a secrets file you create
>   that exports `EXTERNAL_API_URL` / `EXTERNAL_API_KEY`, or export those vars
>   in the env.
