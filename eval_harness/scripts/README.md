# Maintenance utilities

The scripts in this directory operate on local evaluation artifacts. They do not
submit remote jobs or manage cloud resources.

| Script | Purpose |
|---|---|
| `build_venv_cu121.sh` | prepare a CUDA-compatible Python environment for GPU verification |
| `kb_fast_metric.py` | derive KernelBench correctness and optional performance fields |
| `kb_ref_timing_A100.py` | generate a local reference-timing artifact for optional performance analysis |
| `_rescore_extract.py` | shared extraction helpers for existing model outputs |
| `rescore_tbg.py` | rerun TritonBench scoring from saved artifacts |
| `rescore_verilog.py` | rerun VerilogEval scoring from saved artifacts |
| `_rtllm_selfmatch_smoke.py` | exercise the RTLLM verifier with a reference candidate |

All paths are derived from the repository root, so the scripts can be launched from
any working directory. Credentials must be supplied through environment variables or
a local secrets file excluded from version control.

