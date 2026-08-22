# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-21

First public release.

### Added
- One-proxy, two-stage (generate → stop engine → shard-verify) eval harness for
  code LLMs across Verilog (VerilogEval, RTLLM 2.0, ArchXBench v1.5,
  RealBench-Module, CVDP) and Triton/CUDA (KernelBench L1/L2/L3, TritonBench-G)
  benchmarks.
- `run_all.sh` / `summarize.sh` orchestration with a single `N_GPU` knob.
- Two modes: external OpenAI-compatible API (`--api`) and local vLLM.
- Hardened `verify/` suite with a closed-blocklist infra-failure classifier
  (`verify/core.py`) shared across all scored benchmarks.
- `setup/` toolchain + dataset bootstrap, Dockerfile, and a per-host
  `config.local.sh` override mechanism.

### Notes
- Defaults are host-agnostic (repo-relative paths, `python3` interpreters).
  Override paths, venvs, toolchain, and gateway credentials in
  `config.local.sh` (see `config.local.sh.example`).
