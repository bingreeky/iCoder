# Qwen3.6-27B VerilogEval Results

**Model**: Qwen3.6-27B (local vLLM, tensor-parallel=2, GPU 1&2)  
**Date**: 2026-04-25

## Pass Rate (%)

| Setting | spec-to-rtl | code-complete-iccad2023 |
|---------|:-----------:|:-----------------------:|
| n=1, temp=0, 0-shot | 67.31 | 71.79 |
| n=1, temp=0, 1-shot | 69.87 | 67.31 |
| n=20, temp=0.85, 0-shot | 64.74 | 67.31 |
| n=20, temp=0.85, 1-shot | 65.38 | 68.59 |

## Notes

- All runs: `--with-model=Qwen3.6-27B`, zero extra rules (`--with-rules` not set)
- n=1 results reflect pass@1 with greedy decoding
- n=20 results reflect pass@1 across 20 samples (any sample passing = pass)
- Build directories: `build/<task>-qwen-<setting>/`
- Raw per-problem results in each build dir: `summary.txt` / `summary.csv`
