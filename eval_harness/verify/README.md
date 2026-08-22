# Verification architecture

The `verify` package adapts benchmark-native checks to a common result model. It is
used directly by benchmark runners and does not require a separate verification
service.

## Trust boundary

Candidate source code, compiler output, runtime output, tracebacks, and process exit
codes are untrusted. They cannot declare themselves to be infrastructure failures or
remove themselves from a scoring denominator.

[`core.py`](core.py) centralizes the boundary:

- `finalize_failure_classification(...)` accepts infrastructure classifications only
  from trusted harness channels;
- `is_infra(...)` provides one predicate for downstream aggregation;
- `run_sandboxed(...)` starts a separate process group, applies resource limits,
  enforces a timeout, and terminates remaining child processes;
- temporary working directories keep candidate artifacts away from the repository.

These controls reduce accidental interference but do not make generated code safe.
Run the harness inside an isolated, disposable environment as described in the
[security guidance](../README.md#security).

## Modules

| Module | Benchmark family | Primary oracle |
|---|---|---|
| `tritonbench.py` | TritonBench | recursive numerical comparison with the reference implementation |
| `kernelbench.py` | KernelBench | the benchmark's official reference evaluator |
| `icarus.py` | RTLLM | simulator transcript matched against a prepared verdict profile |
| `realbench.py` | RealBench | normalized simulator transcript matched against a versioned profile |
| `cvdp.py` | CVDP | benchmark testbench completion |
| `archxbench.py` | ArchXBench | sample-level classification around the benchmark runner |
| `verilogeval.py` | VerilogEval | result cleanup around the upstream analyzer |

Anti-cheat signals are recorded as diagnostic fields. Correctness remains tied to
the benchmark oracle unless a module explicitly documents a different policy.

## Verdict profiles

RTLLM and RealBench can use precomputed profiles produced by
[`profiles/build_profiles.py`](profiles/build_profiles.py). Build profiles on the
same pinned toolchain used for evaluation.

```bash
python verify/profiles/build_profiles.py \
  --backend rtllm \
  --rtllm-root "$RTLLM_ROOT" \
  --repeats <count>

python verify/profiles/build_profiles.py \
  --backend realbench \
  --realbench-manifest <manifest.json> \
  --repeats <count>
```

Configure the resulting artifacts with `RTLLM_VERDICT_PROFILE` or
`REALBENCH_VERDICT_PROFILE`.

## Local checks

The pure-logic checks can run without benchmark datasets or a GPU:

```bash
python -m verify.core
python -m verify.tritonbench
python -m verify.kernelbench
python -m verify.icarus
python -m verify.realbench
python -m verify.cvdp
python -m verify.archxbench
python -m verify.verilogeval
```

End-to-end verification additionally requires the benchmark datasets, compilers,
simulators, and GPU stack used by the selected runners.

