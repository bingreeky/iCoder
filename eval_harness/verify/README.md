# verify/ — hardened RLVR verifier suite (local, on-disk)

Local Python ports of the upstream RLVR verifier logic
(`verifiers/{verify_server_v2,icarus_benchmark_verifier,realbench_verifier,
cvdp_verifier,kernel_payload_verifier,reward_kernel_v2}.py`), called directly by
the `benches/` drivers — **no FastAPI server, no sealed-payload transport**.
Refs are read from disk; the caller is responsible for pinning them.

## The one classification gate

[`core.py`](core.py) is the single source of truth for the infra/model split and
the subprocess sandbox:

- `INFRA_CODES` — a closed 12-code blacklist. Infra is **only** assertable by a
  trusted channel (`trusted_control_plane`, `trusted_cpu_wrapper`); everything
  candidate-controlled (text, tracebacks, exit codes, backend diagnostics) is
  forced to **model failure** by `finalize_failure_classification`.
- `is_infra(result)` — the unified "skipped" predicate. Replaces the three
  divergent `_skipped` definitions that caused the TBG 137/139 denominator drift.
- `run_sandboxed(...)` — `subprocess.Popen(start_new_session=True)` +
  RLIMIT_FSIZE/CPU/AS/NOFILE + `killpg(SIGKILL)` on timeout.

Everything else in `verify/` flows its results through `finalize`, so there is
exactly one definition of "infrastructure" across the whole harness.

## Per-benchmark modules

| module | benchmark | oracle | anti-cheat | notes |
|---|---|---|---|---|
| `tritonbench.py` | TritonBench-G/T | `_nested_allclose` (recursive) | triton launch-exit-hook counter (feature-detected — triton 2.3.x gracefully degrades to `None`), `identity_hack`, `framework_delegation` | deterministic seeding; `per_variant_timeout = max(180, t/2)` autotune floor — **the denominator-stability fix** |
| `kernelbench.py` | KernelBench | `kernelbench.eval.eval_kernel_against_ref` (official, 32 trials) | launch counter + `identity_hack` + `framework_delegation` (hard gate — no separate reward layer) | subprocess under `KERNELBENCH_PY` (KB venv has torch+triton+litellm); counter is a prefix to the candidate module so it counts candidate launches only; "every post-bootstrap exception is a model failure" |
| `icarus.py` | RTLLM | **stdout SHA256** vs verdict profile | `unsafe_runtime_construct` ($display/$dumpvars/DPI-C ban) | replaces the fragile `re.search(r"\b(pass|passed)\b")` regex — a model $display-ing "pass" can no longer fake a pass |
| `realbench.py` | RealBench | **normalized stdout SHA256** vs version-pinned profile | same unsafe-construct ban | `normalize_transcript` strips non-deterministic verilator walltime/CPU lines; profile refuses to judge if runtime verilator version ≠ profile version (infra, not model) |
| `cvdp.py` | CVDP | pytest exit code (`returncode==0`) | `_ALLOWED_ENV` whitelist (forces `SIM=icarus`), cocotb 1.x→2.x compat rewrite, unsafe-construct ban | the oracle was already an exit code; this adds the env-lockdown + cocotb compat the upstream suite wraps around it |

VerilogEval v2 and ArchXBench are **intentionally NOT** routed through here —
they stay on the existing eval_harness paths (per the alignment scope).

## profiles/ — one-time verdict-profile precompute

RTLLM and RealBench need their correctness oracle **precomputed**: run the
golden reference through the pinned toolchain once, capture the stdout SHA256.
`build_profiles.py` does this and writes a profile JSON the runner loads at
judge time.

```bash
# RTLLM (one-time, on the eval box where RTLLM_ROOT + iverilog v12 live):
IVERILOG12_BIN=.../iverilog12/bin python verify/profiles/build_profiles.py \
    --backend rtllm --rtllm-root $RTLLM_ROOT --repeats 3

# RealBench (manifest-driven; verilator version pinned):
python verify/profiles/build_profiles.py --backend realbench \
    --realbench-manifest rb_manifest.json --repeats 3
```

Then point the runner at the profile:
- RTLLM: `export RTLLM_VERDICT_PROFILE=$RTLLM_ROOT/rtllm_verdict_profile.json`
  (absent → `run_rtllm.py` falls back to the legacy regex, so nothing breaks
  before the one-time precompute).
- RealBench: `export REALBENCH_VERDICT_PROFILE=.../realbench_verdict_profile.json`.

## Smoke tests (no GPU / no dataset / no toolchain needed)

Each module's pure-logic gates are self-testable offline:
```bash
python -m verify.core && python -m verify.tritonbench && python -m verify.kernelbench \
  && python -m verify.icarus && python -m verify.realbench && python -m verify.cvdp
```
Full compile+run paths run on the eval box (GPU for KB/TBG; iverilog/verilator
for RTLLM/RealBench; cocotb+pytest for CVDP).
