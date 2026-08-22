"""verify/ — hardened correctness verifiers, local (non-server) edition.

This package ports the verify-worker + classification + anti-cheat logic from
the upstream RLVR verifier suite (verifiers_unzip/verifiers/) into local
Python modules called directly by eval_harness/benches/run_*.sh drivers.

Design contract vs. the upstream server edition:
  * No FastAPI server, no httpx client, no sealed-payload transport. Bench
    drivers call these functions in-process; refs are read from disk and pinned
    by SHA256 (verify.core.sha256_file) instead of by payload allowlist.
  * The correctness *oracle* is preserved verbatim per benchmark:
      - TBG/KB  : _nested_allclose on the reference result dict (+launch/identity
                  /framework-delegation anti-cheat for the Triton backends).
      - RTLLM/RealBench : stdout-sha256 verdict profile (precomputed under a
                  pinned toolchain). This IS the oracle — there is no on-disk
                  fallback, so a verdict-profile build step is mandatory.
      - CVDP    : pytest returncode == 0.
  * Failure classification is the upstream explicit-infra blacklist verbatim:
    only a closed set of trusted infra_codes on a trusted channel count as
    infrastructure; everything else is a model failure. One definition, in one
    place — imported by both the merge scripts and summarize.sh, so the
    denominator can never drift (the original 137/139 TBG bug).

Public surface:
  core.base_result / infra_failure / model_failure / finalize_failure_classification
  core.trusted_backend_infra_code
  core.is_infra(result)            -> bool  (excluded from denominator)
  core.is_skipped(result)          -> bool  (alias for the merge/summarize path)
  core.run_sandboxed(args, cwd, timeout, ...)  -> (rc, out, err, timed_out)
  core.sha256_bytes / sha256_file
  core.INFRA_CODES / CLASSIFICATION_POLICY
"""

from .core import (
    INFRA_CODES,
    CLASSIFICATION_POLICY,
    VERIFIER_VERSION,
    base_result,
    infra_failure,
    model_failure,
    trusted_backend_infra_code,
    finalize_failure_classification,
    is_infra,
    is_skipped,
    run_sandboxed,
    sha256_bytes,
    sha256_file,
)

__all__ = [
    "INFRA_CODES",
    "CLASSIFICATION_POLICY",
    "VERIFIER_VERSION",
    "base_result",
    "infra_failure",
    "model_failure",
    "trusted_backend_infra_code",
    "finalize_failure_classification",
    "is_infra",
    "is_skipped",
    "run_sandboxed",
    "sha256_bytes",
    "sha256_file",
]
