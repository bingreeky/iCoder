"""verify/verilogeval.py — closed-blacklist infra carve-out for VerilogEval.

VerilogEval's *verifier* (the official verilog-eval Makefile → sv-generate →
iverilog → sv-iv-analyze that produces `summary.txt: pass_rate`) stays as-is.
This module ports only the *failure classification*: detecting which samples
are infrastructure failures (excluded from the denominator) and recomputing a
`pass_rate_clean` that carves them out, routed through `verify.core`.

Why this is needed (the gap this closes):
  Before, VE had NO infra carve-out at all. sv-generate retries a failed gateway
  call 10× (`for _ in range(10)`); if all 10 fail, `resp` is never bound and the
  script crashes, leaving a 2-byte empty `.sv`. sv-iv-analyze then counts that
  sample as 0 pass **in the denominator** — so a gateway hiccup unfairly tanks
  the VE score. The other 5 hardened benches all carve infra; VE was the one
  scored column with no such insurance.

Closed-blacklist discipline (the part that makes this safe):
  * An empty `.sv` alone is NOT a trusted infra signal — it is candidate-
    controllable (the model could have returned nothing). So empty-content
    alone stays a model failure (in the denominator), exactly as before.
  * The ONLY trusted infra signal is sv-generate's own retry-exhaustion: the
    programmatic fact that its loop printed "LLM query failed, retrying" ≥10×.
    That count is harness-generated (not model output), so it is a trusted
    control-plane signal → `scheduler_failure` on `trusted_control_plane`.
  * Each infra sample is routed through `core.infra_failure` +
    `finalize_failure_classification`, so the closed-blacklist gate vets it.

This runs at summarize time (summarize.sh:ve), reading `summary.txt` + the
per-sample `*-sv-generate.log` already on disk. run_verilogeval.sh is NOT
touched, so queued runs are unaffected; the clean recompute applies to whatever
lands on disk, and falls back to the raw `pass_rate` when there are no infra
samples (the common, clean case — verified 1:1 against the raw score on a
clean reference run).

References (upstream, verifiers_unzip/verifiers/):
  verify_server_v2.py:634-696  _base_result / _infra_failure / _model_failure
  verify_server_v2.py:737-790  _finalize_failure_classification (the gate)
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any

from .core import (
    finalize_failure_classification,
    infra_failure,
    is_infra,
)

EVAL_BACKEND = "verilogeval"

#: sv-generate prints this once per retry on a failed gateway call
#: (verilog-eval/scripts/sv-generate:453 & :466 — both the direct-OpenAI and
#: the langchain paths). range(10) ⇒ at most 10 retries; ≥10 occurrences means
#: the loop exhausted without a successful `break`, so `resp` was never bound
#: and the script crashed → that sample got no generation = infra. The count is
#: harness-generated text (not model output), hence a trusted control-plane signal.
_RETRY_LINE = "LLM query failed, retrying"
_RETRY_EXHAUSTED = 10

#: `Prob001_zero                  [04/04](100%)  0.50 ....`  → (Prob001_zero, 4, 4)
_PROB_RE = re.compile(r"^(Prob\S+)\s+\[(\d+)/(\d+)\]")

#: `Prob086_lfsr5/Prob086_lfsr5_sample01-sv-generate.log` → (Prob086_lfsr5, sample01)
_SAMPLE_LOG_RE = re.compile(r"^(Prob\S+)_sample(\d+)-sv-generate\.log$")


def detect_infra_samples(task_dir: str) -> dict[str, int]:
    """Per-problem count of gateway-infra samples (retry-exhausted).

    Returns {problem_name: n_infra_samples}. A sample is infra iff its
    sv-generate.log contains ≥ _RETRY_EXHAUSTED retry lines — the trusted
    signal that the gateway failed every retry and the model never answered.
    """
    per_problem: dict[str, int] = {}
    for log in glob.glob(os.path.join(task_dir, "Prob*", "*-sv-generate.log")):
        fname = os.path.basename(log)
        m = _SAMPLE_LOG_RE.match(fname)
        if not m:
            continue
        problem = m.group(1)
        try:
            with open(log, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if text.count(_RETRY_LINE) >= _RETRY_EXHAUSTED:
            per_problem[problem] = per_problem.get(problem, 0) + 1
    return per_problem


def _classify_infra(problem: str, sample: int) -> dict[str, Any]:
    """Route one VE infra sample through the closed-blacklist gate."""
    res = infra_failure(
        infra_code="scheduler_failure",
        eval_backend=EVAL_BACKEND,
        classification_channel="trusted_control_plane",
        info=f"veval_infra:gateway_retry_exhausted",
        error_type="gateway_retry_exhausted",
        original_info=f"{problem}_sample{sample}",
    )
    return finalize_failure_classification(res)


def parse_summary(summary_path: str) -> list[tuple[str, int, int]]:
    """Per-problem (name, npass, nsamples) from sv-iv-analyze's summary.txt."""
    out: list[tuple[str, int, int]] = []
    try:
        with open(summary_path, errors="replace") as f:
            for line in f:
                m = _PROB_RE.match(line.strip())
                if m:
                    out.append((m.group(1), int(m.group(2)), int(m.group(3))))
    except OSError:
        return []
    return out


def recompute_pass_rate_clean(task_dir: str) -> dict[str, Any] | None:
    """Recompute pass_rate with gateway-infra samples carved out.

    Returns None if summary.txt is absent/unparseable (caller falls back to the
    raw pass_rate). Otherwise:
      pass_rate         — reproduced mean(npass/nsamples)*100 (validates the parser
                          against sv-iv-analyze's own pass_rate line)
      pass_rate_clean   — mean over non-fully-infra problems of
                          npass/(nsamples - n_infra)*100
      n_problems        — #problems in summary.txt
      n_infra           — total infra samples (across all problems)
      n_fully_infra     — #problems where every sample was infra (excluded from
                          the clean mean's problem count, mirroring ArchX's
                          scored_des rule)
    """
    summary_path = os.path.join(task_dir, "summary.txt")
    probs = parse_summary(summary_path)
    if not probs:
        return None
    infra_per = detect_infra_samples(task_dir)

    total_pass = 0.0
    clean_sum = 0.0
    n_clean_problems = 0
    n_infra = 0
    n_fully_infra = 0
    for name, npass, nsamples in probs:
        ni = infra_per.get(name, 0)
        n_infra += ni
        total_pass += (npass / nsamples) if nsamples else 0.0
        denom = nsamples - ni
        if denom <= 0:
            # Every sample was infra → the model never got a chance on any of
            # them → exclude this problem from the clean mean entirely.
            n_fully_infra += 1
            continue
        clean_sum += (npass / denom)
        n_clean_problems += 1

    n = len(probs)
    pass_rate = round(100.0 * total_pass / n, 2) if n else 0.0
    pass_rate_clean = round(100.0 * clean_sum / n_clean_problems, 2) if n_clean_problems else 0.0
    # Sanity-classify one infra sample through the gate (records policy version
    # + confirms the closed blacklist accepts scheduler_failure). Count only —
    # the denominator effect is the math above; this also exercises the gate so
    # a future upstream change to INFRA_CODES surfaces here, not silently.
    if infra_per:
        _classify_infra(next(iter(infra_per)), 0)
    return {
        "pass_rate": pass_rate,
        "pass_rate_clean": pass_rate_clean,
        "n_problems": n,
        "n_infra": n_infra,
        "n_fully_infra": n_fully_infra,
    }


def _self_test() -> None:
    import tempfile

    # 1) parser: a clean summary (no infra) reproduces pass_rate, clean == raw.
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "summary.txt"), "w").write(
            "Prob001_zero                  [04/04](100%)  0.50 .\n"
            "Prob002_one                   [02/04]( 50%)  0.50 .\n"
            "pass_rate             =      75.00\n"
        )
        # no sv-generate.logs → no infra
        r = recompute_pass_rate_clean(d)
        assert r is not None
        assert r["pass_rate"] == 75.0  # mean(1.0, 0.5)*100
        assert r["pass_rate_clean"] == 75.0  # unchanged (no infra)
        assert r["n_infra"] == 0

    # 2) one infra sample carved: Prob002 had 4 samples, 1 infra → 2/(4-1)=66.7%
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "summary.txt"), "w").write(
            "Prob001_zero                  [04/04](100%)  0.50 .\n"
            "Prob002_one                   [02/04]( 50%)  0.50 .\n"
        )
        p2 = os.path.join(d, "Prob002_one")
        os.makedirs(p2)
        # 10 retry lines = exhausted = infra
        open(os.path.join(p2, "Prob002_one_sample01-sv-generate.log"), "w").write(
            "ERROR: LLM query failed, retrying in 20 seconds\n" * 10
            + "Traceback (most recent call last):\n UnboundLocalError: resp\n"
        )
        r = recompute_pass_rate_clean(d)
        assert r["n_infra"] == 1
        # Prob001: 4/4=100%; Prob002 clean: 2/(4-1)=66.67%; mean=83.33
        assert r["pass_rate_clean"] == round(100 * (1.0 + (2 / 3)) / 2, 2)
        assert r["pass_rate"] == 75.0  # raw mean(1.0,0.5)*100

    # 3) fully-infra problem excluded from clean mean's problem count.
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "summary.txt"), "w").write(
            "Prob001_zero                  [04/04](100%)  0.50 .\n"
            "Prob002_one                   [00/04](  0%)  0.00 .\n"
        )
        p2 = os.path.join(d, "Prob002_one")
        os.makedirs(p2)
        open(os.path.join(p2, "Prob002_one_sample01-sv-generate.log"), "w").write(
            "LLM query failed, retrying\n" * 10)
        open(os.path.join(p2, "Prob002_one_sample02-sv-generate.log"), "w").write(
            "LLM query failed, retrying\n" * 10)
        open(os.path.join(p2, "Prob002_one_sample03-sv-generate.log"), "w").write(
            "LLM query failed, retrying\n" * 10)
        open(os.path.join(p2, "Prob002_one_sample04-sv-generate.log"), "w").write(
            "LLM query failed, retrying\n" * 10)
        r = recompute_pass_rate_clean(d)
        assert r["n_fully_infra"] == 1
        assert r["n_infra"] == 4
        # only Prob001 in the clean mean → 100%
        assert r["pass_rate_clean"] == 100.0

    # 4) a sample that retried <10 then succeeded is NOT infra (model answered).
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "summary.txt"), "w").write(
            "Prob001_zero                  [04/04](100%)  0.50 .\n")
        p1 = os.path.join(d, "Prob001_zero"); os.makedirs(p1)
        open(os.path.join(p1, "Prob001_zero_sample01-sv-generate.log"), "w").write(
            "LLM query failed, retrying\n" * 3 + "problem = Prob001\n")
        r = recompute_pass_rate_clean(d)
        assert r["n_infra"] == 0  # 3 < 10 → not infra
        assert r["pass_rate_clean"] == 100.0

    # 5) gate vets the infra code (closed blacklist).
    res = _classify_infra("Prob001", 1)
    assert is_infra(res) and res["infra_code"] == "scheduler_failure"

    print("verify.verilogeval smoke: 5/5 passed")


if __name__ == "__main__":
    _self_test()
