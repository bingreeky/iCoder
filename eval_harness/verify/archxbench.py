"""verify/archxbench.py — closed-blacklist classification adapter for ArchXBench.

ArchXBench's *verifier* (the iverilog compile / golden-compare / self-check-tb
assertion-% in run_archx.py:verify_candidate) stays as-is — it is the original
working oracle and is NOT ported into verify/. What this module ports is the
*failure classification*: the decision of whether a sample's failure is an
infrastructure failure (excluded from the denominator) or a model failure
(zero reward, stays in the denominator).

Before this adapter, run_archx.py set a free boolean `r["infra"]` at two sites
(missing testbench, query exception). That was roughly right but **not unified
through verify.core** — nothing enforced the closed-blacklist guarantee that
candidate-controllable text can never authorize an infra exclusion. This module
routes the per-sample result through `core.infra_failure` /
`finalize_failure_classification`, so:

  * the infra decision comes from a trusted harness-set trigger
    (`_infra_trigger`), never from candidate text (stdout/traceback/error msg);
  * `finalize_failure_classification` is the single gate — any infra claim not on
    a trusted channel + trusted code is demoted to a model failure;
  * every sample carries `classification_policy` / `verifier_version` /
    `failure_origin` / `infra_code`, matching the other hardened benches.

The denominator math in run_archx.py (`scored = K - infra`) is unchanged — it
now reads the gate-derived `r["infra"]`. Behaviour on the two existing trusted
triggers is identical (infra stays True); the change is the *guarantee*.

References (upstream, verifiers_unzip/verifiers/):
  verify_server_v2.py:634-696  _base_result / _infra_failure / _model_failure
  verify_server_v2.py:737-790  _finalize_failure_classification (the gate)
"""
from __future__ import annotations

from typing import Any

from .core import (
    INFRA_CODES,
    base_result,
    finalize_failure_classification,
    infra_failure,
    is_infra,
)

EVAL_BACKEND = "archxbench"

#: Trusted harness triggers (set by run_archx.py, NOT by candidate text) → the
#: closed-enum infra code they map to. Adding a trigger here is the ONLY way a
#: new infra exclusion can appear; finalize_failure_classification still vets it.
_TRIGGERS: dict[str, str] = {
    # The benchmark dataset is missing its testbench for this design — a
    # reference/configuration gap discovered before the model is ever called.
    "no_testbench": "trusted_reference_configuration",
    # The LLM gateway raised (connection drop / 5xx) on every retry — the
    # model never got to answer. The exception TYPE is the trusted signal
    # (URLError/HTTPError/ConnectionError); the error string is recorded but
    # is never the basis for the infra claim.
    "query_failed": "scheduler_failure",
}


def classify_archx_sample(r: dict[str, Any]) -> dict[str, Any]:
    """Stamp closed-blacklist classification fields onto an ArchX sample result.

    Idempotent: a result already carrying `classification_policy` (fresh result
    classified this run, or a resumed result from a post-fix run) is returned
    unchanged. Resumed results from a pre-fix run have no `classification_policy`
    and no `_infra_trigger`; they keep their legacy `infra` bool verbatim
    (apply_result reads it directly) — those were set at the same two trusted
    sites, so the legacy value is already correct and need not be re-gated.

    Mutates and returns `r`. After this call `r["infra"]` is the gate-derived
    value (is_infra on the finalized core result), so run_archx.py's denominator
    math (`scored = K - infra`) and apply_result both consume the gated value.
    """
    if r.get("classification_policy"):
        return r  # already classified (idempotent)
    trigger = r.get("_infra_trigger")
    if trigger in _TRIGGERS:
        code = _TRIGGERS[trigger]
        res = infra_failure(
            infra_code=code,
            eval_backend=EVAL_BACKEND,
            classification_channel="trusted_control_plane",
            info=f"archx_infra:{trigger}",
            error_type="archx_infra",
            original_info=str(r.get("error") or trigger),
        )
    else:
        # Non-infra: a real model result. `correct` = compiled AND fully passed
        # (assertion % == 100 / golden compare PASS). Partial-t samples are model
        # failures (they stay in the denominator); ArchX's own syntax/t metrics
        # remain the source of truth — core's `correct` is only the policy record.
        syntax = bool(r.get("syntax"))
        t = float(r.get("t") or 0.0)
        correct = syntax and t >= 100.0
        res = base_result(
            EVAL_BACKEND,
            correct=correct,
            compiled=syntax,
            info="pass" if correct else "fail",
        )
    res = finalize_failure_classification(res)
    r["infra"] = is_infra(res)
    r["infra_code"] = res.get("infra_code")
    r["failure_origin"] = res.get("failure_origin")
    r["classification_policy"] = res.get("classification_policy")
    r["classification_reason"] = res.get("classification_reason")
    r["verifier_version"] = res.get("verifier_version")
    return r


def _self_test() -> None:
    # Trusted triggers → infra (gate accepts trusted channel + trusted code).
    r1 = classify_archx_sample({"design": "d", "k": 0, "syntax": 0, "t": 0.0,
                                "error": "no testbench", "_infra_trigger": "no_testbench"})
    assert r1["infra"] is True
    assert r1["failure_origin"] == "infrastructure"
    assert r1["infra_code"] in INFRA_CODES
    r2 = classify_archx_sample({"design": "d", "k": 1, "syntax": 0, "t": 0.0,
                                "error": "query: URLError...", "_infra_trigger": "query_failed"})
    assert r2["infra"] is True and r2["infra_code"] == "scheduler_failure"
    # A candidate-text "infra" claim with NO trusted trigger → demoted to model.
    r3 = classify_archx_sample({"design": "d", "k": 0, "syntax": 0, "t": 0.0,
                                "error": "compile: blah", "infra": True})  # bogus legacy infra
    assert r3["infra"] is False  # closed blacklist: no trusted trigger → model fail
    assert r3["failure_origin"] == "model"
    # A passing model result.
    r4 = classify_archx_sample({"design": "d", "k": 0, "syntax": 1, "t": 100.0, "error": ""})
    assert r4["infra"] is False and r4["failure_origin"] is None
    assert r4["classification_reason"] == "success"
    # A partial-t model result (compiled but not full pass) → model fail, in denom.
    r5 = classify_archx_sample({"design": "d", "k": 0, "syntax": 1, "t": 40.0,
                                "error": "func partial"})
    assert r5["infra"] is False and r5["failure_origin"] == "model"
    # Idempotent: re-classifying a stamped result is a no-op (values unchanged).
    r6 = classify_archx_sample(dict(r1))
    assert r6["infra"] is True
    assert r6["infra_code"] == r1["infra_code"]
    assert r6["failure_origin"] == r1["failure_origin"]
    print("verify.archxbench smoke: 7/7 passed")


if __name__ == "__main__":
    _self_test()
