# Final model manifest

The manifest identifies what was produced and how it can be checked. It points to
large artifacts rather than copying them.

Record:

- project, run, model, and checkpoint artifact IDs;
- Human Prior release, project-profile, capability-lock, active final-context,
  controller-transition history, cross-log cutoff vector, Task Queue, experiment
  registry, and waiver-decision digests;
- complete stage-scaffold lineage from base through the applicable SFT, OPSD, and
  RLVR nodes; an optional-stage bypass is a typed not-applicable node carrying
  its decision ID rather than a missing parent;
- model architecture, tokenizer, configuration, and content digests;
- optimizer/scheduler state and whether continued training is supported;
- source-code and training-framework revisions;
- Data, SFT-target, OPSD-experience, and RLVR-prompt manifest digests as
  applicable; every not-applicable field carries the matching bypass decision;
- verifier contracts, official-harness profiles, reward adapters, and versions;
- frozen evaluation definition and result artifact IDs;
- regression profile against every stage handoff;
- checkpoint conversion, load, and inference smoke results;
- unresolved measurements, known defects, limits, and intended use;
- internal-selection decision, `release_status: not-released`, and any
  authorization already available at freeze time; and
- exact procedures that recheck integrity and loadability.

A missing release approval is recorded as not released, not as pending success.
The frozen manifest is never edited in place. A later publish/upload produces an
append-only release attestation containing the manifest digest, model artifact
digest, scoped human authorization, release task/attempt IDs, external locator,
observed remote digest, verification time, and released/failed outcome. If a
consumer needs those fields inside a manifest, create a successor manifest that
references both the frozen predecessor and attestation. Changing any
identity-bearing or release-status field creates a new manifest version.
