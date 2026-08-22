---
name: finalization
description: Final evaluation, integrity closure, checkpoint selection, and controlled handoff of a post-trained code model. Load after RLVR produces candidate checkpoints, when comparing the full SFT--OPSD--RLVR lineage, or before declaring, publishing, or uploading a final model.
---

# Model finalization

Training completion and model completion are different states. This stage freezes
measurement, checks cross-stage regressions, proves artifact lineage and
loadability, and prepares a model for human-authorized handoff. It does not run a
new optimization merely to improve a final table.

Enter only through an active `FINAL_EVAL` or `MODEL_FREEZE` context and queue
task. If invoked directly without one, load `auto-post-training` and bootstrap or
resume the controller first. Load `verification` before consuming frozen
verifier profiles, denominator policy, unresolved samples, or evaluation
integrity evidence.

Read [the final model manifest](references/final-model-manifest.md) before
creating finalization queue items.

## Entry requirements

- shortlisted checkpoints have immutable artifact and parent IDs;
- the final evaluation split, sampling protocol, extraction, and verifier
  profiles are frozen and contamination-audited;
- required experiments in the registry are evidence-complete or validly waived
  where their predeclared policy permits by a scoped `human-gate-verdict` that
  names rationale, affected claims, and replacement evidence, or resolve through a unique supersession chain to
  one of those states; failed-blocking and invalidated experiments do not satisfy
  entry; and
- known infrastructure defects cannot silently change the denominator.

## Finalization queue

1. evaluate every shortlisted checkpoint under the same frozen matrix;
2. compare the selected checkpoint with the base and every applicable SFT/OPSD
   handoff, showing both gains and regressions rather than only an aggregate; an
   optional-stage bypass is represented by its explicit decision, not a phantom
   checkpoint;
3. audit unresolved samples, truncation, task coverage, and high-scoring outputs;
4. verify checkpoint completeness, tokenizer/config compatibility, conversion,
   load, inference, and intended resume state;
5. close code, data, configuration, verifier, reward, and checkpoint lineage;
6. record known defects, limitations, unsupported task properties, and intended
   use; and
7. produce an immutable pre-release final model manifest and queue a nonblocking
   human release gate.

If the frozen evaluation is invalid, enter `VERIFICATION_AUDIT` or return to the
responsible data state. If a candidate has unacceptable regressions, return to checkpoint
selection or the responsible training stage through an explicit decision. Do not
change the final evaluation to rescue a preferred checkpoint.

## Selection

Select against a predeclared multidimensional regression profile. The highest
aggregate is not automatically the model that moves forward. Report the number
and depth of regressions, task-family coverage, unresolved fraction, and whether
improvement reflects new capability or reliability on already-solvable tasks.

## Authority

The Agent may prepare and internally select a checkpoint inside the approved
project envelope. Publishing, uploading, or overwriting an existing remote model
requires an explicit human gate. Approval moves the controller only to
RELEASE_AUTHORIZED. A release task must then execute within that scope and verify
the external locator and content digest before the controller may claim
RELEASED.

## Exit

MODEL_FREEZE closes only when the manifest, frozen results, regression audit,
load test, known-defect record, and artifact digests are complete.
Frozen-evaluation integrity, checkpoint identity, and load/inference verification
are `nonwaivable-for-completion`; changing the deliverable instead requires a new
human-approved project-profile version and a new finalization context.
Experiments establishing those same properties use the matching nonwaivable
policy; a `superseded` experiment counts only through its completed successor.
COMPLETE_INTERNAL means the internal project deliverable exists. RELEASED is a
separate state reached only after both explicit human authorization and a
verified external release action; neither is inferred from internal completion.
