# RSI control plane

Read this reference when starting or resuming a project, when a stage needs to
change, or when it is unclear what the Agent may decide without a person.

## What the Human Prior is

The manifest-listed `prior_files` portion of the installed Skill bundle is a
versioned, read-only procedural prior. It may
specify the research program, available components, invariants, evidence needed
for a decision, and the boundary of Agent authority. It must not be rewritten by
a live experiment merely because the experiment produced a useful result.

A run combines six sources of context:

1. the frozen Human Prior and its digest;
2. a human-approved project profile and capability snapshot;
3. current controller state and the active Task Queue;
4. evidence-bearing artifacts and decisions produced so far; and
5. learned memory whose append sequence is included in the context cutoff vector; and
6. separately digested calibrated-note bindings approved for this run.

The last five belong to the run, not to the static Human Prior. A learned result may
be promoted into a later Human Prior version only after human review.

The Human Prior digest is fixed by the static prior-files list in the package
manifest. Loading a reference later does not change the release or its digest.
Bootstrap also verifies the manifest `release_digest`, which binds controller
selection, prior boundaries, calibrated-note policy, and runtime policy. The two
digests together identify the procedural release; neither may be silently
substituted on resume.
Calibrated notes always remain outside the static Human Prior and its digest in
this release. A typed, provenance-bearing binding may add an eligible note only
as a separately digested run-context source.

## Bootstrap without a capability lock

Project discovery necessarily begins before a complete profile and capability
lock exist. Create an immutable bootstrap record containing the prior release ID
and digest, the human request, facts explicitly supplied so far, unresolved
identities, and the authority available for read-only discovery.

Bootstrap may create only bootstrap, audit, and human-gate tasks, plus one
bootstrap-bound transition task fixed to the internal
`PROJECT_AUDIT -> DATA_AUDIT` handoff. These tasks may use a bootstrap-context
ID and a null Stage Context Packet. The transition stays blocked until the
approved profile, capability lock, bootstrap closure evidence, and every input
needed to construct the first Stage Context are frozen; the running transition
constructs and digests that packet transactionally and has no external side
effect. Bootstrap tasks cannot
launch training, mutate external systems, or claim that a missing resource
exists.

After inventory and approval produce the project profile and capability lock,
prepare the initial `PROJECT_AUDIT -> DATA_AUDIT` transition against the
bootstrap-context digest, create the first full Stage Context Packet, evaluate
readiness, seed DATA_AUDIT, and commit the transition. This one-way bootstrap transition removes the circular
requirement that inventory already have the context it is meant to create.

## Human and Agent authority

The human fixes the project objective, base-model identity, allowed data and
external services, held-out evaluation, total resource envelope, permissions,
and the stage scaffold. The Agent may choose experiments, data operators,
mixtures, context variants, learning-signal arms, ordinary checkpoint selection,
and stage transitions inside those bounds when the required evidence exists.

Queue a human gate before:

- changing any human-fixed project-profile field, including the objective or
  completion definition, base-model identity/architecture, held-out evaluation,
  allowed data/services, resource envelope, permissions, or stage scaffold;
- acquiring a new external model, service, credential, or compute allocation;
- exceeding a stage or single-experiment budget boundary;
- deleting, overwriting, publishing, or pushing an artifact outside the run;
- changing a governing rule of the frozen Human Prior; or
- declaring a final model ready for external release.

Silence is not approval. A blocked human gate remains blocked while other
independent, authorized queue items may continue.

## Persistent state graph

The controller records one current state and an append-only history. The normal
path is:

~~~text
PROJECT_AUDIT
  -> DATA_AUDIT -> DATA_PILOT -> DATA_BUILD -> DATA_CURATE
  -> SFT_FILTER -> SFT_TARGETS -> SFT_SMOKE -> SFT_TRAIN -> SFT_EVAL
  -> OPSD_TARGET -> OPSD_CONTEXT -> OPSD_OBJECTIVE -> OPSD_TRAIN -> OPSD_EVAL
  -> RLVR_VERIFIER -> RLVR_REWARD -> RLVR_SMOKE -> RLVR_TRAIN -> RLVR_EVAL
  -> FINAL_EVAL -> MODEL_FREEZE -> COMPLETE_INTERNAL
  -> RELEASE_AUTHORIZED -> RELEASE_EXECUTING -> RELEASED
                          -> RELEASE_FAILED
  RELEASE_AUTHORIZED -> COMPLETE_INTERNAL  (expired, revoked, or changed scope)
  RELEASE_FAILED -> RELEASE_EXECUTING  (valid authorized operational retry)
                 -> COMPLETE_INTERNAL  (new or expired authority required)
~~~

`VERIFICATION_AUDIT` is a cross-stage interrupt state reachable from any active
training/finalization state and from COMPLETE_INTERNAL or a pre-release state.
Its transition records `return_stage_state`; after measurement integrity
closes, the controller returns there or explicitly routes to the stage that must
replace invalid evidence. `RLVR_VERIFIER` remains the normal RLVR integration
state and does not stand in for verifier work used by Data, SFT, OPSD, or final
evaluation.

If a post-freeze audit invalidates evidence, transition COMPLETE_INTERNAL,
RELEASE_AUTHORIZED, RELEASE_EXECUTING (after safely stopping), or RELEASE_FAILED
to `VERIFICATION_AUDIT`, invalidate the frozen manifest artifact, and re-enter
FINAL_EVAL with a successor context/manifest. A defect discovered after RELEASED
cannot rewrite history: open a release incident/known-defect record and human
gate for withdrawal, correction, or successor release.

This is a scaffold, not a one-way schedule. A transition record may return to an
earlier state when its trigger is explicit:

- missing coverage, verifier weakness, or uniformly uninformative groups return
  to the relevant Data state or `VERIFICATION_AUDIT`;
- SFT failures on unseen task types return to Data, while failures on represented
  types may open an OPSD target;
- if SFT closes without an eligible OPSD target, an optional OPSD stage may be
  bypassed only by a recorded evidence-backed decision; a required OPSD stage
  remains blocked until a scoped human waiver or scaffold change;
- an OPSD context without a measured advantage returns to context probing; a
  harmful objective returns to objective design rather than silently continuing;
- reward/validation disagreement returns RLVR to reward and verifier audit;
- invalid measurement blocks model-selection and finalization.

Every transition names the evidence, decision, input artifact, output artifact,
and next gate. The Agent never infers completion from the absence of queued work.

## The recursive research loop

Every stage uses the same outer loop:

1. **Observe.** Read the registered artifacts, raw outputs, failure composition,
   resource state, and unresolved measurements.
2. **Diagnose.** State a falsifiable explanation of the bottleneck. Separate
   capability failure, data coverage, measurement failure, and infrastructure.
3. **Propose.** Enumerate credible interventions supplied by the stage Skill and
   choose the smallest experiment that can distinguish the live explanations.
4. **Register.** Put the probe in the Task Queue with a baseline, controlled
   factor, metric, budget, expected artifacts, and decision it will enable.
5. **Execute.** Run only after dependencies, authority, measurement, and smoke
   gates are satisfied.
6. **Verify.** Check artifact integrity and whether the experiment actually
   measured its registered question before interpreting the number.
7. **Decide.** Retain, revise, or reject the intervention. Preserve negative and
   failed decisions instead of rewriting history.
8. **Learn and route.** Write evidence-bound learned memory, evaluate the stage
   transition guard, and create the next queue items.

A training run is therefore never a free-standing action. It has an upstream
question and a downstream evaluation and decision task.

## Transition guard

Before changing state, confirm all of the following:

- the current context snapshot and Skill digest are recorded;
- required experiments for the decision are evidence-complete with closed
  artifacts, carry a valid scoped human waiver and replacement evidence, or
  resolve through a unique supersession chain to one of those states;
- failed-blocking or invalidated evidence is absent from the decision basis;
- the measurement definition, denominator, verifier version, and split are fixed;
- required baselines and controls are comparable;
- checkpoint, data, code, configuration, and verifier lineage are recoverable;
- regressions and known defects are visible, not hidden in an aggregate;
- the proposed next state has its entry requirements; and
- every human gate required by this specific transition has a matching scoped
  approval record. An optional or downstream release gate does not block
  COMPLETE_INTERNAL.

If one is missing, queue the missing audit or evidence task rather than guessing.

## Completion

Training is not project completion. COMPLETE_INTERNAL requires the finalization Skill to
close the frozen evaluation, regression audit, checkpoint load test, artifact
lineage, and known-defect report. A model may be internally selected while still
not being authorized for external publication or upload. A scoped human gate
creates RELEASE_AUTHORIZED, not RELEASED. RELEASED requires a separate external
action task plus locator, content digest, and post-write verification;
before RELEASE_EXECUTING, recheck that the unique active human authorization is
unexpired, unsuperseded, and matches the exact target/action digest. Otherwise
return to COMPLETE_INTERNAL and request new authority.
RELEASE_FAILED preserves COMPLETE_INTERNAL and may be retried only under the
recorded authorization and retry policy. Before retry, transition through a new
release context that reconciles the remote target and confirms the original
scope is still valid. If authority expired or the target/action changes, return
to COMPLETE_INTERNAL and queue a new human gate and release task.
