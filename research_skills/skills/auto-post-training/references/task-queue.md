# Agent Task Queue

Read this reference when creating, prioritising, executing, reviewing, or
resuming work. The queue is the executable form of the RSI program: it records
what the Agent intends to learn or produce, why the work is allowed, and what
evidence closes it.

## Queue record

Each task is immutable in identity. Status and result fields are appended or
updated through a recorded event; do not silently repurpose a task.

~~~yaml
task_id: OPSD-CONTEXT-003
parent_task_id: OPSD-TARGET-001
task_spec_version: 1
materialized_revision: 1
stage_state: OPSD_CONTEXT
kind: probe
status: ready
required_for_internal_completion: true
waiver_policy: allowed-with-scoped-human-gate
priority_class: P3_DISCRIMINATING_PROBE
ready_sequence: 27
tie_break_key: OPSD-CONTEXT-003
research_question: Does task-matched privileged evidence change token preference?
hypothesis: Matched context outperforms a shape-matched shuffled control.
skill_refs:
  - opsd
context_snapshot_id: ctx-opsd-002
bootstrap_context_id: null
bootstrap_context_digest: null
dependencies:
  - task_id: OPSD-TARGET-001
    acceptable_statuses: [done]
    required_artifact_ids: [recoverable-task-pool-v1]
    required_decision_verdicts: [retain]
    bound_digests: {}
inputs:
  - artifact: recoverable-task-pool-v1
controlled_factor: context identity
baseline_or_control: within-domain shuffled context
experiment_group_id: opsd-context-ablation-001
metrics:
  - matched-minus-shuffled response log-likelihood
acceptance_or_stop_rule: decided before execution
budget:
  resource_limit: project-profile value
  wall_clock_deadline: project-profile value
risk_class: reversible-experiment
approval_required: false
authority_basis: project-profile-id plus capability-lock-id
retry_policy:
  operational_failure_classes: [registered-transient-class]
  max_attempts: bounded-value
  wall_clock_ceiling: bounded-value
  authority_decision_id: pre-execution-decision-id
current_attempt_id: null
action_spec:
  capability_id: scorer-capability-id
  entrypoint: registered-entrypoint
  config_artifact_id: config-artifact-id
  working_directory: registered-workdir
  declared_outputs: [scored-response-table, context-integrity-audit]
  external_side_effects: bounded-description
expected_artifacts:
  - scored-response-table
  - context-integrity-audit
decision_enabled: retain, revise, or reject this context component
result_artifact_ids: []
decision_id: null
waiver_decision_id: null
successor_task_id: null
next_task_ids: []
~~~

Project-specific values belong in the run record. The Skill supplies fields and
decision criteria, not fabricated thresholds or resource identities.

## Task kinds

| Kind | Purpose | Required closure |
|---|---|---|
| bootstrap | Establish the run, unresolved inventory, and first human/discovery tasks before a capability lock exists | approved project profile, capability lock, and first full context packet |
| audit | Establish whether data, measurement, resources, or lineage are trustworthy | audit artifact and explicit pass/block decision |
| probe | Cheapest experiment that separates competing explanations | controlled result and interpretation decision |
| build | Produce a dataset, verifier, service, or configuration | integrity checks and registered artifact |
| smoke | Exercise the real path at reduced scale | end-to-end completion and expected observables |
| train | Run an approved optimization configuration | checkpoint and telemetry manifests |
| evaluate | Measure a dataset, checkpoint, verifier, or reward | per-item records, denominator, unresolved count |
| decide | Compare evidence and select retain/revise/reject | append-only decision record |
| transition | Change controller state and create a new context snapshot | transition record and next queue seed |
| finalize | Freeze, regression-test, package, or hand off the model | final model manifest |
| release | Execute an explicitly authorized external publish/upload | release attestation with external locator, digest, and verification |
| human-gate | Request authority the Agent does not possess | explicit approval or rejection |

## Required fields by task kind

Every task has identity, version, stage state, kind, status, priority, context or
bootstrap context, dependencies, authority, retry and waiver policies, budget/risk, expected artifacts,
acceptance rule, and decision enabled. Additional required fields are:

| Kind | Additional required fields |
|---|---|
| bootstrap | known facts, unresolved inventory, permitted discovery, profile/lock outputs |
| audit | audit target, source of truth, checks, pass/block rule |
| probe | research question, hypothesis, controlled factor, baseline/control, metrics |
| build | input artifacts, build contract, output schema, integrity checks |
| smoke | production-path digest, reduced scale, expected observables |
| train | input checkpoint/data/config, algorithm, launcher, save/eval cadence |
| evaluate | checkpoint or system under test, frozen split/protocol, denominator policy |
| decide | proposal ID, evidence roles, alternatives, verdict set |
| transition | from/to state, guard checklist, required decision and artifacts |
| finalize | frozen evaluation and manifest requirements |
| release | human authorization ID/scope, target capability, immutable model/manifest digests, post-write verification |
| human-gate | requested authority, reason, scope, cost/risk, allowed responses |

Any post-bootstrap executable task also requires an action specification bound
to a capability ID, registered entry point, configuration artifact, working
directory, declared outputs, and external side effects. Runtime attempt records
store a redacted command or request with credential references only. Secret
resolution occurs inside the approved execution boundary and no secret value is
persisted in queue, attempt, log, or artifact metadata.

Before the capability lock exists, a bootstrap-bound audit may reference only a
provisional, read-only discovery capability declared verbatim in the bootstrap
record. Its exact action specification and digest live in that record instead of
depending on a not-yet-created runtime artifact. It cannot submit training,
mutate an external system, or be reused after the lock closes. All
post-bootstrap actions bind locked capability IDs.

Context-snapshot ID may be null only for bootstrap, audit, and human-gate tasks,
plus the single internal transition task fixed to
`PROJECT_AUDIT -> DATA_AUDIT`, when they bind a non-null immutable
bootstrap-context ID and digest. That transition remains blocked until the
approved profile/lock, bootstrap closure evidence, and all inputs to the first
Stage Context Packet are frozen; its running transaction constructs and digests
that packet, and it cannot perform an external action. Post-lock tasks
set both bootstrap fields null and bind an immutable Stage Context Packet;
pre-lock tasks do the inverse. Exactly one authority form is valid.

## Statuses

- **proposed** — useful work identified but not yet dependency-complete;
- **blocked** — a named missing input, authority, or external condition prevents it;
- **ready** — dependencies and authority are satisfied;
- **running** — the action has begun and its run identifier is recorded;
- **waiting** — external work is active; polling and monitoring remain in scope;
- **review** — artifacts exist but integrity or interpretation is not closed;
- **done** — acceptance criteria and downstream record are complete;
- **failed** — the task contract cannot be satisfied or no valid authorized
  retry will be made after attempt review;
- **rejected** — evidence showed the proposed intervention should not proceed;
- **waived** — a required task will not run under an explicit human waiver
  decision naming scope, rationale, affected claims, and replacement evidence,
  and only when the task's waiver policy permits it;
- **superseded** — a later task replaced it without erasing its history.

An unchanged external queue or running job is not a failure. Keep the task in
waiting and monitor it according to experiment-ops.

Legal task-status edges are:

- proposed -> blocked, ready, rejected, waived, or superseded;
- blocked -> proposed, ready, rejected, waived, or superseded;
- ready -> running, blocked, rejected, waived, or superseded;
- running -> waiting or review;
- waiting -> running or review; and
- review -> done, failed, rejected, blocked, waived, or ready only for an
  authorized operational retry.

Done, failed, rejected, waived, and superseded are terminal. They never return to
an executable status. A later correction or scientific retry creates a successor
task and may append a supersession link, preserving the terminal record.

## Queue events and replay

`queue/events.jsonl` is the source of truth; `queue/tasks.yaml` is only a
materialized view. Replay from an empty queue must reproduce every task
scientific specification and materialized revision,
status, dependency resolution, attempt/result/decision link, waiver,
supersession, and selected-next pointer.

Every mutation appends one typed event before the materialized view changes:

~~~yaml
event_id: queue-event-id
sequence: monotonic-run-sequence
task_id: task-id
task_spec_version: 1
expected_materialized_revision: 4
event_type: task-created-or-status-transition-or-readiness-evaluated-or-attempt-linked-or-result-linked-or-decision-linked-or-waiver-linked-or-task-superseded-or-next-selected
from_status: ready
to_status: running
attempt_id: attempt-id-or-null
caused_by_decision_id: decision-id-or-null
artifact_ids: []
linked_transition_id: transition-id-or-null
successor_task_id: task-id-or-null
payload: type-specific-complete-record-or-change
activation_transition_id: transition-id-or-null
timestamp: timestamp
idempotency_key: stable-key-for-the-intended-mutation
previous_chain_digest: sha256-value
record_digest: sha256-value
~~~

`task-created` carries the complete immutable scientific specification and
establishes materialized revision zero. Every accepted later event must match the
current revision and increments it by exactly one. A specification or scientific
configuration change creates a successor task rather than mutating the old one.
`readiness-evaluated` carries dependency task spec versions, observed revisions,
artifact and decision digests, and the resulting ready/block reason. When that
event changes the task into `ready`, its own queue sequence becomes the task's
`ready_sequence` and is persisted in the payload. That value is immutable while
the task remains ready; a later `review -> ready` retry receives the sequence of
that new readiness event. Link events carry the exact IDs they add.
`next-selected` carries the candidate ready-set digest, priority result, and
selected task. The candidate digest is computed with the runtime digest
convention over a canonical array containing each candidate's task ID, spec
version, materialized revision, status, priority class, ready sequence,
tie-break key, and dependency/authority digest, sorted by the scheduler's full
selection key. A task seeded by a prepared controller transition stays dormant
until the matching transition commits.

For `task-superseded`, `successor_task_id` names exactly one immutable replacement.
That field is null only when a dormant seed is cleaned up after its authoritative
activation transition aborts, in which case the payload names the aborted
transition and `supersession_reason: aborted-activation`. Every other successor
link is unique and acyclic; divergent successors, cycles, or a missing chain tip
block readiness and closure. Supersession preserves the old task and its result;
it never mutates either scientific specification in place.

A dormant seed's `required_for_internal_completion` flag becomes effective only
when its activation transition commits. If that transition aborts, replay must
append the transition-caused supersession event; closure may exclude the seed
only after both the authoritative abort and that cleanup event exist.

For `task-created`, `expected_materialized_revision` and `from_status` are null;
its payload supplies the initial proposed/blocked status, and replay rejects an
already existing task ID. All later events supply the exact current revision;
events that change status also supply the exact current status.

Replay events by sequence, reject duplicate idempotency keys, and fail visibly if
the expected materialized revision or from-status does not match. Sequence assignment and
materialization need one run-scoped writer or an equivalent atomic lock. Never
repair a conflict by guessing which write came first. A field that has no
canonical event payload cannot be changed.

## Attempts

`attempts.jsonl` is also event-sourced. Each real execution has an immutable
attempt ID even when several attempts belong to one scientific task:

~~~yaml
attempt_event_id: event-id
sequence: monotonic-attempt-sequence
attempt_id: attempt-id
task_id: task-id
task_spec_version: 1
context_snapshot_id: context-id-or-null
bootstrap_context_id: bootstrap-context-id-or-null
bootstrap_context_digest: sha256-or-null
expected_attempt_revision: 2
event_type: attempt-created-or-submitted-or-observed-or-checkpoint-linked-or-terminal
from_state: null-or-created-or-submitted-or-running-or-waiting-or-unknown
to_state: created-or-submitted-or-running-or-waiting-or-succeeded-or-failed-or-cancelled-or-unknown
configuration_digest: sha256-value
ordinal: bounded-integer
submission_idempotency_token: stable-token
external_scheduler_or_job_handle: handle-or-null
resume_checkpoint_or_cursor: artifact-id-or-null
failure_class: registered-class-or-null
retry_authority_decision_id: pre-execution-decision-id-or-null
artifact_ids: []
timestamp: timestamp
idempotency_key: stable-key
previous_chain_digest: sha256-value
record_digest: sha256-value
~~~

`attempt-created` carries the immutable redacted invocation, configuration
digest, retry-policy binding, and establishes revision zero. Launch order is
fixed: create the attempt, link it as current, move the task `ready -> running`,
then submit externally with the stable idempotency token, and finally append the
submitted event/handle. If a crash occurs before the external handle is recorded,
reconcile by that token rather than submitting again. Every accepted event checks
and increments the attempt revision. One run-scoped writer or equivalent lock
governs sequence assignment.

For `attempt-created`, `expected_attempt_revision` and `from_state` are null and
`to_state` is `created`; replay rejects an existing attempt ID. Every later event
supplies the exact current revision.

Attempt creation uses exactly one authority binding: a full Stage Context ID
post-lock, or bootstrap-context ID plus digest pre-lock. Later events inherit that
immutable binding and may not switch it.

Legal attempt-state edges are `created -> submitted|unknown|failed|cancelled`,
`submitted -> running|waiting|unknown|succeeded|failed|cancelled`,
`running -> waiting|unknown|succeeded|failed|cancelled`,
`waiting -> running|unknown|succeeded|failed|cancelled`, and
`unknown -> running|waiting|succeeded|failed|cancelled` after reconciliation.
Observed/checkpoint-link events may preserve a nonterminal state. Succeeded,
failed, and cancelled are terminal; an operational retry receives a new attempt
ID and never reopens a terminal attempt.

Replay the attempt log to recover its last state, external handle, lifecycle
times, resume checkpoint or cursor, failure class, retry authority, and
attempt-scoped artifacts. A scientific configuration change creates a new task;
only operationally equivalent retries remain attempts of the same task.

The queue never launches a new attempt while an earlier one may still be active.
Resolve an unknown external handle or idempotency token first.

Because queue and attempt logs cannot commit atomically, replay validates every
task/current-attempt pair before selection. A terminal attempt whose task is
still running/waiting causes one deterministic, idempotent queue event keyed by
`attempt-terminal:<attempt-id>` that links its outcome/artifacts and moves the
task to review. An unlinked `created` attempt with no external execution is an
attempt-log-first crash: append a deterministic cancellation/no-side-effect
reconciliation event keyed by its attempt ID and leave the task ready. If an
unlinked attempt has any external handle, token match, or evidence that execution
may have started, block selection for an integrity audit instead of cancelling
or relaunching it. A ready task with a linked `created` attempt is a pre-submit crash:
cancel that attempt, append a no-side-effect result link and authorized readiness
event, then retain ready. A running task with a `created` attempt is
reconciled by submission token: append submitted/observed if execution exists, or
cancel the attempt and move the task through review before an authorized retry.
A ready task with an active/submitted attempt is invalid. A ready task may retain
a terminal current attempt only when its terminal result link and explicit
authorized review -> ready (or pre-submit no-side-effect) event are present;
otherwise it is blocked. A review task with a nonterminal attempt, an unlinked
external execution, or a task pointing to a missing attempt is also blocked for
integrity audit. These repairs never launch work and may not change the
scientific specification.

## Priority order

Choose among ready tasks by the following persisted priority class:

1. **P0_INTEGRITY_SAFETY** — blockers that make evidence or execution invalid;
2. **P1_RECOVERY_MONITORING** — recover or safely monitor active work;
3. **P2_CLOSE_ACTIVE_EVIDENCE** — evaluate and decide completed attempts;
4. **P3_DISCRIMINATING_PROBE** — cheapest probes that unlock a decision;
5. **P4_SMOKE** — end-to-end smoke for an approved production path;
6. **P5_PRODUCTION** — production data generation or training; and
7. **P6_OPTIONAL** — work that blocks neither a claim nor a transition.

Within a class, choose the lowest ready-sequence and then tie-break key. Persist
the selected-next-task event and current pointer. A project-profile override must
name its decision or human approval and resolves to another recorded class or
tie-break order; it is never an invisible scheduling preference.

Do not launch a larger run merely because its configuration is available. A
production task becomes ready only after its measurement, data, resource, smoke,
and budget gates close.

## Dependency contracts

A dependency edge specifies acceptable terminal statuses, required artifact and
decision roles, and the exact digests against which readiness was evaluated.
Done satisfies an edge only when its required outputs exist and remain
integrity-valid at the readiness cutoff. Artifact invalidation invalidates the
recorded readiness result and blocks the edge until a named successor artifact
and, where the producing scientific contract changed, a successor task are
evaluated in a new context. Failed never
satisfies an edge unless a downstream diagnostic task explicitly accepts that
failure artifact. Rejected may satisfy a decision or transition edge only when
listed, while it cannot implicitly unblock production. Waived satisfies an edge
only when both `waived` and the exact waiver-decision ID are declared in that
edge; a waiver never fabricates an artifact or positive result.

Dependencies form an acyclic graph. Superseding an upstream task either binds the
dependent to a named successor version or blocks it for reevaluation. Record a
readiness event containing the dependency versions and digests; an upstream
artifact change invalidates that event.

## Queue composition across the Code Model program

### Project audit lane

- create the bootstrap record and freeze the Human Prior release digest;
- inventory unresolved identities through audit and human-gate tasks;
- approve the project profile and resolve the capability lock;
- inventory repositories, models, data, official harnesses, compute, storage,
  permissions, and active work;
- audit held-out evaluation and verifier validity;
- create the first context snapshot and controller state.

### Shared verification lane

- classify each task by executable properties before selecting a harness profile;
- bind source-specific official files, commands, toolchain, positive-verdict rule,
  integrity digests, and failure taxonomy;
- challenge the contract with known-good, known-bad, parser, isolation, and
  attribution fixtures;
- register separate Data, SFT, OPSD, RLVR, and final-evaluation consumption
  adapters without changing the raw tri-state verdict; and
- enter `VERIFICATION_AUDIT` whenever a profile, denominator, or attribution rule
  becomes suspect, then invalidate dependent evidence explicitly.

### Data lane

- inventory seeds and establish ancestor-level train/evaluation separation;
- register task properties, references, harnesses, licences, and lineage;
- pilot candidate evolution methods on a stratified sample;
- compare yield, novelty, executable validity, difficulty movement, and cost;
- build with the selected operators and retain rejection reasons;
- deduplicate, compose, decontaminate, and freeze the executable task pool.

### SFT lane

- measure the base model by repeated verified sampling;
- identify a capability-gap subset and build verified teacher targets;
- validate target format, reasoning span, loss mask, and length policy;
- run one-node or reduced-scale smoke, then the approved training task;
- evaluate checkpoints with failure composition and select the handoff.

### OPSD lane

- name a weakness using SFT evidence and mine recoverable failures;
- audit candidate privileged-context components and leakage controls;
- run matched controls before training on a context signal;
- compare update-direction and token-credit arms one factor at a time;
- monitor weight movement, policy drift, collapse, and the named weakness;
- select, reject, or revise the OPSD intervention and checkpoint.

### RLVR lane

- validate task-property verifier contracts and official environment profiles;
- adversarially audit reward eligibility, exploit gates, and failure attribution;
- select tasks with eligible within-group adapted-reward variance while preserving
  raw tri-state verdicts;
- smoke the full rollout-verification-update path;
- train while monitoring reward, validation, unresolved samples, degenerate
  groups, truncation, and high-reward outputs;
- select a checkpoint or return to verifier, reward, or data work.

### Finalization lane

- run the frozen evaluation matrix on every shortlisted checkpoint;
- perform cross-stage regression, integrity, load, and inference tests;
- close dataset, code, configuration, verifier, and checkpoint lineage;
- record known defects and limits;
- produce the final model manifest and queue the human release gate.

## Comparison groups

Related ablations share an experiment-group identifier. The queue freezes the
common checkpoint, dataset, sampling settings, verifier, metric, and budget, and
names the single controlled factor. A decision task may compare only registered,
comparable arms. If an upstream artifact changes, dependent arms become blocked
until they are rerun or the comparison is explicitly abandoned.

## Failure and retry

An attempt failure appends its immutable attempt outcome and moves the task to
review; it does not by itself declare the scientific task failed. First queue a
diagnosis or classify the failure using an existing closed taxonomy. When an
operationally equivalent retry is permitted by the task's pre-execution retry
policy and authority-decision ID, record the retry event and move review to ready
before creating a new attempt ID. A scientific change
creates a new task and decision trail. Mark the task failed only when no valid
retry will be made or the registered task contract cannot be satisfied.

No failed attempt automatically reproduces itself. Queue status events reference
attempt IDs, while lifecycle detail remains append-only in `attempts.jsonl`; the
materialized task view points only to the current attempt.

## Queue closure

Every build, train, or probe task must be followed by integrity/evaluation and a
decision. Every stage must end in a transition task. COMPLETE_INTERNAL is valid
only when every required queue item is done/rejected with evidence whose required
artifacts remain integrity-valid at the closure cutoff, replaced by a
unique acyclic `successor_task_id` chain whose tip is done/rejected with valid
evidence or validly waived, or waived directly, and every required experiment is
evidence-complete, validly waived, or superseded by a unique successor chain that
ends in one of those states. A failed task or `failed-blocking`
or `invalidated` experiment remains unresolved even though its attempt is terminal. Waiver
requires a scoped `human-gate-verdict` naming rationale, affected claims, and
replacement evidence. External release is nonblocking for internal completion
and uses a separate release state.

A task marked `nonwaivable-for-completion` cannot be bypassed while retaining the
same completion definition or claim. A human may approve a new project-profile
version that removes the affected claim or changes the deliverable, but that
creates a new context and closure evaluation; it is not a waiver that makes
missing evidence true.

A waived successor closes the predecessor chain only when the successor and
every completion requirement it replaces permit waiver and one scoped
human-gate verdict names the full chain, affected claims, and replacement
evidence. Supersession cannot be used to turn a nonwaivable requirement into a
waivable one.

If a task's human waiver later expires, is superseded/revoked, or has invalidated
replacement evidence, the terminal waived record stops satisfying closure. Create a
successor task in the active context and append its supersession link; do not
reopen or silently preserve the invalid waiver.
