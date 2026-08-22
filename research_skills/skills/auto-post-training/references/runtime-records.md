# Runtime records and context assembly

Read this reference when starting a run, resuming after context loss, registering
an artifact, or writing learned memory. These records live outside the installed
Skill and are scoped to one end-to-end RSI run.

## Recommended run layout

~~~text
post-training-runs/<run-id>/
  bootstrap.yaml
  bootstrap/current.yaml
  bootstrap/events.jsonl
  bootstrap/contexts/<bootstrap-context-id>.yaml
  run.yaml
  project-profile.yaml
  capabilities.lock.yaml
  state/current.yaml
  state/transitions.jsonl
  queue/tasks.yaml
  queue/events.jsonl
  attempts.jsonl
  contexts/<context-id>.yaml
  cutoffs/<cutoff-vector-id>.yaml
  experiments/registry.yaml
  experiments/events.jsonl
  decisions.jsonl
  artifacts.jsonl
  memory/index.jsonl
  memory/data/
  memory/sft/
  memory/opsd/
  memory/rlvr/
  memory/verification/
  memory/experiment-ops/
  memory/finalization/
  known-defects.yaml
~~~

The paths are illustrative; the project profile may choose another root. The
logical separation and append-only boundaries are required.
Additional scoped components use their canonical Skill name under `memory/` and
are indexed through `memory/index.jsonl`; no component writes an unindexed note.

## Runtime digest convention

All append-only event schemas in this run, including the queue and attempt logs
defined in `task-queue.md`, include `previous_chain_digest` and `record_digest`.
Canonicalize the event mapping as UTF-8 JSON with recursively
sorted object keys and compact separators after removing only `record_digest`;
arrays retain order. The first event uses 32 zero bytes, rendered as 64 lowercase
hex zeros, for `previous_chain_digest`. Each later event copies the preceding
event's `record_digest`. Compute SHA-256 over the canonical bytes and store the
lowercase hex result as `record_digest`. IDs, sequences, timestamps, and
idempotency keys are inside the digest.

The component tail digest is the last record digest, or the zero genesis for an
empty log. A cutoff-vector digest is SHA-256 over the same canonical JSON form of
the vector after removing only `cutoff_vector_digest`. YAML may be the displayed
or stored human-readable syntax, but implementations digest its parsed data model
through this JSON convention. Artifact content hashes remain hashes of the
artifact bytes and are not replaced by event-record digests.

Every immutable structured authority record — bootstrap context, Stage Context
Packet, project profile, capability lock, and run record — uses the same canonical
parsed-data convention without chaining, removing only its own top-level digest
field before hashing. All nested IDs/digests remain included. The package manifest
continues to use its separately declared release-digest algorithm.

All component appenders participate in the same run-global commit/snapshot lock.
A defined cross-log operation holds it through its ordered event sequence; the
cutoff creator holds it while reading every tail and writing the vector. Per-log
writer rules remain necessary for replay but do not replace this shared lock.

Bootstrap is the only pre-lock context and authority record. Pre-lock queue,
attempt, decision, and artifact events may already exist, but each must bind the
bootstrap-context ID/digest and remain within its read-only discovery authority.
Bootstrap binds the
static prior-files digest, manifest release digest, current human request, known facts, unresolved
inventory, discovery authority, and any provisional read-only discovery
capabilities explicitly supplied by the human. It may seed only
bootstrap/audit/human-gate queue items plus the one internal, bootstrap-bound
`PROJECT_AUDIT -> DATA_AUDIT` transition task defined in `task-queue.md`. Create
run.yaml and the first Stage
Context Packet only after the profile and lock close. Provisional capabilities
expire at that transition and never authorize mutation or training.
Until then, immutable `bootstrap.yaml` is the initial `PROJECT_AUDIT` context and
records its ID/digest; the unique validated bootstrap event-chain tip is the
active authority after any successor. `bootstrap/current.yaml` is only its cache.
`run.yaml`, a full Stage Context Packet, and `state/current.yaml` do not yet
exist. The append-only event logs provide durable resume history without becoming
an additional source of authority.

If a bootstrap human gate supplies a new locator or read authority, do not edit
the current bootstrap context. Create an immutable successor context containing
its parent ID/digest, the scoped human-gate verdict, exact new provisional
capability, updated unresolved inventory, and unchanged release digests; append a
`successor-context-created` event with sequence, idempotency key, context digest,
and parent digest; then materialize `bootstrap/current.yaml`. Only a unique
human-authorized successor chain is active. Old tasks remain bound to their old
context and receive successor tasks under the new one; a pre-lock Agent decision
cannot extend discovery authority by itself.

The authoritative bootstrap event shape is:

~~~yaml
bootstrap_event_id: event-id
sequence: monotonic-bootstrap-sequence
event_type: initial-context-created-or-successor-context-created
bootstrap_context_id: context-id
bootstrap_context_revision: integer
bootstrap_context_digest: sha256-value
parent_context_id: null-or-parent-id
parent_context_digest: null-or-parent-digest
human_gate_decision_id: null-for-initial-or-required-for-successor
timestamp: timestamp
idempotency_key: stable-key
~~~

Sequence one anchors immutable `bootstrap.yaml` as revision zero. Every successor
file lives under `bootstrap/contexts/`, increments its parent's revision, and
must match the scoped human event. Replay rejects a missing parent, digest
mismatch, duplicate idempotency key, or divergent children and materializes only
the unique chain tip. Every pre-lock task specification, attempt creation, Agent
decision, and artifact registration carries both that tip's ID and digest; later
events bind transitively through the immutable task/attempt/artifact identity. An
ID alone never grants authority.

## Project profile and capability lock

The human-approved project profile states:

- project objective and completion definition;
- base model, permitted teacher models, and fixed stage scaffold;
- allowed data sources, prohibited sources, held-out evaluation, and licences;
- repositories, code revisions, training and evaluation entry points;
- allowed verifier/harness sources and immutable correctness, integrity, and
  reporting invariants; any verifier profile fixed directly by the human is
  listed here, while Agent-integrated profiles are versioned runtime artifacts;
- compute, storage, wall-clock, and monetary envelopes;
- read/write/publish permissions and approval boundaries; and
- where credentials are referenced, never their secret values.

At run start, give the approved profile an immutable version and digest, then
resolve its resource identities into a capability lock. If a resource identity
changes, create a new profile/lock version and decision record; never edit
history to make an older experiment appear to have used the new resource.

Each capability entry records:

~~~yaml
capability_id: stable-id
type: repository-or-model-or-data-or-verifier-or-compute-or-storage-or-launcher
identity: exact-human-approved-identity
version_or_revision: exact-value
locator: path-or-endpoint-without-secret
access: read-write-execute-or-submit
external_side_effects: bounded-description
credential_reference: reference-only-or-null
verified_at: timestamp
verification_artifact_id: artifact-id
status: available-or-degraded-or-blocked
~~~

The lock also records project-level budget and approval boundaries. An unverified
or missing identity remains blocked; the Agent does not fill it from a similarly
named resource.

Each project-profile version is immutable. Inside its allowlist, the Agent may
assemble and validate a task-contract/harness-profile pair, then register its
configuration and digests as artifacts. Introducing an external harness, dataset,
toolchain, or correctness rule outside that envelope requires a human-gate task
and a new profile/lock version; it is never written back silently as if it had
been part of the initial Human Prior.

When a human gate activates a new profile/lock version, update the active pointer
only through a recorded controller transition. Nonterminal tasks and contexts
remain bound to their old versions and do not inherit new authority.
Proposed/blocked/ready tasks may be superseded immediately. Tasks already in
review move through `review -> blocked -> superseded` under a recorded profile-change
decision. Running/waiting
attempts are monitored or cancelled only within existing authority, then move
through review -> blocked -> superseded. Their evidence remains labelled with
the old versions. A new context packet creates successor tasks bound to the new
versions; immutable tasks are never rebound in place. Every successor that could
duplicate old external work starts blocked on the old attempt's terminal
reconciliation and review/supersession, unless a recorded integrity decision
proves the actions and side effects are disjoint.

## Current state

The current-state record is the mutable controller-state cache. Queue, experiment,
and handover caches are separate materialized views and are never authoritative.
It includes:

~~~yaml
current_stage_state: SFT_EVAL
return_stage_state: null
input_checkpoint_artifact: base-or-prior-checkpoint-id
active_context_snapshot_id: context-id
active_task_ids: []
selected_next_task_id: task-id-or-null
open_blockers: []
next_gate: checkpoint-selection
human_prior_digest: sha256-value
manifest_release_digest: sha256-value
active_project_profile_id: project-profile-version-id
active_project_profile_digest: sha256-value
capability_lock_digest: sha256-value
active_cutoff_vector_id: cutoff-vector-id
last_decision_id: decision-id
last_replayed_queue_sequence: monotonic-sequence
last_replayed_experiment_sequence: monotonic-sequence
last_committed_transition_sequence: monotonic-sequence
~~~

Every materialized-summary mutation follows its corresponding authoritative
append-only event (transition, queue, experiment, attempt, decision, artifact,
or memory index) so the history can be reconstructed.

### Transition events and crash recovery

`state/current.yaml` is a cache, never the authority. The append-only transition
log records preparation and commitment:

~~~yaml
transition_event_id: event-id
transition_id: stable-transition-id
sequence: monotonic-transition-sequence
event_type: prepared-or-committed-or-aborted
idempotency_key: stable-key
source_transition_task_id: task-id
source_transition_task_spec_version: integer
source_transition_task_materialized_revision: integer
expected_from_state: OPSD_EVAL
expected_bootstrap_context_digest: null-or-sha256-value
to_state: RLVR_VERIFIER
return_state: null-or-recorded-state
guard_artifact_ids: []
decision_id: decision-id
project_profile_digest: sha256-value
capability_lock_digest: sha256-value
new_context_id: context-id
new_context_digest: sha256-value
seed_task_events_and_record_digests: {}
timestamp: timestamp
reason_or_abort_cause: bounded-text
~~~

Under the run-global lock, first verify that the source transition task is the
selected running task at the bound spec version and materialized revision. Then
capture a cutoff vector from the pre-transition tails, preallocate context/task/event IDs, construct the context against that
vector, construct the ordered canonical `task-created` seed events against the
current queue tail, and compute all context/event digests. Then append `prepared`
with the precomputed context digest and exact seed event ID-to-record-digest map;
persist the immutable context and append those byte-equivalent seed events,
marking the tasks dormant under the transition ID; append `committed` only after their
digests match. Still under the lock, append deterministic, idempotent queue
events keyed by the transition ID that link the committed transition and its
decision to the source task, move it `running -> review`, validate the committed
transition as its result, and then move it `review -> done`; finally materialize
current state. A committed event is the source
of truth even if the cache update crashes. On resume, replay the last committed
event. An uncommitted preparation is either completed idempotently when every
guard and digest still matches, or receives an `aborted` event; its dormant tasks
never become ready. Replay of a committed transition repairs any missing source-task
link or terminal event without executing the transition again. An aborted
transition instead moves its source task to `review` with the recorded cause; a
new attempt or successor transition task requires the normal retry or scientific-change
decision. Under the same recovery lock, every aborted seed receives an
idempotent task-superseded event caused by that transition; replay repairs any
missing cleanup event before closure. This ordering prevents a half-written stage transition from
selecting both the old and new stage, and prevents the prepared event from being
included in a context digest that recursively contains that same event.

For the initial `PROJECT_AUDIT -> DATA_AUDIT` transition,
`expected_from_state` is `PROJECT_AUDIT`, current state is expected not to exist,
and `expected_bootstrap_context_digest` must match the authoritative bootstrap
record. Its source is the unique bootstrap-bound internal transition task
permitted by `task-queue.md`; the task binds the same bootstrap digest and the
newly approved profile/lock digests and has no external attempt. Its committed
event creates the first current-state cache and full Stage
Context Packet. Later transitions set the bootstrap digest to null and validate
the last committed transition instead.

## Stage Context Packet

Create an immutable packet after bootstrap whenever a stage is entered or
materially re-entered.
It contains:

- controller stage, entry reason, objective, and input checkpoint;
- Human Prior version/digest and manifest release digest;
- active project-profile version/digest and capability-lock version/digest;
- capability-lock version and resource envelope;
- fixed human constraints and Agent-selectable variables;
- input datasets, splits, lineage, verifier contracts, and metric definitions;
- relevant prior decisions and artifacts up to an explicit cutoff;
- selected learned-memory IDs and why each is applicable;
- selected calibrated-note bindings, exact digests, and approval/provenance IDs;
- known defects, open blockers, baselines, controls, and unresolved questions;
- success, stop, rollback, and human-escalation conditions; and
- required outputs and the initial queue task IDs.

The packet records references and hashes rather than copying large model or data
artifacts into context. Nothing created after its cutoff may be represented as
knowledge available at that decision point.

Preallocated context/output IDs and seed-task declarations from the same prepared
transition are a narrow exception: they are transaction-local forward references,
not evidence or prior knowledge. They are usable only after commit and only when
their exact digests match the prepared event. No observation, result, or decision
created after the cutoff receives this exemption.

### Cross-log cutoff vector

Do not compare bare sequence numbers from different logs or rely on wall-clock
ordering. Every context packet contains an immutable cutoff vector:

~~~yaml
cutoff_vector_id: stable-id
cutoff_vector_digest: sha256-value
bootstrap_sequence: integer
transition_sequence: integer
queue_sequence: integer
attempt_sequence: integer
experiment_sequence: integer
decision_sequence: integer
artifact_sequence: integer
memory_sequence: integer
component_tail_digests: {}
~~~

Each value is the inclusive tail visible from its named append-only log, and the
tail digests prevent a same-number rewrite. Decisions, artifacts, and the memory
index receive their own monotonic append sequences. Context assembly may cite
only records at or before every corresponding component cutoff. Later decisions
refer to a cutoff-vector ID/digest rather than an ambiguous timestamp or untyped
sequence.

Cutoff vectors are immutable standalone records under `cutoffs/`. To create one,
hold the run-global snapshot lock, capture every component log's current tail
sequence and chained tail digest, write the canonical vector and its digest, then
release the lock. Context entry uses one vector, while a later proposal or verdict
creates a fresh vector immediately before that decision so post-entry experiment
evidence can be cited. Referencing an absent or digest-mismatched vector blocks
the decision; a vector is never reconstructed from current tails after the fact.

### Calibrated-note binding

Bundled notes outside `prior_files` are unavailable by default. A context packet
may load one only through a typed binding:

~~~yaml
note_path: bundle-relative-path
content_sha256: exact-manifest-matching-digest
note_version: provenance-bearing-version
provenance_artifact_id: source-run-or-human-record
evidence_artifact_ids: []
applicability_scope: model-data-toolchain-task-properties
project_profile_approval_id: required-human-approval-id
bound_at_context_cutoff: cutoff-vector-id
~~~

A note marked `legacy-unbound` in the manifest is not Agent-loadable merely
because its filename or prose appears relevant. It becomes eligible only when
the human-approved project profile binds its exact digest and supplies the
missing version, provenance, evidence, and applicability scope. Otherwise, only
a dedicated provenance-audit task may inspect it, and its claims cannot justify
an experiment or action. This prevents bundled history from being mistaken for
initial Human Prior. The binding also becomes ineligible when its
`project_profile_approval_id` is expired, superseded, revoked, ambiguous, or no
longer the unique active authority. Its selecting context and dependent
readiness views then block; further use requires a successor context and a new
active human approval rather than silent carryover.

## Experiment registry

`experiments/events.jsonl` is authoritative and `registry.yaml` is a materialized
cache. The registry defines which scientific questions and controls must close
before a stage or internal project completion:

~~~yaml
experiment_id: experiment-id
experiment_spec_version: 1
materialized_revision: 0
context_snapshot_id: context-id
project_profile_digest: sha256-value
capability_lock_digest: sha256-value
experiment_group_id: comparison-group-or-null
research_question: question
hypothesis: falsifiable-statement
required_for_stage_transition: true
required_for_internal_completion: true
waiver_policy: allowed-with-scoped-human-gate-or-nonwaivable-for-completion
required_for_claim_ids: []
baseline_and_controls: []
independent_variable: one-controlled-factor
metrics_and_denominators: []
dataset_split_and_digests: []
budget_ceiling: bounded-value
dependencies: []
status: planned-or-active-or-evidence-complete-or-failed-blocking-or-invalidated-or-waived-or-superseded
task_ids: []
artifact_ids: []
closure_decision_id: null
waiver_decision_id: null
successor_experiment_id: null
~~~

Every mutation appends a typed event:

~~~yaml
experiment_event_id: event-id
sequence: monotonic-experiment-sequence
experiment_id: experiment-id
experiment_spec_version: 1
expected_materialized_revision: integer-or-null-for-create
event_type: experiment-created-or-status-transition-or-task-linked-or-artifact-linked-or-closure-linked-or-invalidation-linked-or-waiver-linked-or-superseded
from_status: status-or-null
to_status: status-or-null
payload: complete-create-record-or-type-specific-link
caused_by_decision_id: decision-id-or-null
timestamp: timestamp
idempotency_key: stable-key
~~~

Creation carries the complete immutable scientific specification, requires a new
ID, and establishes revision zero. Every later event checks and increments the
revision. Legal status edges are `planned -> active|waived|superseded`,
`active -> evidence-complete|failed-blocking|waived|superseded`, and
`failed-blocking -> active|waived|superseded` only under a recorded operational
retry, replacement, or human-waiver decision. A later integrity/authority audit
may make the exceptional `evidence-complete|waived -> invalidated` transition,
citing the audit artifact or superseded/revoked human decision;
`invalidated -> superseded|waived` only after
a replacement is named or a scoped human waiver closes the affected claim.
Superseded is terminal; evidence-complete and waived are closed unless later
invalidated, while invalidated remains blocking. One
run-scoped writer or equivalent lock controls
sequence assignment, and replay from an empty registry must reproduce its
materialized cache.

A valid negative scientific result is `evidence-complete` when measurement and
integrity close and its decision rejects the intervention. An operationally
failed experiment is `failed-blocking`, and evidence later shown invalid is
`invalidated`; neither is evidence-complete. It
blocks every required transition until a replacement experiment closes or an
explicit human waiver states scope, rationale, affected claims, and replacement
evidence. A waiver does not convert missing evidence into a positive result.
Waived is legal only when the experiment's predeclared waiver policy permits it.
A waiver's expiry, replacement-evidence invalidation, or authority
supersession/revocation propagates an invalidation event to every experiment that
consumed it.
A superseded required experiment closes no requirement by itself: its event must
name `successor_experiment_id`, and closure follows that successor chain until an
evidence-complete or validly waived experiment is reached. Cycles or divergent
successors block closure. A waived chain tip closes the predecessor requirement
only when every experiment requirement in the chain permits waiver and the
scoped human verdict names the full chain and affected claims. Supersession
cannot make a nonwaivable experiment waivable; changing that completion
definition requires a new human-approved project-profile version and context.

An experiment never combines tasks or artifacts from different context/profile/
lock bindings. A profile change leaves old evidence historically valid only under
its old binding and creates a successor experiment for continued work in the new
context. Active old attempts may close safely, but their outputs cannot be added
to the successor comparison without a newly registered compatibility experiment.

## Decision events

Decisions are append-only events rather than records edited from pending to
complete. A proposal is written before execution; a later verdict cites it.

~~~yaml
decision_id: decision-id
decision_sequence: monotonic-decision-sequence
record_type: proposal
created_at: timestamp
actor_type: agent
principal_id: stable-agent-id
authority_basis: project-profile-and-context-ids-or-bootstrap-discovery-authority
stage_state: OPSD_OBJECTIVE
context_snapshot_id: context-id-or-null
bootstrap_context_id: bootstrap-context-id-or-null
bootstrap_context_digest: sha256-or-null
comparison_group_id: comparison-group-or-null
observations_and_artifact_ids: []
hypothesis: falsifiable statement
alternatives_considered: []
selected_action: action
rationale: why the evidence supports it
expected_outcome: registered before execution
budget_committed: bounded amount
source_task_ids: []
evidence_cutoff: cutoff-vector-id
evidence_cutoff_digest: sha256-value
idempotency_key: stable-key
record_digest: sha256-value
~~~

After execution append:

~~~yaml
decision_id: decision-verdict-id
decision_sequence: monotonic-decision-sequence
record_type: verdict
created_at: timestamp
actor_type: agent
principal_id: stable-agent-id
authority_basis: project-profile-and-context-ids-or-bootstrap-discovery-authority
context_snapshot_id: context-id-or-null
bootstrap_context_id: bootstrap-context-id-or-null
bootstrap_context_digest: sha256-or-null
proposal_decision_id: decision-id
comparison_group_id: comparison-group-or-null
source_task_ids: []
source_attempt_ids: []
result_artifact_ids: []
evidence_roles_and_integrity: []
evidence_cutoff: cutoff-vector-id
evidence_cutoff_digest: sha256-value
verdict: retain-or-revise-or-reject
rationale: interpretation against the registered expectation
next_task_ids: []
idempotency_key: stable-key
record_digest: sha256-value
~~~

Before the initial transition, an Agent proposal/verdict may close only a
bootstrap-bound read-only audit. It sets Stage Context to null, records the exact
bootstrap ID/digest and discovery authority, and cannot authorize mutation,
training, or an external side effect. After the lock closes, the bootstrap
alternative is invalid and every Agent decision binds a full context/profile.

Human authority uses a distinct event; it is never represented by an Agent
verdict:

~~~yaml
decision_id: human-gate-verdict-id
decision_sequence: monotonic-decision-sequence
record_type: human-gate-verdict
created_at: timestamp
actor_type: human
principal_id: stable-project-owner-or-delegate-id
authority_basis: authorization-record-id
gate_task_id: human-gate-task-id
decision_purpose: waiver-or-profile-change-or-budget-or-release-or-other-scoped-gate
requested_scope_digest: sha256-value
verdict: approved-or-rejected-or-approved-with-conditions
approved_scope: bounded-actions-resources-and-budget
rationale: human-provided-or-faithfully-recorded-reason
conditions: []
expires_at: timestamp-or-null
affected_claim_ids: []
replacement_evidence_artifact_ids: []
evidence_cutoff: cutoff-vector-id
evidence_cutoff_digest: sha256-value
idempotency_key: stable-key
record_digest: sha256-value
~~~

The authorization record preserves the user message, signed system record, or
other auditable source that established the principal and authority. Human-gate,
waiver, profile-change, and release transitions accept only this event type with
matching scope; a scientific `retain` verdict cannot satisfy them.

All decision records receive one monotonic sequence under a run-scoped writer;
duplicate idempotency keys or decision IDs are rejected. A correction or changed
judgment appends an immutable supersession event:

~~~yaml
decision_id: supersession-event-id
decision_sequence: monotonic-decision-sequence
record_type: decision-supersession
actor_type: agent-or-human
principal_id: stable-principal-id
authority_basis: context-or-authorization-record
superseded_decision_id: old-id
replacement_decision_id: new-id
rationale: correction-reason
evidence_cutoff: cutoff-vector-id
evidence_cutoff_digest: sha256-value
idempotency_key: stable-key
record_digest: sha256-value
~~~

The active record at a cutoff follows the unique supersession chain. Conflicting
successors block use pending an authority audit; an Agent cannot supersede a
human authority event. This preserves both earlier records and separates what
the Agent expected before an experiment from what it concluded afterward.

Keep decisions that were wrong. They are part of the recursive improvement
trajectory and prevent the same experiment from being rediscovered as new.

## Artifact record

Datasets, code snapshots, configurations, verifier profiles, checkpoints,
telemetry, and evaluations receive stable artifact IDs. `artifacts.jsonl` is an
append-only event log:

~~~yaml
artifact_event_id: event-id
artifact_sequence: monotonic-artifact-sequence
artifact_id: stable-artifact-id
artifact_spec_version: 1
context_snapshot_id: context-id-or-null
bootstrap_context_id: bootstrap-context-id-or-null
bootstrap_context_digest: sha256-or-null
expected_materialized_revision: integer-or-null-for-register
event_type: artifact-registered-or-integrity-observed-or-invalidation-linked-or-artifact-superseded
integrity_from: null-or-unverified-or-valid-or-invalid-or-invalidated
integrity_to: unverified-or-valid-or-invalid-or-invalidated
successor_artifact_id: stable-artifact-id-or-null
payload: full-registration-or-observation-or-successor-link
caused_by_decision_id: decision-id-or-null
timestamp: timestamp
idempotency_key: stable-key
~~~

Registration requires a new ID, carries the full immutable identity, and
establishes revision zero. Later events check and increment the revision.
Registration uses exactly one authority binding: a full Stage Context post-lock,
or bootstrap-context ID plus digest for a permitted pre-lock audit artifact.
Later events inherit the immutable binding.
Integrity may move `unverified -> valid|invalid` and `valid -> invalidated` when
a later audit cites contrary evidence. Invalid or invalidated bytes never become
valid in place; repaired or rebuilt bytes receive a successor artifact ID.
An `artifact-superseded` event names exactly one `successor_artifact_id`; the
replacement must have its own registration, content digest, integrity history,
and authority binding. The link does not alter the predecessor's historical
integrity state, so its `integrity_from` and `integrity_to` both carry that same
state. Registration and integrity events set the successor field to null.
Successor chains are unique and acyclic; conflicting children, cycles, missing
registrations, or digest mismatches block use. Dependents never follow a
successor implicitly: task readiness and experiment compatibility are evaluated
again against the named replacement. An invalidated predecessor remains
invalidated after supersession. Replay rejects conflicting event IDs,
idempotency keys, revisions, or successor links.

The registration payload records at least:

- artifact type, locator, content hash, creation time, producer task, and producer
  attempt where execution occurred, plus its monotonic artifact sequence;
- parent artifacts and full stage lineage;
- code, configuration, data, model/tokenizer, and verifier digests as relevant;
- integrity status and the command or procedure that rechecks it; and
- whether it is internal, reportable, publishable, or still approval-gated.

Checkpoint records additionally state optimizer/scheduler state, resume
eligibility, conversion history, and evaluation IDs. Dataset records preserve
ancestor identity so every derivative inherits its train/evaluation separation.
Artifact invalidation invalidates every cached readiness or completion view that
used it, blocks dependent task readiness and run closure, and moves any dependent
evidence-complete experiment to `invalidated` through its own event. Recovery
creates or selects a named successor artifact/task/context as required; the old
done status alone cannot make invalid evidence satisfy a dependency. Every active
learned-memory item that cites the artifact also receives a memory invalidation
event; a context that selected it is superseded before further execution. A
calibrated-note binding whose provenance or evidence artifact is invalidated
becomes ineligible immediately: its selecting context and dependent readiness
views are superseded, and further use requires a successor context with a renewed
human-approved binding to valid provenance/evidence.

## Learned memory

A live run never edits the installed Human Prior. It writes a scoped memory item:

~~~yaml
memory_id: memory-id
memory_sequence: monotonic-memory-sequence
memory_spec_version: 1
materialized_revision: 0
stage: rlvr
applicability_scope: task-properties, toolchain, and model family
claim: what was learned
mechanism: why it is believed to occur
supporting_artifact_ids: []
supporting_decision_ids: []
limitations: where it may fail to transfer
revalidation: how to check it still holds
created_at: timestamp
status: active-or-invalidated-or-superseded
~~~

`memory/index.jsonl` is authoritative and each event has:

~~~yaml
memory_event_id: event-id
memory_sequence: monotonic-memory-sequence
memory_id: memory-id
memory_spec_version: 1
expected_materialized_revision: integer-or-null-for-create
event_type: memory-created-or-status-transition-or-superseded
from_status: null-or-active-or-invalidated
to_status: active-or-invalidated-or-superseded
payload: complete-item-or-invalidation-or-successor-link
caused_by_decision_id: decision-id-or-null
caused_by_artifact_ids: []
timestamp: timestamp
idempotency_key: stable-key
~~~

`memory-created` carries the complete
immutable item and establishes revision zero. Later `status-transition` or
`superseded` events include the expected revision, causing decision/artifact IDs,
idempotency key, and successor memory ID when applicable. Legal status edges are
`active -> invalidated|superseded` and `invalidated -> superseded`. A changed
claim or scope creates a successor item rather than editing memory in place.
Replay rejects divergent successors and materializes only the unique active
record at the cutoff vector.

If memory selected by the active context is later invalidated, preserve the
historical context/cutoff but block dependent task readiness immediately. Create
a successor context and successor tasks that exclude or replace the memory
before continuing; invalidated memory cannot remain an operative premise merely
because the earlier context once selected it.

The next context packet may retrieve only memories that match its scope and
precede its cutoff. Promotion into a future Skill version is a human-reviewed
release action, not an automatic consequence of one successful run.

## Resume order

After context loss, read in this order:

If no initial transition has committed, authenticate the bootstrap record and
successor-context event chain, reconstruct its unique active context/digest, then
replay each pre-lock decision/artifact/queue/attempt log in its own chain order,
validate cross-log references and bootstrap bindings under the snapshot rules,
materialize them by bound bootstrap context, repair every task--attempt
join, and reconcile permitted read-only external discovery. Select new work only
from bootstrap, audit, or human-gate tasks whose
provisional authority is still valid. Continue PROJECT_AUDIT, or prepare the
initial DATA_AUDIT transition once profile and lock close. Do not assume a full
Stage Context, active profile pointer, or post-lock authority; pre-lock
experiment or learned-memory events are invalid and block bootstrap closure.
Before bootstrap closure, every older-context task must be terminal/superseded
and every old attempt safely reconciled; the active tip controls authority, not
whether historical events are replayed.

After the initial transition has committed:

1. verify the manifest release and static-prior digests from bootstrap;
2. replay decision, artifact, memory, experiment, queue, and attempt events and
   validate every cross-reference and cutoff digest;
3. replay committed transition events against those exact event/digest guards to
   reconstruct controller state and reject or finish any prepared transition;
4. reconcile every task/current-attempt pair, including terminal attempts missing
   their derived review/result queue event, by external handle or submission
   idempotency token before selecting work;
5. read the reconstructed active Stage Context Packet, active profile/lock,
   known defects, and decisions/results since its cutoff;
6. retrieve only the learned-memory IDs and calibrated-note bindings already
   selected by the active immutable context and included in its cutoff; newer
   applicable knowledge requires a successor context; and
7. load the specific stage Skill needed for the next deterministically selected
   ready task.

Before acting, verify the active state plus at least one artifact/resource claim
against their sources. If either disagrees, queue a state-reconciliation audit
rather than continuing from stale materialized files.
