---
name: verification
description: Cross-stage executable verification for code-model data, supervised targets, OPSD feedback, and RLVR reward. Load when integrating or auditing a verifier, mapping task properties to an official harness profile, classifying failures, or changing how a stage consumes a verdict.
---

# Verification

Verification is a shared measurement system, not an RLVR-only component. Data
uses it to admit task--oracle pairs, SFT uses it to select teacher trajectories,
OPSD uses it to build failure experience and optional outcome signals, and RLVR
turns its eligible verdict into reward.

Enter through an active controller context and queue task. If this Skill is
invoked directly without them, load `auto-post-training`, bootstrap or resume the
run, and do not execute a verifier against project assets yet.

The static Human Prior defines how these decisions must be governed; the
human-approved project profile fixes the run's allowed sources plus correctness
and integrity invariants. Within that boundary, the Agent may construct a
versioned task-contract/harness-profile artifact, compare admissible stage
adapters, and revise consumption only through registered experiments. A new
external harness, dataset, toolchain, or correctness rule is outside this
authority and requires a human gate plus a new project-profile/capability-lock
version.

Read [task contracts](references/task-contracts.md) when classifying a new task
family or designing the dispatcher. The existing RLVR notes on verification
service and heterogeneous execution are a separate calibrated-note source class,
not learned memory and not static Human Prior in this release. Load them only
through the typed binding defined in runtime records.

Load [verification service](../rlvr/notes/verification-service.md) only when the
context packet binds that service implementation, and load
[multi-backend verification](../rlvr/notes/multi-backend-verification.md) only
when the queued task actually spans those backend assumptions. Otherwise, use
the governing rules in this Skill and the registered project artifacts.

## Separate task semantics from benchmark identity

Classify the executable obligation first: artifact boundary, interface scope,
state and timing, external dependencies, expected observation, correctness
criterion, and whether a calibrated performance ordering exists.

A benchmark name selects an official harness profile, payload layout, toolchain,
and reporting convention. It does not define the conceptual verifier class. Two
benchmarks with the same task properties may reuse a contract while retaining
separate official-environment profiles; tasks in one benchmark may require more
than one contract.

## Normalized request and verdict

Normalize only the envelope, not the judging logic.

A request names the task ID and lineage, task-contract ID, harness profile,
candidate artifact, stage, execution budget, and expected integrity digests.

A verdict records:

- pass, fail, or unjudged as mutually distinct states;
- the explicit positive evidence required by the task contract;
- compile, launch, simulation, functional, and performance observations that
  actually exist for that contract;
- diagnostics safe to expose at the consuming stage;
- exploit or contract-violation indicators;
- failure attribution and retry history;
- backend, harness, toolchain, payload, and input digests; and
- enough identity to reproduce or independently audit the decision.

Absence of a positive verdict is never silently converted into pass. An event is
unjudged only when a closed, trusted infrastructure taxonomy supports that
attribution; candidate-controlled output cannot decide that classification.

## Stage consumption adapters

Keep the underlying verdict stable and make consumption explicit:

- **Data:** admit only when the reference passes the declared contract and the
  verifier has sufficient discriminative strength. Record gate and rejection
  evidence.
- **SFT:** keep a teacher trajectory only after a positive task-native verdict.
  Verification filters targets; it does not choose the teacher or capability-gap
  policy by itself.
- **OPSD:** expose only the approved diagnostic granularity, and separately map
  judged outcomes into an optional training signal. Preserve the raw verdict so
  context and objective ablations remain possible.
- **RLVR:** map eligible, judged outcomes into the registered reward contract.
  Unjudged trajectories follow the sentinel/masking contract and never become an
  ordinary negative reward.

Do not overwrite the raw verdict with a stage-specific scalar. Store the adapter
version and both representations.

## Audit before scale

For every task-contract and harness-profile pair:

1. run known-good and known-bad fixtures;
2. confirm the denominator, excluded and unjudged counts, and reference ceiling;
3. align the official environment and record intentional deviations;
4. test payload completeness and tool resolution at the point of use;
5. adversarially test exploit detectors and their obvious evasions;
6. verify the trusted boundary for failure attribution;
7. run a joint smoke with every already-integrated profile; and
8. register the verifier and fixture artifacts before a training task may depend
   on them.

Generated code is untrusted in every stage. Apply process, resource, output,
filesystem, credential, and network boundaries according to the project profile;
do not reserve containment for RLVR merely because RLVR calls the service most.

## Agent decisions

The Agent may decide, with evidence:

- which task contract matches a task family;
- whether a new official-harness profile is required;
- which diagnostics are useful and safe for OPSD;
- whether a meaningful partial or performance ordering exists;
- which detector closes a stated violation without rejecting valid behavior; and
- when unresolved rates or integrity failures require stopping downstream work.

Changing held-out evaluation, weakening an integrity rule, or replacing an
official correctness criterion requires a human gate.

## Exit condition

A verifier integration is ready for downstream use only when its request,
verdict, fixtures, environment profile, integrity digests, failure taxonomy,
stage adapter, and joint smoke are registered. Plausible numbers from an
unversioned harness do not satisfy this condition.
