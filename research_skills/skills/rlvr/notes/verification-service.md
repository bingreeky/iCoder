# Building the verification service

> **Provenance class: calibrated reference.** Cross-stage governing rules now live
> in the verification Skill. This file is not part of the static Human Prior in
> this release; load it only through a separately digested, provenance-bearing
> calibrated-note binding in the Stage Context Packet.

Mechanisms for running execution-based reward as a service that a training loop calls every
step. The [RLVR Skill](../SKILL.md) says what the reward must do; this says how the thing that produces it is
built.

**This is the component where a bug produces plausible numbers rather than an error.** Almost everything below exists to make a failure loud.

---

## Executing untrusted code

Every request runs code you did not write. It will, with no intent to attack you, exhaust
memory, fill disks, spawn children that outlive it, run forever, and crash in ways that take
the worker with it.

- **A fresh process group per request, terminated as a group.** Killing only the process you
  spawned leaves its children holding the accelerator, and the leak is invisible until the
  pool is exhausted.
- **Resource limits applied inside the child before it executes anything** — address space,
  CPU time, output file size, descriptor count. Size them per backend; simulation and kernel
  execution have very different legitimate footprints.
- **Cap captured output.** A model looping on a token emits until something stops it, and
  unbounded capture kills the worker instead of failing the request.
- **Arm the fatal-signal handler in the child and capture its dump.** Without it, a crash that
  takes down the interpreter is indistinguishable from a hang.
- **A start-up deadline distinct from the execution deadline.** A worker that never signalled
  readiness is a different failure from one that ran too long, and conflating them spends the
  full execution timeout on a fast failure.

---

## Slot pools

Give each contended resource its own pool: a FIFO of slots, one executor bound per slot, and a
router that sends each request to the pool its backend needs. Simulation and accelerator work
must not share a queue.

**Return the slot in a `finally`.** A request that raises must not leak its slot, or the pool
degrades by one slot per exception until it deadlocks with no error at all.

**Rebuild a broken executor in place.** When a pool's executor has died, discard it and
construct a new one for the next request rather than failing everything after the first crash.
Assume nothing survives a crash — that assumption is what keeps the recovery simple.

**Validate the request at the API boundary, before dispatch.** Malformed requests should be
rejected where you can still attribute the rejection to yourself, rather than becoming a
mysterious worker failure later.

---

## Attribution and trust boundaries

The classification of a failure as infrastructure decides whether the sample is masked out of
training, which is why the [RLVR Skill](../SKILL.md) calls it an attack surface. This is the mechanism.

**Name your trusted channels explicitly.** Request validation before dispatch, and your own
wrapper around the worker, are trusted. Anything produced inside the environment the
candidate's code ran in is not — **including the structured result your own worker returned
from there**, because the candidate shared that process.

**Re-classify at the boundary.** Strip results arriving from an untrusted channel down to a
canonical set of fields and decide the classification yourself, discarding whatever
classification they carried. An untrusted component may report facts; it may not report
verdicts about whose fault something was.

**Keep the infrastructure causes as a closed, versioned enumeration.** For example, a frozen
profile might contain exactly scheduler rejection, worker-bootstrap timeout, payload-digest
mismatch, missing pinned runtime dependency, and trusted-reference-environment failure.
Anything outside that declared enum is not infrastructure-unjudged by default. Adding a new
cause creates a new profile/adapter version and triggers comparison revalidation; candidate
output can never extend the enum.

**For harnesses that run candidate code in-process, attribution needs the call stack.** Walk it
and decide whether the innermost frame is in the candidate or in your harness. Pin the harness
frames you consider known-good — an error raised at a known construction site is the
candidate's, not yours — otherwise arbitrary harness errors get charged to the model.

---

## Retry mechanics

**The deadline is the bound, not the attempt count.** Attempt count multiplied by request
timeout is the real worst case; with generous values on both, a single black-holed host or
wedged worker stalls one training step for hours while every accelerator idles. Set a
wall-clock deadline for the whole sequence and stop when it expires, whatever the attempt
count says.

**Split the budget by fault class**, and track it as the running maximum over the classes seen
so far in a sequence — so that one stray deterministic fault in the middle of a transient
outage does not truncate the retry that would have succeeded.

**Beware the shared client under reset.** If several threads reset a shared connection pool
during an outage, one thread will use a client another just closed and receive a bare
library-level error rather than a network error. Classify it as transient. Note the perverse
interaction: raising the transient budget makes this race more likely, so it appears exactly
when you are trying to survive an outage.

---

## Integrity, and failing closed

**Check verification inputs against an allowlist of hashes** loaded at start-up, and hash the
allowlist itself.

**Each backend carries a profile**: the reference's output fingerprints, the pinned tool
versions, the expected input count. A mismatch should fail the backend **at start-up rather
than at request time** — a verifier that starts and then judges everything wrong is far more
expensive than one that refuses to start.

**Hash a configuration file before and after reading it** and reject the load if it changed. A
concurrent writer otherwise gives you a half-old, half-new verifier and no error.

---

## Monitoring

**Track the unresolved fraction as a running rate over a recent window**, not as a total, with
two levels: one that logs and continues, one that halts. Print the rate periodically with both
thresholds visible, so the number is interpretable in the log without looking anything up.

**Record per sample, always, not only when something is wrong**: the backend, the correctness
verdict, every exploit-detector output, whether this was an infrastructure failure and after
how many attempts, the response length before extraction, the extracted length, and whichever
domain-specific diagnostic the backend produces. A bounded ring of recent samples costs
nothing and is the difference between diagnosing a run and rerunning it.

---

## Validation differs from training, deliberately

**Preserve unresolved validation samples as unjudged in the per-item record.** The training
sentinel exists only for masking during advantage computation and must never enter a metric.
A frozen reporting policy may count unjudged items as zero when the official criterion
requires it, but this is a derived metric: keep the raw tri-state verdict, denominator policy,
and unresolved fraction visible.

**Log validation results to their own file** with enough identity to regroup them afterwards —
the item identifier, the backend, the benchmark subdivision. Compute metrics from that file
rather than from whatever the trainer aggregated, so a metric definition can be changed without
rerunning anything.

**Where a benchmark defines its own success criterion, compute against that criterion**, not
against your training reward. They can differ, and reporting the reward as if it were the
benchmark's metric is an error nobody catches from the outside.

---

## Batch composition

When one dataset is much larger than the others, ordering the batch naively lets it occupy
every early slot. Interleave by backend so each is represented throughout. This matters for
anything computed over a window — the unresolved rate, degenerate-group fractions, throughput —
because otherwise those statistics describe whichever backend happened to land there.

---

## Checking these still hold

1. **Re-check that the advantage consumer still honours the sentinel** after any framework
   upgrade. This is the failure that turns masking into a large negative reward, and nothing
   reports it.
2. **Re-test slot return under exception** after touching the pool. A leaked slot per exception
   is invisible until the pool deadlocks.
3. **Re-verify the trust boundary** after any change to where verification runs. If a component
   moves into the candidate's environment, its output silently stops being trustworthy.
4. **Re-check the pinned harness frames** used for in-process attribution after a harness
   upgrade. They are location-based and they drift.
