# Keeping long runs comparable

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding whose environment also appears in
> the capability lock and Stage Context Packet.

Mechanisms for making a sequence of multi-day runs mean something when taken together. The
general operational method is in the [Experiment Operations Skill](../SKILL.md).

**The expensive failure here is not a crashed run, it is a finished run whose numbers cannot
be attributed to anything.**

---

## Write the manifest before the model initialises

Record, before anything loads: the revision of the code, hashes of the configuration and of
the data files, the verifier's endpoint and version, and every setting that could move a
number.

Writing it first matters. A manifest produced at the end describes the state you believe you
were in; a manifest produced before initialisation describes the state you were actually in,
including the parts that a later crash would have obscured.

---

## The experiment identifier is immutable

**Any change to code, configuration, data, verifier or reward is a new experiment**, not a
continuation of the old one.

The rule that follows and is easy to violate under pressure: **never edit a manifest so that an
old run resembles a new configuration.** The manifest is the only record of what actually
produced those numbers. Editing it does not make the old run comparable; it makes it
uninterpretable, and it does so silently — the file still parses, the numbers still plot.

When you are tempted, the thing you actually want is a new identifier and a note saying which
run this supersedes.

---

## Resume within the experiment, and append

Resume only from a checkpoint under the same identifier. After loading, confirm the immutable
fields still match before continuing — a resume that silently picks up different code is
indistinguishable in the logs from a run that behaved differently.

**Append a resume event; do not replace the original manifest.** The history of restarts is
part of what happened.

**Do not merge telemetry files across restarts.** They contain overlapping step ranges and
merging them produces a curve that reads as a single run with implausible discontinuities. Keep
them separate and align at read time.

---

## A durable checkpoint is not a risk-free resume

A checkpoint being fully written means the state survived. It does not mean resuming from it
will work.

If the step that failed involved anything beyond the optimiser — synchronising weights to a
separate inference engine, remapping a cache, reallocating a pool — resuming at that step
re-runs the same code path that failed, before any training or validation happens. The failure
reproduces immediately and looks like a new problem.

**When a run dies at a step boundary, establish which phase of the step it died in before
resuming there.** If it died in the part that resume re-runs first, resume from an earlier
checkpoint or fix the cause; do not retry the same boundary and expect a different outcome.

Validate a checkpoint before relying on it: that it is complete, that it loads, and — after any
format conversion — that the converted artifact loads too. All three fail in different ways and
the third is usually discovered by whoever tries to evaluate it.

---

## Selecting a checkpoint across mixed sources

**Do not use best-of-k metrics on data drawn from several sources with different sample
counts.** The comparison is not valid: the source contributing the most samples dominates the
statistic, and a checkpoint can win on a number that mostly describes one component of the
mixture.

Report the composition alongside the number, and state explicitly when sampling was partial or
uneven rather than presenting a single figure.

**Where a benchmark defines its own success criterion, use that criterion**, not the reward you
trained against. They can differ — a benchmark may count an outcome as passing that your reward
scores below full marks, or vice versa — and reporting one as the other is an error nobody
outside your team can catch.

---

## Checking these still hold

1. **Try resuming a run deliberately**, on a cheap configuration, before you need to. The first
   real resume should not be the first resume.
2. **Confirm the manifest still records everything that can move a number** after any change to
   the configuration surface. New settings do not add themselves.
3. **Re-check that a converted checkpoint loads** after a framework upgrade, rather than at the
   point where someone wants to evaluate it.
