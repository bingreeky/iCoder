---
name: project-docs
description: Writing down what survives you losing your own context. Load early in a long-running project, at any decision point worth recording, and whenever you resume work and need to reconstruct where things stood.
---

# Project documentation

You are running a process that lasts far longer than what you can hold at once. Your
context will be compressed or lost several times before the project ends, and the version
of you that continues will have the codebase, the job history, and whatever you wrote
down.

In an RSI run, enter through an active controller context and queue task. If invoked directly
without them, load `auto-post-training` first. After interruption, replay authoritative
runtime transition, experiment, queue, attempt, decision, artifact, and memory records before
using the prose documents below; handover text is a convenience, not a recovery source.

So the question for anything you consider recording is: **could this be recovered by
re-reading the code and the logs?** If yes, do not write it — it will drift and mislead.
If no, it is the only copy and it must be written now.

The things that cannot be recovered are consistently the same: why you chose what you
chose, what a number means, what you already tried and ruled out, and what you currently
believe is broken.

---

## Write at the decision point

Write when the decision is made, not at the end.

Reconstructing afterwards from artifacts recovers what happened and loses what you
believed at the time — which is the part that makes a record useful, because it is what
lets a later reader see which conclusions rested on evidence that has since changed.

This is also cheap insurance against the specific failure of re-running an experiment you
already ran, having lost the memory of its result.

---

## Kinds, and what each one is for

These have different rules and should not be mixed into one document.

**Definitions.** How each metric is computed, what its denominator is, what is excluded and
why. These exist to be frozen. Their entire value is that they do not change, so that
numbers from different weeks are comparable. When one must change, change it explicitly
and say what it was before.

**Results.** The numbers, together with every annotation needed to keep them from being
misread — what produced them, under which definitions, with what known caveats. A number
without its conditions attached will eventually be quoted without them.

**Known defects.** Things you have found and not fixed, with the evidence and an estimate
of impact. This is the highest-value document and the one most often skipped. Without it,
the same defect is rediscovered from scratch, treated as new, and investigated at full
cost — possibly by you.

**Handover state.** Where things stand right now: what is running, what is blocked on what,
what the next intended action is. Unlike the others this one is live and should be
overwritten rather than appended to. A stale handover is worse than none.

**Procedures.** Sequences you have executed more than once and will execute again. When a
procedure has enough preconditions that you would get it wrong from memory, record it first
as evidence-bound learned memory. It may be proposed for a future human-reviewed Skill
release; see the stage controller section on writing learned memory and proposing Skill
updates.

---

## Everything you write down will drift

Code changes, paths move, configurations are replaced. A document that was accurate when
written becomes wrong without any event marking the transition, and a confidently wrong
document is more damaging than a missing one, because it is believed.

So **record how to check that a document still holds**, next to what it claims. Which file
the claim is derived from, what command reproduces the number, what you would look at to
confirm the defect is still present. A claim that carries its own verification method
degrades into a task; one that does not degrades into a false belief.

Prefer pointing at the source over copying from it. A path to the code that implements a
rule ages better than a paraphrase of the rule.

---

## Picking up after you have lost your context

When you come back without memory of what you were doing, read in this order:

1. **reconstructed runtime state** — replay the authoritative records first
2. **handover state** — compare prose against the replayed state
3. **known defects** — so you do not investigate something already understood
4. **definitions** — before reading any number
5. **results** — which will now be interpretable

Then, before acting on any of it, check one claim against reality. If the document says a
job is running, look for the job. That single check tells you how much to trust the rest,
and it is cheaper than discovering the gap three steps later.

---

## The experiment log

The log is a deliverable, and it is separate from the documents above: those describe
current state, the log describes the sequence of decisions.

In an RSI run, the Task Queue records proposed and active work, the artifact registry records
what the work produced, and the decision log records what the evidence changed. Keep their IDs
linked; prose notes are not substitutes for these runtime records.

At each decision point record what you observed, what you concluded, what you did, and
what you expected to happen. Return later and record what actually happened.

Keep the decisions that turned out wrong. A choice that was correct on the evidence
available and wrong in hindsight is more informative than one that simply worked, and
removing it makes the record less useful in exactly the way that matters — it hides the
reasoning under a sequence of outcomes.
