---
name: opsd
description: On-policy self-distillation against a named weakness. Load when considering whether to enter the OPSD stage, when the objective is falling and the model is not changing, when choosing what sets the direction of the update, and when judging whether the targeted weakness has moved.
---

# OPSD

One model plays both roles. The student produces trajectories from the bare prompt. A frozen
teacher — the same weights, conditioned on an approved privileged view the student does not
get — scores those same tokens. The view may contain an audited direction, failed attempts,
or verifier evidence; it need not and, when the project forbids leakage, must not contain a
known-good solution. The update moves the student toward evidence preferred under that view.

Enter only through an active `OPSD_*` context and queue task. If invoked directly without
one, load `auto-post-training` and bootstrap or resume the controller first. Load
`verification` before building failure histories, exposing diagnostic evidence, or scoring
outcomes.

**On-policy** here means the trajectories are the student's own, rather than a stronger
model's or a dataset's. That is the part that matters. It does not mean they have to be
regenerated at every step.

Bundled notes are calibrated implementation references. Load one only through an eligible
typed calibrated-note binding whose exact digest/provenance precede the active context cutoff
and whose assumptions match. Write newly learned asymmetries,
directions, and diagnostics to run-scoped memory rather than editing this Skill.

Load [offline OPSD](notes/offline-opsd.md) only when the context packet binds that
implementation and the queued experiment uses its offline, tensor-parallel assumptions.
Otherwise treat it as unbound legacy memory, not the method definition.

---

## Do not enter without a named target

This stage needs something to aim at. If you cannot state the weakness — and what evidence
identified it — you are running another training pass under a different name, and you will
not be able to tell afterwards whether it worked.

Before starting, have all three:

- **the weakness, named** specifically enough that you could recognise it in an output
- **a measurement of it** taken before training, so that "it moved" is a claim you can check
- **a reason to believe the teacher's advantage bears on it** — if the student's failure is
  not the kind of failure that the permitted privileged evidence could prevent, the
  asymmetry is irrelevant and there is no channel through which this can help

---

## Construct the privileged view as a controlled decision space

Do not start from one assumed context. Register components separately: bare prompt,
task-specific direction or plan, failed trajectory history, verifier stage or mismatch,
coarse repair guidance, bounded input-level counterexamples, and task-independent domain
advice. A reference solution is a separate high-leakage component and is excluded unless the
project profile explicitly permits it.

For every component, record what it may see, what it must omit, its ordering, length, and
leakage audit. Compare a candidate against the nearest control that changes one factor. Before
training, score the same responses under matched, within-domain shuffled, and shape-matched
neutral views so that task-specific information is separated from length and formatting.

The Agent selects the smallest composition whose incremental value survives these controls.
The selected view and unsuccessful variants belong to decisions and learned memory, not to
the frozen prior.

---

## The obvious formulation can fail by not moving at all

Write down the objective directly — minimise a divergence between the student's distribution
and the teacher's over the response tokens — and you get something that trains, reports a
falling loss, and changes the model by almost nothing.

**The failure may be insufficient movement rather than instability.** A run can report a
falling loss while its relative parameter update remains below a predeclared, project-specific
meaningful-change threshold. The loss curve alone does not distinguish that case.

Two consequences, and the second is the one that catches people:

**Instrument weight movement from the first step**, not just loss and gradient norm. Read the
optimiser's parameters before and after the step and record the relative change. It is the
only quantity that distinguishes *learning slowly* from *not learning*.

**Check that the instrument works before trusting it.** Under a distributed optimiser the
update lands on a sharded high-precision master copy and is copied back to the working
precision asynchronously, so reading the working-precision parameters around the step
can return a stale or zero reading — a false negative that looks identical to the failure you
are hunting. Probe the optimiser's own parameter groups and validate the probe on a known
update before trusting it. An invalid instrument and a non-updating model can produce the
same reading; neither should be assumed in advance.

---

## What runs: precompute everything, then update off it

The shape that works here separates generation, scoring and training into phases that never
run at the same time:

```
1  roll out     several candidates per problem from the current student, bare prompt,
                sampled hot
2  score        a teacher-forcing pass over those exact tokens, producing both the
                teacher's logprobs under the privileged context and the student's own
                under the bare prompt
3  weight       whatever decides how much each token counts (see below)
4  train        a clipped importance-weighted update over the precomputed logprobs
```

Nothing is generated during training and the teacher is not forwarded during training. That
is what frees the whole accelerator for the update, which in turn is what makes long
trajectories trainable rather than truncated.

**The price is drift**, and you pay it explicitly with an importance ratio between the policy
that sampled and the policy being updated, clipped. Then **measure the drift rather than
assuming a schedule** — the mean log-ratio, the fraction of tokens hitting the clip, the
largest ratio in a batch. When they climb past what you decided in advance was tolerable,
re-roll. This turns "how often should I regenerate" from a guess into a reading.

**The two logprobs must come from the same computation, not merely the same weights.** If the
sampling engine produces one and a teacher-forcing pass produces the other, they disagree by
more than the ratio can absorb and the first step comes out orders of magnitude from one,
where it should be exactly one. Score with the operation the trainer will use, and assert
that first-step ratio. It is free and it catches the entire class.

**Discard truncated rollouts.** A trajectory that hit the length limit is an incomplete
answer; moving toward it teaches the model to stop early.

---

## What sets the direction is a decision, and an open one

A teacher conditioned on the answer is not an oracle. It will sometimes prefer tokens that
are fluent and wrong, and nothing in its own signal separates those from the rest.

There is more than one thing you can put in front of the update, and which is right is not
settled:

- **the teacher's per-token gap alone.** Needs nothing but the teacher, so it is available in
  domains you cannot verify. Inherits the teacher's mistakes. This is the direct offline
  analogue of the divergence objective, and the reasonable default.
- **an outcome signal alone**, from a verifier, broadcast across the trajectory.
- **both** — the outcome sets the sign, the teacher's gap modulates which tokens within the
  trajectory earn the credit.

**Build these as a runtime switch rather than choosing one.** They share everything except
the few lines that produce the per-token coefficient, so making them selectable costs almost
nothing — and it is the only thing that lets the comparison happen at all. Default to the one
that needs least, so that a missing verifier degrades to a runnable configuration instead of
an error.

If you do compare, compare the combination against the **outcome-only** version, not against
the teacher-only one: going from teacher-only to the combination changes two things at once,
and the outcome signal will account for most of the difference by itself. See *Compare against
the baseline that differs by one factor* in the stage controller.

---

## Watch for the student learning the shape instead of the skill

The student can reduce the objective by imitating the teacher's surface form — its phrasing,
its structure, its length — without acquiring what is underneath. This appears as the
objective improving while the targeted measurement does not.

Reading outputs distinguishes these and nothing else reliably does.

**Name a collapse indicator before you start.** A model narrowing onto whatever it already
says shows up in the confidence it assigns to its own sampled tokens, rising monotonically.
Pick the quantity, decide what healthy looks like, and watch it from step one. Note that the
obvious choice may not survive your parallelism — a per-token entropy needs the full
vocabulary, which is exactly what a vocabulary-sharded layout does not give you cheaply, so
check that whatever you picked is actually computable before you rely on it.

**Be careful with normalisation.** The teacher's preference is often weak and consistent —
small per-token gaps pointing the same way. Normalisation designed for strong noisy signals
can invert exactly that. If you whiten or standardise, verify on a small run that it did not
destroy the signal, and default to off.

---

## Both outcomes are results

You are done when the named weakness has measurably moved, **or** has measurably failed to
move.

The second is a real finding. It says this instrument does not address this weakness, which
is what you need in order to choose the next move rather than repeating this one with
different settings. A stage that produces a clean negative result has done its job.

Report against the weakness you named at the start, not against whatever moved. Aggregate
metrics improve for reasons unrelated to the target, and accepting that as success loses the
only thing this stage was set up to tell you.
