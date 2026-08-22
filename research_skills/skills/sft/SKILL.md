---
name: sft
description: Supervised fine-tuning on verified data. Load when entering the SFT stage, when choosing a length budget or a parallelism layout, when deciding whether SFT has saturated, and when deciding whether a disappointing result is a data problem or a training problem.
---

# SFT

This stage teaches the model the shape of a correct answer and the behaviours that
generalise from your data. Its results reflect the data it was given more directly than any
later stage, which makes it the best instrument you have for finding out what your data
actually contains.

Enter only through an active `SFT_*` context and queue task. If invoked directly without
one, load `auto-post-training` and bootstrap or resume the controller first. Load
`verification` before filtering teacher trajectories or evaluating checkpoints.

Bundled notes are calibrated setup references, not automatically part of the current run's
initial prior. Load one only through an eligible typed calibrated-note binding whose exact
digest, version, provenance, and environment match the context packet. Write
new findings to run-scoped learned memory with experiment and artifact IDs.

Load [Megatron multi-node setup](notes/megatron-multinode.md) only when the context packet
binds that implementation and the queued run uses its tensor-parallel or multi-node
assumptions. It is not a default claim about the current infrastructure.

---

## Fix the format contract before training, not after

SFT is where the model learns what an answer looks like, and that shape becomes an
assumption in every stage after it: the verifier's parser, the teacher prompts in OPSD, the
reward extraction in RLVR.

Fix it before the first run, then leave it alone between rounds unless you are prepared to
re-measure everything, because **a change in output format is indistinguishable from a
change in capability** in any aggregate metric. A model that got better and a model that
started wrapping its answer differently produce the same movement.

If you must change it, re-evaluate the earlier checkpoints under the new contract before
comparing.

---

## Capability-gap supervision is a selectable cold-start component

When the goal is to teach a reasoning pattern the base model does not reliably exhibit,
construct the candidate subset relative to both student and teacher capability rather than
sampling teacher traces indiscriminately.

Register the student and teacher identities, prompt template, sampling settings, repeated
sample count, extraction, and task-native verifier. A common admissible policy is to keep a
task only when the student has no verified success under the registered sample budget and the
teacher has at least one. Retain a verified complete teacher trajectory and record tasks for
which neither model succeeds. The project profile or an Agent decision supplies the sample
budget and trajectory-selection rule; this Skill does not hard-code model names or values.

Report the full count chain from executable pool through student failures, teacher successes,
format filtering, deduplication, and final training pairs. This distinguishes a deliberate
capability-gap curriculum from a teacher-generated mixture whose selection cannot be audited.

---

## Prefer verified targets to reference solutions

Where you can verify, build the training targets by sampling from a capable model and
keeping only what passes — not by taking the dataset's reference answer. A verified rollout
is correct *and* in a form a model produces naturally; a reference solution is frequently
neither.

Count what never produced a passing sample. That set is your genuinely hard problems, and
its size is one of the more useful numbers you will have when deciding what to do after this
stage.

Where you cannot verify a domain, reusing existing high-quality pairs directly is
reasonable. Keep the two provenances distinguishable in the data, because when a result
looks strange you will want to know which part of the mixture produced it.

---

## Confirm what the loss is actually computed over

Before the first real run, establish in code — not in documentation or a config summary —
which tokens contribute to the loss: prompt tokens, response tokens, or some tagged span
inside the response.

This is worth the time because getting it wrong produces a run that trains, converges,
reports a falling loss, and does not improve the model. There is no error message for
training on the wrong span.

---

## Choose the length budget from the data, not the model

The maximum sequence length is not a capability setting to be maximised. It is a decision
about which tail of your data you are willing to lose, and it is priced steeply.

**Take it from the length distribution of your actual data.** Find the point that covers all
but a negligible fraction — a fraction small enough that dropping those examples cannot
change the mixture. That is your budget.

**The cost is superlinear.** Raising the cap well past what the data needs can multiply step
time for no gain, because you pay for the longest sequence the configuration allows on
every step, not for the ones you actually have. Measure step time at two candidate caps
before committing; the difference is usually much larger than the length difference
suggests.

**Drop over-length examples; do not truncate them.** Truncation cuts the end off an answer,
leaving a training example that reasons and never concludes. The model learns to do the
same, and nothing reports that this happened. Find the flag that controls this and confirm
it, because dropping is rarely the default.

---

## The training configuration is part of correctness, not only of speed

A parallelism layout that is valid arithmetic can still be a layout that does not run.

**A configuration verified at one scale can deadlock at twice that scale**, because the
layout that fit inside a machine now spans machines, and the communication pattern that was
free over an internal interconnect is not. The failure is a hang inside a collective, with
no error and no traceback. When you scale node count, treat the layout as a new
configuration to be validated, not a parameter to be doubled.

**The batch size can also deadlock**, not merely run out of memory, when the pipeline
schedule's warm-up depth depends on it. If a batch size hangs, try a smaller one before
concluding the problem is elsewhere.

**Run the smallest authorized representative smoke first, every time.** It may be one node,
fewer devices, or reduced data/steps, but it must still exercise the real tokenizer, loss
mask, checkpoint, launcher, and communication path relevant to production. This catches
configuration errors before the expensive queue and scale path; do not assume a single-node
tier exists or faithfully represents the locked topology.

Record which configurations you have actually seen complete. That list is worth more than
any reasoning about what should work.

---

## The data mixture shows through

SFT reflects its input mixture more faithfully than later stages do. A category that is most
of the mixture produces a model disproportionately good at that category, and this reads as
a capability profile when it is really a composition artifact.

Record the composition alongside every checkpoint — how much came from which sources, in
what proportion. When a downstream stage produces an uneven result, this is the first thing
to check, and reconstructing it afterwards is unreliable.

---

## Read the failure composition, not just the score

The aggregate score is the least informative thing this stage produces.

A shift from *malformed output* to *well-formed and wrong* is real progress that a pass rate
will not show. A shift from *wrong* to *refuses to attempt* is a regression that a pass rate
will also not show. Both change what you should do next, and only the composition of
failures distinguishes them.

Track the composition across checkpoints, not only within one.

---

## Deciding where to go next

**Saturated** — gains flattened and the failure composition no longer shifting. This is
sufficient reason to move on; see the stage controller. Do not keep training a saturated
stage looking for a reason to leave it.

**Failures dominated by problem types absent from the data** — return to data, and return
with a specific coverage gap rather than a request for more data in general.

**Failures dominated by problems the model has seen and gets wrong** — this is a weakness
you may be able to name, which is what the next stage needs.

The distinction between the last two is worth making carefully, because "the model is bad at
X" and "the model has never seen X" produce identical scores and opposite next moves. Check
whether the data contains items of that type before concluding it is a capability problem.
