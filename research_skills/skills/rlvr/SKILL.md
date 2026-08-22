---
name: rlvr
description: Reinforcement learning against a mechanical verifier. Load when entering the RLVR stage, when designing or changing a reward, when deciding how to handle samples the verifier could not judge, when setting a length budget, and when training reward rises without validation following.
---

# RLVR

In every earlier stage the verifier was an instrument you used to read results. Here it
becomes the objective itself. Every property of it — what it measures, what it fails to
measure, how it behaves when it breaks, and what the model can make it say — is now a
training decision, whether or not you made it deliberately.

Enter only through an active `RLVR_*` context and queue task. If invoked directly without
one, load `auto-post-training` and bootstrap or resume the controller first.

Budget accordingly. The centre of gravity of this stage is not the algorithm; it is making
*execution is the reward* hold up. Most of the difficulty is in the reward and the
infrastructure around it.

Load verification for the cross-stage request, verdict, task-contract, official-profile,
integrity, and failure-attribution rules. Bundled RLVR notes are calibrated implementation
references and require an eligible typed binding with exact digest, provenance, and
applicability; record newly integrated profiles, exploits, and relaxations as run-scoped learned
memory with verifier and experiment artifacts.

Load [verification service](notes/verification-service.md) only when the context packet
binds that service version. Load
[multi-backend verification](notes/multi-backend-verification.md) only when the queued task
spans those backend assumptions. Neither note overrides the shared `verification` Skill.

---

## You get what you measure, and only that

The model will find the highest-reward strategy available to it. If a dimension you care
about is not in the reward, the strategy that ignores that dimension is not a bug — it is
correct play against the objective you actually wrote.

**Before training, list what you want and check each one appears in the reward.** A quality
you want but do not measure will not merely fail to improve. It will degrade, because effort
spent on it is effort not spent on what scores.

There is a distinction here that decides what you can do about it:

- the model **violating a rule you stated** — that is reward hacking, and gates can catch it
- the model **exploiting a rule you never stated** — no gate helps, because nothing is being
  violated

The second is more common and is only fixable by changing the objective. **Anti-cheat
machinery is not a substitute for a missing dimension in the reward.** If you find yourself
writing a detector for behaviour that is technically compliant and obviously not what you
wanted, you are patching the wrong layer.

---

## Reward hacking

Expect the first run on a fresh base model to find the cheapest path rather than the intended
one. Two shapes to expect: **submitting the reference implementation** under a new name
without doing the work at all, and **doing the work vacuously** — an operation that satisfies
the form while computing nothing. Both pass a naive check of "the answer contains the right
construct" combined with "the output is correct".

**Make exploits hard gates, not deductions.** An exploit that retains partial or performance
credit still supplies a useful path to the model. A detected exploit emits a
candidate-caused contract violation before any layered credit is considered. The registered
adapter must make it no more favourable than an ordinary judged failure; do not invent a
special scalar that can accidentally rank the exploit above failure.

**Trust a runtime signal over anything read from the answer.** Evidence that the work actually
happened is the only kind the answer cannot talk its way around; anything derived from
inspecting the answer can be fooled by an answer written to be inspected. Static checks are
supplements to adversarial fixtures, never replacements.

**Test every detector against the obvious evasion, not just the obvious exploit.** Whatever
you write, try the one-step-more-indirect version before trusting it. Instrumentation placed
one level too high gets bypassed by wrappers; placed one level too low it counts work that
never ran.

### Do not stack a gate on a path the metric already closes

Before adding a detector, ask whether the reward already makes that exploit unprofitable.

If a task is scored on speed, delegating the work back to the thing you were asked to replace
**cannot win** — such an implementation cannot outrun what it delegates to. The metric
forecloses the exploit. A purity gate on top buys no protection and does real harm: genuine
answers are mapped to failure.

**Judge each task against its own definition of a violation.** Two benchmarks in the same
domain can define the task differently — one requiring that all computation happen in the new
form, another explicitly granting freedom to replace only what you choose. A gate that is
correct for the first is a bug for the second.

The end state is not a relaxation of standards. It is a principle: **each task is judged
against its own definition of a violation, and no redundant gate is stacked on a path the
metric itself already closes.**

### The invalidation path is an attack surface

A sample marked *unjudged* is masked out and receives no gradient. A judged failure is mapped
through the frozen reward adapter. Avoiding an update can be preferable to receiving the
registered failure outcome, so if the model can influence which status it gets, that is an
exploit — and a well-hidden one, because it looks like infrastructure noise rather than
gaming.

**Whether a failure counts as infrastructure must be decided from signals the candidate
cannot produce.** Anything written by the answer's own execution is not such a signal. This
makes attribution a reward-design decision, not merely an engineering one.

### Shape of the reward

Put the exploit indicators in front as hard gates. Behind them, a layered reward gives a
gradient toward the first success — but only where the domain admits meaningful partial
success. Where correctness is binary, do not invent intermediate states; you would be
rewarding something you cannot define.

Keep `pass`, `fail`, and `unjudged` in the raw verifier record. A versioned stage adapter,
fixed before comparison, maps only judged outcomes into scalar reward; unjudged uses a
masked sentinel outside the reward range. Scalar values are properties of that adapter, not
meanings of the raw verdict.

---

## When the verifier cannot answer

Verification is a real distributed system. Compilers crash, simulations hang, accelerators run
out of memory, connections drop, and inputs can be built wrong. **None of these is
distinguishable numerically from the model having answered incorrectly** — both appear as "no
correct result was obtained".

Generated code is also untrusted code. It will, with no intent to attack you, exhaust memory,
fill disks, spawn children that outlive it, and crash in ways that take the worker with it.
Isolating and bounding it is a precondition for this stage, not an optimisation.

### An ordinary scalar is a lie with a direction

Filling an unjudged sample with any in-range scalar asserts something false: that you checked
the answer and observed the judged outcome represented by that scalar.

It also injects noise that is **systematic rather than random**. One malformed input makes an
entire class of problems receive an ordinary failure-like outcome forever, which reads exactly
like a model that cannot do that class of problem. And infrastructure failures concentrate on
the samples that take longest and do most — the slow, the large, the ambitious — so treating
them as judged failures punishes exactly the behaviour you are trying to produce.

### The sentinel is a contract between two components

Mark unjudged samples with a value **outside the range any legitimate reward can take**, then
carry the mark through: exclude them from the group statistics, force their own advantage to
zero, and divide the loss by the count of valid trajectories rather than the nominal batch
size. The effect should be equivalent to the sample never having been drawn.

**Both halves must exist or neither works.** The component that verifies marks what it could
not judge; the component that computes advantages excludes it. With only the first, the
sentinel is not a mask — **it is an extreme unintended reward value**, and you have built a
new failure. When you introduce a sentinel, confirm the consumer honours it before running
anything.

### Which way to lean when attribution is unclear

Work through the failure classes explicitly, asking of each whether it originates in the
candidate's code or in your environment, and keep an **explicit list of infrastructure
causes**: only enumerated causes count as yours, everything else is attributed to the model.

The asymmetry is deliberate. Mistaking your fault for the model's understates the model, which
you will notice. Mistaking the model's fault for yours **hides a real capability gap** by
masking those samples out of training entirely — you quietly stop training on a class of
problem and nothing tells you.

### Where fail-closed begins

A small fraction of unjudged samples is a condition to neutralise and continue. A large
fraction is a condition to stop.

The reason is not severity, it is validity: neutralising assumes the surviving samples still
represent the batch. Past some rate that assumption fails, and continuing means training on a
distribution you are no longer measuring. Decide the threshold before you start, monitor the
rate from the first steps, and make crossing it halt rather than warn.

### Retry by failure class, and bound it by wall-clock

Two kinds of failure look identical at the call site and deserve opposite treatment.
**Transport failures** produced nothing and may well succeed on repeat. **Deterministic
failures** — version mismatch, malformed payload, an unsupported construct — will fail
identically forever, and every retry is a real verification job consuming real capacity.

**Express the budget as a deadline, not as an attempt count.** Attempts multiplied by a
request timeout is the real worst case, and with generous values on both, one unreachable host
stalls a step for hours while the whole cluster idles.

---

## Groups that teach nothing

When every sample in a group receives the same score there is no variance within it, the
advantage is zero throughout, and the item contributes nothing — whether all failed or all
succeeded.

**Monitor the fraction of degenerate groups from the first steps.** It is one of the few
metrics that tells you something is wrong before the loss curve does.

When most groups have identical eligible adapted rewards and therefore zero within-group
advantage, ordinary trainer tuning cannot recover the missing signal. First verify that the
frozen adapter and verifier preserve every legitimate partial/performance distinction. If
they do, return to Data to move task difficulty; otherwise return to reward or verification
audit. Raw all-fail or all-pass labels alone do not prove degeneracy.

---

## The length budget

Context length here is not a throughput knob to be turned up when convenient. It is pinned
from both sides by measurement.

### Too small a budget silently edits your validation set

Frameworks commonly discard over-long prompts and report it in a single log line without
raising. On the training split that is a rounding error. On the validation split it can remove
most of a benchmark — after which every number you report for it is computed over the short
remainder while every table still describes it as the whole set.

This is hardest to notice exactly when it matters most: if the cut-off lands in a dense part
of the length distribution, nothing about the remainder looks truncated.

**Measure the longest prompt in every split, set the budget above it, and confirm the filter
dropped nothing.** A context budget is not only an efficiency parameter — it silently changes
what you are measuring.

### Before assuming truncation costs you signal, bucket by length

Reinforcement learning lengthens outputs, and truncation is scored as failure, so a model
becoming more capable and more verbose can produce a falling score. That is the standard worry
and it is worth checking rather than assuming.

**Bucket sampled rollouts by response length and look at the mean score in each bucket.** If
score decays with length and has already reached the registered failure floor *before* the
cap, then beyond that point the model is looping rather than solving: lowering the cap only
relabels the same failure outcome from *wrong* to *truncated*, costs no reward signal, and buys a great deal of wall-clock, because a step's
generation time is set by its single slowest sequence. If score is still healthy near the cap,
you are losing real answers and the budget must rise.

Either way, **track the truncation rate as a first-class metric next to reward**, and check it
before concluding anything about capability from a falling curve.

### Bound the trajectory, not its parts

The binding constraint is memory per trajectory, not batch size. Once the micro-batch is a
single sequence, no batching change can help — **one trajectory on its own either fits or does
not.** Peak memory per token is usually far higher than intuition suggests.

Budget prompt and response **together** rather than capping each. The physical constraint
applies to the trajectory, so splitting the limit both fails to preserve the margin and wastes
it: tasks with short prompts could have been given room to generate and are not.

**Do not trust a single rank's memory reporting.** Balancing total tokens across ranks does not
equalise the longest single sequence on a rank, and that is what sets the peak.

---

## Making the verdict credible

If reward comes from execution, whether execution results can be trusted is the foundation of
the whole system.

**Match each benchmark's official environment rather than imposing one of your own.** Tool
versions are part of the verdict, different benchmarks pin different ones, and aligning also
finds things nothing else would — including problems that were unwinnable for reasons of your
own packaging and had been counted as model failures.

**Absence of a positive verdict on a completed, judged execution is a failure, not a pass.**
Self-checking tests that report success by printing something must be read as failing when
that something is missing after the harness completes. A trusted infrastructure event that
prevents judgment remains unjudged; defaulting either class to pass corrupts the verdict.

**Use behavioural equivalence as an auxiliary diagnostic or versioned training adapter when
the official tests under-check.** It is stronger and can reject a differently implemented but
valid answer. Do not silently replace held-out or official correctness with it; changing that
criterion requires the verification audit and a human gate.

**Cross-check against an independent rule, holding everything else constant.** Feed the
identical extracted answer through both rules with prompting, sampling and extraction fixed,
so the only variable is the rule. Align tool versions first, or you will read environment
differences as rule disagreements.

**Fail closed on integrity, never degrade.** When a recorded hash, a pinned version or an
expected input count does not match, stop — do not fall back to a weaker check, because a
weaker check produces numbers that look fine.

---

## Expect the reward to grow

Reward rarely stays on one benchmark. Expect the cost of each addition to fall almost entirely
on the verification side rather than on the algorithm, and expect several benchmarks to demand
mutually incompatible environments simultaneously.

Two consequences worth planning for from the start: **the task-property contract plus its
official-harness profile is the unit, and everything above it is a dispatcher** — benchmark
identity selects files, environment and reporting but does not replace semantic analysis;
normalise only the request and the verdict. **Every integration ends with a joint smoke across
all profiles**, because integrations interact through shared resources and the regression
rarely lands in the one just touched.

---

## Leaving this stage

You are done when training reward and validation have both flattened and reading outputs shows
no capability appearing that was not there before.

Before reporting, confirm the run was measuring what you think:

- the truncation rate at the end, against the start
- the fraction of degenerate groups over the run
- the fraction of samples that went unjudged, and whether that fraction drifted
- that no split was silently filtered
- a sample of high-reward outputs, read directly

A reward curve that rose while validation stayed flat is not a partial success. It means the
model found something in your reward you were not intending to pay for, and the outputs will
show you what it was.
