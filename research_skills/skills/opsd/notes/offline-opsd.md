# Offline OPSD on a tensor-parallel trainer

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding in the Stage Context Packet. A live
> run writes new findings to run-scoped learned memory.

Specifics for running self-distillation as four offline phases on a Megatron-style
tensor-parallel trainer with a separate sampling engine. The method is in the
[OPSD Skill](../SKILL.md).

---

## The four phases, and why they are separate processes

```
1  student rollout       several candidates per problem, sampled hot, bare prompt
2  teacher scoring       teacher-forcing forward; produces teacher_lp and old_lp
3  verify + advantage    run each candidate; build the signed, group-centred advantage
4  offline training      importance-weighted clipped update over the scored rows
```

They are separate because their resource profiles are incompatible, not for tidiness. One
wants a sampling engine and no gradients; another wants gradients and no sampling engine.
Trying to hold both resident is the single largest source of trouble here.

---

## The sampling engine will not share the card

This is usually presented as a property of sampling engines. It is more often a property of
your model.

An unusual architecture — a hybrid that is not plain attention — narrows which frameworks
support it at all, and then narrows the engine configuration that works with it: which
attention backend, which graph-capture mode, and whether the engine may free its cache
between calls. Once cache-freeing is off, the engine holds its reservation for the life of
the process, and a trainer that pages weights back for an update fails against it.

**So the chain runs architecture → viable engine configuration → memory behaviour → whether
training and sampling can coexist.** When you hit this, look at the top of that chain rather
than tuning the bottom of it: the utilisation fraction is not the free variable you think it
is, and the configuration that works was probably arrived at after a deadlock.

The standard training loop builds the engine **even when it never samples**. If you are
training offline you must actively prevent that construction — stubbing the managers that
create it, and masking whatever flag the worker uses to decide it is a sampling role.
Disabling sampling in configuration is not the same as not building the engine.

The payoff is not only avoiding the failure: with the engine gone, the whole card is
available, which is what makes long reasoning chains trainable without truncating them.

---

## Scoring must use the trainer's own forward, not the sampling engine's

Two reasons, and the second is the one that will waste your time.

**Memory.** Asking a sampling engine to echo logprobs materialises a vocabulary-sized float
per token. At the sequence lengths this stage exists to support, that is tens of gigabytes
for a single sequence, and it fails.

**Calibration.** The reference logprob is compared against a training-time logprob through
an importance ratio. If the two come from different code paths, they disagree by more than
the ratio can absorb, and step one — where the ratio should be exactly one — comes out
orders of magnitude off. Same weights is not enough; it must be the same operation.

**Check it.** On the first step, the mean ratio should be one to within numerical noise.
This assertion is free and it catches the entire class of misalignment above. If it is not
one, stop; nothing downstream is meaningful.

---

## Logprobs under tensor parallelism

With the vocabulary sharded across ranks, each holds a slice of the logits. Gathering the
full vocabulary to take a softmax is prohibitive at these vocabulary sizes.

Decompose instead: take the per-shard maximum and all-reduce it for a numerically stable
shift, exponentiate locally, all-reduce the sum, and normalise in place. Every vocabulary
sum becomes a local partial plus one collective. The same decomposition works for a
divergence between two distributions, with a hand-written backward.

Gradient-check the result against a single-rank full-vocabulary reference before trusting
it. This code is easy to get subtly wrong and the error looks like a training problem.

---

## Collectives must not be conditional on data

Padding a batch up to the data-parallel world size leaves rows that carry nothing. The
instinct is to skip them.

**Do not skip them — run them and scale the contribution to zero.** A rank that skips work
while its peers do not desynchronises the tensor-parallel reduction, and the job deadlocks
inside a collective with no error and no traceback. The general rule: mask the contribution,
never the collective.

This applies to anything else conditioned on data — an empty response after filtering, a
row whose verifier returned nothing. Return a zero that is still connected to the graph.

---

## Keep a problem's candidates in one batch

Group-relative centring subtracts the mean of a problem's own candidates. If the batch is
smaller than the group, the group is split across batches, the centring is computed over a
subset, and nothing reports it.

Shuffle at the group level rather than the row level, and make the batch size a whole number
of groups.

---

## Discard truncated rollouts

A rollout that hit the length limit is an incomplete answer. Two harms compound: the
verifier judges it on what it managed to emit, and distilling toward it teaches the model to
stop early. Default to dropping them and count how many you dropped — a rising rate means
the length budget is now binding and the stage is quietly training on a biased subset.

---

## What to log, and how to read it

Three groups, and they answer different questions.

**Is the offline batch still usable?** The mean log-ratio between the sampling policy and the
current one, the fraction of tokens hitting the clip, and the maximum ratio in the batch. All
three rise together as the student drifts. Decide the tolerance before the run; when it is
crossed, regenerate rather than continuing.

**Is the model collapsing?** The confidence assigned to its own sampled tokens. Flat is
healthy; monotone rise means it is narrowing onto whatever it already says. This is the
sentinel — pick it before you start rather than looking for one afterwards.

**Is there anything left to learn from this batch?** The mean teacher-minus-student gap. When
it plateaus the batch is exhausted, independently of whether the loss is still moving.

---

## Advantage construction

Keep the arms selectable at runtime rather than committing to one: the teacher gap alone,
the verifier outcome alone, and the combination. They share everything except the few lines
that build the per-token advantage, so making them a switch costs almost nothing and is what
lets the comparison happen at all. Default to the one that needs least — the gap — so that a
missing verifier degrades to a runnable configuration instead of an error.

The rest of this section describes the verifier-anchored construction, which is the one with
moving parts.

The registered OPSD adapter consumes the raw tri-state verdict. Among judged,
candidate-caused failures, it may distinguish a contract-defined compile/launch failure from
a response that ran and was wrong when that ordering is predeclared and audited. Trusted
infrastructure failures remain `unjudged` and are masked or neutralized by the registered
adapter; they never inherit a candidate-failure grade.

A continuous quality measure, where one exists, is normalised **only across the candidates
that passed**, using a median-and-deviation normaliser rather than mean-and-standard-
deviation. Outliers are common here and a single extreme value otherwise dominates its
group.

Groups where every candidate passed and no quality measure applies contribute nothing. That
is correct — there is no variance to learn from — but count them, because a rising share
means the problems have become too easy for this stage.

Problems with no verifier are kept with zero advantage rather than dropped, so they appear
in the accounting instead of silently shrinking the denominator.

---

## Checking these still hold

1. **Re-run the step-one ratio check** after any change to how logprobs are produced on
   either side. It is the cheapest guard here and it silently stops being true.
2. **Confirm the sampling engine is still being built by default** when you do not want it.
   This is a property of the surrounding framework, not of your code, and it changes under
   you on upgrade.
3. **Re-measure the memory reservation** before assuming the offline split is still
   necessary. If the engine learns to release memory, the main reason for this architecture
   goes away.
4. **Re-run the scrambled-context audit** (see the [OPSD Skill](../SKILL.md)) whenever the teacher's context is
   reformatted. The audit tests the content, but its verdict depends on the format being
   held constant between real and scrambled.
