# Execution reward across heterogeneous task contracts

> **Provenance class: calibrated reference.** Cross-stage governing rules now live
> in the verification Skill. This file is not part of the static Human Prior in
> this release; load it only through a separately digested, provenance-bearing
> calibrated-note binding in the Stage Context Packet.

Specifics for a setting where reward comes from compiling and running the answer, across
several benchmarks in more than one domain. The general method is in the
[RLVR Skill](../SKILL.md).

---

## What makes this hard is the plurality, not the execution

Executing one benchmark's answers is a day of work. The cost arrives when the reward spans
several, because each ships its own judging protocol and its own pinned environment, and they
do not agree with each other.

Concretely, in this setting: several benchmarks across two domains, judged variously by
simulation against recorded waveforms, by coroutine-driven testbenches, by self-checking
testbenches with a build system and golden models, by a different simulator compiling and
running real designs, and by execution on an accelerator with tensors compared against a
reference. **Several major versions of the same simulator run simultaneously**, each matching
a different benchmark's official environment, differing enough in language support that
swapping them changes verdicts.

The design consequence: **the task-property contract paired with an official-harness profile
is the unit, and everything above it is a dispatcher.** Benchmark identity selects concrete
files, pinned environment and reporting conventions; it does not replace analysis of the
artifact boundary, integration scope and correctness evidence. Normalise only the request and
the verdict.

The operational consequence: verification splits into a pool running simulators and a pool
running accelerator work, so that the two never contend for the same resource.

---

## Aligning to an official environment is not a formality

Two things it produces beyond conformance.

**Tool resolution has to be controlled at the point of use.** Where a runner picks up whichever
version appears first on the search path, the fix is to prepend the correct directory in the
child environment for that backend, while other backends keep resolving their own pinned
version through explicit paths. Changing it globally breaks the others.

**Aligning finds broken inputs.** Compiling the way the official harness does — including with
its plain flags rather than your preferred stricter ones — is how you discover that some of
your verification inputs were incomplete: a testbench present, the design path empty, and a
substantial number of problems aborting on a missing input file before the candidate was ever
loaded. **Those were unwinnable however good the answer, and they had been counted as model
failures the whole time.** Carrying the whole problem directory rather than a curated subset
is the fix.

The general lesson for this setting: when you package a benchmark's problems into your own
payload format, you will drop something the official harness relied on, and it will present as
a model weakness.

---

## Adding a backend is a fixed procedure

Doing this repeatedly is what the architecture is for, so make the sequence explicit:

1. freeze a snapshot you can roll back to
2. implement that backend's verifier against its official harness
3. write its data preprocessing and reward wiring
4. fold it into the mixed dataset
5. run a joint smoke across **all** backends, not only the new one

The last step is the one that gets skipped and the one that catches the regression, because
integrations interact through shared pools, shared toolchain paths and shared configuration.

---

## Where the detectors actually go

The [RLVR Skill](../SKILL.md) says to place the runtime signal where every execution path must pass and nothing
else does, and to test detectors against the obvious evasion. Both are more specific than they
sound.

**The runtime signal.** The instrumentation point one level too high gets bypassed: wrappers
that add autotuning or heuristic dispatch carry their own version of the entry point and call
the underlying implementation directly, so all decorated work goes uncounted. The point one
level too low fires on a lookup or on a warm-up compilation and counts work that never ran.
The boundary that works is the compiled dispatcher's exit hook — invoked only after the launch
returns without error — which covers direct, autotuned and heuristic paths alike. Test both
directions: that decorated work is counted, and that a compile-without-run is not.

**The vacuous-answer detector.** Applied only inside the bodies that are supposed to do the
work. Starting from the variable that receives the input, propagate along assignment chains and
ask whether what is finally written back is still the same value. **The propagation must survive
intermediate reassignment** — an earlier version followed only direct assignments, and inserting
one intermediate variable was enough to slip past it.

**The delegation detector.** A syntactic scan over statements outside the bodies that are
supposed to do the work, looking for compute calls into the framework being replaced. Output
allocation and synchronisation are deliberately permitted. This is a conservative heuristic and
is positioned as a supplement to adversarial fixtures, not a replacement.

---

## The reward ladder

Kernel-style backends can support a layered reward, because there is a meaningful sequence of
partial success: correct and faster than the reference, correct, runs but produces the wrong
answer, does not build. The exploit indicators sit in front of all of it as hard gates.

RTL-style backends are binary. "Behaviourally identical to the reference" has no meaningful
intermediate state, and inventing one would mean rewarding something you cannot define.

Do not force a common shape across backends. The ladder should follow what the domain
actually admits.

---

## The three context findings, in this setting

These came from measurement rather than from reasoning, and each pinned the budget from a
different side.

**A prompt budget below the data's longest prompt deleted most of a benchmark from the
validation split** while leaving the training split essentially untouched. The filter reported
it in one log line and did not raise. What made it hard to see was the cut-off landing in a
dense region: the longest surviving prompt and the shortest discarded one were only a few
hundred tokens apart, so the remainder did not look truncated. Raising the budget above the
measured longest prompt filters nothing.

**Bucketing tens of thousands of rollouts by response length showed score decaying
monotonically and reaching zero before the cap**, with no full marks at all in the band
approaching it. Beyond the cap, uniformly zero. So lowering the cap relabelled zeros without
changing the reward signal, and bought back the tail of generation time — which matters
because a step's generation phase is set by its single slowest sequence, and in most steps
that sequence sat at the cap.

**A run died in the actor backward pass with a single long sequence on one rank having
consumed nearly the whole card.** Peak attributed to roughly a megabyte per token, dominated
by gradient-checkpointed layer inputs and by attention intermediates wider than the hidden
size, which grow with length faster than intuition suggests. The micro-batch was already one
sequence, so nothing about batching could help. This is what produced the shift to bounding
the whole trajectory rather than prompt and response separately — and under a trajectory
budget, short-prompt tasks automatically receive the room that a fixed response cap had been
denying them.

---

## Checking these still hold

1. **Re-run the joint smoke across all backends** after any change to a single one. The shared
   surfaces — pools, toolchain paths, payload formats — are where the regression lands, not in
   the backend you touched.
2. **Re-verify each backend against its official harness** after a toolchain update. The pinned
   versions are part of the verdict; an upgrade that looks harmless changes what passes.
3. **Re-measure the longest prompt per split** whenever the data changes, and confirm the
   filter dropped nothing. This one silently stops being true.
4. **Re-bucket score against length** after a substantial capability change. The cap was
   derived from a measured curve, and the curve moves.
