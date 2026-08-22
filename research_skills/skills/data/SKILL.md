---
name: data
description: Building a pipeline that produces and expands verified training items in a domain where correctness can be checked mechanically. Load when entering the data stage, when designing the synthesis pipeline or its gates, and whenever another stage sends you back here — SFT because the model has never seen a kind of problem, RLVR because sampled groups contain no eligible adapted-reward variation.
---

# Data

This stage produces items you can train on and trust. You will almost certainly have to
build the pipeline that produces them.

Enter only through an active `DATA_*` context and queue task. If invoked directly without
one, load `auto-post-training` and bootstrap or resume the controller first. Load
`verification` before admitting or executing any task--reference--verifier item.

The hard part is not producing problems. It is that a problem whose parts disagree looks
exactly like one whose parts agree. It passes every check you did not think to run, it
teaches the model something false, and it surfaces weeks later as a result you cannot
explain.

Expect most of your code and nearly all of your debugging to be in rejection, not
generation. If the pipeline you build is mostly generation, you have not yet found the
failure modes.

Bundled notes hold calibrated references for particular seed families. Load one only through
an eligible typed calibrated-note binding whose exact digest, provenance, and applicability
are recorded in the active context snapshot. During a live run,
write new findings to evidence-bound learned memory outside the installed Skill; do not
edit the Human Prior in place. See runtime records in auto-post-training.

Load [kernel seed evolution](notes/kernel-seeds.md) only when the context packet binds that
implementation and the seed artifact is a framework-level GPU-kernel task. Load
[RTL seed evolution](notes/rtl-seeds.md) only when it binds that implementation and the
artifact is an RTL fragment/module with its declared test contract. Neither note is a
default source of project facts.

---

## What an item is

Three parts, and they must agree:

- **the instruction** — everything the model is allowed to see
- **the reference solution**
- **the verifier** — what decides whether an answer is correct

An item is worth exactly as much as its weakest part. A correct reference paired with a
verifier that tests something else is not a partly-good item; it is a wrong item that will
not announce itself.

---

## Why you came here changes what you do

You will enter this stage more than once and the right work differs each time.

**First entry.** You have seeds and no pipeline. Everything below applies.

**Sent back from SFT** because failures are dominated by problem types the model has never
seen. The work is *coverage* — find which types are absent and generate along that axis.
More of what you already have will not move it.

**Sent back from OPSD** because a named weakness has no items to exercise it, or because the
items it has lack what the teacher's privileged context needs. The work is *targeted* rather
than broad: you need problems of one specific kind, and you probably need them solved as well
as stated, which is a harder requirement than anything upstream asked for.

**Sent back from RLVR** because sampled groups contain identical eligible adapted rewards and
therefore zero within-group advantage. The work is *signal placement*, not coverage. Diagnose
the frozen adapter before changing difficulty: uniformly low rewards may need easier or
scaffolded items, uniformly high rewards may need harder or more discriminating items, and
collapsed partial/performance rewards may require a verifier/reward audit. Preserve raw
verdicts while moving the pool toward a band with measurable within-group reward variation.

Do not start generating until you can say which of these you are in.

---

## Split ancestors before evolution

Freeze train, development, and held-out evaluation at the root-task level before generating
derivatives. Every evolved instruction, reference, test, verifier, and teacher trajectory
inherits its ancestor identity and split. A held-out root never becomes training data through
paraphrase, mutation, composition, or a newly generated harness.

Record source, licence, root identity, parent lineage, and permitted use in the seed registry.
Deduplication within the generated pool does not replace this separation. Before leaving the
stage, audit prompt, reference, test, harness, interface, and structural overlap against the
held-out roots and register the result as a decontamination artifact.

---

## Take stock before you generate

Existing verified data is worth more than anything you will synthesise, because someone
has already paid the verification cost and usually the debugging cost too.

Establish what you hold: which datasets, how many items, which come with verifiers, and
which of those verifiers you have actually run. Synthesis expands what you have. It is a
poor way to obtain a first item.

---

## Decide the method before you build the pipeline

**Do not start from a method you have read about.** Start from the seeds you actually
hold, work out what can usefully be done to them, then try the candidates at small scale
before committing engineering to any of them.

The pipeline in the next section is expensive to build. Building it around a method that
turns out to yield nothing is the most costly mistake available to you here, and it is
avoidable with a bounded pilot whose cost is small relative to production construction.
Choose its budget from coverage needs, uncertainty, and the project envelope rather than a
fixed wall-clock promise.

### Read the seeds first

Characterise what you have before proposing anything. Seed metadata is usually richer than
you expect and is the cheapest information in the project — category, difficulty, source,
size, structure of the reference solutions, how much any of it varies.

Ask specifically:

- **which dimension is under-covered** — that is what a method has to produce, and it is
  the only reason to prefer one method over another
- **how much structure the solutions have** — this decides whether structural operators are
  even available to you, or only numeric ones
- **how long and how uniform the references are** — short, highly structured items behave
  very differently under synthesis than long discursive ones
- **what the verifier depends on** — this tells you which methods invalidate it

### What a method costs is decided by one thing

Whether it invalidates the existing verifier or any surface of the task contract.

- **Methods that do not change what the problem asks** may inherit the verifier and official
  profile digests, but must still check every changed surface: instruction--verifier
  agreement, extraction, interface, formatting, lineage, and leakage. They avoid semantic
  verifier regeneration; they do not bypass validation.
- **Methods that change what the problem asks** require regenerating the verifier, and then
  everything under *Gates* below applies to them.

The cost difference is project-dependent. Measure model calls, verifier regeneration,
gate failures, retries, and cost per surviving item in a pilot before scaling; do not assume
a fixed multiplier from the method family alone. Know which contract surface a proposal
changes before proposing it.

### Families worth considering

A menu, not a prescription. The first two preserve the verifier; the rest do not.

**Rephrase the instruction only.** Same problem, different wording. Correctness is
preserved by construction and you get no new problems — you buy robustness to phrasing and
nothing else. The cheapest thing you can run, and worth running.

**Enrich the target side.** Do not create new problems; make existing ones teach more. For
instance, derive a specification from the reference solution and place it ahead of the
answer as the reasoning the model should have produced. The verifier is untouched. This is
often the best yield per unit cost when your item count is adequate but your targets are
thin, and it is easy to overlook because it does not look like data synthesis.

**Mutate the solution, then derive the instruction.** The workhorse for producing genuinely
new problems. Mechanics in the next section.

**Evolve the instruction, then regenerate the solution.** A strong model produces a new
reference to match the evolved instruction. Passes gates less often than mutation, and
buys diversity that mutation cannot reach because it is not anchored to an existing
solution's shape.

**Compose a harder specification.** Do not edit an existing solution at all. Attach
additional requirements to a root problem's specification, then generate the entire triplet
fresh against the augmented spec. This is the only family that raises a problem's
structural complexity rather than perturbing its behaviour, and it produces items further
from their parents than mutation can.

### Operators are a catalogue you write down, not a prompt you improvise

Whichever families you pick, enumerate the specific changes as a fixed catalogue, each with
its parameter choices and the classes of problem it applies to. Improvising the change per
item gives you no way to measure which changes work, and no way to balance the final set.

Operators tend to come in two characters, and you want both:

- **perturb the computation** — shift a range, clamp a boundary, invert a region, change a
  type's width, alter a threshold
- **augment the structure** — add a handshake, add buffering, add a timeout-and-retry path,
  add arbitration between competing requestors

The second kind raises difficulty in a way the first cannot: it adds machinery the solution
did not have, rather than changing numbers it already had. It is also harder to generate
and harder to verify, which is why it is the one that gets skipped.

### Pilot before you commit

Take a small, stratified set of seeds and run every candidate method end to end with the gates
switched on. Size the pilot to cover the live seed/operator strata and discriminate useful
yield under the registered budget; do not hard-code a count. Measure:

- **yield after gates**, per method and per operator
- **how far survivors actually moved** from their parents
- **cost per surviving item**
- **which gate rejected the most**, per method — this tells you whether a low yield means
  the method is bad or your gate is miscalibrated

Then weight by what you measured, and drop what produced nothing.

Treat a null-yield operator family as a valid negative result. Derive final operator weights
from the registered pilot evidence rather than from design intuition, and verify that the
gates actually ran before interpreting unexpectedly uniform outcomes.

---

## The shape of the pipeline

Build it as a sequence of stages with an explicit state per item and an explicit rejection
reason. Each stage answers one question and can only reject:

```
seed audit          does this seed have everything an item needs?
      ↓
plan                which mutations apply to this seed, and with which parameters?
      ↓
generate            produce the triplet
      ↓
static gates        does it compile, lint, and expose the interface it declares?
      ↓
verifier strength   does the verifier actually discriminate?
      ↓
contract audit      does the instruction state everything the verifier checks?
      ↓
dedup               is this new?
      ↓
difficulty          is it solvable, and harder than its parent?
      ↓
compose             which survivors go into the set?
```

Two properties of this arrangement matter more than the individual stages.

**Order by cost, cheapest first.** Compilation costs milliseconds, a solver panel costs
minutes. Every item rejected early is capacity you get to spend on a survivor.

**Rejection is a state, not a deletion.** Give every item a status and a reason, keep the
rejected ones, and count them by reason. You will need those counts constantly — they are
how you find out that one gate is doing all the work, or that a method you believed in
yields nothing. A pipeline that reports only successes cannot be debugged, only rebuilt.

---

## Build two abstractions before writing any method

These are what make the pipeline survive contact with a second dataset, and retrofitting
them is expensive.

**One seed record, and one adapter per dataset that produces it.** Fix the fields a seed
must carry — identifier, source, instruction, reference solution, tests, whatever the
evaluator needs, and a free metadata slot. Each dataset gets a small adapter that reads its
own layout and yields that record. Everything downstream sees only the record.

**A registry of methods behind one interface.** A method takes a seed and returns some
number of expanded items. Methods are then pluggable, comparable, and share the gates.

Without both, dataset-specific handling leaks into the generation code and each new dataset
costs as much as the first. Some of the verification will genuinely be dataset-specific —
what "run it" means differs by domain — so isolate that behind one function per dataset
rather than letting it spread.

---

## Generation

### Expand from the solution, not the instruction

The obvious move is to take an instruction and rewrite it into a harder one. In a
verifiable domain this quietly destroys your data: the instruction changes, the reference
and the verifier do not, and nothing reports an error. You have manufactured items whose
parts disagree, at scale, with every gate green.

Go the other way:

```
mutate the reference solution
      ↓
derive the instruction from the mutated reference
      ↓
generate the verifier from the same mutation
```

The reference is now the thing that changed first, and the other two are derived from it
rather than left behind by it.

This is what makes the difference between the two mutation families above: evolving the
instruction is only safe when a strong model regenerates the reference against it
afterwards. Rewriting the instruction and keeping the old reference is the failure this
section describes.

### Write the mutation contract before you write anything else

Do not describe the change in prose and let two separate model calls interpret it. They
will interpret it differently, and the difference is exactly a broken item.

Write the change once, as structure, and feed it **verbatim** to both the call that edits
the solution and the call that produces the verifier. It should pin down at minimum:

- what the change is, and what its boundary is
- what property held before, and what property must hold after
- what the verifier must check in order to detect the difference
- what must not change — and always include the interface here, or the model will drift it
  and break everything downstream

Enumerate the operators explicitly and give each one its parameter choices. Then expand
seeds against them **deterministically** — derive the choice from a hash of the seed
identifier rather than sampling. You get reproducibility, and you get even coverage across
operators instead of a pile-up on whichever is first.

### Use two model roles, and keep them separate

The solution side becomes your training target. It must come from **one fixed model**, or
you are training on a mixture of styles and will not be able to attribute anything.

The instruction side is different: rotating several models there buys diversity in phrasing
at no cost to consistency, because the instruction is input rather than target.

Keep these configurable separately. They have different requirements and you will want to
change one without the other.

### Extract what has an exact form; do not generate it

Anything with an exact written form — interfaces, signatures, widths, types, polarity and
timing conventions — should be lifted mechanically from the reference, not restated by a
model. A model asked to describe an interface paraphrases it, and the paraphrase loses
precision without losing fluency. One dropped width is one word in the instruction and
total failure at evaluation.

Reserve generation for the parts that genuinely require judgement.

### Keep the solution out of the instruction

An instruction derived from a solution tends to leak the solution's structure, and an item
that names its own decomposition trains the model to follow instructions rather than solve
problems.

Lint for it. Build the blacklist from the reference itself — its internal names, its
component names — rather than from a fixed list, add the structural phrases that give away
a decomposition, and whitelist vocabulary common enough to be meaningless. Allow one
regeneration on a lint failure before giving up on the item.

**Compression is not a leak defense.** Summarising the behaviour into a few sentences does
remove structure, but it also drops rules, and it drops them precisely on the items that
carry the most rules — state tables, encodings, protocols. You lose correctness on your
most valuable items and you still need the lint. Let the lint do the work.

### Derive the instruction from the solution alone

If you intend to measure how far generated items have moved from their source, the step
that writes the instruction must not see the original instruction. Otherwise it copies,
the distance you measure shrinks, and you conclude the pipeline is faithful when you have
only measured it against itself.

---

## Gates

### Static gates

Compile, lint, and check that the artifact exposes the interface it claims. These are
nearly free and catch a large fraction. Run them first, stop at the first failure.

### Does the verifier actually verify?

**This is the gate most likely to be missing, and its absence is invisible.** Everything
else confirms the reference passes the verifier. Nothing so far rules out a verifier that
would pass anything.

Test it by breaking things on purpose. Generate mutants of the reference — targeted
corruptions of the behaviour this item is supposed to be about, plus generic ones — and run
each past the verifier. A mutant the verifier accepts is a behaviour it does not check.

Accept the item on three conditions:

- **enough mutants were valid** — if too few of them compiled, the score is noise
- **the kill rate clears a floor.** Calibrate this on your own items rather than adopting a
  number: too high rejects items whose verifier is adequate, too low admits verifiers that
  check almost nothing
- **no mutant of the class this item is about survives.** Mark those separately. A verifier
  that catches generic damage but misses the specific property the item was built around is
  precisely the useless case, and an overall rate will hide it

Without this gate, an empty verifier passes your pipeline and every item built on it is
worthless in a way no later stage detects.

### Does the instruction state everything the verifier checks?

The dual of leakage. If the verifier requires a behaviour the instruction never asked for,
you are penalising the model for not reading your mind, and no amount of training fixes it.

A model judge works here: give it the instruction, the reference, the verifier and the list
of properties the mutation was supposed to introduce, and require it to confirm every one
is both stated and checked. Require an explicit enumeration rather than a verdict — a judge
asked for a boolean returns an optimistic boolean.

### Is the change real, and bounded?

Gate the size of the change on **both** sides, and do it on both the source and the
behaviour.

- **on the source** — a lower bound catches no-ops, an upper bound catches a model that
  rewrote the whole thing instead of applying the operator
- **on the behaviour** — run the original and the mutated reference on the same inputs and
  measure how much of the output differs. A lower bound catches a change that is textually
  real and behaviourally nothing; an upper bound catches a mutation that replaced the task
  rather than modifying it

The no-op is the one that will get past you. It passes every correctness check, because it
is correct. It is a duplicate wearing a new name, and only a lower bound finds it.

### Is it solvable in the form you want?

If answers must be expressible in a particular form, check the item admits one. Rejecting
by structure before generation is far cheaper than discovering at evaluation time that
nothing valid can be written.

### Never repair on a semantic signal

This is the failure that costs the most and shows the least.

When a generated verifier disagrees with the reference, you have two choices. Discarding is
safe. Feeding the mismatch back to the model to repair is not: given the mismatch, the
model adjusts the verifier until it accepts whatever the reference actually does. You get a
reference and a verifier that agree perfectly and are both wrong, with every gate green.

The line is not how serious the error is. It is **whether the error carries information
about behaviour**:

- **syntax and compilation errors carry none.** Safe to feed back. Restate the contract when
  you do, or the repair drifts off the change you asked for.
- **semantic mismatches carry behaviour.** They are a verdict, not a repair signal. Discard.

Any repair loop with access to the reference's actual behaviour will converge on describing
it rather than checking it.

Note this is a constraint on the *loop*, not on simulation. Running the reference against
its verifier as an accept/reject gate is fine and you should do it. What you must not do is
hand the model the result and ask for a fix.

### Strict gates discard your hardest items first

Gates tuned for the common case fail disproportionately on difficult seeds, because
difficult seeds produce larger and messier changes. If you retry a fixed number of times
and then give up, the items you lose are systematically the ones you most wanted, and your
set drifts easy without any signal that it did.

Only non-semantic heuristics such as diversity or predeclared change-size bounds may receive a
registered wider final-attempt arm, and every such admission stays labelled for comparison.
Never relax task correctness, positive-verdict, leakage, split, licence, containment, or
integrity invariants to improve yield. A change to those governing rules requires the
verification/profile authority path, not an adaptive retry.

---

## Properties of the finished set

### Difficulty is measured, not assumed

"I applied a harder mutation" is not evidence. Measure it, with a fixed solver and fixed
sampling settings, so a difference in success rate is attributable to the items rather
than the setup.

Two conditions, pulling opposite ways:

- **solvable** — a strong model solves it in a small number of attempts. An unsolvable item
  is not a hard item, it is a broken one, and it will drag every aggregate it enters.
- **harder than its parent** — a solver that gets the parent fails the child. Comparing
  against the parent rather than an absolute bar is what makes this robust to the solver you
  happened to pick.

An item failing the first is broken; failing the second is a duplicate.

Which band you need depends on the stage you are feeding. SFT tolerates items the model
cannot yet solve. RLVR does not — an item nothing solves and an item everything solves are
equally worthless there, because neither produces variance to learn from.

### Deduplicate along more than one axis

Different axes catch different duplicates, and any one alone lets through families of
near-identical items that inflate your counts and skew the mix. Use several and take the
union:

- exact hashes of the instruction and of the solution — catches the trivial case
- an interface signature — catches items differing only in prose
- a structural signature of the solution — catches items differing only in naming
- n-gram overlap for near-duplicates, applied within items sharing an interface

Deduplicate against parents as well as siblings, or a mutation that round-tripped back to
its original will be counted as new.

### Composition is a decision, not a by-product

The set you train on is not "everything that passed". Compose it in this order:

1. **coverage first** — every seed that produced anything contributes at least one item,
   so the set does not silently lose whole regions to whichever seeds were productive
2. **then caps** — per seed, per (seed, operator), and per operator overall
3. **then steer the mixture** toward the ratios you want across methods and sources
4. **then fill to the target size**

Coverage before ratio, always. A ratio satisfied by over-sampling a few productive seeds
looks correct in the summary and is narrow in reality.

Record the final composition alongside the data. You will need it to interpret what SFT
does next, and reconstructing it afterwards is unreliable.

---

## Making it run

The pipeline is long, expensive, and partly non-deterministic. These are what make it
finishable.

**Check the environment before spending anything.** Write a preflight that verifies every
external dependency cheaply and prints a table: interpreter version, importable packages,
required binaries on the path, accelerators present, credentials and endpoints set,
registries loading, the model endpoint answering. Separate hard failures from warnings —
a missing simulator is fatal for one dataset and irrelevant for another. Run it before every
real run. The alternative is discovering a missing binary after an hour of generation.

**Default to a dry run.** Make the flag that spends money the one you have to set
explicitly.

**Cap each seed's wall-clock time.** Without it a single pathological seed blocks a slot
indefinitely. On timeout, record the seed and move on.

**Flush every completed seed to disk immediately.** Do not accumulate in memory and write
at the end — long runs are killed. Append under a lock and the file is always valid.

**Write failures to a sidecar file, and make it directly re-runnable.** One record per
failed seed with its identifier and the error, and a flag that takes that file as input and
retries only those seeds. Failures cluster by cause — a quota exhaustion strands a hundred
seeds that would all succeed an hour later, and without this you rerun everything.

**Retry by error class, not by count.** Rate-limit and quota errors should back off on the
timescale the quota actually refills, which is far longer than a normal retry. Transport
errors retry fast. Anything deterministic should not retry at all — it will fail
identically and each attempt costs real capacity.

**Spend cost where it discriminates.**

- *hoist* — any expensive setup shared across variants of a seed runs once at seed level,
  not once per variant
- *stage cheap-then-expensive* — a first pass with few trials and wide bounds rejects most
  candidates; only survivors pay for the precise pass. Cache the original's outputs between
  the two so it is computed once
- *keep warm workers* — where startup dominates (interpreter, accelerator init,
  compilation), hold a pool of ready processes and feed them tasks rather than paying
  startup per item
- *bail early on a dead operator* — if the first attempts show no change at all, the
  operator does not apply to this seed. Stop instead of exhausting the budget

**Keep measuring yield at scale, not only in the pilot.** Per-method and per-operator
survival rates drift once you leave the seeds you piloted on, and a method that worked on
the easy third of your seeds can collapse on the rest. Recount periodically and re-weight.

One case is worth knowing in advance because it produces no error: **on short, structured
problems, methods that rewrite at the level of wording tend to round-trip back to the
original.** The item is valid, passes every gate, and duplicates its parent. It only shows
up as lost yield if you are deduplicating against parents.

---

## Producing the training set

Surviving the gates makes an item correct. It does not yet make it a training example.

**Convert to the training format and validate the conversion.** Check every record has the
turns it needs and that none of them is empty. This is a mechanical check and it will catch
a surprising number of rows.

**Decide what the target side actually is.** The reference solution is one option. Better,
where you can afford it: sample a solution from a strong model, verify it, and use the
first one that passes. That target is verified correct *and* in a form a model produces
naturally, which the reference frequently is not. Sample a few times per item, keep the
first that passes, and record which items never yielded one — they are your genuinely hard
set and you want to know their size.

**If the target model reasons before answering, keep the reasoning.** It is the part of the
trajectory worth training on, and it is easy to discard by accident when extracting the
answer.

**Deduplicate once more at the end**, on the final pairs. Items that were distinct upstream
can collapse to the same training example after formatting.

---

## Leaving this stage

You are ready when you hold verified items covering the capability you intend to train, at
a difficulty appropriate to the stage receiving them, and you can state:

- rejection counts by reason, and which gate rejected the most
- the final composition — how much from which seeds, methods and sources
- how many items had no verified solution at all
- what you know is still wrong with the set

The last one matters. There is no point at which the data is clean, and a stage reporting no
known defects has usually not looked. Carry the known defects forward as an explicit list,
so that when a downstream result looks strange you check that list before concluding it is
the model.
