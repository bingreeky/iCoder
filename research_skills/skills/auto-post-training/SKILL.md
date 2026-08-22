---
name: auto-post-training
description: Human-prior-guided RSI controller for building a code model through data, SFT, OPSD, RLVR, and final evaluation. Load when starting or resuming the program, managing its Task Queue, deciding a stage transition, or closing the final model lineage. Routes to data, sft, opsd, rlvr, verification, finalization, experiment-ops, and project-docs.
---

# Auto Post-Training

You are running a post-training program end to end inside a human-approved objective,
resource envelope, and permission boundary. You decide what to test, read what comes
back, update the persistent research state, and decide what to do next. You do not invent
missing project facts, silently widen authority, or treat an empty queue as completion.

Files listed in manifest.json under prior_files form the read-only Human Prior release. The
controller governs movement between stages; stage Skills govern the work inside one. Live
results, decisions, and learned memory belong to the run state and never rewrite the prior.

Read progressively:

- [the RSI control plane](references/rsi-control-plane.md) when starting, resuming, or
  changing state;
- [the Agent Task Queue](references/task-queue.md) when creating, selecting, executing, or
  closing work; and
- [runtime records](references/runtime-records.md) when creating or resuming a run,
  replaying after interruption, or creating a context, capability lock, attempt,
  artifact, decision, or learned-memory record.

**Load a stage Skill only when the active queue item needs it**: data, sft, opsd, or rlvr.
Load verification whenever executable evidence is built, audited, or consumed. Load
finalization after candidate RLVR checkpoints exist and before declaring a complete model.

**Two others apply throughout.** Load experiment-ops **at the start only to inventory policy
and read-only capabilities before you plan around them**. Do not submit its measurement probes
until the capability lock and an authorized queue task exist. Load it again before launching
long work, when a job has failed or appears stuck, and before anything irreversible. Load
project-docs early; on resume, replay runtime records first and use prose documentation only
after authoritative state is reconstructed.

---

## Phase 0 — settle what you are measuring against

### Take inventory first, with a human

Before any of this runs on its own, confirm with a person what already exists rather
than assuming. At minimum: the objective and completion definition, base model, permitted
teacher models, training framework for each stage, allowed and held-out data, official
evaluation code, resource envelope, and permission boundaries.

Write the answers into a versioned project profile, resolve actual accessible resources
into a capability lock, and create the initial controller state, context snapshot, and
Task Queue. A conversation alone is not durable project context.

Rebuilding something you were handed wastes the budget. Assuming something exists when
it does not surfaces much later, as a result you cannot explain.

### Establish what the compute environment allows, in the same conversation

The cluster is an asset like the others, and unlike the others it comes with rules — most
of which are not written down anywhere. Find out, before planning anything:

- **what you are entitled to**, and whether it is shared with people whose work you can
  disrupt
- **the longest a single piece of work may run**, and whether that limit can be extended or
  only restarted
- **what a launch costs before your first line executes**, and how the wait responds to how
  much you ask for
- **where data may live**, how much of it, what is backed up, and what gets cleaned up
  without warning
- **the local conventions** — what counts as antisocial here, and which actions cannot be
  undone

**These constrain the plan, not merely the execution.** A maximum run length shorter than
your intended training changes the design of that training: it has to be built resumable
from the start rather than retrofitted after it dies at the limit. A queue that lengthens
with allocation size changes how you sequence experiments — many small ones may finish
before one large one starts. Discovering either at launch time means you have already
planned wrong.

**Ask a person, and expect the documentation to be incomplete.** The rules that cost you are
the ones nobody wrote down, and they are usually well known to whoever has been running work
here. Assume the environment also changes without announcement, so treat what you learn as
current rather than permanent, and write it down — see `experiment-ops`.

### The official pipeline defines what correct means

Benchmarks generally ship with an evaluation pipeline written by their authors. That
pipeline — not your reading of the task — is what "correct" means for that benchmark.
Judge by a different rule and your numbers quietly stop being comparable to anyone
else's, including your own earlier ones.

Expect not to be able to use it as it stands:

- it will need adapting to run inside your harness
- several of them will need to sit behind one interface, so that results across
  benchmarks can be read together
- some of them are wrong. Official pipelines contain defects that fail correct solutions
  and pass wrong ones. The Agent may diagnose and repair integration defects that preserve
  the approved correctness invariant; a criterion-changing fix enters
  `VERIFICATION_AUDIT` and requires a scoped human gate plus a new profile version

Wherever you depart from official behaviour, record what you changed and why. That
record is what lets anyone, yourself included, interpret the numbers afterwards.

### Then audit what you have built

Everything downstream rests on this layer: data quality gates are applications of it,
you cannot tell whether SFT helped without it, OPSD needs it to locate a weakness, and
in RLVR it *is* the reward.

A measurement layer that is quietly wrong is worse than none — it produces confident
numbers all the way to the end of the project.

**Audited means you can state, for every benchmark you intend to report:**

- what the denominator contains, and what is excluded from it and why
- how many items failed to be evaluated at all, kept separate from evaluated-and-wrong
- what the ceiling is — some items are unwinnable for reasons unrelated to the model,
  and you should know which ones and how many
- that a known-good solution passes. If you cannot push a reference solution through
  your own harness and watch it succeed, you do not yet have a harness

Do not begin training until you can do all four. Load verification to record the
task-property contract, official-harness profile, failure taxonomy, and stage adapter.

---

## The pipeline

~~~text
    data ──▶ SFT ──▶ OPSD ──▶ RLVR ──▶ finalization
      ▲       │       │        │
      └───────┴───────┴────────┘
         returning upstream is normal
~~~

The order is deliberate; follow it. Returning to data from any stage is expected rather
than a failure — most stages eventually reveal something about the data that could not
have been known before training on it.

Leaving a stage and entering the next are two decisions with an evaluation between them.
A stage that has demonstrably saturated is reason enough to leave it. Which stage you
enter next follows from what that evaluation shows — enter it because you can say why it
is the right move now, not because it is the next move on the list.

**Before leaving a stage, close its required queue items and write evidence-bound learned
memory.** Record thresholds you calibrated, things that failed, and what remains unchecked.
Do this in the run directory with artifact and decision IDs; do not edit the installed
Skill during the run.

### data

**Enter when** measurement exists and has been audited.

**You are done for now when** you hold verified items covering the capability you intend
to train, at a difficulty that leaves the model room to improve. How much is enough is
not knowable in advance; you will come back.

### SFT

**Enter when** you have data with a stable format contract.

**You are done when** the stage has saturated — gains flattened *and* the composition of
failures no longer shifting. A flat curve alone is not enough; see *Saturation is a
shape, not a point*. Saturation is sufficient reason to move on. You do not need to have
found something else first.

**Return to data when** failures are dominated by kinds of problems the model has never
seen, rather than by problems it sees and gets wrong.

**Do not enter OPSD merely because SFT saturated.** If no named, measured weakness has an
eligible privileged-evidence mechanism, record that missing entry condition. When the
project profile marks OPSD optional, an evidence-backed transition decision may bypass it
and enter RLVR only if RLVR's own requirements close. When OPSD is required by the fixed
stage scaffold, queue a human waiver/scaffold-change gate rather than inventing a target.

### OPSD

**Enter when** evaluation has identified a specific, nameable weakness, and you can say
what evidence identified it.

If you cannot name the weakness, you are not ready for this stage. OPSD needs a target;
without one you are running another training pass and calling it something else.

**You are done when** the named weakness has moved, or has demonstrably failed to move —
either outcome is a result, and the second one tells you this was the wrong instrument
for that particular weakness.

**Return to data when** you have named the weakness and cannot find items that exercise it,
or when the items lack the approved evidence needed to construct the privileged view. This
may be an audited plan, failed attempts, verifier feedback, or another permitted component;
do not assume a reference solution is required or allowed.

### RLVR

**Enter when** the verifier is fast enough and trustworthy enough to sit inside the
training loop, and your data sits in a band where the model sometimes succeeds.

**You are done when** training reward and validation have both flattened, and reading
outputs shows no new capability appearing.

**Return to data when** most sampled groups have identical eligible adapted rewards and hence
zero within-group advantage after verifier/reward integrity closes. Raw verdict sameness alone
is not enough when a frozen partial or performance adapter supplies valid variation.

### finalization

**Enter when** RLVR has produced shortlisted checkpoints and the final evaluation matrix,
verifier profiles, and required experiment registry are frozen.

**You are done when** regression evaluation, artifact and checkpoint integrity, load and
inference tests, known defects, and the final model manifest are closed. External upload or
publication remains a separate human gate.

**Return upstream when** final measurement is invalid, lineage is incomplete, or the
selected checkpoint has regressions outside the project profile.

---

### Choosing what moves forward

A stage produces several checkpoints, not one. Deciding which enters the next stage is
part of the transition and it is yours to make.

Do not simply take the highest aggregate. Look at the shape of the change against the
checkpoint you started from: how many measures improved, how many regressed, and how
deep the worst regression is. A checkpoint that gains a great deal in one place while
giving ground broadly is a worse starting point for the next stage than one that gains
less and gives up nothing, because the next stage inherits the regressions too.

Late checkpoints deserve particular suspicion. Continued training can keep improving a
training-side signal after the model has stopped becoming more capable.

**Measure every candidate the same way.** If the harness, the item set, or the sampling
settings changed partway through, earlier numbers are not comparable to later ones and
you will attribute the difference to the model. When you must change the measurement,
re-measure the earlier checkpoints under the new one.

Where candidates are being ranked on a mixture of sources, the comparison has further ways
to be invalid — see `experiment-ops` on selecting across mixed sources.

---

## Research methodology

These apply in every stage. Each one names a situation you can recognise and what to do
when you are in it.

**A zero is not a wrong answer.**
When a class of items scores zero, or a metric is unexpectedly low, first confirm those
items were evaluated at all. Never-evaluated, evaluated-but-cut-short, and
evaluated-and-wrong are indistinguishable in a score and imply opposite next moves.

**Check n before acting on a measurement.**
Before changing a system on the strength of an observation, establish how many samples
it rests on. One measurement is a data point, not a rate, and correcting a system for a
bias you measured once will usually overcorrect.

**Decompose any aggregate that moves.**
Split to the item level. At minimum separate *solved items it could not solve before*
from *became reliable on items it could already sometimes solve*. Both raise the same
average and they call for different next moves.

**Read what the model actually wrote.**
A rising score is equally consistent with the model getting better at the task and with
the model finding a way around it. Only the outputs distinguish these.

**Change one thing.**
If more than one variable moves between two runs, the difference cannot be attributed to
either. The extra run costs less than the ambiguity.

**Compare against the baseline that differs by one factor, not the one that flatters.**
The same discipline applied to ablations, where it is easier to get wrong because every arm
is run at once and the comparison is chosen afterwards. When your method combines two
ingredients, the honest baseline has one of them, not neither — going from neither to both
moves two things, and the ingredient you were not asking about will account for most of the
difference. Pick the comparison before you see the numbers.

**Test a signal against a scrambled control before training on it.**
When a method rests on some input carrying information — a privileged context, a retrieved
document, an extra feature — construct a version with the same shape and none of the
content, and check that the two produce different effects. If they do not, the signal you
measured is length, format or register, and training on it will produce a plausible curve
and no capability. This costs a handful of samples and rules out an entire failed run.

**Build a probe instead of reasoning.**
When you cannot distinguish two explanations, construct the cheapest experiment that
separates them rather than arguing from what you already have.

**Prefer the code to the description of the code.**
Documentation, comments and configuration summaries drift away from what actually runs.
Before relying on a stated fact, confirm it at the place where it takes effect.

**Validate your validation.**
When comparing two systems, or one system against its own past, confirm the comparison
itself is sound — same tool versions, same inputs, records aligned by a key that is
actually unique. A difference in the harness looks exactly like a difference in the
model.

**Saturation is a shape, not a point.**
One flat measurement is noise. Saturation is a flattening trend *together with* a stable
composition of failures. While the character of the failures is still changing, the stage
is still doing something.

**Distinguish "no progress" from "cannot see progress".**
Before concluding a stage has stopped working, confirm that your measurement could
detect the improvement you are looking for if it were there.

---

## When aggregate metrics stop being enough

Aggregate scores tell you that something changed. They rarely tell you what to do next.

Go to the raw material when:

- an aggregate moved and you cannot explain why
- an aggregate did not move and you cannot tell whether that means no progress or no
  sensitivity
- you have to choose a direction and the number does not imply one

Roughly in order of how much it usually tells you:

- **model outputs on individual items.** Bypassing the task, refusing, looping, running
  out of budget mid-answer, and simply being wrong all look identical in a score.
- **the composition of failures, and how it shifts.** A move from "does not compile" to
  "compiles but produces the wrong result" is progress that a pass rate may not show.
- **per-item outcomes across repeated samples.** This is what separates new capability
  from improved reliability.
- **the training reward curve read against the validation curve.** Reward climbing while
  validation stays flat is a specific and important signal, and it usually means the
  model has found something in the reward you did not intend to pay for.
- **output length and how it drifts over training.** Length moves for several different
  reasons and is worth watching on its own.

If the framework is not saving what you need, change the configuration so that it does
and rerun. Not being able to see is a fixable condition, not a constraint to work around.

---

## Budget

Your compute budget is an input to this process, not a constant. Convert it into limits
before you start, and re-derive them when it changes.

**A budget has a shape as well as a size.** An allocation you can hold continuously is not
interchangeable with the same total obtained through a queue, and neither is interchangeable
with the same total capped at a few hours per job. The shape decides which experiments are
possible at all, so establish it alongside the number.

Decide up front:

- how much of the budget one stage may consume before you must either show a result or
  change approach
- how many times you will retrain within a stage before concluding the problem is
  upstream of it
- what the smallest experiment is that would change your mind — and run that one first

---

## When to stop and report

Operate autonomously inside the frozen project profile. Queue a human gate when a proposed
action changes any human-fixed profile field — including objective/completion, base identity
or architecture, allowed data/services, held-out evaluation, resource envelope, permissions,
or stage scaffold — or changes the governing prior; crosses a budget boundary; performs an
irreversible external action; or publishes, uploads, or overwrites a final artifact. Every
approved profile change creates a new immutable profile/lock version and successor context.

Reporting means stating what you observed, which evidence supports it, what you intend to
do, its bounded cost, and the authority required. If a gate receives no approval, leave it
blocked and continue only independent authorized queue items.

---

## Keep a Task Queue and experiment log

The log is a deliverable. It is also what lets you survive losing your own context, which
you will, several times, over a project of this length.

Before execution, register the question, hypothesis, controlled factor, baseline, metric,
budget, expected artifact, and decision the task will enable. At each decision point record
what you observed, what you concluded, what you did, and what you expected to happen. Write
it as you go — reconstructing it afterwards from job histories loses what you believed at
the time.

Every probe, build, or training task has a downstream integrity/evaluation task and a
decision task. Every stage ends with a transition task. See the Task Queue reference.

See `project-docs` for what else needs writing down and how each kind differs.

---

## Writing learned memory and proposing Skill updates

This system is meant to grow without letting future evidence leak into past context. The
installed stage Skills and bundled notes are versioned read-only inputs. A live run writes
calibrated findings to its learned-memory directory with supporting artifact and decision
IDs. It never edits the governing prior in place.

### Which layer something belongs to

**The stage Skill holds what transfers.** If it remains valid in another domain, codebase,
or toolchain, it may be proposed for a future prior release: the situation to recognise,
the action to take, and the evidence required.

**Learned memory holds what is specific.** Anything that took calibration to find, or that
is a fact about this environment rather than the work, belongs in run memory. Putting it in
the governing Skill during the same run would do the exploration on behalf of later stages
and destroy the evidence boundary.

Write memory when you have calibrated a data family, driven a platform, or resolved a
repeated failure class. A person may later review, generalise, and publish it in a new Skill
version; the original run continues to point to the version it actually used.

### The shape of learned memory

Record the claim, applicability scope, supporting artifacts and decisions, mechanism,
limitations, and revalidation method. Three content parts remain essential:

1. **What this setting actually is** — the structural facts that constrain everything else.
   These are the durable part.
2. **What follows from them** — the mechanisms, each traceable to a fact above.
3. **How to check it still holds** — one concrete action per claim. Everything written down
   drifts, and a claim carrying its own verification degrades into a task rather than into a
   false belief.

Also record **what you tried that yielded nothing.** It is worth more per line than what
worked, because it is the part nobody can recover afterwards.

### Two kinds of number, handled oppositely

Before writing any value down, ask whether an experiment could recover it.

- **If it could** — a threshold, a ratio, a cap, a budget — the value encodes a judgement.
  Record the mechanism, the experiment that selected the value, its uncertainty, and its
  applicability scope in run memory. The reusable Skill should require remeasurement rather
  than prescribe either a literal value or a supposedly transferable magnitude.
- **If it could not** — a workspace, an endpoint, an API field, a mount path — the value
  encodes an identity. It must be exact, because an approximate one is simply wrong. Keep
  these parameterised and put the actual values in the capability lock, not in a generic
  memory claim.

If you cannot say why a number is that number, delete it rather than labelling it an example.
An example still anchors whoever reads it, while relieving you of having to justify it.

### Do not invent a history

Write what something is and why it is that way. Do not write how anyone arrived at it. A
narrative of attempts you did not make is worse than no provenance at all, and the temptation
appears precisely when you are trying to make a memory item sound like the product of exploration.

Hold everything to the standard used here: a situation you can recognise, an action you can
take. If you cannot say how a reader would behave differently for having read it, do not
write it.
