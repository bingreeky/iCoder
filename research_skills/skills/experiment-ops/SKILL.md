---
name: experiment-ops
description: Running work on shared compute — launching, monitoring, attributing failures, and acting safely on things already running. Load before launching a long job, when a job has crashed or appears stuck, when a fix does not seem to have taken effect, and before any irreversible operation.
---

# Experiment operations

This is about the machinery rather than the science. It applies in every stage, and it is
where a large share of elapsed time goes if it is handled badly — not in compute, but in
runs that had to be repeated for reasons unrelated to what they were testing.

Enter through an active controller context and queue task. If invoked directly
without them, load `auto-post-training` and bootstrap or resume the run before
launching, modifying, or monitoring project work.

While PROJECT_AUDIT still uses bootstrap authority, restrict this Skill to
reading supplied policy and local read-only inventory. Do not submit timing,
queue, image, or resource probes until the capability lock and an authorized
post-bootstrap task exist.

The organising idea: **the cost of a mistake here is set by how long it takes to surface,
not by how serious it is.** A configuration error that stops the job in ten seconds is
free. The same error surfacing after eight hours costs a day, and it costs it twice if you
resubmit without finding the deeper cause.

---

Bundled notes hold calibrated platform and run-hygiene references. Treat them as versioned,
read-only inputs and load them only when their environment matches the capability lock. A live
run records newly observed platform behavior in evidence-bound learned memory outside the
installed Skill.

Load [compute platform](notes/compute-platform.md) only when an eligible note binding names
that exact platform implementation. Load [run hygiene](notes/run-hygiene.md) only when the
active launcher, scheduler, and storage assumptions match its binding. A `legacy-unbound`
note cannot guide execution until the approved project profile and context packet supply the
provenance binding required by runtime records. A future Skill release may instead move a
reviewed note into its static `prior_files` list.

---

## Learn the rules before you plan around them

Every cluster has house rules, and they are only partly written down. The written part tells
you how to submit work; the unwritten part tells you what will actually happen when you do.

**Split the question in two: ask what cannot be measured, measure what can.**

Ask a person — whoever has been running work here — for the things that are policy rather
than behaviour:

- what your allocation actually is, and who else draws on it
- whether there is a hard limit on how long one piece of work may run, and whether it can be
  extended or only restarted
- which storage is durable, which is scratch, what is quota'd, and what gets cleaned up
- what is considered antisocial: which resources not to occupy, what needs announcing, whose
  work you could break
- who to tell when something is wrong with the platform rather than with your job

Then measure the rest yourself rather than asking for estimates, because behaviour differs
from policy and the answer you get will be an average of someone's memories:

- **start-up cost** — submit a job that does nothing and time it from submission to first
  line of output. That number sets your iteration unit for the rest of the project.
- **queue response** — submit the same trivial job at two different allocation sizes and
  compare the wait. This tells you whether testing small is worth it here, which is usually
  the single most useful operational fact.
- **what the environment actually contains** — print the mounts, the visible devices, the
  library paths and the tool versions from inside a real job, rather than trusting the image
  description.

All of it is cheap, and all of it is the kind of thing you would otherwise learn by having a
week-long run die.

**Write down what you find**, including the date, in run-scoped learned memory. Clusters
change without announcement — a quota is resized, an image is updated, a mount moves — so a
record of what was true, when, and under which capability lock lets you notice that something
moved rather than concluding your code broke.

Bundled platform references are examples of calibrated knowledge, not writable state. For a
different platform, create a scoped memory record rather than adapting the installed file;
the specifics will not transfer, though the questions do.

## Know which execution tier you are in

Discover the execution tiers and actions present in the capability lock; do not assume a
particular scheduler model. A platform may expose interactive machines, batch jobs, both, or
neither. When both exist, their iteration costs commonly differ:

- an **interactive** tier — you hold a machine and iterate in seconds
- a **batch** tier — you submit a job, and every change costs a full start-up before your
  first line runs

Repeatedly debugging through a high-startup tier is expensive. Each iteration may pay image
pull, scheduling, and environment setup before reaching the error.

**Use the smallest authorized representative path until the workflow runs clean, then submit
the production path.** If an equivalent interactive tier exists, use it for fast diagnosis.
If only batch exists, use a minimal queued smoke rather than inventing an unavailable tier.

Two consequences worth planning for:

**Different tiers may not be the same environment.** When more than one locked path exists,
compare mounts, paths, shells, devices, images, and inherited variables. Treat translation
between paths as a separately smoked step rather than assuming equivalence.

**Capture the environment once it is right.** If image or snapshot creation is supported and
authorized, register that immutable artifact. Otherwise register the exact existing image,
package lock, launcher configuration, and observed environment digest; do not mutate platform
state merely to follow this guidance.

For any leased resource, discover lifetime and renewal/termination policy before depending on
it and record the expiry. If no such resource exists, this check is not applicable.

## Before you launch

Sort what can go wrong by when it would appear.

Anything that fails in the first seconds — a missing path, a malformed argument, an import
error — needs no special care. Launch and let it tell you.

Anything that can only fail late deserves work up front. Before a long run, verify locally
everything that can be verified locally: that inputs exist and parse, that outputs can be
written where you expect, that the shapes match, that a single step completes. A short run
of the real thing at reduced scale is worth more than any amount of reading.

**The question to ask is not "is this correct" but "if this is wrong, when will I find
out".** Only the second one tells you how much verification to buy.

---

## Launching

**The launch path can have preconditions that have nothing to do with your job.** A
launcher may check repository cleanliness, environment state, or quotas before it will
start anything. When one of these is unmet, the symptom is that jobs stop starting — which
looks like a scheduling problem and is not. Bind the run to an authorized immutable code
artifact or a verified clean worktree. Commit only files already in the task's mutation scope
and only when that repository change is explicitly authorized; never sweep unrelated user
changes into a launcher workaround. If you changed the environment, check whether the
launcher inspects it.

**Every resource dimension you request can block you.** Over-requesting any one of them —
accelerators, CPU, memory, shared memory, local storage — leaves the job queued
indefinitely, and the queue does not usually tell you which dimension is unsatisfiable.
When a job will not schedule, check every dimension of the request against what the
partition actually offers, rather than assuming it is the obvious one.

---

## While it runs

**Do not modify anything a running job reads.** Scripts are commonly read incrementally
rather than loaded whole, so editing a file that a running job is executing corrupts that
execution in ways that produce confusing failures far from the edit. Copy, edit the copy,
launch the copy.

**Know how a fix would take effect before concluding it did not work.** There are at least
two mechanisms and they behave oppositely:

- *read at runtime* — files the job loads when it starts. A mutable path could make a queued
  job pick up an unregistered change; this is prohibited. Cancel or hold it, register a new
  immutable code/config artifact and successor task/experiment as needed, then submit a new
  attempt.
- *fixed at submission* — arguments, resource requests, environment captured at submit
  time. A change still requires a registered artifact/config and authorized successor or
  operational retry; editing source never retroactively changes queued work.

Guessing wrong produces the most expensive symptom in this whole area: **the fix appears
to have been applied and the old behaviour continues**, with nothing anywhere reporting a
problem. Establish which mechanism applies before you spend time doubting the fix itself.

**Queued is not stuck.** Long waits on shared compute are usually genuine contention, and
resubmitting throws away accumulated queue position — often making the wait longer while
appearing to be action. When you cannot tell the difference, first register and authorize a
bounded probe task: a job of the same shape and resource request running a trivial command.
If the probe schedules, your
request is satisfiable and you are waiting. If it does not, the request is the problem.
Measure it rather than guessing.

Since queue time scales with what you ask for, **request the smallest allocation that can
reproduce the problem** when you are testing rather than producing. A test that schedules
immediately at low resource beats an accurate one that waits an hour.

**An inventory call that returns nothing is not proof that nothing exists.** Listing APIs
on these platforms are frequently incomplete, paginated in surprising ways, or filtered by
something you did not set. Before concluding a job or instance is gone, query it directly by
identifier. Acting on a false empty — recreating something that already runs, or assuming a
resource was released — is expensive in both directions.

**And a non-zero exit code is not proof that a command failed.** Cluster bring-up commands
routinely time out client-side while the thing they started comes up fine a few seconds
later — most often on a cold or loaded filesystem, which is exactly when you are already
suspicious. Tearing down and retrying on that signal is how a working cluster gets destroyed
repeatedly.

**Verify the state you wanted, not the command that was supposed to produce it.** After a
bring-up step, poll for the observable consequence — the coordinator answering, the expected
number of workers registered — with a bounded number of attempts, and fail loudly only if
that never arrives. Then the control plane's own reporting stops being on the critical path
in either direction.

**Some states do not accept the action you want.** A resource still queuing often cannot be
stopped, only deleted; one mid-transition may accept neither. When an action is refused,
read the state before retrying the action.

---

## When it crashes

**Do not fix only the last line.** The final error is frequently a consequence rather than
a cause, and a job that crashed at hour eight will crash again at hour eight if you patch
the symptom and resubmit.

Read backwards from the end until you find the first thing that was not normal. Check
whether the failure is reproducible cheaply. Then decide what to change — once, and with a
reason.

Two failures worth ruling out early, because they masquerade as everything else: running
out of memory (frequently reported as something unrelated) and running out of disk (which
tends to surface as a corrupted write much later than the disk actually filled).

---

## Checkpoint and validation strategy

This is a design decision made before the run, not a default to accept.

**Save frequency and retention policy together set the upper bound on what a crash costs
you.** Decide that bound deliberately: how much progress are you willing to lose, and how
much storage will the retained checkpoints consume over the full run.

**Validation frequency should be an integer multiple of save frequency.** Otherwise a
share of your validation points refer to weights that no longer exist, and the most
interesting one always turns out to be among them.

Confirm early — from the filesystem, not the configuration — that checkpoints are actually
being written where you expect and are loadable. A run that saved nothing usable is
discovered at the end, which is the worst possible time.

---

## Before anything irreversible

Deleting data, stopping long-running work, overwriting a directory: establish what the
target actually is and who is using it before acting, not after.

If what you find does not match how it was described to you — a directory that holds more
than expected, a job that something else appears to depend on, a file you did not create —
stop and report rather than proceeding. The description being wrong is itself the finding.

Prefer moving to deleting where possible. It costs storage and buys back the mistake.
