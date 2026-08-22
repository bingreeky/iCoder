# Kernel seeds

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding in the Stage Context Packet. A live
> run writes new findings to run-scoped learned memory.

Specifics for generating from seeds whose reference is a program to be rewritten into
accelerator kernels. The general method is in the [Data Skill](../SKILL.md).

---

## What these seeds are

Not a problem and an answer. Each seed is a module containing a model class plus two
functions that produce its constructor arguments and its inputs. The harness imports the
module, builds the model from those functions, and runs it.

The consequence that shapes everything else: **the harness reaches into the reference by
name.** The class name, the forward signature, the constructor signature, and the bodies of
both input-producing functions are part of the contract with the evaluator, not part of the
answer. A mutation that touches any of them yields an item that still imports, still runs,
and fails at evaluation for reasons unrelated to the mutation.

Models rename things when asked to modify them. The first gate exists for this.

Seeds also carry a difficulty level and a problem identifier. Stratify the pilot by level —
sampling the first N seeds gets you the easy end, which behaves differently under mutation
than the rest.

---

## Operators

Two levels, and the split matters more than the individual operators.

**Model level** — change what the computation does: offset a range, clamp a boundary,
invert or shift a region, scale a subregion, inject a bias, perturb an epsilon, swap a
threshold.

**Kernel level** — change how the computation is organised: shift a constant, repartition
the program-id space, offset the block mapping, widen a dtype, post-process in the wrapper.

Kernel-level operators produce harder items and fail gates more often, because they break
launch configuration in ways that surface only at runtime. Decide the mix knowing both.

**Do not chain to a fallback operator when one exhausts its attempts on a seed.** The usual
cause is that the operator does not apply to that seed at all, and rolling to the next one
buys several more expensive attempts at the same conclusion. Detect it instead: if the first
attempts produce approximately zero difference, stop. A small attempt budget plus early
detection beats a large budget.

---

## Gates, in order

Each is cheaper than the next, and the spread is wide — the last costs on the order of a
thousand times the first. The ordering is not cosmetic.

**1 — interface identity.** Class name, forward signature, constructor signature, and both
input-producing function bodies, byte-for-byte against the parent. Free, and it rejects the
most.

**2 — source difference, bounded both ways.** Character-level diff computed on the *forward
body only*, after parsing and re-emitting the source. Diffing the whole file does not work:
reformatting and comment churn swamp the signal.

On the final attempt, **relax the lower bound by roughly an order of magnitude** rather than
losing the seed. A tight lower bound rejects hard seeds systematically — on those the model
makes a small, careful change, which is what you want and what a lower bound punishes.

**3 — it runs.** Import, construct, one forward pass, under a short timeout. Cheap insurance
before anything expensive. **Cap concurrency here at the number of accelerators**: this stage
is cold-start bound, and oversubscribing produces driver-level contention rather than
throughput.

**4 — expressible in the target form.** AST walk over the forward body, rejecting
comprehensions, unbounded loops, and anything that pulls a tensor back into host code
(`.item()`, `.tolist()`, `.cpu()`, `.numpy()`). Such items are valid in the source language
and cannot be written as the kernel being asked for. Rejecting them here is much cheaper
than discovering it when solvers repeatedly produce nothing valid.

**5 — behaviour changed, and not too much.** Run parent and child on the same inputs; count
the fraction of output elements differing beyond tolerance.

Run this in two passes rather than one:

- **cheap pass** — a few trials, bounds wide enough to reject only the clearly-identical and
  the clearly-unrelated
- **full pass** — several times the trial count, tight bounds, survivors only

Cache the parent's outputs by a hash of its source between the two, so the parent is
evaluated once per seed rather than once per variant.

**The full pass needs a timeout far above what its extra trials imply.** Accelerator
initialisation dominates, so it times out on cold start rather than on compute. Scaling the
timeout with the trial count is the wrong model and will cost you a run.

### Where the dict-returning variants differ

For seed families whose harness returns a keyed result rather than a tensor, gate 5 counts
differing keys instead: require at least one key to change, and well short of all of them.

These need **a tolerance around two orders of magnitude looser** than elementwise tensor
comparison. Reduction order differs between implementations, and a tolerance tight enough
for the tensor case rejects correct rewrites. Gate 2's bounds also run wider on both ends
here.

---

## Cost

Cold-start compilation runs tens of seconds and dominates everything. Three consequences,
each worth more than any generation-side change:

**Warm workers.** A pool of processes, one per accelerator, each pinned and each having paid
initialisation once before accepting work. Per-task subprocesses pay it every time.

**Hoist per-seed work out of the per-variant loop.** Anything shared across a seed's variants
— reference smoke tests especially — runs once at seed level. Running a smoke test per
framing serialises on one accelerator and exceeds the per-seed cap, which fails *every*
variant of that seed. In the aggregate that reads as the method failing rather than the
budget being wrong.

**Cap per-seed wall clock, and set it per dataset.** The heaviest seeds need two to three
times the median. A single global cap either wastes time on the light datasets or truncates
the heavy ones.

---

## Checking these still hold

The structural facts above are more durable than anything derived from them, but they are
not permanent.

1. **Confirm the harness still reaches into the reference by name.** Gate 1 exists entirely
   because of this. If the harness changes to accept a declared entry point, gate 1 becomes
   unnecessary and is only costing you items.
2. **Measure cold-start cost directly.** Everything under *Cost* rests on it, and a toolchain
   upgrade can invalidate all of it at once.
3. **Confirm gate 4's reject list still names real constraints** — what is expressible in the
   target form changes with the target's version.
4. **Disable the lower bounds for one run** and count the no-ops that get through. Near zero
   means the seeds have changed character and the bounds are now only costing items.
