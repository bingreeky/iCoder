# Multi-node Megatron-style SFT

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding in the Stage Context Packet. A live
> run writes new findings to run-scoped learned memory.

Specifics for full fine-tuning a large model across nodes on a shared cluster with a
tensor-parallel trainer. The general method is in the [SFT Skill](../SKILL.md).

**Almost everything that goes wrong here fails as a hang, not as an error.** Budget
your debugging accordingly, and build the guards before you need them.

---

## Where the parallelism layout is allowed to sit

The layout is not just a throughput decision. Whether it runs at all depends on where its
communication lands relative to machine boundaries.

**Tensor parallelism must stay inside a machine.** It is the most communication-intensive
axis and it assumes an internal interconnect. A layout that spreads it across nodes will
either crawl or hang.

**Pipeline parallelism is the axis that breaks when you scale out.** Doubling node count
usually keeps the product of the intra-node axes constant and pushes pipeline stages across
the boundary. Point-to-point exchange between stages is then travelling over the cluster
fabric, and on a cluster whose fabric is not healthy this hangs — inside the send/receive
exchange, with no error.

The consequence is counter-intuitive and worth stating plainly: **a layout that works at one
scale can be impossible at twice the scale, and the fix is to remove pipelining entirely
rather than to tune it.** Absorbing the extra ranks into data parallelism instead costs
nothing in correctness and usually gains throughput, because data-parallel reduction is a
much friendlier communication pattern than staged point-to-point.

When a job hangs, get a stack sample from a worker before changing anything. It will name
the collective, and the collective names the axis.

---

## Sub-group collectives can work while world-group ones hang

This is the failure that will cost you the most time, because every partial test passes.

The parallelism groups — tensor, pipeline, data — each form their own communicator, and
those can be entirely healthy over a high-performance fabric. The first collective that
spans *every* rank is a different communicator, and it can fail to form when per-node device
naming is inconsistent or a vendor plugin ignores the device selection you configured.

Symptom: training initialises, sub-groups exchange fine, and the job hangs at the first
global operation — often something incidental like an object gather in the optimiser setup,
which sends you looking in the wrong place.

**The reliable workaround is to force the whole job onto plain sockets over a named
interface**, disabling the high-performance transport and its plugin. It is slower per
collective and it completes. Take the throughput loss and open a ticket; this is
infrastructure, not your configuration.

Before spending a day on this, test the hypothesis directly: a tiny script that forms a
world-group collective across all nodes reproduces it in seconds and tells you whether to
keep looking at your training code at all.

---

## Preprocess the dataset outside the distributed context

Frameworks commonly build a tokenised cache on first use, guarded by a barrier so only one
rank does the work. On a cluster where the world-group collective is unreliable, or where
the cache lives on shared storage with cross-node locking, that barrier is exactly where you
deadlock — before a single training step.

**Build the cache in a separate single-process run**, with no distributed initialisation at
all, then distribute the result to node-local storage on every node and point the training
job at the local copy. Every rank then hits a warm cache, no barrier is needed, and the
cross-node lock never comes into play.

This also makes the cost visible. Tokenising a large corpus is not free, and hiding it
inside the first training step makes every launch look slow for reasons nobody can see.

---

## Both launch paths need the environment built the same way

A non-interactive remote shell does not inherit the environment an interactive one gets.
Library paths in particular are commonly set by a profile that will not run, and the symptom
is a worker reporting a driver or runtime version mismatch while the same command works when
typed by hand.

Have the launcher re-establish the environment explicitly, and have the training script do
it again. The redundancy costs nothing and removes a class of failure that only ever appears
on the worker nodes, which are the ones you are not watching.

---

## Guards worth having before you need them

**Cap the size of anything handed to an external tool.** A model that degenerates into
repetition emits until it hits the token limit, and the result can be orders of magnitude
larger than any legitimate answer. Refuse it by size before invoking a compiler or simulator
rather than discovering it as a hung verification worker.

**Kill process trees, and tolerate the race.** Verification subprocesses spawn children.
Killing the parent on timeout leaves the children holding the resource. Kill the tree, and
write the kill to tolerate a target that has already exited — under concurrency that race
happens constantly and otherwise surfaces as a spurious verification error.

**Sweep stale temporary directories on a age threshold.** Killed workers leave them behind.
Sweeping by age rather than unconditionally avoids racing a verification that is still
running.

---

## Checking these still hold

1. **Re-test the world-group collective after any cluster change.** It is a property of the
   infrastructure and it can be fixed without anyone telling you, at which point the
   socket workaround is pure cost.
2. **Re-measure step time against the length cap** when the data changes. The cap was chosen
   against a particular length distribution and new data moves the tail.
3. **Confirm the over-length policy is still dropping rather than truncating** after any
   framework upgrade. It is rarely the default and it reverts quietly.
4. **Re-run a single-node smoke before every scale-out**, not only after changes you think
   are risky. It is minutes, and the failures it catches cost hours of queue time.
