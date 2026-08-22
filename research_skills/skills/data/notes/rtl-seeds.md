# RTL seeds

> **Provenance class: calibrated reference.** This file is not part of the static
> Human Prior in this release. Load it only through a separately digested,
> provenance-bearing calibrated-note binding in the Stage Context Packet. A live
> run writes new findings to run-scoped learned memory.

Specifics for generating from seeds whose reference is a hardware module and whose verifier
is a testbench. The general method is in the [Data Skill](../SKILL.md).

---

## What these seeds are

A specification, a reference module, and a testbench that instantiates the module and
reports pass or fail through its output.

Two structural facts follow, and between them they account for most of the pipeline:

**The port list is the contract.** The testbench instantiates by port name. A generated
module whose ports differ from what the testbench expects does not fail informatively — it
fails to elaborate, or worse, elaborates with an implicit net and produces plausible wrong
results. Every operator's "do not change" set must name the module name and the port list
explicitly, or the model will drift them while doing exactly what you asked.

**Some interface properties are one word in the instruction and total failure at
evaluation.** Reset polarity and synchronicity, and every port's bit width, are of this kind:
flip synchronous to asynchronous, or active-high to active-low, and every test fails while
the specification still reads as correct English. Extract these mechanically from the
reference and place them in the instruction verbatim. Never let a model restate them —
paraphrase loses precision here without losing fluency.

Seeds also admit a coarse classification by static features — whether the design holds
state, whether it has a protocol, whether it streams, whether it shares a resource. This
classification is worth computing because operators are not universally applicable, and
applying one to a design class it does not fit produces a run of confident failures that
look like model weakness.

---

## This family composes rather than mutates

Unlike seed families where you edit the reference and derive the instruction from the edit,
the productive approach here is to **attach requirements to the specification and generate
the whole triplet fresh** against the augmented spec — module, testbench and instruction all
produced from the same augmented description, none of them inherited.

This is the difference between perturbing a design and making it a harder design. Editing
an existing module gets you variations on its behaviour. Composing gets you a module that
needs machinery the parent did not have.

The operators are correspondingly structural. Each attaches a mechanism, brings its own
ports, and states what the testbench must now check:

- wrap a datapath in a latency-bearing controller with start and done signalling
- add a valid/ready handshake with backpressure
- add buffering between producer and consumer
- add a timeout with a retry and recovery path
- add arbitration between competing requestors, or burst behaviour

Each needs parameter choices attached — depths, latencies, retry counts, arbitration
disciplines — and the choice should be derived deterministically from the seed identity
rather than sampled, so a single variant per seed still spreads evenly across operators
instead of piling onto the first one.

**Prose-level evolution does not work on this family, and it fails silently.** Rewriting the
wording of these specifications produces items that pass every gate and are their parents:
the problems are short and structured enough that a paraphrase round-trips. Measured
retention for structural operators sits near the ceiling and for prose-level ones near zero
— that gap is the finding, and it only becomes visible if you deduplicate against parents.
Weight accordingly, and do not assume it transfers to longer or less structured seeds.

---

## Roles, and what each is allowed to see

Four generation roles, and the separation between them is load-bearing:

- the **module** is generated from the augmented specification alone
- the **testbench** is generated from the specification and the module
- the **instruction** is generated from the specification alone — never from the module, or
  it leaks the implementation's structure into the problem statement
- the **judge** sees all four and audits them against each other

The instruction role's blindness to the module is what keeps the leak lint from being the
only defense. The testbench role's sight of the module is what lets it instantiate the exact
interface — and is also why the testbench must never be repaired using a semantic mismatch
against that module. See the [Data Skill](../SKILL.md); this family is where that failure
mode was named.

---

## Gates, in order

**Interface schema.** The module declares the name and ports the specification requires.
Free, catches the most.

**Compiles.** The module and testbench compile together.

**The reference passes its own testbench.** Run it. Unlike the repair loop — which must never
see a semantic result — this is an accept/reject gate and it is the cheapest three-way
consistency check available.

**Lints clean, and elaborates structurally.** A separate linter and a structural elaboration
check each catch things the compiler accepts: inferred latches, width mismatches, unconnected
or multiply-driven nets. These are the defects that produce a module which simulates
correctly on the testbench you generated and fails on any other.

**The testbench discriminates.** Mutation testing — see below. Without it the four gates
above are all satisfiable by a testbench that reports success unconditionally.

**The instruction states what the testbench checks.** A judge enumerates the obligations the
operator introduced and confirms each is both stated in the instruction and checked by the
testbench. Require the enumeration, not a verdict.

**Not a duplicate.** Four axes, because they catch different things: a hash of the
instruction, a hash of the module, a hash of the port signature, and a signature of control
structure — counts of processes, case statements, conditionals, non-blocking assignments,
registers. Items differing only in naming collide on the last one and on nothing else.

**Harder than its parent.** A strong model must solve the child at least once, establishing
it is solvable at all; and a panel must solve the child strictly less often than the parent.
Comparing against the parent rather than an absolute bar is what makes this robust to
whichever solver you picked.

---

## Mutation testing, since this is where it exists

Generate mutants of the reference by corrupting the behaviour the operator introduced, plus
generic corruptions as a fallback for operators without specific patterns. Run each past the
testbench. A mutant that survives is a behaviour the testbench does not check.

Three things decide acceptance, and the third is the one that matters:

- **enough mutants were valid.** Mutants that fail to compile tell you nothing and must be
  excluded from the denominator, not counted as killed.
- **the kill rate clears a floor.** Calibrate it; it trades verifier strictness against yield.
- **no mutant of the operator's own class survives.** Mark these separately from the generic
  ones. A testbench that catches generic damage while missing the specific property the item
  was built around is precisely the useless case, and an aggregate kill rate hides it
  completely.

The third condition is why per-operator mutant patterns are worth writing rather than
relying on generic corruption alone.

---

## Composing the set

Coverage first — every root that produced anything contributes — then caps per root, per
(root, operator), and per operator overall, then fill toward the target.

Rank within a root by the difficulty measurement rather than by gate scores. A high kill
rate means the testbench is good, not that the item is valuable.

---

## Checking these still hold

1. **Confirm the testbench still instantiates by port name.** Every interface gate rests on
   this. A harness that binds positionally, or that wraps the module, changes what the
   contract is.
2. **Re-measure prose-level retention before dismissing it.** The near-zero result is tied to
   these seeds being short and structured. On a longer or more discursive seed family it may
   not hold, and prose-level methods are cheap enough to be worth rechecking.
3. **Check that lint and structural elaboration still reject things the compiler accepts.**
   Toolchain updates move this line, and if they converge, one of the gates is now free
   coverage you are paying for.
4. **Confirm mutants still fail to compile at a low rate.** A rise means the mutation
   patterns have drifted out of step with the language or the generator, and the kill rate is
   being computed over a shrinking and biased denominator.
