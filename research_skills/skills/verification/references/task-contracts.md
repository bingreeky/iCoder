# Task-property verification contracts

Use this classification before writing or selecting a backend. It prevents
benchmark labels from standing in for the executable obligation.

## Common property record

Record for every task:

- generated-artifact boundary and required entry point;
- interface schema, types, shapes, ports, timing, and protocol obligations;
- whether state persists across observations;
- external modules, files, models, drivers, simulators, and accelerators;
- compilation, elaboration, launch, and runtime stages that apply;
- observation source and explicit positive-verdict rule;
- reference or golden behavior and permitted implementation freedom;
- meaningful partial outcomes or performance measures, if any; and
- candidate-controllable channels that cannot be trusted for attribution.

The resulting property record selects a contract. A source-bound official
harness profile then binds exact files, commands, versions, timeouts, extraction,
and reporting. A benchmark label is metadata for locating that profile, not a
verifier level or correctness definition.

## RTL contracts

RTL verification usually progresses with integration scope:

1. **Code fragment or isolated combinational block.** Check syntax, interface,
   widths, and direct input/output behavior. There is little persistent state or
   external integration.
2. **Self-contained module.** Compile and elaborate the declared module, then
   simulate state, reset, timing, handshake, and task-specific behavior through
   the supplied testbench.
3. **Integrated subsystem or system.** Build the required design context, models,
   files, clocks, protocols, and testbench environment before judging the
   candidate. Distinguish a candidate failure from a missing dependency or
   harness failure.

These are task properties, not benchmark tiers. A benchmark may contain only one
class or a mixture. Correctness remains binary when the official executable
contract exposes only behavioral acceptance; compilation alone is not invented
as semantic partial credit unless the downstream objective explicitly and
honestly defines it.

## GPU-kernel contracts

Kernel verification separates three questions:

1. **Functional equivalence.** Run the candidate on declared shapes, dtypes,
   tolerances, and edge cases and compare with the trusted reference.
2. **Execution eligibility.** Confirm the requested generated computation really
   launched and did not satisfy the task through a prohibited delegation,
   vacuous identity path, or reference reuse.
3. **Performance ordering.** Only after correctness and eligibility, measure the
   task-defined latency or speedup under a calibrated environment and protocol.

A task that asks only for functional code need not inherit performance reward. A
task whose official metric already makes delegation uncompetitive should not
receive a redundant purity gate. The contract follows the permitted solution
space and observable evidence.

## Profile registry

Each official-harness profile records:

~~~yaml
profile_id: stable-id
task_contract_ids: []
source_revision: exact-revision
payload_schema: versioned-schema
toolchain_versions: {}
reference_fixture_ids: []
negative_fixture_ids: []
positive_verdict_rule: explicit-rule
infrastructure_failure_enum: []
integrity_digests: {}
stage_adapter_versions: {}
~~~

Adding a profile does not change an existing task contract. Changing the meaning
of correctness creates a new contract or version and invalidates dependent
comparisons until they are re-evaluated.
