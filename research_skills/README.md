<h1 align="center">iCoder Research Skills</h1>

<p align="center">
  <strong>Human-authored operating knowledge for agent-led model post-training</strong>
</p>

<p align="center">
  <img alt="Bundle version 0.2.0" src="https://img.shields.io/badge/bundle-v0.2.0-8B5E0C?style=flat-square">
  <img alt="Stages: Data, SFT, OPSD, RLVR" src="https://img.shields.io/badge/stages-Data%20%E2%86%92%20SFT%20%E2%86%92%20OPSD%20%E2%86%92%20RLVR-B87E14?style=flat-square">
  <img alt="Domains: RTL and GPU kernels" src="https://img.shields.io/badge/domains-RTL%20%2B%20GPU%20kernels-E5C982?style=flat-square">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#skill-map">Skill map</a> ·
  <a href="#how-the-bundle-works">Workflow</a> ·
  <a href="#human-prior-and-calibrated-notes">Prior boundary</a> ·
  <a href="manifest.json">Manifest</a>
</p>

---

## Overview

This directory publishes the Research Skills used by the iCoder project to turn
expert model-development practice into an executable starting point for an AI
agent. The bundle follows the technical report's central design:
**high-density prior, low-frequency intervention**.

Human experts provide the objective, resource and permission envelope, task and
verifier requirements, and reusable operating procedures. The agent works inside
that boundary to select experiments, run and diagnose them, update the persistent
research state, and decide what to try next.

The bundle supports the full development path:

```text
Data ──▶ SFT ──▶ OPSD ──▶ RLVR ──▶ Finalization
  ▲       │       │        │
  └───────┴───────┴────────┘
       evidence may send the process upstream
```

These files are operating instructions, not a pretrained model or a one-command
training system. A real run still requires project-specific models, datasets,
official harnesses, compute, credentials, and explicit authorization boundaries.

## Quick start

### 1. Install the Skill directories

Clone the repository, then copy the published Skill directories into your Codex
personal skills directory:

```bash
git clone https://github.com/bingreeky/iCoder.git
cd iCoder

mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R research_skills/skills/. "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Review any same-named directories before copying so that an existing personal
Skill is not replaced unintentionally. Start a new Codex conversation after
installation so the Skill catalog is refreshed.

### 2. Start through the controller

Invoke the controller rather than entering a training stage directly:

```text
$auto-post-training

Bootstrap a new post-training run.
Objective: <target capability and completion criteria>
Base model: <checkpoint and tokenizer>
Allowed teachers: <models and access constraints>
Data: <allowed sources and held-out evaluation>
Verification: <official harnesses and correctness rules>
Resources: <compute, storage, time, and budget>
Permissions: <allowed writes, launches, and human approval gates>
```

The controller first freezes the human-approved project profile and capability
lock. It then creates evidence-bearing queue items and loads a stage Skill only
when the active task requires it. To resume a run, invoke the same controller and
point it to the existing runtime-record directory.

### 3. Inspect a Skill without running the pipeline

Every entrypoint is ordinary Markdown and can be reviewed directly on GitHub.
The controller is
[`auto-post-training/SKILL.md`](skills/auto-post-training/SKILL.md); the complete
set is linked below.

## Skill map

| Skill | Role in the project | Load when |
|---|---|---|
| [`auto-post-training`](skills/auto-post-training/SKILL.md) | Governs the RSI control plane, persistent Task Queue, stage transitions, and final lineage. | Starting, resuming, routing, or closing a run. |
| [`data`](skills/data/SKILL.md) | Builds a shared pool of executable task--reference--verifier items and evolves it under ordered admission gates. | Constructing data, closing a coverage gap, or moving tasks toward the policy frontier. |
| [`sft`](skills/sft/SKILL.md) | Produces a verified reasoning cold start while auditing format, loss masking, length, composition, and distributed training integrity. | Building or evaluating the SFT stage. |
| [`opsd`](skills/opsd/SKILL.md) | Tests same-weight privileged-context self-distillation against a named weakness, with leakage controls and executable evidence. | Designing, diagnosing, or selecting an OPSD intervention. |
| [`rlvr`](skills/rlvr/SKILL.md) | Converts task-native execution into online reward while handling exploitation, missing verdicts, group variation, and trajectory budgets. | Designing or running verifier-grounded RL. |
| [`verification`](skills/verification/SKILL.md) | Defines the cross-stage task contract, official-harness profile, stable pass/fail/unjudged verdict, and stage adapters. | Building, auditing, or consuming executable evidence in any stage. |
| [`finalization`](skills/finalization/SKILL.md) | Freezes evaluation, compares the full checkpoint lineage, checks artifact integrity, and prepares a controlled release handoff. | Selecting and closing the final model. |
| [`experiment-ops`](skills/experiment-ops/SKILL.md) | Handles safe launches, monitoring, failure attribution, checkpoint policy, and shared-compute constraints. | Before long jobs, after failures, or before irreversible operations. |
| [`project-docs`](skills/project-docs/SKILL.md) | Preserves decisions, definitions, defects, handover state, and the auditable experiment history. | At decision points and whenever work resumes after context loss. |

The most important supporting references are:

- [`rsi-control-plane.md`](skills/auto-post-training/references/rsi-control-plane.md)
  for controller states and transition authority;
- [`task-queue.md`](skills/auto-post-training/references/task-queue.md) for bounded,
  evidence-bearing work items;
- [`runtime-records.md`](skills/auto-post-training/references/runtime-records.md)
  for replayable state, artifacts, decisions, and learned memory;
- [`task-contracts.md`](skills/verification/references/task-contracts.md) for
  separating task semantics from benchmark-specific harness profiles; and
- [`final-model-manifest.md`](skills/finalization/references/final-model-manifest.md)
  for checkpoint and release closure.

## How the bundle works

The Skills separate stable human guidance from decisions made during execution.

| Layer | What it contains | Who controls it during a run |
|---|---|---|
| **Human Prior** | Workflow invariants, task contracts, evidence requirements, resource and permission boundaries. | Frozen after bootstrap; the agent cannot rewrite it. |
| **Agent loop** | Hypotheses, experiment registrations, queue items, stage transitions, checkpoint choices, and failure-driven revisions. | Agent-controlled inside the approved envelope. |
| **Executable verification** | Task-native compiler, simulator, testbench, numerical, and performance evidence with provenance. | Raw verdicts remain stable; stage adapters define how evidence is consumed. |
| **Runtime state** | Attempts, artifacts, decisions, defects, and learned memory. | Written outside the installed Skill bundle and replayed on resume. |
| **Human gates** | New authority, large budget commitments, irreversible operations, and external release. | Require explicit human approval. |

Verification is shared across the pipeline. Data uses it to admit executable
items, SFT filters teacher trajectories, OPSD constructs failure-conditioned
experience and optional outcome signals, and RLVR maps eligible verdicts to
reward. The conceptual unit is the **task-native execution contract**; a
benchmark name selects an official harness, payload, toolchain, and reporting
profile rather than defining the verifier class.

## Human Prior and calibrated notes

[`manifest.json`](manifest.json) is the release boundary. Its `prior_files`
entries form the immutable Human Prior and are bound by `prior_digest`.
Experiments and newly learned rules must be recorded in the run state instead of
editing those files in place.

Files under `notes/` are calibrated implementation references. In this release
they are marked `legacy-unbound` and `default_load: false`; they are **not part of
the initial Human Prior**. A note becomes eligible only when a human-approved
project profile binds its exact digest, provenance, version, evidence, and
applicability scope in the active context packet.

This separation keeps the attribution boundary auditable: prior knowledge is
known before the run, while conclusions produced by the agent remain attached to
the experiments that established them.

## Repository layout

```text
research_skills/
├── manifest.json              version and Human Prior digest
├── README.md                  public guide and Skill index
└── skills/
    ├── auto-post-training/    controller and persistent runtime contracts
    ├── data/                  executable task-pool construction
    ├── sft/                   verified reasoning cold start
    ├── opsd/                  privileged-context self-distillation
    ├── rlvr/                  execution-grounded reinforcement learning
    ├── verification/          cross-stage verifier semantics
    ├── finalization/          checkpoint and release closure
    ├── experiment-ops/        shared-compute operations
    └── project-docs/          auditable research records
```

## Safety and scope

The workflows compile and execute model-generated code. Use isolated,
disposable workers with explicit filesystem, process, resource, credential, and
network controls. Installing the Skills grants no authority to launch jobs,
modify remote assets, spend a budget, or publish a model; those permissions must
be supplied by the active project profile and confirmed at the relevant human
gate.
