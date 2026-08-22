<p align="center">
  <img src="eval_harness/docs/assets/icoder-27b.png" alt="iCoder-27B" width="560">
</p>

<h3 align="center">Recursive AI-Led Development of Frontier Industrial Coding Model</h3>

<p align="center">
  <em>High-density expert priors · Low-frequency human intervention · Auditable agent-led experimentation</em>
</p>

<p align="center">
  <sub>Cheng Yang · Guibin Zhang · Jiayang Lyu · Shangyuan Liu · Linfeng Zhang · Qibing Ren · Shuicheng Yan</sub>
</p>

<p align="center">
  <img alt="Paradigm: human-guided, agent-led" src="https://img.shields.io/badge/paradigm-human--guided%20%7C%20agent--led-8B5E0C?style=flat-square">
  <img alt="Domains: RTL and GPU kernels" src="https://img.shields.io/badge/domains-RTL%20%2B%20GPU%20kernels-B87E14?style=flat-square">
  <img alt="Eval harness license: MIT" src="https://img.shields.io/badge/eval%20harness-MIT-2B2D3A?style=flat-square">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#framework">Framework</a> ·
  <a href="#repository-components">Components</a> ·
  <a href="eval_harness/README.md">Eval Harness</a> ·
  <a href="#security-and-governance">Security</a>
</p>

---

## Overview

iCoder studies a practical path toward agent-led model development. Human experts
encode reusable operational knowledge as **Research Skills**; within explicit
resource, verification, and governance boundaries, an agent selects experiments,
executes them, diagnoses failures, and revises a multi-stage post-training program.

The project targets **industrial coding**: RTL design and GPU kernel optimization,
where generated code is judged by task-native executable toolchains rather than
surface similarity.

> [!NOTE]
> iCoder is **human-guided and agent-led**. It is not presented as fully autonomous
> recursive self-improvement. Humans define the objective, approved resources,
> verifier semantics, and release authority; the agent leads experimental decisions
> within that envelope.

## Framework

<p align="center">
  <img src="eval_harness/docs/assets/framework.png" alt="Research Skills, self-evolving agent loop, and frontier model" width="100%">
</p>

<p align="center">
  <sub>Research Skills provide an executable human prior; the agent iterates through data evolution and post-training under task-native verification.</sub>
</p>

The framework separates knowledge supplied before the loop from evidence acquired
during the loop:

| Layer | Responsibility |
|---|---|
| **Human prior** | Encode model-training procedures, permission boundaries, invariants, and reusable operations as versioned Research Skills. |
| **Agentic experimentation** | Select, execute, diagnose, and revise experiments while maintaining an auditable decision trail. |
| **Executable verification** | Ground decisions in task-native compilers, simulators, testbenches, and numerical oracles. |
| **Governance** | Preserve artifact lineage, verifier integrity, resource controls, and human release authority. |

### Development path

| Data | SFT | OPSD | RLVR |
|---|---|---|---|
| Evolve and admit executable tasks through verifier-backed gates. | Cold-start reasoning from verified teacher trajectories. | Convert demonstrated self-correction into persistent capability. | Refine the policy with task-conditioned, verifiable rewards. |

The stages form a feedback loop, not a one-way assembly line. Evaluation can return
the process to data construction, verifier work, or an earlier training decision
whenever the available evidence is insufficient.

## Repository components

| Component | Role |
|---|---|
| [`eval_harness/`](eval_harness/) | Runs code-model evaluations across RTL and GPU-kernel benchmarks using local vLLM or an OpenAI-compatible endpoint. |
| [`research_skills/`](research_skills/) | Houses the versioned Research Skills that encode the executable human prior. Maintained by Yangcheng. |

### Evaluation harness

The harness turns generated artifacts into task-native evidence:

```text
model endpoint → candidate generation → isolated verification → structured artifacts
```

It separates generation from resource-intensive verification, records verdict and
provenance information, and keeps benchmark-specific logic behind a shared
orchestration interface. See the [complete harness guide](eval_harness/README.md).

### Research Skills

Research Skills turn recurring expert intervention into versioned, executable
procedures. They cover the post-training control plane and stage-specific operations
while keeping mutable experiment state outside the prior itself.

The bundle is governed by [`research_skills/manifest.json`](research_skills/manifest.json).
Its documentation and maintenance remain with Yangcheng.

## Getting started

Clone the repository and enter the evaluation harness:

```bash
git clone https://github.com/bingreeky/iCoder.git
cd iCoder/eval_harness
```

Then follow the [toolchain, dataset, and endpoint setup](eval_harness/README.md#quick-start).

## Repository layout

```text
iCoder/
├── eval_harness/       evaluation and executable-verification infrastructure
├── research_skills/    versioned Research Skills and release manifest
└── README.md            project overview
```

## Security and governance

The evaluation harness compiles and executes model-generated code. Run it only in an
isolated, disposable environment with strict filesystem, process, resource, and
network controls. Never place personal data, model credentials, or unrelated secrets
on an evaluation host.

Research automation is constrained by explicit capabilities. New datasets, external
side effects, verifier semantics, and public releases remain human-authorized
decisions. See the harness [security guidance](eval_harness/README.md#security) for
operational details.

## Citation and release resources

Citation metadata for the evaluation harness is available in
[`eval_harness/CITATION.cff`](eval_harness/CITATION.cff). Paper and model release
links will be added when their public artifacts are available.

## License

The evaluation harness is released under the [MIT License](eval_harness/LICENSE).
Vendored components and benchmark datasets retain their original licenses; review
the [third-party notices](eval_harness/NOTICE) before redistribution.
