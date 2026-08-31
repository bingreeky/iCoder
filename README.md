<p align="center">
  <img src="eval_harness/docs/assets/icoder-27b.png" alt="iCoder-27B" width="400">
</p>

<h3 align="center">Recursive AI-Led Development of Frontier Industrial Coding Model</h3>

<p align="center">
  <em>Research Skills · Agent-led post-training · Executable evaluation</em>
</p>

<p align="center">
  <img alt="Paradigm: human-guided, agent-led" src="https://img.shields.io/badge/paradigm-human--guided%20%7C%20agent--led-8B5E0C?style=flat-square">
  <img alt="Domains: RTL and GPU kernels" src="https://img.shields.io/badge/domains-RTL%20%2B%20GPU%20kernels-B87E14?style=flat-square">
  <a href="https://huggingface.co/i-Coder"><img alt="Model: iCoder-27B" src="https://img.shields.io/badge/model-iCoder--27B-E5C982?style=flat-square"></a>
  <img alt="Eval harness license: MIT" src="https://img.shields.io/badge/eval%20harness-MIT-2B2D3A?style=flat-square">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="https://huggingface.co/i-Coder/iCoder-27B/blob/main/Coder_Tech_Report.pdf">Paper</a> ·
  <a href="#framework">Framework</a> ·
  <a href="#repository-components">Components</a> ·
  <a href="research_skills/README.md">Research Skills</a> ·
  <a href="eval_harness/README.md">Eval Harness</a> ·
  <a href="https://huggingface.co/i-Coder">Model</a> ·
  <a href="#security">Security</a>
</p>

---

## Overview

iCoder is a research project on agent-led model development for RTL design and GPU
kernel optimization. This repository publishes the project's evaluation harness and
Research Skills bundle. The full technical report is available as a
[PDF](https://huggingface.co/i-Coder/iCoder-27B/blob/main/Coder_Tech_Report.pdf), and model artifacts are available as
[iCoder-27B](https://huggingface.co/i-Coder).

The project targets **industrial coding**: RTL design and GPU kernel optimization,
where acceptance is determined by specialized executable toolchains and deployment
quality rather than surface similarity alone.

> [!NOTE]
> iCoder is **human-guided and agent-led**, not a fully autonomous self-improvement
> system. Its workflows require user-provided objectives, task instructions,
> verifiers, resource limits, and an isolated execution environment.

## Framework

<p align="center">
  <img src="eval_harness/docs/assets/framework.png" alt="iCoder project framework" width="100%">
</p>

<p align="center">
  <sub>Research Skills guide the agent loop; task-native execution feeds evidence back into experimentation.</sub>
</p>

The project framework separates instructions supplied before an experiment from
evidence produced during execution:

| Element | Function |
|---|---|
| **Research Skills** | Provide versioned task instructions, workflow constraints, and verifier requirements. |
| **Agent loop** | Applies those instructions, runs experiments, and reacts to execution feedback. |
| **Executable verification** | Uses task-native compilers, simulators, testbenches, and numerical oracles to produce structured outcomes. |
| **Model release** | Makes the resulting iCoder-27B artifacts available through Hugging Face. |

### Development path

Data construction first creates a shared executable task pool. Parameter updates
then proceed through **SFT → OPSD → RLVR**:

| Stage | Role |
|---|---|
| **Data construction** | Prepares executable RTL and GPU-kernel tasks and checks them with domain toolchains. |
| **SFT (Supervised Fine-Tuning)** | Uses verified teacher solutions to establish task capability. |
| **OPSD (On-Policy Self-Distillation)** | Trains on feedback collected from the model's own execution attempts. |
| **RLVR (Reinforcement Learning with Verifiable Rewards)** | Uses domain-specific execution outcomes as training signals. |

The stages form a feedback loop: evaluation can send the process back to data
construction, verifier work, or an earlier training decision. This repository
focuses on the released evaluation and Research Skills components; it does not
include the complete training infrastructure.

## Repository components

| Component | Role |
|---|---|
| [`eval_harness/`](eval_harness/) | Runs code-model evaluations across RTL and GPU-kernel benchmarks using local vLLM or an OpenAI-compatible endpoint. |
| [`research_skills/`](research_skills/) | Provides the versioned task instructions and workflow constraints used by the project. |

### Evaluation harness

The harness turns generated artifacts into task-native evidence:

```text
model endpoint → candidate generation → isolated verification → structured artifacts
```

It separates generation from resource-intensive verification, records structured
outcomes and provenance fields, and keeps benchmark-specific logic behind a shared
orchestration interface. See the [harness guide](eval_harness/README.md).

## Getting started

Clone the repository once:

```bash
git clone https://github.com/bingreeky/iCoder.git
cd iCoder
```

To use the agent workflow, install the Skill directories and start with the
`auto-post-training` controller. The [Research Skills guide](research_skills/README.md#quick-start)
explains installation, bootstrap inputs, the Human Prior boundary, and the role
of every Skill.

To run model evaluation, enter `eval_harness/` and follow the
[toolchain, dataset, and endpoint setup](eval_harness/README.md#quick-start).

## Repository layout

```text
iCoder/
├── eval_harness/       evaluation and executable-verification infrastructure
├── research_skills/    versioned Research Skills and release manifest
└── README.md            project overview
```

## Security

The evaluation harness compiles and executes model-generated code. Run it only in an
isolated, disposable environment with strict filesystem, process, resource, and
network controls. Never place personal data, model credentials, or unrelated secrets
on an evaluation host.

See the harness [security guidance](eval_harness/README.md#security) for operational
details.

## Citation and release resources

```bibtex
@techreport{yang2026icoder,
  title  = {iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model},
  author = {Cheng Yang and Jiayang Lyu and Shangyuan Liu and Guibin Zhang and
            Jiong Lin and Xinlei Yu and Junchi Yan and Shuicheng Yan and
            E, Weinan and Linfeng Zhang and Linfeng Zhang and Qibing Ren},
  year   = {2026},
  month  = aug,
  type   = {Technical Report},
  url    = {https://huggingface.co/i-Coder/iCoder-27B}
}
```

Citation metadata for the evaluation harness is available in
[`eval_harness/CITATION.cff`](eval_harness/CITATION.cff). Model artifacts are
available from the [i-Coder organization on Hugging Face](https://huggingface.co/i-Coder).

## License

The evaluation harness is released under the [MIT License](eval_harness/LICENSE).
Vendored components and benchmark datasets retain their original licenses; review
the [third-party notices](eval_harness/NOTICE) before redistribution.
