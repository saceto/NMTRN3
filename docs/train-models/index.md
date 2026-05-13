# Model Training with Nemotron Steps

This section documents how to run supervised fine tuning (SFT), parameter-efficient fine tuning (PEFT), reinforcement learning (RL) alignment, and post-training optimization with Nemotron *steps*. Each step packages manifests in `step.toml`, YAML under its directory, and entry logic you invoke through the `nemotron step` command line interface (CLI).

## What a Step Is

A *step* is a packaged unit with a stable identifier, such as `sft/automodel`, NeMo Run job metadata, and configuration files.
You run steps with the Nemotron CLI.
Remote execution uses an environment profile and NeMo Run, as described in [Execution through NeMo Run](../nemo_runspec/nemo-run.md).

## Capabilities at a Glance

| Area | Step Names | Role |
|------|-------------------------------|------|
| SFT | `sft/automodel`, `sft/megatron_bridge` | Supervised fine tuning from chat-formatted JSON Lines (JSONL) or packed Apache Parquet |
| PEFT | `peft/automodel`, `peft/megatron_bridge` | Adapter training with a smaller trainable surface |
| RL | `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, `rl/nemo_rl/rlhf` | Alignment after a supervised fine tuning (SFT) policy exists |
| Optimize | `optimize/modelopt/quantize`, `optimize/modelopt/prune`, `optimize/modelopt/distill` | Compression and quality recovery |

## Learning Path

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc
This guide covers installation expectations, the environment profile, and how to run your first sample job with the `tiny` configuration.
+++
`Beginner`
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
These walkthroughs give hands-on first runs for individual training steps.
+++
`Beginner`
:::

:::{grid-item-card} How-To Guides
:link: how-to/index
:link-type: doc
These guides are task-focused. They explain how to pick a backend, wire data, and run optimization steps.
+++
`Intermediate`
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc
This material explains artifacts, training stacks, and how steps differ from recipes.
+++
`Concepts`
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
This material is for lookup. It lists the step catalog, parameters, and configuration conventions.
+++
`Lookup`
:::

:::{grid-item-card} Model Training with Agents
:link: using-skill
:link-type: doc
Use the customize skill with a YAML-first plan: repo steps, then configs, then code only for gaps.
+++
`Workflow`
:::

::::

## Quick Links

- [Model Training with Agents](using-skill.md) describes how to work with the `nemotron-customize` skill for multi-stage training plans and YAML-first deliverables.
- [Getting Started](getting-started.md) describes how to verify the CLI and run a tiny configuration.
- [Execution through NeMo Run](../nemo_runspec/nemo-run.md) describes profiles, attached and detached runs, and clusters.
- [Nemotron CLI Overview](../nemotron/cli.md) describes how the wider CLI relates to configuration and overrides.
