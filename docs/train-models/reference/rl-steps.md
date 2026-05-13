# Reinforcement Learning (RL) Steps Reference

All three steps live under `rl/nemo_rl/` and emit `checkpoint_megatron`. Each manifest lists exact `[[consumes]]` fields, including the policy checkpoint and dataset shape.

## `rl/nemo_rl/dpo`

- Manifest: `src/nemotron/steps/rl/nemo_rl/dpo/step.toml`
- Consumes: preference `training_jsonl`, warm-start `checkpoint_megatron`
- Produces: `checkpoint_megatron`
- Operator notes: `src/nemotron/steps/rl/nemo_rl/dpo/SKILL.md`

## `rl/nemo_rl/rlvr`

- Manifest: `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml`
- Consumes: verifiable prompt `training_jsonl`, warm-start `checkpoint_megatron`
- Produces: `checkpoint_megatron`
- Operator notes: `src/nemotron/steps/rl/nemo_rl/rlvr/SKILL.md`

## `rl/nemo_rl/rlhf`

- Manifest: `src/nemotron/steps/rl/nemo_rl/rlhf/step.toml`
- Consumes: prompt `training_jsonl`, warm-start `checkpoint_megatron`, reward model `checkpoint_hf`
- Produces: `checkpoint_megatron`
- Operator notes: `src/nemotron/steps/rl/nemo_rl/rlhf/SKILL.md`

## Category Overview

The files `src/nemotron/steps/rl/SKILL.md` and `src/nemotron/steps/rl/nemo_rl/SKILL.md` document reward selection, NeMo Gym toggles, and sample commands.

## Related Reading

- [Choose an RL Alignment Step](../how-to/choose-rl-step.md)
- [Execution through NeMo Run](../../nemo_runspec/nemo-run.md)
