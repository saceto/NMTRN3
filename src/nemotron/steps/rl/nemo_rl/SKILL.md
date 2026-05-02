---
name: nemotron-rl-nemo-rl
description: Navigate the Nemotron rl/nemo_rl family for DPO, RLVR or GRPO, and RLHF. Use when working with NeMo-RL launchers, Ray execution, data schemas, reward configuration, NeMo-Gym, rollout settings, and Megatron checkpoint alignment outputs.
---

# NeMo-RL

Use the `rl/nemo_rl` family for post-SFT alignment that produces Megatron checkpoints.

## Pick A Method

- Use `dpo` for offline preference pairs and a reference-policy KL objective.
- Use `rlvr` for GRPO with verifiable rewards such as math answers, tests, or tool results.
- Use `rlhf` for GRPO with a learned judge or generative reward model through NeMo-Gym.

## Workflow

1. Read the method step's `step.toml` for consumed artifacts and main knobs.
2. Prepare data with `prep/rl_prep` when HF placeholders or split sharding are needed.
3. Prefer `config/tiny.yaml` for runner validation.
4. Use NeMo-Gym configs when custom resource-server or GenRM rewards are required.
5. Check `src/nemotron/steps/patterns/validate-rl-rewards-before-scale.md` before scaling DPO, RLVR, or RLHF jobs.

## Local Files

- `rl/nemo_rl/dpo/step.py`, `rl/nemo_rl/dpo/config/default.yaml`, `rl/nemo_rl/dpo/config/tiny.yaml`
- `rl/nemo_rl/rlvr/step.py`, `rl/nemo_rl/rlvr/config/default.yaml`, `rl/nemo_rl/rlvr/config/tiny.yaml`, `rl/nemo_rl/rlvr/config/nemo_gym.yaml`
- `rl/nemo_rl/rlhf/step.py`, `rl/nemo_rl/rlhf/config/default.yaml`, `rl/nemo_rl/rlhf/config/tiny.yaml`

## Guardrails

- Keep policy checkpoints in Megatron format unless a config explicitly expects HF.
- Keep reward-model serving, policy rollout, and validation data paths explicit in YAML.
- Treat Ray and NeMo-Gym setup as part of the experiment, not incidental infrastructure.
