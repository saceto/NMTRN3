---
name: nemotron-prep
description: Navigate Nemotron data preparation steps for SFT packing, pretraining bin/idx tokenization, and RL prompt or preference sharding. Use when choosing, configuring, validating, or chaining prep steps before pretrain, SFT, PEFT, DPO, RLVR, or RLHF training.
---

# Nemotron Prep

Use this skill to choose and configure data preparation under `src/nemotron/steps/prep/`.

## Route

| Need | Step | Produces |
| --- | --- | --- |
| Pack chat JSONL for Megatron-Bridge SFT or PEFT | `prep/sft_packing` | `packed_parquet` |
| Tokenize text into Megatron pretraining shards | `prep/pretrain_prep` | `binidx` plus `blend.json` |
| Resolve and shard RL prompt or preference data | `prep/rl_prep` | `training_jsonl` |

## Workflow

1. Read the target step's `step.toml` for artifacts, parameters, strategies, and references.
2. Start with `config/tiny.yaml` for smoke tests and `config/default.yaml` for production shape.
3. Keep tokenizer, chat template, sequence length, split names, and shard policy aligned with the downstream trainer.
4. Inspect sample outputs before launching expensive training.
5. Check `src/nemotron/steps/patterns/prepared-data-is-tokenizer-locked.md` when reusing or changing prepared artifacts.
6. Use each step's `step.toml [reference]` section for upstream repos and documentation.

## Local Files

- `prep/sft_packing/step.py`, `prep/sft_packing/config/default.yaml`, `prep/sft_packing/config/tiny.yaml`
- `prep/pretrain_prep/step.py`, `prep/pretrain_prep/config/default.yaml`, `prep/pretrain_prep/config/tiny.yaml`
- `prep/rl_prep/step.py`, `prep/rl_prep/config/default.yaml`, `prep/rl_prep/config/tiny.yaml`

## Guardrails

- Repack SFT data whenever tokenizer, chat template, or `pack_size` changes.
- Rebuild pretraining bin/idx data whenever the tokenizer changes.
- Materialize remote data during RL prep when the training cluster cannot reach the source.
- Treat prepared artifacts as reproducible, versioned data products.
