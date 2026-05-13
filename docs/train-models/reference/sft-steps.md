# Supervised Fine Tuning (SFT) Steps Reference

## `sft/automodel`

- Manifest: `src/nemotron/steps/sft/automodel/step.toml`
- Consumes: `training_jsonl` (required)
- Produces: `checkpoint_hf`
- Notable parameters: the `peft` parameter accepts `lora` or `null` for adapter-style training versus full fine tuning in the same step
- Operator notes: `src/nemotron/steps/sft/automodel/SKILL.md`

## `sft/megatron_bridge`

- Manifest: `src/nemotron/steps/sft/megatron_bridge/step.toml`
- Consumes: `packed_parquet` (required)
- Produces: `checkpoint_megatron`
- Operator notes: `src/nemotron/steps/sft/megatron_bridge/SKILL.md`

## Category Overview

The file `src/nemotron/steps/sft/SKILL.md` summarizes backend choice, sample commands, and pattern references for tokenizer alignment and evaluation discipline.

## Related Reading

- [Choose an SFT Backend](../how-to/choose-sft-backend.md)
- [Configuration Conventions](config-conventions.md)
