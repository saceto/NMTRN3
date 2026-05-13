# Parameter-Efficient Fine Tuning (PEFT) Steps Reference

## `peft/automodel`

- Manifest: `src/nemotron/steps/peft/automodel/step.toml`
- Consumes: `training_jsonl` (required)
- Produces: `checkpoint_lora`
- Operator notes: `src/nemotron/steps/peft/automodel/SKILL.md`

## `peft/megatron_bridge`

- Manifest: `src/nemotron/steps/peft/megatron_bridge/step.toml`
- Consumes: `packed_parquet` and `checkpoint_megatron` base, both required
- Produces: `checkpoint_lora` in Megatron adapter layout
- Operator notes: `src/nemotron/steps/peft/megatron_bridge/SKILL.md`

## Category Overview

The file `src/nemotron/steps/peft/SKILL.md` covers adapter merge discipline, rank and alpha defaults, and when to stay on AutoModel compared with Megatron Bridge.

## Related Reading

- [Choose a PEFT Backend](../how-to/choose-peft-backend.md)
- [Configuration Conventions](config-conventions.md)
