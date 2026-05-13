# Step Catalog (Training)

The following step identifiers live under `src/nemotron/steps/` in the `sft`, `peft`, `rl`, and `optimize` directories. Each row gives the manifest path on disk.

| Category | Step id | Manifest path |
|----------|---------|----------------|
| SFT | `sft/automodel` | `src/nemotron/steps/sft/automodel/step.toml` |
| SFT | `sft/megatron_bridge` | `src/nemotron/steps/sft/megatron_bridge/step.toml` |
| PEFT | `peft/automodel` | `src/nemotron/steps/peft/automodel/step.toml` |
| PEFT | `peft/megatron_bridge` | `src/nemotron/steps/peft/megatron_bridge/step.toml` |
| RL | `rl/nemo_rl/dpo` | `src/nemotron/steps/rl/nemo_rl/dpo/step.toml` |
| RL | `rl/nemo_rl/rlvr` | `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml` |
| RL | `rl/nemo_rl/rlhf` | `src/nemotron/steps/rl/nemo_rl/rlhf/step.toml` |
| Optimize | `optimize/modelopt/quantize` | `src/nemotron/steps/optimize/modelopt/quantize/step.toml` |
| Optimize | `optimize/modelopt/prune` | `src/nemotron/steps/optimize/modelopt/prune/step.toml` |
| Optimize | `optimize/modelopt/distill` | `src/nemotron/steps/optimize/modelopt/distill/step.toml` |

Adjacent prep steps often appear in the same pipelines, but they are not part of this catalog. Those prep identifiers include `prep/sft_packing` and `prep/rl_prep`. Conversion steps such as `convert/megatron_to_hf` are also outside this list.

## Related Reading

- [Supervised Fine Tuning Steps Reference](sft-steps.md)
- [Configuration Conventions](config-conventions.md)
