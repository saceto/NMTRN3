# Model Optimization Steps Reference

## `optimize/modelopt/quantize`

- Manifest: `src/nemotron/steps/optimize/modelopt/quantize/step.toml`
- Consumes: `checkpoint_hf`
- Produces: `checkpoint_megatron`
- Notable parameters: `args.export_quant_cfg` selects the quantization recipe string that the installed Megatron Bridge integration accepts. The parameter `args.calib_size` sets calibration sample count.
- Operator notes: `src/nemotron/steps/optimize/modelopt/quantize/SKILL.md`

## `optimize/modelopt/prune`

- Manifest: `src/nemotron/steps/optimize/modelopt/prune/step.toml`
- Consumes: `checkpoint_hf`
- Produces: `checkpoint_hf`
- Operator notes: `src/nemotron/steps/optimize/modelopt/prune/SKILL.md`

## `optimize/modelopt/distill`

- Manifest: `src/nemotron/steps/optimize/modelopt/distill/step.toml`
- Consumes: teacher and student as `checkpoint_hf`. The optional `binidx` artifact supplies real distillation data when you do not rely on mock data.
- Produces: `checkpoint_megatron`
- Operator notes: `src/nemotron/steps/optimize/modelopt/distill/SKILL.md`

## Category Overview

The files `src/nemotron/steps/optimize/SKILL.md` and `src/nemotron/steps/optimize/modelopt/SKILL.md` describe ordering between prune and distill, hardware targets, and limitations of mock-data sample runs.

## Related Reading

- [Run Post-Training Optimization](../how-to/run-optimization.md)
- [Configuration Conventions](config-conventions.md)
