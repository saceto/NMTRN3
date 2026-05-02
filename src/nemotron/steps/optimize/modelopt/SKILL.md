---
name: nemotron-optimizer-modelopt
description: Navigate Nemotron optimize/modelopt steps for ModelOpt quantization, distillation, and pruning through Megatron-Bridge. Use when working under the ModelOpt optimization family, choosing compression order, setting calibration or distillation data, or validating optimized checkpoint outputs.
---

# ModelOpt Optimization

Use the `optimize/modelopt` family when NVIDIA Model Optimizer drives checkpoint compression or recovery.

## Pick A Step

- Use `quantize` for PTQ recipes such as FP8 or NVFP4 and Megatron checkpoint output.
- Use `prune` for Minitron-style target-parameter search or fixed architecture pruning.
- Use `distill` to transfer quality from a teacher to a student, often after pruning or quantization.

## Workflow

1. Decide final deployment hardware and checkpoint format first.
2. Use pruning to change architecture, quantization to change numeric format, and distillation to recover quality.
3. Pass newly supported upstream flags through `extra_args` instead of editing wrappers.
4. Keep teacher, student, calibration data, and output paths explicit in config.
5. Check `src/nemotron/steps/patterns/representative-calibration-before-optimization.md` before judging quality.
6. Check `src/nemotron/steps/patterns/distill-after-structural-compression.md` when compressed quality needs recovery.

## Config Nuances

- Quantization and pruning rely on wrapper-level W&B logging because their upstream scripts do not expose the same native W&B flags as distillation.
- Keep the ModelOpt checkout and installed package in sync; clone-and-install the checkout before applying local compatibility patches.
- Keep compatibility patches narrow and named after the upstream mismatch they address, such as `moe_grouped_gemm` for Megatron-Bridge loader drift.
- Use launch configs as plumbing checks only; calibration size, MMLU scoring, and distillation data must be representative before quality claims.

## Local Files

- `optimize/modelopt/quantize/step.py`, `optimize/modelopt/quantize/config/default.yaml`, `config/fp8.yaml`, `config/nvfp4.yaml`, `config/tiny.yaml`
- `optimize/modelopt/distill/step.py`, `optimize/modelopt/distill/config/default.yaml`, `config/tiny.yaml`
- `optimize/modelopt/prune/step.py`, `optimize/modelopt/prune/config/default.yaml`, `config/tiny.yaml`

## Guardrails

- Do not treat launch-validation configs as quality signals.
- Use representative calibration or distillation data before judging model quality.
- Preserve the full-precision baseline and eval results.
