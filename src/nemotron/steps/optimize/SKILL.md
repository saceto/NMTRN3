---
name: nemotron-optimizer
description: "Choose and configure Nemotron optimization steps using NVIDIA ModelOpt and Megatron-Bridge: quantization, distillation, and pruning. Use when reducing inference cost, compressing checkpoints, recovering quality, targeting FP8 or NVFP4 hardware, or preparing optimized Megatron or HuggingFace outputs."
---

# Nemotron Optimizer

Use this skill to choose a ModelOpt optimization path.

## Route

| Need | Step | Output |
| --- | --- | --- |
| FP8 or NVFP4 post-training quantization | `optimize/modelopt/quantize` | `checkpoint_megatron` |
| Teacher-student quality transfer | `optimize/modelopt/distill` | `checkpoint_megatron` |
| Structured architecture pruning | `optimize/modelopt/prune` | `checkpoint_hf` |

## Workflow

1. Decide target hardware, serving stack, checkpoint format, and quality budget first.
2. Read the target step's `step.toml` and `config/default.yaml`.
3. Smoke with `config/tiny.yaml` when present, or use mock-data and short-iteration overrides for distillation.
4. Convert or export checkpoints when the downstream consumer requires a different format.
5. Check `src/nemotron/steps/patterns/representative-calibration-before-optimization.md` before judging optimized quality.
6. Check `src/nemotron/steps/patterns/distill-after-structural-compression.md` when pruning or quantization creates a quality gap.

## Guardrails

- Distill after pruning or quantization when quality recovery matters.
- Use representative calibration or distillation data for meaningful quality conclusions.
- Preserve the full-precision baseline and evaluation record.
