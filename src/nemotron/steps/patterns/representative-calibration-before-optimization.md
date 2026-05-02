---
id: representative-calibration-before-optimization
title: "Use representative data before judging optimization quality"
tags: [optimize, quantization, distillation, eval]
triggers:
  - "A quantization, pruning, or distillation run is being evaluated for quality."
  - "A tiny optimization config passed and someone wants to treat it as evidence."
  - "Calibration or distillation data does not match the deployment domain."
steps: [optimize/modelopt/quantize, optimize/modelopt/prune, optimize/modelopt/distill]
confidence: high
---

## When to apply

Apply this to every ModelOpt quality decision. Tiny configs and mock data are useful for plumbing, but they say almost nothing about the optimized model's real behavior.

Use it when judging FP8 or NVFP4 quantization, structured pruning, teacher-student distillation, or quality recovery after compression.

It is especially important when the deployment domain has long contexts, code, tool calls, multilingual data, or other activation patterns that a generic calibration slice may not cover.

## What to do

Keep a full-precision baseline checkpoint and evaluation record before optimizing.

Choose calibration or distillation data that resembles the deployment domain, sequence lengths, and prompt formats. Include examples that stress the model's expected serving patterns.

Record the optimization recipe, calibration data revision, teacher path, student path, export dtype, target hardware, and conversion commands with the artifact.

Evaluate both quality and deployment metrics: benchmark accuracy, latency, throughput, memory footprint, context length, and serving compatibility.

Run export or conversion smoke tests immediately after producing an optimized checkpoint. A quantized or pruned checkpoint can train or export successfully and still fail in the serving path.

Do not judge quality from calibration loss alone. Run task evals and representative generation checks.

## Exceptions

For wrapper development, mock data and tiny configs are enough to validate command construction and file outputs.

If no representative data exists yet, use the closest available proxy and label the result as provisional. Do not make deployment claims from proxy-only validation.

## References

- Pair with `eval-bookends` to compare optimized checkpoints against their baseline.
- Pair with `production-export-trt` when the optimization target is serving efficiency.
- Pair with `distill-after-structural-compression` when pruning or low-precision quantization needs quality recovery.
