---
id: distill-after-structural-compression
title: "Plan distillation after structural compression"
tags: [optimize, pruning, distillation, quality-recovery]
triggers:
  - "A model has been pruned to a smaller architecture."
  - "Quantization or pruning creates a quality gap that matters for deployment."
  - "A compressed student needs to recover behavior from a stronger teacher."
steps: [optimize/modelopt/prune, optimize/modelopt/quantize, optimize/modelopt/distill]
confidence: high
---

## When to apply

Use this when compression changes model capacity or creates a measurable quality gap. Pruning removes architecture capacity, and aggressive quantization can expose sensitivity in specific layers or domains.

Apply it when the optimized model is intended for production or benchmark comparison, not just a plumbing test.

This pattern is most relevant after structured pruning and sometimes after quantization-aware recovery plans. Distillation turns compression from a one-step artifact conversion into a quality-recovery workflow.

## What to do

Keep the strongest appropriate full-precision checkpoint as the teacher. Confirm license, tokenizer, and behavior compatibility before using it as the source of truth.

Make teacher, student, data, and output paths unambiguous in config. Avoid reusing ambiguous names such as `model` or `checkpoint` for multiple roles.

Choose distillation data that reflects deployment use. Generic pretraining text may not recover instruction following, tool use, reasoning style, or domain terminology.

Run a short representative distillation job before the full budget. Verify loss movement, checkpoint save, and downstream eval signal.

Compare the compressed student against both the baseline teacher and the pre-distillation compressed checkpoint. This shows whether distillation is actually recovering quality.

## Exceptions

If pruning or quantization preserves quality within the deployment tolerance, distillation may be unnecessary.

If the compressed checkpoint is only a temporary experiment, a full distillation run may be overkill. Still record that no quality-recovery stage was run.

## References

- Pair with `representative-calibration-before-optimization` before making quality claims.
- Pair with `eval-bookends` to quantify recovery.
- For pruning, also verify architecture divisibility before spending distillation budget.
