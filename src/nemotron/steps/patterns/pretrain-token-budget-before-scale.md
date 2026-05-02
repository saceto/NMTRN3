---
id: pretrain-token-budget-before-scale
title: "Define the pretraining token budget before scaling"
tags: [pretrain, planning, scaling]
triggers:
  - "You are planning pretraining or continued pretraining beyond a smoke test."
  - "A pretrain config is being scaled from tiny/local execution to multi-GPU or multi-node execution."
  - "You need to choose between AutoModel and Megatron-Bridge for a pretraining run."
steps: [prep/pretrain_prep, pretrain/automodel, pretrain/megatron_bridge]
confidence: high
---

## When to apply

Use this before any serious pretraining or continued-pretraining run. Pretraining cost is driven by tokens, sequence length, batch size, and restart policy; those choices should be explicit before the cluster request grows.

Apply it when moving from a working tiny config to a production-shaped config, or when deciding whether a HuggingFace-native AutoModel run is enough versus a Megatron-Bridge run with distributed parallelism.

It is especially important for continued pretraining. CPT can damage a strong base model if the token budget, learning rate, or validation data is chosen casually.

## What to do

Define the target token budget first. Record sequence length, effective global batch, train iterations, warmup tokens or steps, learning-rate schedule, expected wall-clock, and checkpoint cadence.

Choose the backend based on the required scale and artifact format. Prefer AutoModel when HF-native checkpoints and quick iteration matter. Prefer Megatron-Bridge when model size, sequence length, or throughput requires tensor, pipeline, context, expert, or sequence parallelism.

Preserve a clean validation corpus. Do not train on validation examples, and do not tune data filters against the final test set.

Run a short representative job before the long run. Verify data access, throughput, validation loss movement, checkpoint save, and checkpoint restore.

For continued pretraining, use a lower learning rate than training from scratch and keep the base tokenizer stable unless the experiment explicitly studies tokenizer replacement.

## Exceptions

Tiny configs are still enough for runner validation. Do not require a full token-budget exercise when the goal is only import, launch, or config smoke testing.

For exploratory research, the budget can be approximate at first, but it still needs to be documented well enough that another run can reproduce the scale.

## References

- Pair with `prepared-data-is-tokenizer-locked` before reusing existing bin/idx artifacts.
- Pair with `eval-bookends` so CPT gains and regressions are measured against the base checkpoint.
- Use `checkpoint-before-convert` if the pretraining output crosses HF and Megatron formats.
