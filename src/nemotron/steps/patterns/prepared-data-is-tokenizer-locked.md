---
id: prepared-data-is-tokenizer-locked
title: "Treat prepared data as tokenizer-locked"
tags: [prep, tokenizer, data-artifacts]
triggers:
  - "You are reusing packed Parquet or bin/idx data after changing the tokenizer, chat template, or sequence length."
  - "A downstream trainer reports shape, vocabulary, EOS, loss-mask, or data-prefix mismatches."
  - "You need to decide whether an existing prepared dataset is still compatible with a new training config."
steps: [prep/sft_packing, prep/pretrain_prep, sft/megatron_bridge, peft/megatron_bridge, pretrain/automodel, pretrain/megatron_bridge]
confidence: high
---

## When to apply

Apply this whenever a training pipeline consumes materialized data artifacts rather than raw JSONL or text. Packed Parquet and bin/idx outputs are not generic datasets; they encode tokenizer, template, length, split, and sometimes loss-mask assumptions.

This matters most at handoff points: `prep/sft_packing` into Megatron-Bridge SFT or PEFT, and `prep/pretrain_prep` into AutoModel or Megatron-Bridge pretraining.

Use it when a run changes model family, tokenizer path, chat template, sequence length, EOS handling, or role formatting. Any of those changes can invalidate data that otherwise still looks readable on disk.

## What to do

Record the tokenizer id or path, tokenizer revision, chat template, pack size or sequence length, split policy, shard count, and source blend path with the prepared artifact.

For SFT packing, keep `pack_size` equal to downstream `seq_length`. Inspect packed rows for EOS handling, role boundaries, and loss masks before treating a training failure as a model or runner problem.

For pretraining bin/idx, rebuild the artifact when the tokenizer changes. Check token counts, empty-document rates, dtype, split names, and the emitted `blend.json` before launching a long job.

For RL prep, inspect a few records from every split. DPO data should preserve prompt, chosen, and rejected ordering. RLVR data should carry explicit verifier fields. RLHF data should keep prompt data separate from reward-model config.

Use tiny configs for plumbing only. A tiny prepared artifact can prove the runner starts, but it does not prove quality, throughput, or distributional coverage.

## Exceptions

If only execution infrastructure changes, such as local versus cluster executor, the data artifact may still be valid. Verify paths and permissions rather than rebuilding immediately.

If the downstream trainer reads raw JSONL directly, as AutoModel SFT and PEFT do, do not add SFT packing just for consistency. The compatibility boundary is different.

## References

- Pair with `pack-variable-length` when the question is whether packing is useful.
- Pair with `multilingual-tokenizer-check` when the corpus includes non-English or mixed-language data.
- This pattern explains many late failures that appear during training but originate in prep.
