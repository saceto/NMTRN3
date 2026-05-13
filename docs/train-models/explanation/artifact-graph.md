# Artifact Graph

Training steps declare typed inputs and outputs so pipelines can reason about compatibility without reading every Python file.

## Where Types Are Defined

The file `src/nemotron/steps/types.toml` lists artifact kinds such as `training_jsonl` and `checkpoint_megatron`. It gives short descriptions and optional `is_a` or `convert_to` edges. When a step manifest names a type in `[[consumes]]` or `[[produces]]`, that name must align with that graph, or you must insert an explicit conversion step between stages.

For example, running `uv run nemotron step show sft/automodel` shows the consumes and produces information:

```{code-block} text
:emphasize-lines: 8-12

────────────────────────────── sft/automodel — SFT Training (AutoModel) ──────────────────────────────
/path/to/nemotron/src/nemotron/steps/sft/automodel

Supervised fine-tuning with the AutoModel stack for HF-format models and JSONL
datasets that already use OpenAI chat-format messages. Supports full SFT and
LoRA-style adapter tuning from the same step.

Consumes
  • training_jsonl — Instruction data in JSONL with a messages field

Produces
  • checkpoint_hf — HuggingFace checkpoint directory (full model or adapter-style PEFT output)

Parameters
  • peft (default=null) — Use 'lora' for adapter tuning, or 'null' for full fine-tuning.

Runspec
  launcher: torchrun
  image: -
  resources: nodes=1 gpus_per_node=4
  config dir: /path/to/nemotron/src/nemotron/steps/sft/automodel/config
  default config: default
```

## Common Acyclic Paths

Typical supervised paths include the following chains:

- JSON Lines (JSONL) AutoModel line: `training_jsonl` → `sft/automodel` → `checkpoint_hf`
- Packed Megatron line: `training_jsonl` → packing prep → `packed_parquet` → `sft/megatron_bridge` → `checkpoint_megatron`

A typical alignment path starts from a `checkpoint_megatron` policy, adds preference or reward-side data, runs one of the `rl/nemo_rl/...` steps, and ends at `checkpoint_megatron`.

A typical compression path starts from `checkpoint_hf`, runs `optimize/modelopt/quantize`, and lands at `checkpoint_megatron`. Add conversion after quantization when the next consumer needs a Hugging Face layout again.

## Tokenizer and Template Lock

Artifacts are not enough for correctness. Tokenizer, chat template, and sequence length must stay consistent across every step that tokenizes text or loads weights for the same model line. A mismatch often appears as plausible loss with poor downstream quality.

## Related Reading

- [Data and Checkpoint Formats](../how-to/data-and-checkpoint-formats.md)
- [Training Stacks](training-stacks.md)
