# Training Libraries

Each step delegates to a concrete training or optimization library.
The library decides how to load models, how to read data, and the native checkpoint layout.

## NeMo AutoModel

The steps `sft/automodel` and `peft/automodel` use the AutoModel library for training that centers on Hugging Face conventions.
Data is usually OpenAI chat-formatted JSONL files that are read as-is.

## Megatron-Bridge

The steps `sft/megatron_bridge` and `peft/megatron_bridge` use Megatron Bridge for large distributed runs.
These steps expect packed Apache Parquet for the packed pipeline.
These steps emit Megatron checkpoints or Megatron-format adapters.

## NeMo RL

Steps under `rl/nemo_rl/` delegate to NeMo RL for alignment algorithms.
They assume a Megatron-format policy checkpoint as the warm start.
Reward handling differs per algorithm.
Refer to [Choose an RL Alignment Step](../how-to/choose-rl-step.md) for selection guidance.

## NVIDIA Model Optimizer

Steps under `optimize/modelopt/` call Model Optimizer flows that are orchestrated next to Megatron Bridge conventions for export.
Quantization targets reduced-precision inference.
Pruning changes architecture.
Distillation transfers quality from a teacher checkpoint to a student checkpoint.

## Choosing a Library

Use the how-to guides for supervised fine tuning (SFT), parameter-efficient fine tuning (PEFT), and reinforcement learning (RL) to map requirements to a stack. Treat stack choice as sticky. Crossing stacks implies conversion steps and new performance tuning, not a single configuration flag change.

## Related Reading

- [Choose an SFT Backend](../how-to/choose-sft-backend.md)
- [Execution through NeMo Run](../../nemo_runspec/nemo-run.md)
