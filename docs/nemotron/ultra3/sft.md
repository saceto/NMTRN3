# Stage 1: SFT

This stage fine-tunes Nemotron 3 Ultra with [Megatron-Bridge](../nvidia-stack.md#megatron-bridge)'s `finetune()` entry point and the existing Ultra recipe:

```text
megatron.bridge.recipes.nemotronh.nemotron_3_ultra.nemotron_3_ultra_sft_openmathinstruct2_packed_config
```

The default Nemotron-side config overrides the Megatron-Bridge OpenMath starter into a paper-style packed-Parquet SFT run. `openmath.yaml` remains available as a smaller fallback/demo config that leaves Megatron-Bridge's built-in OpenMathInstruct-2 dataset path intact.

## Reference configuration

| Setting | Default packed SFT | OpenMath fallback |
|---------|--------------------|-------------------|
| CLI command | `nemotron ultra3 sft` | `nemotron ultra3 sft -c openmath` |
| Runspec name | `ultra3/sft` | `ultra3/sft` |
| Launch | `torchrun` | `torchrun` |
| Container | `~/.cache/nemotron/containers/ultra3-sft.sqsh` (built from `nvcr.io/nvidia/nemo:26.04.01`) | same |
| HF model id | `nvidia/nemotron-ultra-rl-052726` | same |
| Resources | 384 nodes × 8 GPUs | 24 nodes × 8 GPUs |
| Parallelism | TP=2, PP=6, EP=32, ETP=1, CP=8 | TP=2, PP=6, EP=32, ETP=1, CP=1 |
| Sequence length | 294,912 packed tokens | 4,096 packed tokens |
| Global batch | 64 | 128 |

## Data

The default config consumes an externally prepared packed-Parquet artifact:

```yaml
run:
  data: ultra3-sft-data:latest

dataset:
  ultra3_packed_sft_dir: ${art:data,path}
  seq_length: 294912
  packed_sequence_specs:
    packed_sequence_size: 294912
```

The corresponding data-prep stage lives under `src/nemotron/recipes/ultra3/stage1_sft/data_prep.py` and packs the open Ultra/Super-style SFT blend into `train/` and optional `valid/` Parquet shards. The paper-style default requires the data artifact to be packed at 294,912 tokens; use `openmath.yaml` when you want Megatron-Bridge's built-in OpenMathInstruct-2 download/packing path instead of an external artifact.

## Run

Preview the compiled default job config:

```bash
uv run nemotron ultra3 sft -c tiny --run YOUR-CLUSTER --dry-run
uv run nemotron ultra3 sft --run YOUR-CLUSTER --dry-run
```

Submit an attached Slurm job through NeMo-Run:

```bash
uv run nemotron ultra3 sft --run YOUR-CLUSTER
```

Run the OpenMath fallback/demo config:

```bash
uv run nemotron ultra3 sft -c openmath --run YOUR-CLUSTER --dry-run
```

If you have an imported/pretrained checkpoint path, provide it with either `PRETRAINED_CHECKPOINT` or a config override:

```bash
PRETRAINED_CHECKPOINT=/path/to/checkpoint \
uv run nemotron ultra3 sft --run YOUR-CLUSTER

uv run nemotron ultra3 sft --run YOUR-CLUSTER \
  checkpoint.pretrained_checkpoint=/path/to/checkpoint
```

## Slurm wiring

SLURM execution uses the same `nemo_runspec` / NeMo-Run command path as pretraining. No separate training `sbatch` script is required in this repo. Container builds use the shared `nemotron kit slurm build <profile> --recipe ultra3 --stage sft` path (or the fallback `src/nemotron/recipes/ultra3/build.slurm.sh`).

## Source

- Recipe: `src/nemotron/recipes/ultra3/stage1_sft/`
- CLI: `src/nemotron/cli/commands/ultra3/sft.py`
- Back to [Ultra3 overview](./README.md)
