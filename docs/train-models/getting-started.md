# Getting Started with Training Steps

This page walks through one supervised fine tuning (SFT) run on DGX Cloud Lepton using the *tiny* configuration.
The tiny configuration lives in `src/nemotron/steps/sft/automodel/config/tiny.yaml` and is meant for short validation before you scale work.
The goal is to validate wiring, NeMo Run, and your environment profile on real multi-node hardware.

## Prerequisites

- You need a clone of the Nemotron repository with dependencies installed.
  Run `uv sync --all-extras` from the repository root if you have not installed dependencies yet.
- You need access to DGX Cloud Lepton with GPU nodes.
  This path assumes two nodes with eight A100 80 GB GPUs per node, matching the `run.env` block in `src/nemotron/steps/sft/automodel/config/tiny.yaml`.
- You need to set the `HF_TOKEN` environment variable.
- You ran `lep login` after syncronizing dependencies and are logged into Lepton.

## Procedure

1. Create an `env.toml` at the root of the repository like the following example:

   ```{literalinclude} _snippets/input/env.toml
   :language: toml
   ```

   Contact your cluster administrator for the values to substitute for the placeholders.

1. View the step manifest and run specification:

   ```console
   $ uv run nemotron step show sft/automodel
   ```

   ````{dropdown} Example Output
   :icon: code-square

   ```{literalinclude} _snippets/output/gs-show.txt
   ```
   ````

1. Compile the job against your Lepton profile without submitting it.
   The profile name `lepton-sft` must match a table in your root `env.toml`.

   ```console
   $ uv run nemotron step run sft/automodel --config tiny --run lepton-sft --dry-run
   ```

   ````{dropdown} Partial Output
   ```text
   Compiled Configuration

   ╭─────────────────────────────────────────── run ───────────────────────────────────────────╮
   │ env:                                                                                      │
   │   nodes: 2                                                                                │
   │   gpus_per_node: 8                                                                        │
   │   nprocs_per_node: 8                                                                      │
   │   executor: lepton                                                                        │
   │   container_image: nvcr.io/nvidia/nemo-automodel:26.04                                    │
   │   node_group: az-sat-lepton-001                                                           │
   │   resource_shape: gpu.8xa100-80gb                                                         │
   │   remote_job_dir: /mnt/lustre-shared/user/nemotron/.nemotron-jobs
   ...
   ```
   ````

1. Submit the sample SFT job:

   ```console
   $ uv run nemotron step run sft/automodel -c tiny -r lepton-sft
   ```

The sample `tiny` config sets small training and validation splits.
To specify the output path for checkpoints, set `SFT_OUTPUT_DIR` before running or specify the `checkpoint.checkpoint_dir` CLI override.

## Discover Other Steps

List step identifiers the CLI knows about:

```console
$ uv run nemotron step list
```

Other training stacks, for example Megatron Bridge SFT, PEFT, reinforcement learning (RL), or optimization, have their own `consumes` requirements. Use the [How-To Guides](how-to/index.md) and [Reference](reference/step-catalog.md) when you move past this first SFT path.

## Success Checks

- The command `nemotron step show <step_id>` lists `consumes` and `produces` artifact types. Those types must line up with your pipeline when you chain steps.
- A finished sample run leaves logs and job metadata where NeMo Run is configured to write them. See [Execution through NeMo Run](../nemo_runspec/nemo-run.md) for experiment layout.
- If you change tokenizer, template, or sequence length, keep them consistent across every step that touches the same model line. The [Artifact Graph](explanation/artifact-graph.md) page explains why consistency matters.

## Next Steps

- Follow [First SFT Run with AutoModel](tutorials/first-sft-automodel.md) when you need to point `tiny.yaml` at your own data or change the base model.
- Read [Choose an SFT Backend](how-to/choose-sft-backend.md) when you need Megatron Bridge instead of AutoModel.
