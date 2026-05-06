<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

(sdg-getting-started)=
# Generate Your First Synthetic Dataset

::::{grid} 2

:::{grid-item-card}
:columns: 8

**What You'll Build**: A small synthetic SFT chat dataset in OpenAI format--five records grounded in the bundled `sft_topic_seeds.jsonl` seed file, generated through Data Designer against an NVIDIA-hosted LLM endpoint.

^^^

**In this tutorial, you will**:

1. Set up prerequisites: the repository and an NVIDIA API key.
2. Read the bundled pipeline configuration.
3. Run a preview to verify the pipeline and model.
4. Generate a small dataset of five records.
5. Locate and inspect the output JSONL.

{octicon}`clock;1.5em;sd-mr-1` This tutorial requires between 5 and 10 minutes to complete.
:::

:::{grid-item-card}
:columns: 4

{octicon}`flame;1.5em;sd-mr-1` **Sample Prompt**

^^^

Run a 2-record preview of the default SDG pipeline, then generate 5 records and show me the first output record.

:::
::::

## Start Here

- Run all commands from the repository root.
- Data generation uses an NVIDIA-hosted endpoint, so the step needs no local GPUs.
  However, you must set the `NVIDIA_API_KEY` environment variable and you must have network access.

## Prerequisites

- ✅ Repository cloned and `uv sync` complete. Refer to [Quick Start](../index.md) if you have not done this yet.
- ✅ `NVIDIA_API_KEY` for the default model, `nvidia/nemotron-3-nano-30b-a3b`.

## How the Default Pipeline Works

The `src/nemotron/steps/sdg/data_designer/config/default.yaml` combines two sources of variation to generate each record.
A seed topic, such as "safe deployment of AI assistants in enterprise support workflows" or
"ways to monitor data drift in production machine learning systems", is drawn from `.../data/sft_topic_seeds.jsonl`.
A persona category, such as teacher or engineer, is sampled from a fixed category.
Together they anchor the user prompt: a researcher might ask a concise technical question about RAG and a student might ask the same topic more tentatively.

The pipeline generates a matching assistant response and then projects the result into OpenAI chat-format messages.

The full configuration is stored at `src/nemotron/steps/sdg/data_designer/config/default.yaml`.

```{literalinclude} ../../src/nemotron/steps/sdg/data_designer/config/default.yaml
:language: yaml
:lines: 15-
:class: scrollable
```

## Procedure

1. Set your API key:

   ```console
   $ export NVIDIA_API_KEY="<your-api-key>"
   ```

1. Run a two-record preview.
   Preview mode runs the same pipeline against a tiny record count so you can verify the model alias, prompts, and column wiring cheaply before generating at scale.

   ```console
   $ nemotron step run sdg/data_designer -c default preview=true num_records=2
   ```

   The pipeline registers the model alias, generate two rows, and prints a summary:

   ````{dropdown} Example Output
   :icon: code-square

   ```{literalinclude} _snippets/output/preview.txt
   :language: text
   ```
   ````

   The default output path is `./output/sdg/sft.jsonl`.
   You can override by setting `SDG_OUTPUT_DIR` or specifying `output_path=...` on the command line.

   Inspect the output.
   Each line is one chat record.
   The `openai_messages` projection emits a `messages` array plus the seed `topic` and sampled `persona` as metadata for traceability.
   The following shows one sample record from the `sft.jsonl` file.

   ```{literalinclude} _snippets/output/sft_first_record.jsonl
   :language: json
   ```

## Summary

What you learned:

- ✅ Ran a two-record preview to verify the pipeline and model.
- ✅ Generated a five-record SFT chat dataset with `default.yaml`.
- ✅ Located the OpenAI-format JSONL output.

Key takeaways:

- **Preview first.** `preview=true num_records=N` runs the same pipeline against a tiny record count. Use it to iterate on column specs and prompts before scaling `num_records` up.
- **Output format matches the trainer.** The `openai_messages` projection emits records ready for `prep/sft_packing` or AutoModel SFT.

## Next Steps

- **Adapt the pipeline to a domain you care about**: {doc}`how-to/create-domain-dataset`.
- **Preview, generate, and customize output**: {doc}`how-to/run`.
- **Generate preference pairs for DPO**: {doc}`how-to/preference-data`.
- **Dispatch to a cluster**: {doc}`how-to/dispatch-to-cluster` learn about env.toml profiles and container images.
- **Look up flags and config fields**: {doc}`reference/cli-reference`, {doc}`reference/config-schema`.
