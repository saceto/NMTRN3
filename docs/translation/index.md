---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Translate JSON Lines or Parquet corpora with nemotron steps translation, NeMo Curator backends, and optional FAITH scoring."
topics: ["Translation", "FAITH", "NeMo Curator"]
tags: ["Translation", "Documentation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(translation-index)=
# Translation With Nemotron

`nemotron steps translation` translates selected fields in JSON Lines (JSONL) or Apache Parquet files through the checked-in `translate/translation` step: a NVIDIA NeMo Curator reader, `TranslationStage`, and writer pipeline. You can choose a large language model (LLM) with an OpenAI-compatible endpoint, a local neural machine translation (NMT) service over HTTP, Google Cloud Translation, or Amazon Translate, and optionally run *FAITH*, the translation-quality scoring path built into NeMo Curator, in the same invocation.

:::{tip}
New here? Start with {doc}`getting-started`, then use this page as the map to deeper topics.
:::

## When to Use

Use `nemotron steps translation` when you need:

- Localized training or synthetic corpora from translating natural-language fields while preserving structured payloads such as chat turns, tool payloads, and fenced code blocks. Field paths, `output_mode`, and segmentation interact with that behavior; see {doc}`how-to/configure-fields-and-output` and {doc}`explanation/segmentation`.
- Optional FAITH scoring with configurable thresholds and filtering, without a separate evaluation command-line interface (CLI).
- Repeatable configuration by using the checked-in `default.yaml` plus CLI overrides.

## Pipeline Summary

```{mermaid}
flowchart LR
    A[Input JSONL or Parquet] --> B[Curator reader]
    B --> C[TranslationStage]
    C --> D[Curator writer]
    D --> E[Output shards under output_dir]
    C --> F{FAITH enabled?}
    F -->|yes| G[LLM scores segments]
    F -->|no| E
    G --> E
```

## Documentation Series

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tutorial
:link: getting-started
:link-type: doc
Run `nemotron steps translation` end-to-end using `default.yaml` and a sample chat JSONL file.
+++
{bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-to guides
:link: how-to/index
:link-type: doc
Backends, fields and outputs, segmentation, FAITH tuning.
+++
{bdg-secondary}`task-based`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: explanation/index
:link-type: doc
Pipeline architecture, segmentation, FAITH behavior.
+++
{bdg-secondary}`learn`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc
YAML parameters and `nemotron steps translation` CLI.
+++
{bdg-secondary}`lookup`
:::

::::

## All Documentation

````{tab-set}

```{tab-item} Tutorial

| Guide | What you do |
|-------|-------------|
| {doc}`getting-started` | Run translation and FAITH using `default.yaml` and sample JSONL |

```

```{tab-item} How-to guides

| Guide | Focus |
|-------|-------|
| {doc}`how-to/run-llm-translation` | `backend: llm` |
| {doc}`how-to/run-nmt-translation` | `backend: nmt` |
| {doc}`how-to/run-google-aws-translation` | `backend: google` / `aws` |
| {doc}`how-to/configure-fields-and-output` | Field paths and `output_mode` |
| {doc}`how-to/use-fine-segmentation` | `segmentation_mode` |
| {doc}`how-to/run-faith-evaluation` | `faith_eval` block |

```

```{tab-item} Concepts

| Guide | Topic |
|-------|-------|
| {doc}`explanation/pipeline-overview` | End-to-end flow |
| {doc}`explanation/segmentation` | Coarse versus fine |
| {doc}`explanation/faith-evaluation` | FAITH semantics |

```

```{tab-item} Reference

| Guide | Content |
|-------|---------|
| {doc}`reference/translate-config` | `default.yaml` field reference |
| {doc}`reference/cli-translation` | `nemotron steps translation` syntax |
| {doc}`reference/io-format` | Input and output shapes |

```

````

## Limitations and Considerations

- Cost and rate limits: Hosted and cloud LLM backends incur usage; throttle with `max_concurrent_requests` and your provider’s guidance.
- Local execution only: `nemotron steps translation` rejects cluster `--run` and `--batch` modes today.
- Overrides: Use `key=value` dotlist syntax after global flags, not passthrough script arguments.
- Mixed folders: Do not point `input_path` at one directory that contains both `.jsonl` and `.parquet` shards unless you split formats first.

## Quick Paths

1. First run: {doc}`getting-started`
2. Swap backend: {doc}`how-to/index`
3. Lookup flags: {doc}`reference/cli-translation`
