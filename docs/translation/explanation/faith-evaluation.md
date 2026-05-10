---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How FAITH integrates into nemotron steps translation and interacts with translation backends."
topics: ["Translation", "FAITH"]
tags: ["Explanation", "Quality"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# FAITH Evaluation Inside Translation

This page explains how optional FAITH scoring behaves when `faith_eval.enabled` is `true` inside `nemotron steps translation`.

FAITH runs in the same `TranslationStage` invocation as translation. There is no separate command-line interface only for FAITH scoring.

## What FAITH Adds

FAITH scores translation quality segment-by-segment using a large language model (LLM) judge configured alongside your translation backend:

- `faith_eval.threshold` defines the minimum acceptable average score. The starter default is `2.5`, which you should tune per model.
- `faith_eval.segment_level` aligns scoring granularity with translation segmentation for long inputs.
- `faith_eval.filter_enabled` drops failing rows when `true`, which lets you keep high-confidence shards only.

## Why an LLM Client Is Always Required

Whenever `faith_eval.enabled` is `true`, the stage instantiates an OpenAI-compatible client using `server.url`, `server.api_key` or `server.api_key_env`, and `faith_eval.model_name`. If you omit `faith_eval.model_name`, the stage falls back to `server.model`.

Even if `backend` is `nmt`, `google`, or `aws`, FAITH still issues LLM calls. Plan keys and quotas accordingly.

## Merge Semantics

`merge_scores: true` is the default. It attaches FAITH outputs alongside translated columns so reviewers can audit scores without losing structured chat payloads.

## Related YAML Surface

See `faith_eval` in {doc}`../reference/translate-config` and operational recipes in {doc}`../how-to/run-faith-evaluation`.

## Related Pages

- Where FAITH sits in the pipeline: {doc}`pipeline-overview`
- Input and output shapes when scores merge or filter: {doc}`../reference/io-format`
