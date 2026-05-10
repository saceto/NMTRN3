---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Tune faith_eval inside nemotron steps translation runs."
topics: ["Translation", "FAITH"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run FAITH Evaluation

Use this guide when you need to tune thresholds, filtering, or scorer models for FAITH inside `nemotron steps translation` without a separate evaluation command.

FAITH runs inside `TranslationStage` whenever `faith_eval.enabled` is `true`. The `default.yaml` starter profile ships with FAITH enabled. For how FAITH couples to non-LLM backends, see {doc}`../explanation/faith-evaluation`.

## Essential Knobs

| YAML path | Purpose |
|-----------|---------|
| `faith_eval.enabled` | Master toggle. Set `false` when you only need raw translation. |
| `faith_eval.threshold` | Average score floor. Rows failing the threshold are dropped when `filter_enabled` is `true`. |
| `faith_eval.segment_level` | Keeps FAITH aligned with translation segmentation. Recommended for long documents. |
| `faith_eval.filter_enabled` | Enables or disables removing low-scoring rows. |
| `faith_eval.model_name` | Overrides `server.model` for scoring-only workloads. |

## LLM Credentials

FAITH always requires the OpenAI-compatible `server` configuration. Set `NVIDIA_API_KEY` first, for example by replacing `<api-key>` below.

```bash
export NVIDIA_API_KEY="<api-key>"
uv run nemotron steps translation -c default \
  faith_eval.enabled=true \
  faith_eval.threshold=3.0 \
  faith_eval.segment_level=true \
  faith_eval.filter_enabled=true \
  server.model=YOUR_LLM_MODEL_ID \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=hi
```

## Disable FAITH Temporarily

```bash
uv run nemotron steps translation -c default \
  faith_eval.enabled=false \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=hi \
  server.model=YOUR_LLM_MODEL_ID
```

Even with FAITH off, `backend=llm` still needs `server.model` for translation itself.

## Related Pages

- Concept primer: {doc}`../explanation/faith-evaluation`
- Full YAML reference: {doc}`../reference/translate-config`
