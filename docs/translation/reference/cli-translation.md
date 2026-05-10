---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "CLI reference for nemotron steps translation using RecipeTyper conventions."
topics: ["Translation", "CLI"]
tags: ["Reference", "CLI"]
content:
  type: "Reference"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# CLI Reference for Translation

Syntax, global flags, and merge rules for `nemotron steps translation`. Pair this page with {doc}`translate-config` for YAML field meanings.

## Synopsis

```bash
uv run nemotron steps translation [GLOBAL OPTIONS] [-c CONFIG] [DOTLIST_OVERRIDES...]
```

## Global Options

These flags mirror other Nemotron commands:

| Flag | Purpose |
|------|---------|
| `-c NAME`, `--config NAME` | Select `NAME.yaml` inside `src/nemotron/steps/translate/translation/config/` or pass an explicit `*.yaml` path. |
| `-d`, `--dry-run` | Print the merged OmegaConf YAML without executing `TranslationStage`. |
| `-r`, `--run PROFILE` | Not supported for translation. The command exits unless mode stays local. |
| `-b`, `--batch PROFILE` | Not supported for translation under the same restriction. |

Invocation without `-c` loads `default` automatically through `parse_config`.

## Dotlist Overrides

Anything after the global flags that matches `key=value` merges into the YAML dictionary loaded from `default.yaml`. Nested keys use dotted paths:

```bash
uv run nemotron steps translation -c default \
  faith_eval.threshold=3.1 \
  server.model=YOUR_MODEL \
  input_path=/data/text.jsonl \
  output_dir=/data/translated \
  source_language=en \
  target_language=ar
```

## Restrictions Specific to Translation

- Passthrough arguments are not supported.
- Remote and cluster submission is disabled.

## Artifact Overrides

`META.input_artifacts` documents `run.data` overrides for future lineage-aware launches. The current translation CLI reads `input_path` directly from YAML and dotlists instead.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Dry-run printed or translation finished successfully. |
| `1` | Validation failures such as missing languages, illegal backend configuration, unsupported CLI mode, or missing API keys. |

## Related Pages

- YAML keys: {doc}`translate-config`
- Tutorial invocation: {doc}`../getting-started`
