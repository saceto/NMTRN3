---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Use backend=nmt with nemotron steps translation and a local HTTP translation service."
topics: ["Translation", "NMT"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run NMT Translation

Use this guide when translation should run against `backend: nmt` and your own hypertext transfer protocol (HTTP) microservice that performs *neural machine translation (NMT)*.

## Prerequisites

- A running NMT service reachable at the URL you pass as `nmt.server_url`, implementing the contract in the next section.
- If `faith_eval.enabled` is `true`, LLM credentials for FAITH remain required; see {doc}`run-faith-evaluation`.

## Service Contract

NeMo Curator expects:

- `POST /translate` accepting JSON payloads with `texts`, `src_lang`, and `tgt_lang` arrays or lists compatible with the Curator client.
- Tune `nmt.batch_size`, `nmt.timeout`, and `nmt.max_concurrent_requests` once you understand server throughput.

## Procedure

```bash
uv run nemotron steps translation -c default \
  backend=nmt \
  nmt.server_url=http://localhost:5000 \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=hi
```

## FAITH Considerations

If `faith_eval.enabled` stays `true` as in the default `default.yaml`, you must still supply large language model (LLM) credentials through `server` because FAITH scoring uses that client even though translation runs on NMT.

## Related Pages

- Field wiring: {doc}`configure-fields-and-output`
- YAML reference for the `nmt` block: {doc}`../reference/translate-config`
