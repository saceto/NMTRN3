# QA Runbook: Model Customization Playbook RC1

## Purpose

This runbook tells QA how to validate the Model Customization Playbook RC1 workflows as a user would use them.

For every required module, QA should run one direct CLI workflow and one agent-driven workflow. Compare the generated commands, configs, artifacts, logs, and result summaries against the expected behavior in this document.

## Scope

In scope modules:

- General setup
- Translation
- BYOB
- Training
- SDG
- Airgap
- Lepton testing
- Evaluation

Out of scope:

- Full model training to convergence.
- Public benchmark leaderboard validation.
- Production deployment hardening outside the Airgap module.
- Vendor account provisioning, cloud quota, and external service uptime.
- Curator internals, except validating that Nemotron installs and calls a Curator build with experimental translation support.

## Runbook Rules

- Run commands from the repo root.
- Use a fresh `QA_ROOT` for each QA pass.
- If a backend, credential, GPU, Lepton workspace, Docker daemon, or external model endpoint is unavailable, mark that case `BLOCKED` and record the exact missing prerequisite.
- Do not edit checked-in configs for QA. Copy configs to `QA_ROOT`, patch the copies, and run with those copies.
- Do not treat tiny runs as model-quality evidence. Tiny runs validate plumbing, schemas, artifact paths, and executor behavior.
- Capture stdout/stderr, config paths, output artifact paths, row counts, job IDs, and redacted credential evidence for every case.

## Common Setup

### Environment

```bash
export QA_RUN_ID="${QA_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
export QA_ROOT="${QA_ROOT:-/tmp/nemotron-qa-$QA_RUN_ID}"
mkdir -p "$QA_ROOT"/{logs,metadata,artifacts}

export TRANSLATION_MODEL="${TRANSLATION_MODEL:-openai/gpt-oss-120b}"
export FAITH_MODEL="${FAITH_MODEL:-$TRANSLATION_MODEL}"
export TRANSLATION_BASE_URL="${TRANSLATION_BASE_URL:-https://integrate.api.nvidia.com/v1}"
export NMT_SERVER_URL="${NMT_SERVER_URL:-http://localhost:5000}"

export BYOB_MODEL="${BYOB_MODEL:-nvidia/nemotron-3-nano-30b-a3b}"
export SDG_MODEL="${SDG_MODEL:-nvidia/nemotron-3-nano-30b-a3b}"
export EVAL_MODEL_ID="${EVAL_MODEL_ID:-megatron_model}"
export EVAL_URL="${EVAL_URL:-http://0.0.0.0:8080/v1/completions/}"
export EVAL_TOKENIZER="${EVAL_TOKENIZER:-/path/to/checkpoint/tokenizer}"
```

Credential variables used by the runbook:

```bash
# Set these only when running real hosted workflows.
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
export NGC_API_KEY="${NGC_API_KEY:-$NVIDIA_API_KEY}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
```

### Install And CLI Discovery

```bash
uv sync
uv run nemotron --help
uv run nemotron steps --help
uv run nemotron steps list
uv run nemotron steps show --help
uv run nemotron steps run --help
uv run nemotron data --help
uv run nemotron byob --help
```

Optional workflow dependency installs:

```bash
uv sync --extra translation
uv sync --extra byob
uv sync --group run
```

### Step Metadata Snapshot

```bash
for STEP in \
  translate/translation \
  byob \
  data_prep/sft_packing \
  data_prep/pretrain_prep \
  data_prep/rl_prep \
  sft/automodel \
  sft/megatron_bridge \
  peft/automodel \
  peft/megatron_bridge \
  pretrain/automodel \
  pretrain/megatron_bridge \
  rl/nemo_rl/dpo \
  rl/nemo_rl/rlvr \
  rl/nemo_rl/rlhf \
  optimize/modelopt/quantize \
  optimize/modelopt/prune \
  optimize/modelopt/distill \
  sdg/data_designer \
  eval/model_eval \
  env/env_toml
do
  uv run nemotron steps show "$STEP" --json > "$QA_ROOT/metadata/${STEP//\//_}.json"
done
```

Expected result:

- Every command exits successfully.
- Metadata files are valid JSON.
- Step IDs match the paths shown in the command.
- No command requires a model credential just to show metadata.

## Translation

### Test Data

```bash
export TR_ROOT="$QA_ROOT/translation"
mkdir -p "$TR_ROOT/news_en" "$TR_ROOT/mixed_dir"

cat > "$TR_ROOT/news_en/shard_0001.jsonl" <<'EOF'
{"text":"The central bank raised its benchmark interest rate by a quarter point today."}
{"text":"Heavy monsoon rains caused widespread flooding across the western districts."}
EOF

cat > "$TR_ROOT/chat_en.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"The capital of France is Paris."}]}
{"messages":[{"role":"user","content":"Book a flight from Boston to Seattle next Tuesday."},{"role":"assistant","content":"Searching available flights now.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"search_flights","arguments":"{\"from\":\"BOS\",\"to\":\"SEA\",\"date\":\"2026-05-19\"}"}}]}]}
EOF

cat > "$TR_ROOT/chat_code_en.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"Show me a Python snippet that reads a CSV."},{"role":"assistant","content":"```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```"}]}
{"messages":[{"role":"user","content":"Call the weather tool for London."},{"role":"assistant","content":"Issuing the call.","tool_calls":[{"id":"call_w","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\",\"units\":\"metric\"}"}}]}]}
EOF

uv run python - <<PY
from pathlib import Path
import pandas as pd

root = Path("$TR_ROOT")
df = pd.DataFrame(
    [
        {"text": "A solar startup announced a new battery pilot in Nevada."},
        {"text": "Researchers published a study on multilingual model evaluation."},
    ]
)
df.to_parquet(root / "news_en.parquet", index=False)
PY

cp "$TR_ROOT/news_en/shard_0001.jsonl" "$TR_ROOT/mixed_dir/shard_0001.jsonl"
cp "$TR_ROOT/news_en.parquet" "$TR_ROOT/mixed_dir/shard_0002.parquet"
```

### Discovery And Dry Run

```bash
uv run nemotron steps show translate/translation
uv run nemotron steps run translate/translation --help
uv run nemotron steps translation --help
uv run python -c "from nemo_curator.stages.text.experimental.translation import TranslationStage; print(TranslationStage)"

uv run --extra translation nemotron steps translation \
  --dry-run \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/dry" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Expected result:

- Dry run prints resolved YAML.
- `source_language` and `target_language` are explicit.
- No output directory is created by dry run.

### LLM Translation

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for hosted LLM translation}"

uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_llm_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=both \
  reconstruct_messages=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Validate output:

```bash
uv run python - <<PY
from pathlib import Path
import json

root = Path("$TR_ROOT/out_llm_hi")
files = sorted(root.rglob("*.jsonl"))
assert files, f"No JSONL output found under {root}"
rows = []
for path in files:
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
assert rows, "No rows written"
assert any("translated_text" in row or "text" in row for row in rows), rows[0]
print({"files": len(files), "rows": len(rows), "sample": rows[0]})
PY
```

Expected result:

- Output JSONL shards exist.
- Row count matches input when FAITH filtering is disabled.
- Translated text is present.

### Structured Chat Translation

```bash
uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/chat_code_en.jsonl" \
  output_dir="$TR_ROOT/out_chat_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field='messages.*.content' \
  output_mode=both \
  reconstruct_messages=true \
  segmentation_mode=coarse \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Validate output:

```bash
uv run python - <<PY
from pathlib import Path
import json

root = Path("$TR_ROOT/out_chat_hi")
rows = []
for path in sorted(root.rglob("*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
assert rows, "No rows written"
assert any("messages" in row for row in rows), rows[0]
for row in rows:
    for msg in row.get("messages", []):
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                json.loads(call["function"]["arguments"])
print({"rows": len(rows), "tool_json_ok": True})
PY
```

Expected result:

- Tool-call JSON remains parseable.
- Code fences are not corrupted.
- Natural-language message content is translated or accompanied by translated output fields depending on `output_mode`.

### FAITH Annotation

```bash
uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_faith_annotated" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=both \
  reconstruct_messages=false \
  faith_eval.enabled=true \
  faith_eval.filter_enabled=false \
  faith_eval.model_name="$FAITH_MODEL" \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Expected result:

- Output rows are retained.
- FAITH score metadata is present.
- Logs do not print API keys.

### NMT Translation

```bash
: "${NMT_SERVER_URL:?Set NMT_SERVER_URL to an NMT service endpoint}"

uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/news_en" \
  output_dir="$TR_ROOT/out_nmt_hi" \
  source_language=en \
  target_language=hi \
  backend=nmt \
  nmt.server_url="$NMT_SERVER_URL" \
  text_field=text \
  output_mode=both \
  reconstruct_messages=false \
  faith_eval.enabled=false
```

Expected result:

- Translation succeeds without an LLM API key.
- Backend traffic goes to the configured NMT server.

### Parquet Translation

```bash
uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/news_en.parquet" \
  output_dir="$TR_ROOT/out_parquet_hi" \
  source_language=en \
  target_language=hi \
  input_format=parquet \
  output_format=parquet \
  backend=llm \
  text_field=text \
  reconstruct_messages=false \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY

uv run python - <<PY
from pathlib import Path
import pandas as pd

root = Path("$TR_ROOT/out_parquet_hi")
files = sorted(root.rglob("*.parquet"))
assert files, f"No Parquet output found under {root}"
df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
assert len(df) == 2, len(df)
print(df.head().to_dict(orient="records"))
PY
```

Expected result:

- Parquet output is readable.
- Row count is preserved when filtering is disabled.

### Resume And Negative Cases

```bash
uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/out_llm_hi" \
  output_dir="$TR_ROOT/out_resume_hi" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  output_mode=both \
  reconstruct_messages=false \
  skip_translated=true \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

```bash
if uv run --extra translation nemotron steps translation \
  input_path="$TR_ROOT/mixed_dir" \
  output_dir="$TR_ROOT/out_mixed_should_fail" \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
then
  echo "FAIL: mixed-format translation unexpectedly succeeded"
else
  echo "PASS: mixed-format translation failed as expected"
fi
```

Expected result:

- Resume run skips already translated rows where supported by the output schema.
- Mixed JSONL/Parquet input fails with a clear format error.

### Translation Agent Prompt

Ask the agent:

```text
Translate /tmp/nemotron-qa-<run-id>/translation/chat_code_en.jsonl from English to Hindi using the Nemotron translation step. Preserve tool-call JSON, do not filter rows, and show me the exact command before running it.
```

Expected agent behavior:

- Uses `nemotron steps translation` for local execution.
- Uses `text_field='messages.*.content'`.
- Sets `source_language=en` and `target_language=hi`.
- Keeps credentials in environment variables.
- Validates output JSON and row count.

## BYOB

### Test Data And Configs

```bash
export BYOB_ROOT="$QA_ROOT/byob"
mkdir -p "$BYOB_ROOT/input/maths" "$BYOB_ROOT/config" "$BYOB_ROOT/output"

cat > "$BYOB_ROOT/input/maths/overview.txt" <<'EOF'
Algebra studies symbols and the rules for manipulating them. Linear equations relate variables through addition and scalar multiplication.
Finance studies how people, firms, and governments allocate money over time. It includes saving, investing, borrowing, lending, budgeting, and risk management.
EOF

cp src/nemotron/steps/byob/config/tiny.yaml "$BYOB_ROOT/config/byob_mcq.yaml"
cp src/nemotron/steps/byob/config/translate.yaml "$BYOB_ROOT/config/byob_translate.yaml"
```

Patch the BYOB generation config:

```bash
uv run python - <<PY
from pathlib import Path
import os
import yaml

root = Path("$BYOB_ROOT")
cfg_path = root / "config" / "byob_mcq.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["expt_name"] = "byob_mcq_qa"
cfg["input_dir"] = str(root / "input")
cfg["output_dir"] = str(root / "output")
cfg["language"] = "en-US"
model = {
    "alias": "qa_generation_model",
    "model": os.environ.get("BYOB_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
    "provider": "nvidia",
    "inference_parameters": {
        "max_tokens": 1024,
        "max_parallel_requests": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}
cfg["generation_model_config"] = model
cfg["judge_model_config"] = model
cfg["distractor_expansion_model_config"] = model
cfg["distractor_validity_model_config"] = model
cfg["filtering_model_configs"] = {
    "hallucination": [{**model, "alias": "qa_hallucination"}],
    "easiness": [{**model, "alias": "qa_easiness"}],
}
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

Create a standalone benchmark for translation-only validation:

```bash
uv run python - <<PY
from pathlib import Path
import pandas as pd

root = Path("$BYOB_ROOT")
rows = [
    {
        "question_id": "finance-001",
        "question": "What does budgeting help manage?",
        "options": ["Weather", "Money", "Photos", "Music"],
        "answer_index": 1,
        "answer": "Money",
        "cot_content": "Budgeting tracks income and spending.",
        "src": "maths/overview.txt",
        "category": "maths",
    }
]
path = root / "output" / "translation_input" / "benchmark.parquet"
path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(path, index=False)
print(path)
PY
```

Patch the BYOB translation config:

```bash
uv run python - <<PY
from pathlib import Path
import os
import yaml

root = Path("$BYOB_ROOT")
cfg_path = root / "config" / "byob_translate.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["expt_name"] = "byob_mcq_translation_qa"
cfg["dataset_path"] = str(root / "output" / "translation_input" / "benchmark.parquet")
cfg["output_dir"] = str(root / "output")
cfg["source_language"] = "en-US"
cfg["target_language"] = "hi-IN"
cfg["remove_low_quality"] = False
cfg["translation_model_config"]["params"]["model"] = os.environ.get("TRANSLATION_MODEL", "openai/gpt-oss-120b")
cfg["translation_model_config"]["params"]["provider"] = "nvidia"
cfg["translation_model_config"]["params"]["api_key_env"] = "NGC_API_KEY"
cfg["translation_model_config"]["params"]["base_url"] = os.environ.get(
    "TRANSLATION_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

### Discovery

```bash
uv run --extra byob nemotron byob --help
uv run --extra byob nemotron byob --list-families
uv run python src/nemotron/steps/byob/scripts/validate.py --help
```

Expected result:

- `mcq` is listed as a family.
- Help output shows `prepare`, `generate`, `translate`, and `all`.

### BYOB All-Stage Generate

```bash
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY for BYOB generation}"

uv run --extra byob nemotron byob \
  --family mcq \
  --stage all \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

Validate benchmark artifacts:

```bash
find "$BYOB_ROOT/output" -maxdepth 4 -type f | sort

uv run python - <<PY
from pathlib import Path
import pandas as pd

root = Path("$BYOB_ROOT/output")
benchmarks = sorted(root.rglob("benchmark.parquet"))
assert benchmarks, f"No benchmark.parquet under {root}"
df = pd.read_parquet(benchmarks[-1])
required = {"question_id", "question", "options", "answer_index", "answer", "src", "category"}
missing = required - set(df.columns)
assert not missing, missing
assert len(df) > 0, "benchmark is empty"
print({"path": str(benchmarks[-1]), "rows": len(df), "columns": list(df.columns)})
PY
```

Expected result:

- `all` runs prepare followed by generate.
- Seed, raw benchmark, and final benchmark artifacts are written.
- Final benchmark schema contains MCQ fields.

Optional stage-by-stage commands for isolating failures:

```bash
uv run --extra byob nemotron byob \
  --family mcq \
  --stage prepare \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"

uv run --extra byob nemotron byob \
  --family mcq \
  --stage generate \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

### BYOB Resume

```bash
uv run --extra byob nemotron byob \
  --family mcq \
  --stage generate \
  --skip-until JUDGEMENT \
  --config "$BYOB_ROOT/config/byob_mcq.yaml"
```

Expected result:

- Earlier cached stages are reused.
- Later stages regenerate or validate without corrupting final schema.

### BYOB Translation

```bash
: "${NGC_API_KEY:?Set NGC_API_KEY or NVIDIA_API_KEY for BYOB translation}"

uv run --extra byob nemotron byob \
  --family mcq \
  --stage translate \
  --config "$BYOB_ROOT/config/byob_translate.yaml"
```

Validate translation artifacts:

```bash
find "$BYOB_ROOT/output" -maxdepth 5 -type f | sort

uv run python - <<PY
from pathlib import Path
import pandas as pd

root = Path("$BYOB_ROOT/output")
benchmarks = sorted(root.rglob("benchmark.parquet"))
assert benchmarks, f"No translated benchmark.parquet under {root}"
df = pd.read_parquet(benchmarks[-1])
required = {"question_id", "question", "options", "answer_index", "answer", "src", "category"}
missing = required - set(df.columns)
assert not missing, missing
assert len(df) > 0, "translated benchmark is empty"
print({"path": str(benchmarks[-1]), "rows": len(df), "columns": list(df.columns)})
PY
```

Expected result:

- Translated benchmark exists.
- `question_id`, options, answer, and answer index remain structurally valid.
- If `remove_low_quality=false`, row count should match the translation input.

### BYOB Agent Prompt

Ask the agent:

```text
Use BYOB to create a tiny MCQ benchmark from /tmp/nemotron-qa-<run-id>/byob/input and then translate the generated benchmark to Hindi. Use copied configs under /tmp/nemotron-qa-<run-id>/byob/config and do not edit repo configs.
```

Expected agent behavior:

- Uses `nemotron byob --family mcq --stage all` for prepare+generate, or explicitly explains why it is running `prepare` and `generate` separately.
- Uses `nemotron byob --family mcq --stage translate`.
- Patches config copies only.
- Validates final Parquet schema.

## Training

### Test Data

```bash
export TRN_ROOT="$QA_ROOT/training"
mkdir -p "$TRN_ROOT/data" "$TRN_ROOT/output"

cat > "$TRN_ROOT/data/sft_chat.jsonl" <<'EOF'
{"messages":[{"role":"user","content":"Summarize what a budget is."},{"role":"assistant","content":"A budget is a plan for income, spending, saving, and borrowing over time."}]}
{"messages":[{"role":"user","content":"Give one reason to diversify investments."},{"role":"assistant","content":"Diversification spreads risk across assets instead of concentrating it in one place."}]}
EOF

cat > "$TRN_ROOT/data/pretrain.txt" <<'EOF'
Finance studies how people and organizations allocate resources under uncertainty.
Risk management identifies, measures, and mitigates sources of financial loss.
EOF

cat > "$TRN_ROOT/data/preferences.jsonl" <<'EOF'
{"prompt":"Explain diversification in one sentence.","chosen":"Diversification spreads investment risk across multiple assets.","rejected":"Diversification means buying only one stock."}
{"prompt":"What is budgeting?","chosen":"Budgeting plans income and expenses over a period of time.","rejected":"Budgeting predicts tomorrow's weather."}
EOF
```

### Discovery And Dry Runs

```bash
uv run nemotron steps list --json > "$TRN_ROOT/steps.json"
uv run nemotron steps show data_prep/sft_packing --json > "$TRN_ROOT/data_prep_sft_packing.json"
uv run nemotron steps show data_prep/pretrain_prep --json > "$TRN_ROOT/data_prep_pretrain_prep.json"
uv run nemotron steps show data_prep/rl_prep --json > "$TRN_ROOT/data_prep_rl_prep.json"
uv run nemotron steps show sft/automodel --json > "$TRN_ROOT/sft_automodel.json"
uv run nemotron steps show sft/megatron_bridge --json > "$TRN_ROOT/sft_megatron_bridge.json"
uv run nemotron steps show peft/automodel --json > "$TRN_ROOT/peft_automodel.json"
uv run nemotron steps show peft/megatron_bridge --json > "$TRN_ROOT/peft_megatron_bridge.json"
uv run nemotron steps show pretrain/automodel --json > "$TRN_ROOT/pretrain_automodel.json"
uv run nemotron steps show pretrain/megatron_bridge --json > "$TRN_ROOT/pretrain_megatron_bridge.json"
uv run nemotron steps show rl/nemo_rl/dpo --json > "$TRN_ROOT/rl_dpo.json"
uv run nemotron steps show rl/nemo_rl/rlvr --json > "$TRN_ROOT/rl_rlvr.json"
uv run nemotron steps show rl/nemo_rl/rlhf --json > "$TRN_ROOT/rl_rlhf.json"
uv run nemotron steps show optimize/modelopt/quantize --json > "$TRN_ROOT/modelopt_quantize.json"
uv run nemotron steps show optimize/modelopt/prune --json > "$TRN_ROOT/modelopt_prune.json"
uv run nemotron steps show optimize/modelopt/distill --json > "$TRN_ROOT/modelopt_distill.json"

uv run nemotron steps run data_prep/sft_packing -c tiny --dry-run
uv run nemotron steps run data_prep/pretrain_prep -c tiny --dry-run
uv run nemotron steps run data_prep/rl_prep -c tiny --dry-run
uv run nemotron steps run sft/automodel -c tiny --dry-run
uv run nemotron steps run sft/megatron_bridge -c tiny --dry-run
uv run nemotron steps run peft/automodel -c tiny --dry-run
uv run nemotron steps run peft/megatron_bridge -c tiny --dry-run
uv run nemotron steps run pretrain/automodel -c tiny --dry-run
uv run nemotron steps run pretrain/megatron_bridge -c tiny --dry-run
uv run nemotron steps run rl/nemo_rl/dpo -c tiny --dry-run
uv run nemotron steps run rl/nemo_rl/rlvr -c tiny --dry-run
uv run nemotron steps run rl/nemo_rl/rlhf -c tiny --dry-run
uv run nemotron steps run optimize/modelopt/quantize -c tiny --dry-run
uv run nemotron steps run optimize/modelopt/prune -c tiny --dry-run
uv run nemotron steps run optimize/modelopt/distill -c tiny --dry-run
```

Expected result:

- All listed steps are discoverable.
- Dry runs compile configs and show resolved resources.
- No remote job is submitted during dry run.

### Local Prep Smoke

```bash
SFT_OUTPUT_DIR="$TRN_ROOT/output/sft_packing" \
  uv run nemotron steps run data_prep/sft_packing -c tiny

PRETRAIN_OUTPUT_DIR="$TRN_ROOT/output/pretrain_prep" \
  uv run nemotron steps run data_prep/pretrain_prep -c tiny

RL_OUTPUT_DIR="$TRN_ROOT/output/rl_prep" \
  uv run nemotron steps run data_prep/rl_prep -c tiny
```

Validate outputs:

```bash
find "$TRN_ROOT/output" -maxdepth 5 -type f | sort | tee "$TRN_ROOT/output_files.txt"
test -s "$TRN_ROOT/output_files.txt"
```

Expected result:

- Prep outputs are written under `TRN_ROOT/output`.
- Missing tokenizer, dataset, or package failures are clear and actionable.

### W&B Offline Dry Run

```bash
WANDB_MODE=offline \
SFT_OUTPUT_DIR="$TRN_ROOT/output/sft_automodel" \
  uv run nemotron steps run sft/automodel -c tiny --dry-run
```

Expected result:

- Dry-run output includes W&B offline settings where supported.
- No API key is printed.

### Training Agent Prompt

Ask the agent:

```text
Prepare a tiny SFT workflow for /tmp/nemotron-qa-<run-id>/training/data/sft_chat.jsonl. Show the prep step, the AutoModel SFT option, and the Megatron-Bridge option. Do not run full training unless I provide a GPU executor.
```

Expected agent behavior:

- Uses `data_prep/sft_packing` before Megatron-Bridge SFT/PEFT.
- Does not add `data_prep/sft_packing` before AutoModel unless the chosen step requires packed data.
- Uses tiny or dry-run first.
- Explains what requires GPU or remote executor.

## SDG

### Test Data And Config

```bash
export SDG_ROOT="$QA_ROOT/sdg"
mkdir -p "$SDG_ROOT/config" "$SDG_ROOT/data" "$SDG_ROOT/output"

cat > "$SDG_ROOT/data/topic_seeds.jsonl" <<'EOF'
{"topic":"budgeting basics"}
{"topic":"investment diversification"}
EOF

cp src/nemotron/steps/sdg/data_designer/config/tiny.yaml "$SDG_ROOT/config/sdg_tiny.yaml"
```

Patch the copied SDG config:

```bash
uv run python - <<PY
from pathlib import Path
import os
import yaml

root = Path("$SDG_ROOT")
cfg_path = root / "config" / "sdg_tiny.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["output_dir"] = str(root / "output")
cfg["output_path"] = str(root / "output" / "sft-tiny.jsonl")
cfg["num_records"] = 5
cfg["seed_dataset"] = {
    "path": str(root / "data" / "topic_seeds.jsonl"),
    "strategy": "ordered",
    "fields": ["topic"],
}
for model in cfg.get("models", []):
    model["model"] = os.environ.get("SDG_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    model["provider"] = "nvidia"
    model["skip_health_check"] = True
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path)
PY
```

### Discovery And Run

```bash
uv run nemotron steps list --category sdg --json
uv run nemotron steps show sdg/data_designer --json
uv run nemotron steps run sdg/data_designer -c "$SDG_ROOT/config/sdg_tiny.yaml" --dry-run
```

Real generation:

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for SDG generation}"

uv run nemotron steps run sdg/data_designer -c "$SDG_ROOT/config/sdg_tiny.yaml"
```

Validate output:

```bash
uv run python - <<PY
from pathlib import Path
import json

path = Path("$SDG_ROOT/output/sft-tiny.jsonl")
assert path.exists(), path
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert rows, "No SDG rows"
assert all("messages" in row for row in rows), rows[0]
print({"rows": len(rows), "sample": rows[0]})
PY
```

Expected result:

- Dry run compiles without model calls.
- Real run writes OpenAI-style `messages`.
- Invalid endpoint/key failures identify the missing credential or service.

### SDG Agent Prompt

Ask the agent:

```text
Generate five synthetic SFT chat examples about budgeting and diversification using sdg/data_designer. Use a copied config under /tmp/nemotron-qa-<run-id>/sdg/config and validate that the output is JSONL with messages.
```

Expected agent behavior:

- Uses `sdg/data_designer`.
- Copies and patches config instead of editing repo files.
- Runs dry-run before real generation.
- Validates output JSONL schema.

## Airgap

### Plan Commands

Airgap plan mode is the default. Do not pass `--execute` unless QA is intentionally building and saving images.

```bash
export AIRGAP_ROOT="$QA_ROOT/airgap"
mkdir -p "$AIRGAP_ROOT"

uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml

uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target sdg/data_designer:tiny

uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target data_prep/sft_packing:tiny \
  --target sft/megatron_bridge:tiny

uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --target sdg/data_designer:tiny \
  --target data_prep/sft_packing:tiny \
  --target sft/megatron_bridge:tiny
```

Dependency discovery plan:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --stage validate \
  --stage discover-execution-deps \
  --target sdg/data_designer:tiny \
  --target sft/megatron_bridge:tiny
```

Optional connected-machine image build:

```bash
uv run python deploy/nemotron-customizer/airgap/runner.py \
  --config deploy/nemotron-customizer/airgap/airgap.yaml \
  --execute \
  --target sdg/data_designer:tiny \
  --target sft/megatron_bridge:tiny
```

Expected result:

- Plan mode validates selected targets without building images.
- Selected targets and dependencies are printed.
- Models, datasets, checkpoints, and customer files remain external to images.
- Execute mode writes outputs under `deploy/nemotron-customizer/airgap/out/` if Docker and registry access are available.

### Airgap Agent Prompt

Ask the agent:

```text
Create an airgap plan for SDG plus Megatron-Bridge SFT. Validate the plan only; do not build images. Explain which assets remain outside the images.
```

Expected agent behavior:

- Uses `deploy/nemotron-customizer/airgap/runner.py`.
- Omits `--execute` for plan-only validation.
- Selects scoped targets.
- States that models, datasets, checkpoints, and customer data remain on persistent storage.

## Lepton Testing

### Generate Lepton Env File

```bash
export LEPTON_ROOT="$QA_ROOT/lepton"
mkdir -p "$LEPTON_ROOT"

uv run nemotron steps run env/env_toml \
  -c lepton \
  output_path="$LEPTON_ROOT/env.lepton.toml" \
  force=true

export NEMOTRON_ENV_FILE="$LEPTON_ROOT/env.lepton.toml"
```

Expected result:

- `env.lepton.toml` is generated.
- QA reviews and edits workspace, node group, mounts, shapes, and secret passthrough before submitting jobs.

### Lepton Dry Runs

```bash
uv run nemotron steps run translate/translation \
  -c default \
  --dry-run \
  --batch lepton_translate \
  input_path=/mnt/lustre-shared/data/nemotron-qa/tiny.jsonl \
  output_dir=/mnt/lustre-shared/output/nemotron-qa/translated \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  faith_eval.enabled=false \
  server.model="$TRANSLATION_MODEL"

uv run nemotron steps run data_prep/sft_packing \
  -c tiny \
  --dry-run \
  --batch lepton_prep_sft_packing

uv run nemotron steps run sft/automodel \
  -c tiny \
  --dry-run \
  --batch lepton_sft_automodel

uv run nemotron steps run sft/megatron_bridge \
  -c tiny \
  --dry-run \
  --batch lepton_sft_megatron_bridge

uv run nemotron steps run rl/nemo_rl/dpo \
  -c tiny \
  --dry-run \
  --batch lepton_rl_nemo_rl_dpo

uv run nemotron steps run optimize/modelopt/quantize \
  -c tiny \
  --dry-run \
  --batch lepton_optimize_modelopt_quantize
```

### Lepton Real Smoke Commands

Run only after the env file has valid site-specific values:

```bash
uv run nemotron steps run data_prep/sft_packing \
  -c tiny \
  --batch lepton_prep_sft_packing

uv run nemotron steps run translate/translation \
  -c default \
  --batch lepton_translate \
  input_path=/mnt/lustre-shared/data/nemotron-qa/tiny.jsonl \
  output_dir=/mnt/lustre-shared/output/nemotron-qa/translated \
  source_language=en \
  target_language=hi \
  backend=llm \
  text_field=text \
  faith_eval.enabled=false \
  server.url="$TRANSLATION_BASE_URL" \
  server.model="$TRANSLATION_MODEL" \
  server.api_key_env=NVIDIA_API_KEY
```

Expected result:

- Dry runs compile against batch profiles.
- Real runs return Lepton job IDs or complete locally depending on executor behavior.
- Remote logs show mounted paths and do not print secrets.

### Lepton Agent Prompt

Ask the agent:

```text
Convert this local translation validation into a Lepton run using the generated env.lepton.toml. Use the generic steps runner, not the local translation shortcut.
```

Expected agent behavior:

- Uses `nemotron steps run translate/translation`.
- Uses `--batch lepton_translate`.
- Keeps input/output under shared storage.
- Does not use `nemotron steps translation` for Lepton.

## Evaluation

### Test Data

```bash
export EVAL_ROOT="$QA_ROOT/eval"
mkdir -p "$EVAL_ROOT/results"

cat > "$EVAL_ROOT/results/synthetic_results.jsonl" <<'EOF'
{"benchmark":"qa-smoke","sample_id":"1","prediction":"B","answer":"B","score":1.0}
{"benchmark":"qa-smoke","sample_id":"2","prediction":"A","answer":"C","score":0.0}
EOF
```

### Discovery And Dry Run

```bash
uv run nemotron steps list --category eval --json
uv run nemotron steps show eval/model_eval --json

uv run nemotron steps run eval/model_eval \
  -c tiny \
  --dry-run \
  output_dir="$EVAL_ROOT/results-tiny" \
  deployment.url="$EVAL_URL" \
  deployment.model_id="$EVAL_MODEL_ID" \
  deployment.api_key_name=NVIDIA_API_KEY \
  params.limit_samples=1 \
  params.extra.tokenizer="$EVAL_TOKENIZER"
```

Expected result:

- Eval step is discoverable.
- Dry run resolves endpoint/model/tokenizer parameters.
- No endpoint call is made during dry run.

### Hosted Eval Smoke

```bash
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for hosted eval}"
: "${EVAL_URL:?Set EVAL_URL for the eval endpoint}"
: "${EVAL_MODEL_ID:?Set EVAL_MODEL_ID for the eval endpoint}"
: "${EVAL_TOKENIZER:?Set EVAL_TOKENIZER to a tokenizer path visible to the run}"

uv run nemotron steps run eval/model_eval \
  -c tiny \
  output_dir="$EVAL_ROOT/results-tiny" \
  deployment.url="$EVAL_URL" \
  deployment.model_id="$EVAL_MODEL_ID" \
  deployment.api_key_name=NVIDIA_API_KEY \
  params.limit_samples=1 \
  params.extra.tokenizer="$EVAL_TOKENIZER"
```

Validate output:

```bash
find "$EVAL_ROOT/results-tiny" -maxdepth 5 -type f | sort
```

Expected result:

- One-sample eval runs or fails with a clear endpoint/tokenizer/credential error.
- Result files are written under `EVAL_ROOT/results-tiny` when successful.
- Secrets are not printed in logs.

### Evaluation Agent Prompt

Ask the agent:

```text
Set up a one-sample hosted evaluation smoke run for my model endpoint. Ask me for any missing endpoint, model ID, tokenizer path, or API key environment variable before running.
```

Expected agent behavior:

- Uses `eval/model_eval`.
- Runs dry-run first.
- Asks for missing endpoint/model/tokenizer/key instead of inventing values.
- Validates output artifact paths after the real run.

## Reporting

Use these result labels:

| Result | Meaning |
| --- | --- |
| `PASS` | Command ran and all expected artifacts or behavior were observed. |
| `FAIL` | Product behavior violated the expected result. |
| `BLOCKED` | Required credential, endpoint, GPU, Lepton workspace, Docker daemon, image, quota, or external service was unavailable. |
| `NOT RUN` | Intentionally skipped; reason must be recorded. |

For each module, report:

- Direct CLI result.
- Agent-driven result.
- Exact command used.
- Config path and relevant overrides.
- Output artifact paths.
- Row counts or job IDs where applicable.
- Any redacted credential evidence.
- Final status: `PASS`, `FAIL`, `BLOCKED`, or `NOT RUN`.

## Exit Criteria

- General setup commands pass in a clean `uv` environment.
- Translation direct local workflow passes for at least one real backend, or unavailable backend is marked `BLOCKED`.
- BYOB prepare/generate and translation workflows pass, or unavailable hosted model/backend is marked `BLOCKED`.
- Training dry-runs pass for required training paths, and at least one prep smoke writes artifacts.
- SDG dry-run passes, and real generation either writes JSONL or is marked `BLOCKED` for missing endpoint/key.
- Airgap plan validation passes for SDG plus SFT targets.
- Lepton dry-runs compile against generated profiles, and real Lepton smoke is run or marked `BLOCKED`.
- Evaluation dry-run passes, and hosted eval smoke runs or is marked `BLOCKED`.
- Agent-driven runs choose the same module contracts and do not invent missing credentials, endpoints, paths, or model names.
