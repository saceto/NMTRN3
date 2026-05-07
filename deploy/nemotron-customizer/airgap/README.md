# Nemotron Customizer Airgap

This folder is scoped to **Nemotron Customizer** workflows represented by
`src/nemotron/steps/`. It is not a claim that every recipe, cookbook, doc, or
example in this repository is airgap-ready.

The airgap flow has three deliverables:

- **Submitter/runtime image**: local Docker image with repo source, `uv`,
  offline Python wheels, and offline environment defaults.
- **Remote task images**: mirrored Nano3, Super3, RL, or other task-specific
  images selected by the executor profile and pulled by Lepton/DGX/Slurm
  workers.
- **Persistent assets**: models, datasets, git repos, URL payloads,
  checkpoints, and optional remote wheelhouses on storage visible to the
  executor.

Do not bake large models and datasets into the submitter image by default. Keep
large assets on persistent storage and mount them into task containers.

## Simple Stage Map

Use the same stages for one step or a whole workflow:

| Stage | Who runs it | Output | Important rule |
| --- | --- | --- | --- |
| 1. Select and lock | Connected prep machine | `airgap.lock.yaml` | Lock all known `step_id:config` targets together. |
| 2. Fetch runtime | Connected prep machine | `airgap-bundle/runtime/` | Fetch wheels and small support assets; do not pull huge models by default. |
| 3. Stage assets | Customer/executor storage | HF cache, datasets, repos, wheelhouse, init script | Remote jobs read assets from persistent storage, not from the submitter image. |
| 4. Build submitter image | Connected or airgapped build host | Local Docker image | This image submits jobs; Lepton/Slurm workers still use task images. |
| 5. Smoke remote run | From inside submitter image | Remote job logs/checkpoints | Logs must show offline wheelhouse/cache paths and no public fetches. |

`nemotron step airgap plan` prints the same stage map before the asset tables.
Treat the stages as the contract: every command should fit one of these stages,
and every ad hoc fix should be folded back into the relevant stage.

For remote execution, the most important distinction is: the local submitter
image contains source + `uv` + wheels, while task containers consume models,
datasets, checkpoints, repos, and optional wheelhouses from executor-visible
persistent storage.

## Standard Layout

`nemotron step airgap fetch` writes this bundle shape:

```text
deploy/nemotron-customizer/airgap/airgap-bundle/
  runtime/
    wheels/
    requirements-airgap.txt
    requirements-airgap.source.txt
    requirements-build-system.txt
    offline.env
  assets/
    hf-cache/hub/
    repos/
    urls/
```

Only `runtime/` is copied into the submitter image. `assets/` is a staging
layout for local smoke tests or transfer. For remote execution, copy the same
contents to customer persistent storage visible to the executor and mount it at:

```text
/opt/nemotron-airgap/assets
```

The runtime image and docs standardize these container paths:

```bash
HF_HOME=/opt/nemotron-airgap/assets/hf-cache
HF_HUB_CACHE=/opt/nemotron-airgap/assets/hf-cache/hub
NEMOTRON_AIRGAP_REPOS=/opt/nemotron-airgap/assets/repos
```

The customer host path is intentionally not standardized. The standard is the
contents and the container mount point.

## Step-Family Risks

| Step family | Main airgap risks | Preferred delivery |
| --- | --- | --- |
| `prep`, `curate` | HF datasets, tokenizers, local data globs, Ray/temp output | Persistent dataset/model/tokenizer mounts |
| `pretrain`, `sft`, `peft`, `rl` | HF base models, checkpoints, training data, auto-mounted git repos, GPU task images | Mirrored task image plus mounted models/checkpoints/datasets/repos |
| `eval`, `benchmark` | Model endpoints, benchmark data, judge/provider APIs | In-network services and mounted benchmark/model data |
| `sdg`, `translate` | Provider endpoints, generated output paths | Customer-local service endpoints and mounted outputs |
| `convert`, `optimize` | Source checkpoints, export paths, ModelOpt/framework assets | Mounted checkpoints and output directories |

The lockfile records these as `assets`, `services`, `manual_inputs`, and
`unresolved_env`. Treat the lock and plan output as the customer delivery
contract.

## Connected Preparation (Stages 1, 2, and 4)

Run this section on a connected machine with access to git, package indexes,
Docker registries, Hugging Face, and private artifact stores.

### 1. Clone and sync

```bash
git clone "$NEMOTRON_REPO_URL" Nemotron
cd Nemotron
git rev-parse HEAD

uv sync --frozen --no-dev
uv run nemotron step list
```

Record the repo commit SHA with the customer handoff.

### 2. Select the workflow

Prefer locking every known step at once:

```bash
export WORKFLOW_NAME="customer-workflow"
export EXECUTOR_ENV_FILE="env.toml"
export IMAGE_TAG="nemotron-customizer-airgap:latest"

AIRGAP_TARGETS=(
  "prep/sft_packing:tiny"
  "sft/megatron_bridge:tiny"
)

EXECUTOR_PROFILES=(
  "lepton_prep_sft_packing_tiny"
  "lepton_sft_megatron_bridge_tiny"
)

PROFILE_ARGS=()
for profile in "${EXECUTOR_PROFILES[@]}"; do
  PROFILE_ARGS+=(--profile "$profile")
done
```

Inspect each step before locking:

```bash
for target in "${AIRGAP_TARGETS[@]}"; do
  uv run nemotron step show "${target%%:*}"
done
```

### 3. Lock

```bash
uv run nemotron step airgap lock-workflow \
  --name "$WORKFLOW_NAME" \
  --env-file "$EXECUTOR_ENV_FILE" \
  "${PROFILE_ARGS[@]}" \
  -o deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  "${AIRGAP_TARGETS[@]}"
```

The same lock can be produced through the cataloged local stage:

```bash
uv run nemotron step run env/airgap \
  "targets=[\"prep/sft_packing:tiny\",\"sft/megatron_bridge:tiny\"]" \
  "profiles=[\"lepton_prep_sft_packing_tiny\",\"lepton_sft_megatron_bridge_tiny\"]" \
  env_file="$EXECUTOR_ENV_FILE" \
  output_path=deploy/nemotron-customizer/airgap/airgap.lock.yaml
```

Single-step locking is still useful for debugging:

```bash
uv run nemotron step airgap lock sft/megatron_bridge \
  -c tiny \
  -o deploy/nemotron-customizer/airgap/airgap.lock.yaml
```

### 4. Plan and verify

```bash
uv run nemotron step airgap plan deploy/nemotron-customizer/airgap/airgap.lock.yaml

uv run nemotron step airgap verify deploy/nemotron-customizer/airgap/airgap.lock.yaml
```

Resolve the high-signal items before handoff:

- mirror task images into a customer-reachable registry and pin by digest
- pin Hugging Face revisions and git repos to immutable SHAs
- remove online `startup_commands`
- make remote pip policy explicit
- stage models, datasets, repos, wheelhouses, and Lepton init assets on
  persistent storage
- replace cloud provider configs with customer-local endpoints

### 5. Fetch runtime image inputs

```bash
uv run nemotron step airgap fetch deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -b deploy/nemotron-customizer/airgap/airgap-bundle \
  --include-wheels
```

This fetches Python wheels into `runtime/`. It does not download large models,
datasets, or git repos by default. That is deliberate for remote execution:
large assets should be staged directly to customer persistent storage.

Only fetch local assets when producing a portable transfer bundle or running a
local smoke:

```bash
uv run nemotron step airgap fetch deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -b deploy/nemotron-customizer/airgap/airgap-bundle \
  --include-assets
```

### 6. Build the submitter/runtime image

```bash
uv run nemotron step airgap build deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -f deploy/nemotron-customizer/airgap/Dockerfile \
  -t "$IMAGE_TAG" \
  --execute
```

If the selected base image exposes a non-default interpreter, pass
`--python-bin python3.10`, `--python-bin python3`, or a full path.

### 7. Save deliverables

```bash
mkdir -p deploy/nemotron-customizer/airgap/release

docker save "$IMAGE_TAG" | gzip > deploy/nemotron-customizer/airgap/release/runtime-image.tar.gz

tar -czf deploy/nemotron-customizer/airgap/release/metadata.tar.gz \
  deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  deploy/nemotron-customizer/airgap/Dockerfile \
  deploy/nemotron-customizer/airgap/Dockerfile.dockerignore \
  deploy/nemotron-customizer/airgap/README.md

if [ -d deploy/nemotron-customizer/airgap/airgap-bundle/assets ]; then
  tar -czf deploy/nemotron-customizer/airgap/release/asset-bundle.tar.gz \
    -C deploy/nemotron-customizer/airgap/airgap-bundle assets
fi
```

Deliver the runtime image, metadata, optional asset bundle, customer `env.toml`,
and notes for mounts, credentials, services, and registries.

## Customer Airgap Setup (Stages 3 and 4)

### 1. Load the submitter image

```bash
gunzip -c runtime-image.tar.gz | docker load
docker images | grep nemotron-customizer-airgap
```

### 2. Stage persistent assets

Place assets on executor-visible persistent storage. The expected HF cache shape
for model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` is:

```text
<persistent-root>/assets/hf-cache/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
  refs/
  snapshots/
  blobs/
```

For normal directory staging instead of HF cache staging, override the step
config path, for example:

```bash
hf_model_path=/mnt/lustre-shared/airgap/assets/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### 3. Configure Lepton init without GitHub

nemo-run's connected Lepton launcher fetches:

```text
https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh
```

For airgap, use one of these:

```bash
run.env.env_vars.NEMOTRON_LEPTON_INIT_SCRIPT=/mnt/lustre-shared/airgap/assets/lepton/lepton_env_to_pytorch.sh
```

When a Lepton executor profile is included in the lock and
`NEMOTRON_LEPTON_INIT_MODE=skip` is not set, the lock records the init script
as a small support URL asset. It is fetched during normal connected prep; you
do **not** need to fetch every large HF/model asset just to get this script:

```bash
uv run nemotron step airgap fetch deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -b deploy/nemotron-customizer/airgap/airgap-bundle \
  --include-wheels
```

Then copy `airgap-bundle/assets/lepton/lepton_env_to_pytorch.sh` to the
executor-visible asset root, for example:

```text
/mnt/lustre-shared/airgap/assets/lepton/lepton_env_to_pytorch.sh
```

or mount the script at:

```text
/opt/nemotron-airgap/assets/lepton/lepton_env_to_pytorch.sh
```

or, only when the task image/profile already provides the needed distributed
env, skip it:

```bash
run.env.env_vars.NEMOTRON_LEPTON_INIT_MODE=skip
```

### 4. Configure remote pip policy

This policy is understood by Lepton, DGX Cloud/run:ai, and Slurm step
backends. Use it for small Python gaps only; the best production answer is
still a mirrored task image with the right packages already installed.

Preferred:

```toml
pip_extras = ["cosmos-xenna"]
pip_install_mode = "preinstalled"
pip_required_imports = ["cosmos_xenna"]
```

Fallback:

```toml
pip_extras = ["cosmos-xenna"]
pip_install_mode = "offline_wheelhouse"
pip_wheelhouse = "/mnt/lustre-shared/airgap/wheels/nemo-25.11-nano"
pip_no_deps = true
pip_required_imports = ["cosmos_xenna"]
```

Use `pip_no_deps=true` when filling small gaps in a task image. Otherwise pip
can upgrade base packages from the wheelhouse and break the task image.

## Stage 5: True Remote Smoke

A real remote airgap smoke submits from the submitter image. The remote workers
still use the mirrored task image from the selected profile.

Common paths used by the examples:

```bash
export AIRGAP_ROOT=/mnt/lustre-shared/airgap
export AIRGAP_WHEELHOUSE=$AIRGAP_ROOT/wheels/nemo-25.11-nano
export AIRGAP_HF_HOME=$AIRGAP_ROOT/assets/hf-cache
export AIRGAP_HF_CACHE=$AIRGAP_HF_HOME/hub
export AIRGAP_LEPTON_INIT=$AIRGAP_ROOT/assets/lepton/lepton_env_to_pytorch.sh
```

Data prep:

```bash
docker run --rm -it \
  --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run prep/sft_packing \
    -c tiny \
    -b lepton_prep_sft_packing_tiny \
    output_dir=/mnt/lustre-shared/output/test/sft_dataprep_airgap_az \
    force=true \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=offline_wheelhouse \
    run.env.pip_wheelhouse="$AIRGAP_WHEELHOUSE" \
    run.env.pip_no_deps=true \
    'run.env.pip_required_imports=[cosmos_xenna]' \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

SFT:

```bash
docker run --rm -it \
  --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run sft/megatron_bridge \
    -c tiny \
    -b lepton_sft_megatron_bridge_tiny \
    'dataset.packed_sequence_specs.packed_train_data_path=/mnt/lustre-shared/output/test/sft_dataprep_airgap_az/splits/train/*.parquet' \
    checkpoint.save=/mnt/lustre-shared/output/test/sft_megatron_bridge_airgap_az/sft-tiny-2node \
    logger.wandb_project=null \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=preinstalled \
    'run.env.pip_required_imports=[typer,rich,pydantic_settings]' \
    run.env.env_vars.NEMOTRON_AIRGAP_REPOS="$AIRGAP_ROOT/assets/repos" \
    run.env.env_vars.HF_HOME="$AIRGAP_HF_HOME" \
    run.env.env_vars.HF_HUB_CACHE="$AIRGAP_HF_CACHE" \
    run.env.env_vars.HF_HUB_OFFLINE=1 \
    run.env.env_vars.TRANSFORMERS_OFFLINE=1 \
    run.env.env_vars.HF_DATASETS_OFFLINE=1 \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

Pretrain data prep against a pre-staged local blend:

```bash
docker run --rm -it \
  --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run prep/pretrain_prep \
    -c tiny \
    -b lepton_cpu_base \
    blend_path="$AIRGAP_ROOT/assets/datasets/pretrain_smoke/blend.json" \
    tokenizer.model=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    tokenizer.trust_remote_code=true \
    output_dir=/mnt/lustre-shared/output/test/pretrain_idxbin_airgap_az \
    execution_mode=batch \
    tokenization.cpus_per_worker=1 \
    force=true \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=offline_wheelhouse \
    run.env.pip_wheelhouse="$AIRGAP_WHEELHOUSE" \
    run.env.pip_no_deps=true \
    'run.env.pip_extras=[typer,rich,pydantic-settings,cosmos-xenna,obstore,cattrs,portpicker,protobuf]' \
    'run.env.pip_required_imports=[cosmos_xenna,obstore,cattrs,portpicker]' \
    run.env.env_vars.NEMOTRON_LEPTON_INIT_SCRIPT="$AIRGAP_LEPTON_INIT" \
    run.env.env_vars.HF_HOME="$AIRGAP_HF_HOME" \
    run.env.env_vars.HF_HUB_CACHE="$AIRGAP_HF_CACHE" \
    run.env.env_vars.HF_HUB_OFFLINE=1 \
    run.env.env_vars.TRANSFORMERS_OFFLINE=1 \
    run.env.env_vars.HF_DATASETS_OFFLINE=1 \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

CPT on the pretrain data-prep output:

```bash
docker run --rm -it \
  --network host \
  -v "$HOME/.cache/lepton:/root/.cache/lepton:ro" \
  -v "$PWD/env.toml:/workspace/Nemotron/env.toml:ro" \
  -e NEMOTRON_ENV_FILE=/workspace/Nemotron/env.toml \
  "$IMAGE_TAG" \
  uv run nemotron step run pretrain/megatron_bridge \
    -c tiny \
    -b lepton_pretrain_megatron_bridge_tiny \
    dataset.data_paths=/mnt/lustre-shared/output/test/pretrain_idxbin_airgap_az/blend.json \
    checkpoint.save=/mnt/lustre-shared/output/test/pretrain_cpt_airgap_az \
    train.train_iters=1 \
    scheduler.lr_warmup_iters=0 \
    checkpoint.save_interval=1 \
    logger.wandb_project=null \
    'run.env.startup_commands=[]' \
    run.env.pip_install_mode=offline_wheelhouse \
    run.env.pip_wheelhouse="$AIRGAP_WHEELHOUSE" \
    run.env.pip_no_deps=true \
    'run.env.pip_extras=[typer,rich,pydantic-settings,omegaconf]' \
    'run.env.pip_required_imports=[typer,rich,pydantic_settings,omegaconf]' \
    run.env.env_vars.NEMOTRON_LEPTON_INIT_SCRIPT="$AIRGAP_LEPTON_INIT" \
    run.env.env_vars.NEMOTRON_AIRGAP_REPOS="$AIRGAP_ROOT/assets/repos" \
    run.env.env_vars.HF_HOME="$AIRGAP_HF_HOME" \
    run.env.env_vars.HF_HUB_CACHE="$AIRGAP_HF_CACHE" \
    run.env.env_vars.HF_HUB_OFFLINE=1 \
    run.env.env_vars.TRANSFORMERS_OFFLINE=1 \
    run.env.env_vars.HF_DATASETS_OFFLINE=1 \
    'run.env.env_vars.WANDB_PROJECT=""' \
    run.env.env_vars.WANDB_ENABLED=false
```

For one-iteration Megatron Bridge smoke tests, set
`scheduler.lr_warmup_iters=0`. If warmup is `1`, use at least two training
iterations; Megatron requires warmup steps to be less than decay/train steps.

Read the logs with this checklist:

- no `raw.githubusercontent.com`, unless deliberately using connected fallback
- no PyPI/index access; pip should show `--no-index` or `Looking in links`
- no `git clone` from public URLs unless using connected fallback
- no Hugging Face network downloads in offline mode
- HF cache root points at the directory containing `models--...`
- W&B is disabled or configured for an in-network/offline mode

If a run fails with Hugging Face `LocalEntryNotFoundError`, offline mode is
working and the model or tokenizer is missing from the mounted cache.

## Building Inside The Airgap

Use this only when policy requires image construction inside the disconnected
environment. Transfer the repo checkout, `airgap.lock.yaml`,
`airgap-bundle/runtime/`, the Dockerfile, and mirrored base images. Large
`assets/` can still be mounted separately.

```bash
docker load < base-image.tar
cd Nemotron

uv run nemotron step airgap build deploy/nemotron-customizer/airgap/airgap.lock.yaml \
  -f deploy/nemotron-customizer/airgap/Dockerfile \
  -t "$IMAGE_TAG" \
  --execute
```
