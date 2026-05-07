# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for step airgap lock compilation."""

import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from nemo_runspec import execution
from nemotron.cli.commands.step import airgap_cmd
from nemotron.cli.commands.step.airgap_cmd import (
    _git_resolve_target,
    _rewrite_direct_refs_to_locked_versions,
    airgap_app,
    fetch_airgap,
)
from nemotron.steps.airgap import AirgapCompiler, AirgapTarget, build_delivery_plan, lock_to_dict, verify_lock
from nemotron.steps.index import discover_steps


def test_airgap_lock_captures_uv_extra_and_hf_asset() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = AirgapCompiler(repo_root=repo_root).compile(step_id="prep/sft_packing", config_name="tiny")
    data = lock_to_dict(lock)

    assert data["runtime"]["python"]["manager"] == "uv"
    assert data["runtime"]["python"]["uv_version"] == "0.11.1"
    assert "xenna" in data["runtime"]["python"]["extras"]
    hf_asset = next(
        asset
        for asset in data["assets"]
        if asset["kind"] == "hf_model" and asset["id"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    )
    assert hf_asset["delivery"] == "external"
    assert hf_asset["bundle_path"].startswith("assets/hf-cache/hub/models--")
    assert data["runtime"]["bundle_layout"]["runtime_dir"] == "runtime"
    assert data["runtime"]["bundle_layout"]["assets_dir"] == "assets"
    assert data["delivery_plan"]["asset_locations"]["remote_persistent_root"] == "<customer-persistent-asset-root>"


def test_airgap_lock_captures_auto_mount_git_repos() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = AirgapCompiler(repo_root=repo_root).compile(step_id="sft/megatron_bridge", config_name="tiny")
    data = lock_to_dict(lock)
    git_ids = {asset["id"] for asset in data["assets"] if asset["kind"] == "git_repo"}

    assert "Megatron-LM" in git_ids
    assert "Megatron-Bridge" in git_ids
    assert any(asset["kind"] == "docker_image" for asset in data["runtime"]["base_images"])


def test_airgap_warns_when_tiny_training_smoke_warmup_equals_train_iters() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = AirgapCompiler(repo_root=repo_root).compile(
        step_id="pretrain/megatron_bridge",
        config_name="tiny",
        overrides=["train.train_iters=1"],
    )
    codes = {issue["code"] for issue in lock_to_dict(lock)["issues"]}

    assert "megatron_warmup_not_less_than_decay" in codes


def test_airgap_accepts_one_iter_training_smoke_with_zero_warmup() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = AirgapCompiler(repo_root=repo_root).compile(
        step_id="pretrain/megatron_bridge",
        config_name="tiny",
        overrides=["train.train_iters=1", "scheduler.lr_warmup_iters=0"],
    )
    codes = {issue["code"] for issue in lock_to_dict(lock)["issues"]}

    assert "megatron_warmup_not_less_than_decay" not in codes


def test_airgap_workflow_lock_merges_multiple_step_targets() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data = AirgapCompiler(repo_root=repo_root).compile_many(
        [
            AirgapTarget("prep/sft_packing", "tiny"),
            AirgapTarget("sft/megatron_bridge", "tiny"),
        ],
        workflow_name="nano3-sft-pack",
    )
    asset_ids = {asset["id"] for asset in data["assets"]}

    assert data["kind"] == "workflow"
    assert data["step"]["id"] == "nano3-sft-pack"
    assert len(data["steps"]) == 2
    assert "xenna" in data["runtime"]["python"]["extras"]
    assert "Megatron-LM" in asset_ids
    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in asset_ids


def test_airgap_lock_captures_executor_profile(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[cluster]
executor = "slurm"
host = "login.airgap.example"
remote_job_dir = "/lustre/nemotron/jobs"
mounts = ["/lustre/datasets:/datasets"]
env_vars = { HF_HOME = "/lustre/nemotron/hf" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["cluster"],
            env_file=env_file,
        )
    )

    assert data["executors"][0]["profile"] == "cluster"
    assert data["executors"][0]["executor"] == "slurm"
    assert any(service["id"] == "ssh://login.airgap.example" for service in data["services"])
    assert any(item["id"] == "/lustre/nemotron/jobs" for item in data["manual_inputs"])
    assert any(item["id"] == "/lustre/datasets" for item in data["manual_inputs"])


def test_airgap_remote_pip_extras_require_explicit_mode(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
pip_extras = ["cosmos-xenna"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )
    codes = {issue["code"] for issue in data["issues"]}

    assert "remote_pip_extras_implicit" in codes


def test_airgap_remote_pip_preinstalled_mode_is_explicit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
pip_extras = ["cosmos-xenna"]
pip_install_mode = "preinstalled"
pip_required_imports = ["cosmos_xenna"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )
    codes = {issue["code"] for issue in data["issues"]}

    assert "remote_pip_extras_implicit" not in codes
    assert "remote_pip_online_install" not in codes


def test_airgap_remote_wheelhouse_requires_no_deps_or_constraints(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
pip_extras = ["cosmos-xenna"]
pip_install_mode = "offline_wheelhouse"
pip_wheelhouse = "/mnt/lustre-shared/airgap/wheels/nemo-25.11-nano"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )
    codes = {issue["code"] for issue in data["issues"]}

    assert "remote_pip_wheelhouse_deps_mutable" in codes


def test_airgap_remote_startup_online_commands_are_flagged(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
startup_commands = ["python -m pip install cosmos-xenna"]
env_vars = { NEMOTRON_LEPTON_INIT_MODE = "skip" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )
    codes = {issue["code"] for issue in data["issues"]}

    assert "remote_startup_online_command" in codes


def test_airgap_lepton_profiles_warn_without_init_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )
    codes = {issue["code"] for issue in data["issues"]}
    lepton_init_assets = [
        asset
        for asset in data["assets"]
        if asset["kind"] == "url" and asset["id"].endswith("/lepton_env_to_pytorch.sh")
    ]

    assert "lepton_init_script_unset" in codes
    assert len(lepton_init_assets) == 1
    assert lepton_init_assets[0]["delivery"] == "external"
    assert lepton_init_assets[0]["bundle_path"] == "assets/lepton/lepton_env_to_pytorch.sh"


def test_airgap_lepton_skip_mode_does_not_bundle_init_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[lepton_smoke]
executor = "lepton"
container_image = "customer/nano3:latest"
nemo_run_dir = "/mnt/lustre-shared/nemo-run"
env_vars = { NEMOTRON_LEPTON_INIT_MODE = "skip" }
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = lock_to_dict(
        AirgapCompiler(repo_root=repo_root).compile(
            step_id="prep/sft_packing",
            config_name="tiny",
            profiles=["lepton_smoke"],
            env_file=env_file,
        )
    )

    assert not any(
        asset["kind"] == "url" and asset["id"].endswith("/lepton_env_to_pytorch.sh")
        for asset in data["assets"]
    )


def test_airgap_stage_is_discoverable() -> None:
    steps = {step.id: step for step in discover_steps()}

    assert "env/airgap" in steps
    step = steps["env/airgap"]
    assert step.produces[0].type == "airgap_lock"
    assert (step.path / "step.py").exists()
    assert (step.path / "config" / "default.yaml").exists()


def test_airgap_dockerfile_installs_uv_offline_and_uses_baked_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = (repo_root / "deploy/nemotron-customizer/airgap/Dockerfile").read_text(encoding="utf-8")

    assert "syntax=docker/dockerfile" not in dockerfile
    assert "COPY ${AIRGAP_BUNDLE}/runtime/wheels/" in dockerfile
    assert "COPY ${AIRGAP_BUNDLE}/ /opt/nemotron-airgap/" not in dockerfile
    assert "COPY src ./src" in dockerfile
    assert 'pip install --no-index --find-links=/opt/nemotron-airgap/wheels "uv==${UV_VERSION}"' in dockerfile
    assert "uv pip install --python .venv/bin/python --no-index" in dockerfile
    assert "--find-links /opt/nemotron-airgap/wheels" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "HF_HUB_CACHE=/opt/nemotron-airgap/assets/hf-cache/hub" in dockerfile
    assert "NEMOTRON_AIRGAP_REPOS=/opt/nemotron-airgap/assets/repos" in dockerfile
    assert "UV_NO_SYNC=1" in dockerfile
    assert "ARG PYTHON_BIN=python" in dockerfile


def test_airgap_verify_reports_unpinned_refs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = lock_to_dict(AirgapCompiler(repo_root=repo_root).compile(step_id="sft/automodel", config_name="tiny"))
    issues = verify_lock(lock)
    codes = {issue.code for issue in issues}

    assert "floating_git_ref" in codes


def test_airgap_verify_checks_python_bundle_files(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = lock_to_dict(AirgapCompiler(repo_root=repo_root).compile(step_id="prep/sft_packing", config_name="tiny"))

    codes = {issue.code for issue in verify_lock(lock, bundle_dir=tmp_path)}

    assert "bundle_wheelhouse_missing" in codes
    assert "bundle_requirements_missing" in codes
    assert "bundle_offline_env_missing" in codes


def test_airgap_verify_accepts_split_runtime_and_asset_bundle(tmp_path: Path) -> None:
    lock = {
        "runtime": {"python": {"manager": "uv"}},
        "assets": [
            {
                "kind": "hf_model",
                "id": "org/model",
                "repo_type": "model",
                "delivery": "external",
                "bundle_path": "assets/hf-cache/hub/models--org--model",
            }
        ],
    }
    runtime = tmp_path / "runtime"
    wheels = runtime / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "demo-0.1.0-py3-none-any.whl").write_text("", encoding="utf-8")
    (runtime / "requirements-airgap.txt").write_text("demo==0.1.0\n", encoding="utf-8")
    (runtime / "offline.env").write_text("HF_HUB_OFFLINE=1\n", encoding="utf-8")
    (tmp_path / "assets/hf-cache/hub/models--org--model").mkdir(parents=True)

    codes = {issue.code for issue in verify_lock(lock, bundle_dir=tmp_path)}

    assert "bundle_wheelhouse_missing" not in codes
    assert "bundle_requirements_missing" not in codes
    assert "bundle_offline_env_missing" not in codes
    assert "bundle_asset_missing" not in codes


def test_airgap_requirements_rewrite_turns_git_refs_into_local_pins() -> None:
    source = "nemo-run @ git+https://github.com/NVIDIA-NeMo/Run.git@abc123 ; python_version >= '3.10'\n"

    rewritten = _rewrite_direct_refs_to_locked_versions(source, {"nemo-run": "0.10.0+a8425c9"})

    assert rewritten == "nemo-run==0.10.0+a8425c9 ; python_version >= '3.10'\n"


def test_airgap_git_resolve_target_prefers_origin_branch(monkeypatch, tmp_path: Path) -> None:
    """Branch refs resolve to ``origin/<ref>`` after fetch; SHAs/tags pass through."""
    seen: list[list[str]] = []

    def fake_run(args, *_a, **_kw):  # type: ignore[no-untyped-def]
        seen.append(list(args))
        ok = args[-1] == "origin/main"
        return subprocess.CompletedProcess(args, returncode=0 if ok else 1, stdout=b"", stderr=b"")

    monkeypatch.setattr(airgap_cmd.subprocess, "run", fake_run)

    assert _git_resolve_target(tmp_path, "main") == "origin/main"
    assert _git_resolve_target(tmp_path, "abc1234") == "abc1234"
    # Both probes go through `git -C <dir> rev-parse --verify --quiet origin/<ref>`.
    assert seen == [
        ["git", "-C", str(tmp_path), "rev-parse", "--verify", "--quiet", "origin/main"],
        ["git", "-C", str(tmp_path), "rev-parse", "--verify", "--quiet", "origin/abc1234"],
    ]


def test_airgap_cli_lock_workflow_wiring(tmp_path: Path) -> None:
    """``step_app.add_typer(airgap_app, name='airgap')`` wiring + lock-workflow CLI smoke."""
    repo_root = Path(__file__).resolve().parents[2]
    lockfile = tmp_path / "airgap.lock.yaml"

    result = CliRunner().invoke(
        airgap_app,
        [
            "lock-workflow",
            "--name", "smoke",
            "--repo-root", str(repo_root),
            "-o", str(lockfile),
            "prep/sft_packing:tiny",
        ],
    )

    assert result.exit_code == 0, result.output
    assert lockfile.exists()
    data = yaml.safe_load(lockfile.read_text(encoding="utf-8"))
    assert data["kind"] == "workflow"
    assert data["step"]["id"] == "smoke"


def test_airgap_cli_json_output_is_parseable(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lockfile = tmp_path / "airgap.lock.yaml"

    result = CliRunner().invoke(
        airgap_app,
        [
            "lock",
            "prep/sft_packing",
            "-c", "tiny",
            "--repo-root", str(repo_root),
            "-o", str(lockfile),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = yaml.safe_load(result.output)
    assert data["runtime"]["bundle_layout"]["container_hf_cache"] == "/opt/nemotron-airgap/assets/hf-cache/hub"


def test_airgap_delivery_plan_standardizes_downloads_and_remote_staging() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock = lock_to_dict(AirgapCompiler(repo_root=repo_root).compile(step_id="sft/automodel", config_name="tiny"))

    plan = build_delivery_plan(lock)

    assert plan["stages"][0]["stage"] == "1. Select and lock"
    assert plan["stages"][2]["where"] == "customer/executor persistent storage"
    assert plan["asset_locations"]["connected_staging_dir"] == "<airgap-bundle>/assets"
    assert plan["asset_locations"]["remote_container_mount"].endswith(":/opt/nemotron-airgap/assets:ro")
    assert any(asset["kind"] == "hf_model" for asset in plan["download_assets"])
    assert any(asset["customer_action"] == "resolve_into_runtime_wheelhouse" for asset in plan["runtime_assets"])
    assert any(item["known_at_lock_time"] is False for item in plan["customer_inputs"])
    assert any(mount["scope"] == "remote_execution" for mount in plan["standard_mounts"])


def test_airgap_delivery_plan_marks_remote_execution() -> None:
    plan = build_delivery_plan(
        {
            "executors": [{"profile": "cluster", "executor": "slurm", "env_file": "env.toml"}],
            "runtime": {"python": {"manager": "uv"}},
            "assets": [
                {
                    "kind": "hf_model",
                    "id": "org/model",
                    "delivery": "external",
                    "bundle_path": "assets/hf-cache/hub/models--org--model",
                }
            ],
        }
    )

    assert plan["execution"]["mode"] == "remote"
    assert plan["execution"]["asset_fetch_default"] == "remote_stage"
    assert plan["execution"]["remote_executors"][0]["executor"] == "slurm"


def test_airgap_cli_plan_json(tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    lockfile.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "python": {"manager": "uv"},
                    "bundle_layout": {
                        "runtime_dir": "runtime",
                        "assets_dir": "assets",
                        "container_root": "/opt/nemotron-airgap",
                        "container_asset_root": "/opt/nemotron-airgap/assets",
                        "container_hf_home": "/opt/nemotron-airgap/assets/hf-cache",
                        "container_hf_cache": "/opt/nemotron-airgap/assets/hf-cache/hub",
                        "container_repos": "/opt/nemotron-airgap/assets/repos",
                    },
                },
                "assets": [
                    {
                        "kind": "hf_model",
                        "id": "org/model",
                        "delivery": "external",
                        "bundle_path": "assets/hf-cache/hub/models--org--model",
                    }
                ],
                "manual_inputs": [{"kind": "local_path", "id": "/customer/data/train.jsonl"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(airgap_app, ["plan", str(lockfile), "--json"])

    assert result.exit_code == 0, result.output
    plan = yaml.safe_load(result.output)
    assert plan["stages"][-1]["stage"] == "5. Smoke and verify"
    assert plan["download_assets"][0]["id"] == "org/model"
    assert plan["standard_mounts"][0]["scope"] == "local_smoke_test"
    assert plan["standard_mounts"][1]["scope"] == "remote_execution"


def test_airgap_cli_plan_prints_stage_map(tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    lockfile.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "python": {"manager": "uv"},
                    "bundle_layout": {
                        "runtime_dir": "runtime",
                        "assets_dir": "assets",
                        "container_root": "/opt/nemotron-airgap",
                        "container_asset_root": "/opt/nemotron-airgap/assets",
                        "container_hf_home": "/opt/nemotron-airgap/assets/hf-cache",
                        "container_hf_cache": "/opt/nemotron-airgap/assets/hf-cache/hub",
                        "container_repos": "/opt/nemotron-airgap/assets/repos",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(airgap_app, ["plan", str(lockfile)])

    assert result.exit_code == 0, result.output
    assert "Stages" in result.output
    assert "Select and" in result.output
    assert "Smoke and" in result.output


def test_airgap_fetch_tightens_lock_when_revisions_change(monkeypatch, tmp_path: Path) -> None:
    """Default ``--tighten-lock`` rewrites the lockfile after fetch resolves a floating ref."""
    lockfile = tmp_path / "airgap.lock.yaml"
    lock = {
        "assets": [
            {
                "kind": "git_repo",
                "id": "Run",
                "revision": "main",
                "note": "https://example.invalid/Run.git",
                "source": "pyproject.toml",
            },
        ],
        "runtime": {"python": {"manager": "uv"}},
    }
    lockfile.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    resolved_sha = "deadbeef" * 5

    def fake_fetch_assets(lock_data, *, assets_dir, dry_run):  # type: ignore[no-untyped-def]
        for asset in lock_data["assets"]:
            asset["revision"] = resolved_sha
        return ["git mocked"]

    monkeypatch.setattr(airgap_cmd, "_fetch_assets", fake_fetch_assets)

    fetch_airgap(
        lockfile=lockfile,
        bundle_dir=tmp_path / "bundle",
        dry_run=False,
        include_wheels=False,
        include_assets=True,
        tighten_lock=True,
    )

    rewritten = yaml.safe_load(lockfile.read_text(encoding="utf-8"))
    assert rewritten["assets"][0]["revision"] == resolved_sha


def test_airgap_fetch_skips_large_assets_by_default(monkeypatch, tmp_path: Path) -> None:
    lockfile = tmp_path / "airgap.lock.yaml"
    lockfile.write_text(
        yaml.safe_dump(
            {
                "assets": [
                    {
                        "kind": "hf_model",
                        "id": "org/model",
                        "delivery": "external",
                        "bundle_path": "assets/hf-cache/hub/models--org--model",
                    }
                ],
                "runtime": {"python": {"manager": "uv"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fail_fetch_assets(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("external assets should not be fetched by default")

    monkeypatch.setattr(airgap_cmd, "_fetch_assets", fail_fetch_assets)

    fetch_airgap(
        lockfile=lockfile,
        bundle_dir=tmp_path / "bundle",
        dry_run=False,
        include_wheels=False,
        tighten_lock=True,
    )

    assert (tmp_path / "bundle/runtime/offline.env").exists()
    assert not (tmp_path / "bundle/assets").exists()


def test_airgap_fetch_wheels_uses_seeded_pip_helper(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(args, *_a, **_kw):  # type: ignore[no-untyped-def]
        cmd = [str(arg) for arg in args]
        commands.append(cmd)
        if cmd[:2] == ["uv-bin", "export"]:
            Path(cmd[cmd.index("-o") + 1]).write_text("demo==1.0.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(args, returncode=0)

    monkeypatch.setattr(airgap_cmd, "_uv_command", lambda: ["uv-bin"])
    monkeypatch.setattr(airgap_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(
        airgap_cmd,
        "_write_offline_requirements",
        lambda _source, output, **_: output.write_text("demo==1.0.0\n", encoding="utf-8"),
    )
    monkeypatch.setattr(airgap_cmd, "_write_build_requirements", lambda _output, **_: None)

    result = airgap_cmd._fetch_wheels(
        {"runtime": {"python": {"extras": ["xenna"]}}},
        runtime_dir=tmp_path / "runtime",
        dry_run=False,
        repo_root=tmp_path,
    )

    pip_commands = [cmd for cmd in commands if cmd[1:3] == ["-m", "pip"]]
    assert "extras=['xenna']" in result
    assert any(cmd[:2] == ["uv-bin", "venv"] and "--seed" in cmd for cmd in commands)
    assert len(pip_commands) == 2
    assert all(".venv/bin/python" not in cmd[0] for cmd in pip_commands)
    assert "--only-binary=:all:" in pip_commands[0]
    assert "--only-binary=:all:" not in pip_commands[1]


def test_airgap_fetch_no_tighten_lock_keeps_lockfile_untouched(monkeypatch, tmp_path: Path) -> None:
    """``--no-tighten-lock`` opts out of the lock rewrite even when revisions resolved."""
    lockfile = tmp_path / "airgap.lock.yaml"
    lock = {
        "assets": [
            {
                "kind": "git_repo",
                "id": "Run",
                "revision": "main",
                "note": "https://example.invalid/Run.git",
                "source": "pyproject.toml",
            },
        ],
        "runtime": {"python": {"manager": "uv"}},
    }
    lockfile.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    original = lockfile.read_text(encoding="utf-8")

    def fake_fetch_assets(lock_data, *, assets_dir, dry_run):  # type: ignore[no-untyped-def]
        for asset in lock_data["assets"]:
            asset["revision"] = "deadbeef" * 5
        return ["git mocked"]

    monkeypatch.setattr(airgap_cmd, "_fetch_assets", fake_fetch_assets)

    fetch_airgap(
        lockfile=lockfile,
        bundle_dir=tmp_path / "bundle",
        dry_run=False,
        include_wheels=False,
        include_assets=True,
        tighten_lock=False,
    )

    assert lockfile.read_text(encoding="utf-8") == original


def test_airgap_fetch_downloads_lepton_init_without_all_assets(monkeypatch, tmp_path: Path) -> None:
    payload = b"#!/usr/bin/env bash\n"
    lockfile = tmp_path / "airgap.lock.yaml"
    lockfile.write_text(
        yaml.safe_dump(
            {
                "assets": [
                    {
                        "kind": "url",
                        "id": "https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh",
                        "source": "executor:lepton.env_vars.NEMOTRON_LEPTON_INIT_SCRIPT",
                        "delivery": "external",
                        "bundle_path": "assets/lepton/lepton_env_to_pytorch.sh",
                    },
                    {
                        "kind": "hf_model",
                        "id": "nvidia/huge-model",
                        "source": "step:hf_model_path",
                        "delivery": "external",
                        "bundle_path": "assets/hf-cache/hub/models--nvidia--huge-model",
                    },
                ],
                "runtime": {"python": {"manager": "uv"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_stream(_url, target):
        target.write_bytes(payload)
        import hashlib as _hl
        return _hl.sha256(payload).hexdigest()

    monkeypatch.setattr(airgap_cmd, "_stream_url_to_file", fake_stream)

    fetch_airgap(
        lockfile=lockfile,
        bundle_dir=tmp_path / "bundle",
        dry_run=False,
        include_wheels=False,
        include_assets=False,
        tighten_lock=False,
    )

    assert (tmp_path / "bundle" / "assets" / "lepton" / "lepton_env_to_pytorch.sh").read_bytes() == payload
    assert not (tmp_path / "bundle" / "assets" / "hf-cache").exists()


def test_airgap_workflow_lock_with_executor_profile_merges_dataclass_assets(tmp_path: Path) -> None:
    """``compile_many`` + executor profile: merge helpers must accept AirgapAsset dataclasses."""
    repo_root = Path(__file__).resolve().parents[2]
    env_file = tmp_path / "env.toml"
    env_file.write_text(
        """
[cluster]
executor = "slurm"
host = "login.airgap.example"
remote_job_dir = "/lustre/nemotron/jobs"
mounts = ["/lustre/datasets:/datasets"]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    data = AirgapCompiler(repo_root=repo_root).compile_many(
        [AirgapTarget("prep/sft_packing", "tiny")],
        workflow_name="cluster-smoke",
        profiles=["cluster"],
        env_file=env_file,
    )

    assert data["executors"][0]["profile"] == "cluster"
    assert any(item["id"] == "/lustre/datasets" for item in data["manual_inputs"])
    assert any(service["id"] == "ssh://login.airgap.example" for service in data["services"])


def test_airgap_resolve_step_rejects_ambiguous_tail_match(monkeypatch) -> None:
    """Tail-name match must refuse to silently pick a winner when more than one
    step shares the directory name."""
    from types import SimpleNamespace

    from nemotron.steps import airgap as airgap_mod

    fake_steps = [
        SimpleNamespace(id="prep/translate", path=Path("/repo/prep/translate")),
        SimpleNamespace(id="curate/translate", path=Path("/repo/curate/translate")),
    ]
    monkeypatch.setattr(airgap_mod, "discover_steps", lambda: fake_steps)

    with pytest.raises(ValueError, match="ambiguous"):
        airgap_mod._resolve_step("translate")


def test_airgap_python_runtime_warns_when_pyproject_missing(tmp_path: Path) -> None:
    """A wrong --repo-root used to silently emit an empty wheelhouse plan."""
    from types import SimpleNamespace

    from nemotron.steps.airgap import AirgapCompiler

    compiler = AirgapCompiler(repo_root=tmp_path)
    issues: list = []
    spec = SimpleNamespace(
        run=SimpleNamespace(cmd="uv run --extra demo python script.py"),
    )
    runtime = airgap_cmd_module(spec, compiler, issues)

    assert runtime["files"] == []
    assert any(issue.code == "missing_python_metadata" for issue in issues)


def airgap_cmd_module(spec, compiler, issues):
    """Tiny helper to call the private ``_python_runtime`` for the prior test."""
    from nemotron.steps.airgap import _python_runtime

    return _python_runtime(compiler.repo_root, spec=spec, manifest={}, issues=issues)


def test_airgap_lock_workflow_target_spec_supports_per_target_overrides() -> None:
    """``step_id:config+key=val,key=val`` parses into per-target overrides."""
    target = airgap_cmd._parse_target_spec("sft/automodel:tiny+dataset.repo_id=org/repo,trainer.max_steps=10")

    assert target.step_id == "sft/automodel"
    assert target.config_name == "tiny"
    assert target.overrides == ("dataset.repo_id=org/repo", "trainer.max_steps=10")


def test_airgap_lock_workflow_target_spec_keeps_nested_commas() -> None:
    """Hydra values may contain commas inside lists, dicts, or quoted strings."""
    target = airgap_cmd._parse_target_spec(
        'sft/automodel:tiny+dataset.names=["train","eval"],trainer.note="a,b",model.layers={a:1,b:2}'
    )

    assert target.overrides == (
        'dataset.names=["train","eval"]',
        'trainer.note="a,b"',
        "model.layers={a:1,b:2}",
    )


def test_airgap_url_fetch_records_sha256_and_rejects_mismatch(monkeypatch, tmp_path: Path) -> None:
    """``_fetch_url`` must record sha256 and refuse to overwrite the lock when
    the recorded ``expected_sha256`` does not match the downloaded payload."""
    payload = b"hello-world\n"

    def fake_stream(_url, target):
        target.write_bytes(payload)
        import hashlib as _hl
        return _hl.sha256(payload).hexdigest()

    monkeypatch.setattr(airgap_cmd, "_stream_url_to_file", fake_stream)
    asset = {"id": "https://example.invalid/file.tar.gz", "kind": "url"}
    airgap_cmd._fetch_url(asset, assets_dir=tmp_path, dry_run=False)

    import hashlib as _hl
    assert asset["sha256"] == _hl.sha256(payload).hexdigest()

    bad_asset = {
        "id": "https://example.invalid/file.tar.gz",
        "kind": "url",
        "expected_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        airgap_cmd._fetch_url(bad_asset, assets_dir=tmp_path, dry_run=False)


def test_airgap_url_fetch_honors_file_bundle_path(monkeypatch, tmp_path: Path) -> None:
    payload = b"#!/usr/bin/env bash\n"

    def fake_stream(_url, target):
        target.write_bytes(payload)
        import hashlib as _hl
        return _hl.sha256(payload).hexdigest()

    monkeypatch.setattr(airgap_cmd, "_stream_url_to_file", fake_stream)
    asset = {
        "id": "https://raw.githubusercontent.com/leptonai/scripts/main/lepton_env_to_pytorch.sh",
        "kind": "url",
        "bundle_path": "assets/lepton/lepton_env_to_pytorch.sh",
    }

    result = airgap_cmd._fetch_url(asset, assets_dir=tmp_path / "assets", dry_run=False)

    target = tmp_path / "assets" / "lepton" / "lepton_env_to_pytorch.sh"
    assert target.read_bytes() == payload
    assert asset["bundle_path"] == "assets/lepton/lepton_env_to_pytorch.sh"
    assert "lepton_env_to_pytorch.sh" in result


def test_airgap_url_fetch_refuses_non_http_schemes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-http"):
        airgap_cmd._fetch_url({"id": "ftp://example.invalid/file", "kind": "url"}, assets_dir=tmp_path, dry_run=False)


def test_airgap_git_fetch_resets_origin_to_lock_url(monkeypatch, tmp_path: Path) -> None:
    """When the bundle was originally cloned from upstream and the lock note is
    later rewritten to a customer mirror, subsequent fetches must re-point
    origin instead of silently pulling from the old upstream."""
    repo = tmp_path / "repos" / "Run"
    repo.mkdir(parents=True)  # marks "exists" for the existing-repo branch
    seen: list[list[str]] = []

    def fake_run(args, *_a, **_kw):
        seen.append([str(a) for a in args])
        return subprocess.CompletedProcess(args, returncode=0)

    monkeypatch.setattr(airgap_cmd.subprocess, "run", fake_run)
    monkeypatch.setattr(airgap_cmd, "_git_resolve_target", lambda *_a, **_kw: "abc1234")
    monkeypatch.setattr(airgap_cmd, "_git_head_sha", lambda *_a, **_kw: "abc1234")

    airgap_cmd._fetch_git(
        {"id": "Run", "note": "https://customer.invalid/Run.git", "revision": "abc1234"},
        assets_dir=tmp_path,
        dry_run=False,
    )

    assert any(cmd[:5] == ["git", "-C", str(repo), "remote", "set-url"] for cmd in seen)


def test_airgap_pip_no_deps_string_is_trimmed() -> None:
    """``pip_no_deps='true '`` (with trailing whitespace) used to be ignored."""
    from nemo_runspec.execution import _pip_extra_install_args

    assert "--no-deps" in _pip_extra_install_args({"pip_no_deps": "true "})
    assert "--no-deps" in _pip_extra_install_args({"pip_no_deps": " yes\n"})


def test_default_airgap_wheelhouse_matches_steps_constant() -> None:
    """Guard the cross-package literal duplicated to avoid an import cycle."""
    from nemo_runspec.execution import DEFAULT_AIRGAP_WHEELHOUSE
    from nemotron.steps.airgap import AIRGAP_CONTAINER_WHEELHOUSE

    assert DEFAULT_AIRGAP_WHEELHOUSE == AIRGAP_CONTAINER_WHEELHOUSE


def test_airgap_clone_repos_via_tunnel_reads_executor_env(monkeypatch) -> None:
    """The Slurm cluster typically only sets NEMOTRON_AIRGAP_REPOS cluster-side.
    The lookup must check the executor profile, not just os.environ."""
    from types import SimpleNamespace

    from nemo_runspec import execution as exec_mod

    monkeypatch.setattr(
        "nemo_runspec.config.resolvers.get_git_mounts",
        lambda: {
            "Megatron-LM": {
                "url": "https://x.invalid/Megatron-LM.git",
                "ref": "main",
                "target": "/opt/megatron-lm",
            }
        },
    )
    monkeypatch.delenv("NEMOTRON_AIRGAP_REPOS", raising=False)

    class FakeResult(SimpleNamespace):
        ok: bool
        stdout: str

    seen: list[str] = []

    class FakeTunnel:
        def run(self, cmd, *a, **kw):
            seen.append(cmd)
            if cmd.startswith("test -d /lustre/airgap/repos/Megatron-LM/.git"):
                return FakeResult(ok=True, stdout="exists\n")
            return FakeResult(ok=True, stdout="")

    env = {"airgap_repos": "/lustre/airgap/repos"}
    mounts = exec_mod.clone_git_repos_via_tunnel(FakeTunnel(), "/jobs", env=env)

    assert mounts == ["/lustre/airgap/repos/Megatron-LM:/opt/megatron-lm"]


def test_airgap_clone_repos_via_tunnel_accepts_positional_env(monkeypatch) -> None:
    """Older call sites may pass the executor env as the third positional arg."""
    from types import SimpleNamespace

    from nemo_runspec import execution as exec_mod

    monkeypatch.setattr(
        "nemo_runspec.config.resolvers.get_git_mounts",
        lambda: {"Run": {"url": "https://x.invalid/Run.git", "ref": "main", "target": "/opt/run"}},
    )
    monkeypatch.delenv("NEMOTRON_AIRGAP_REPOS", raising=False)

    class FakeTunnel:
        def run(self, cmd, *a, **kw):
            if cmd.startswith("test -d /lustre/airgap/repos/Run/.git"):
                return SimpleNamespace(ok=True, stdout="exists\n")
            return SimpleNamespace(ok=True, stdout="")

    mounts = exec_mod.clone_git_repos_via_tunnel(
        FakeTunnel(),
        "/jobs",
        {"env_vars": {"NEMOTRON_AIRGAP_REPOS": "/lustre/airgap/repos"}},
    )

    assert mounts == ["/lustre/airgap/repos/Run:/opt/run"]


def test_airgap_clone_repos_via_tunnel_preserves_two_arg_local_env_path(monkeypatch) -> None:
    """The original two-argument API still honors submitter-side env vars."""
    from types import SimpleNamespace

    from nemo_runspec import execution as exec_mod

    monkeypatch.setattr(
        "nemo_runspec.config.resolvers.get_git_mounts",
        lambda: {"Run": {"url": "https://x.invalid/Run.git", "ref": "main", "target": "/opt/run"}},
    )
    monkeypatch.setenv("NEMOTRON_AIRGAP_REPOS", "/lustre/airgap/repos")

    class FakeTunnel:
        def run(self, cmd, *a, **kw):
            if cmd == "printenv NEMOTRON_AIRGAP_REPOS":
                raise AssertionError("local env should avoid remote printenv probe")
            if cmd.startswith("test -d /lustre/airgap/repos/Run/.git"):
                return SimpleNamespace(ok=True, stdout="exists\n")
            return SimpleNamespace(ok=True, stdout="")

    mounts = exec_mod.clone_git_repos_via_tunnel(FakeTunnel(), "/jobs")

    assert mounts == ["/lustre/airgap/repos/Run:/opt/run"]


def test_airgap_clone_repos_via_tunnel_falls_back_to_remote_printenv(monkeypatch) -> None:
    """When no executor-side override exists and the submitter lacks the env
    var, probe the SSH login shell so cluster-only exports still work."""
    from types import SimpleNamespace

    from nemo_runspec import execution as exec_mod

    monkeypatch.setattr(
        "nemo_runspec.config.resolvers.get_git_mounts",
        lambda: {"Run": {"url": "https://x.invalid/Run.git", "ref": "main", "target": "/opt/run"}},
    )
    monkeypatch.delenv("NEMOTRON_AIRGAP_REPOS", raising=False)

    class FakeTunnel:
        def run(self, cmd, *a, **kw):
            if cmd == "printenv NEMOTRON_AIRGAP_REPOS":
                return SimpleNamespace(ok=True, stdout="/lustre/airgap/repos\n")
            if cmd.startswith("test -d /lustre/airgap/repos/Run/.git"):
                return SimpleNamespace(ok=True, stdout="exists\n")
            return SimpleNamespace(ok=True, stdout="")

    mounts = exec_mod.clone_git_repos_via_tunnel(FakeTunnel(), "/jobs", env=None)

    assert mounts == ["/lustre/airgap/repos/Run:/opt/run"]


def test_airgap_lock_workflow_cli_accepts_global_overrides(tmp_path: Path) -> None:
    """``lock-workflow`` should expose Hydra dotlist overrides for every target."""
    repo_root = Path(__file__).resolve().parents[2]
    lockfile = tmp_path / "airgap.lock.yaml"

    result = CliRunner().invoke(
        airgap_app,
        [
            "lock-workflow",
            "--name", "smoke",
            "--repo-root", str(repo_root),
            "-o", str(lockfile),
            "prep/sft_packing:tiny",
            "+dataset.foo=bar",  # workflow-wide; ignored by step but must parse
        ],
    )

    assert result.exit_code == 0, result.output
    assert lockfile.exists()


def test_airgap_cloud_git_mounts_prefer_mounted_asset_repos(monkeypatch) -> None:
    def fake_git_mounts() -> dict[str, dict[str, str]]:
        return {
            "Megatron-LM": {
                "url": "https://example.invalid/Megatron-LM.git",
                "ref": "main",
                "target": "/opt/megatron-lm",
            }
        }

    monkeypatch.setattr("nemo_runspec.config.resolvers.get_git_mounts", fake_git_mounts)

    commands = execution._git_mount_commands()

    assert "${NEMOTRON_AIRGAP_REPOS:-/opt/nemotron-airgap/assets/repos}/Megatron-LM" in commands[0]
    assert "cp -a" in commands[0]
    assert "git clone --depth 1 -b main https://example.invalid/Megatron-LM.git /opt/megatron-lm" in commands[0]
