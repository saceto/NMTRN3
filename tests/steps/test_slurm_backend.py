# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nemotron.cli.commands.step.backends.base import JobContext
from nemotron.cli.commands.step.backends.slurm import SlurmBackend


def _ctx(env: dict, *, cmd: str | None = "python {script} --config {config}") -> JobContext:
    return JobContext(
        step_id="prep/rl_prep",
        script_path=Path("/repo/src/nemotron/steps/prep/rl_prep/step.py"),
        train_path=Path("/tmp/train.yaml"),
        spec=SimpleNamespace(
            run=SimpleNamespace(launch="python", cmd=cmd),
            image="test-image",
            resources=SimpleNamespace(nodes=1, gpus_per_node=0),
        ),
        env=env,
        env_vars={},
        passthrough=[],
        startup_commands=[],
        attached=False,
        force_squash=False,
    )


def test_slurm_build_cmd_keeps_existing_command_without_pip_extras() -> None:
    cmd = SlurmBackend._build_cmd(_ctx({}))

    assert cmd == "python main.py --config config.yaml"


def test_slurm_build_cmd_checks_preinstalled_imports() -> None:
    cmd = SlurmBackend._build_cmd(
        _ctx(
            {
                "pip_extras": ["cosmos-xenna"],
                "pip_install_mode": "preinstalled",
                "pip_required_imports": ["cosmos_xenna"],
            }
        )
    )

    assert cmd.startswith("{ python -c ")
    assert "slurm job missing Python imports" in cmd
    assert "cosmos_xenna" in cmd
    assert "pip install" not in cmd
    assert cmd.endswith("; } && python main.py --config config.yaml")


def test_slurm_build_cmd_installs_from_offline_wheelhouse() -> None:
    cmd = SlurmBackend._build_cmd(
        _ctx(
            {
                "pip_extras": ["cosmos-xenna"],
                "pip_install_mode": "offline_wheelhouse",
                "pip_wheelhouse": "/mnt/lustre-shared/airgap/wheels/nemo-25.11-nano",
                "pip_no_deps": True,
                "pip_required_imports": ["cosmos_xenna"],
            }
        )
    )

    assert cmd.startswith(
        "{ python -m pip install --no-index "
        "--find-links /mnt/lustre-shared/airgap/wheels/nemo-25.11-nano "
        "--no-deps cosmos-xenna && python -c "
    )
    assert "cosmos_xenna" in cmd
    assert cmd.endswith("; } && python main.py --config config.yaml")
