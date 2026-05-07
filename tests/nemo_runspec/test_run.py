# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for nemo-run compatibility patches."""

import pytest
from nemo_run.core.execution.lepton import LeptonExecutor

from nemo_runspec.run import patch_lepton_launcher_airgap_init


def test_lepton_launcher_patch_prefers_mounted_init_script(monkeypatch: pytest.MonkeyPatch) -> None:
    original_launch = LeptonExecutor.launch
    monkeypatch.delattr(LeptonExecutor, "_nemotron_airgap_init_patched", raising=False)

    patch_lepton_launcher_airgap_init()

    constants = "\n".join(str(item) for item in LeptonExecutor.launch.__code__.co_consts)

    assert "NEMOTRON_LEPTON_INIT_SCRIPT" in constants
    assert "/opt/nemotron-airgap/assets/lepton/lepton_env_to_pytorch.sh" in constants
    assert "NEMOTRON_LEPTON_INIT_MODE" in constants
    assert "raw.githubusercontent.com/leptonai/scripts" in constants
    # When NEMOTRON_LEPTON_INIT_SCRIPT is set but missing on the worker, the
    # patch must hard-fail rather than silently fall back to the GitHub script
    # (which is exactly what airgap deliveries cannot tolerate).
    assert "does not exist on the worker" in constants
    # Likewise, an explicit airgap marker should refuse to fall back online.
    assert "airgap mode requested but no init script" in constants

    monkeypatch.setattr(LeptonExecutor, "launch", original_launch)
    monkeypatch.delattr(LeptonExecutor, "_nemotron_airgap_init_patched", raising=False)
