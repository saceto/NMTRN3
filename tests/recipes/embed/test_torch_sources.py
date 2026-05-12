"""Validate embed stage CUDA torch source metadata."""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
EMBED_DIR = REPO_ROOT / "src" / "nemotron" / "recipes" / "embed"
TORCH_STAGE_NAMES = [
    "stage0_sdg",
    "stage1_data_prep",
    "stage2_finetune",
    "stage3_eval",
    "stage4_export",
]


def test_embed_torch_stages_pin_linux_torch_to_cu129() -> None:
    cu129_source = [
        {"index": "pytorch-cu129", "marker": "sys_platform == 'linux'"}
    ]

    for stage_name in TORCH_STAGE_NAMES:
        with open(EMBED_DIR / stage_name / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        torch_sources = data["tool"]["uv"]["sources"]["torch"]
        assert torch_sources == cu129_source

        if stage_name == "stage4_export":
            assert data["tool"]["uv"]["sources"]["torchvision"] == cu129_source

        indexes = {entry["name"]: entry["url"] for entry in data["tool"]["uv"]["index"]}
        assert indexes["pytorch-cu129"] == "https://download.pytorch.org/whl/cu129"
        assert "pytorch-cu130" not in indexes


def test_embed_export_lock_pins_linux_torchvision_to_cu129() -> None:
    with open(EMBED_DIR / "stage4_export" / "uv.lock", "rb") as f:
        data = tomllib.load(f)

    torchvision_packages = [
        package
        for package in data["package"]
        if package["name"] == "torchvision"
    ]

    assert any(
        package["version"].endswith("+cu129")
        and package["source"]["registry"] == "https://download.pytorch.org/whl/cu129"
        for package in torchvision_packages
    )


def test_embed_export_stage_matches_finetune_python_and_transformers_ranges() -> None:
    with open(EMBED_DIR / "stage4_export" / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert data["project"]["requires-python"] == ">=3.12,<3.13"
    assert data["tool"]["uv"]["override-dependencies"] == ["transformers>=5.0,<5.2"]


def test_embed_export_lock_matches_finetune_transformers_range() -> None:
    with open(EMBED_DIR / "stage4_export" / "uv.lock", "rb") as f:
        data = tomllib.load(f)

    assert data["requires-python"] == "==3.12.*"

    transformer_packages = [
        package
        for package in data["package"]
        if package["name"] == "transformers"
    ]
    assert [package["version"] for package in transformer_packages] == ["5.1.0"]

    assert not any(
        package["name"] == "nvidia-resiliency-ext"
        and package["version"] == "0.5.0"
        for package in data["package"]
    )
