"""Tests for the Curator-backed translation step runtime."""

from __future__ import annotations

import pytest

from nemotron.steps.translate.translation import step as translation_step


def test_generation_config_keeps_known_fields_and_extra_kwargs() -> None:
    pytest.importorskip("nemo_curator")

    generation_config = translation_step._build_generation_config(
        {
            "max_tokens": 512,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
    )

    assert generation_config.max_tokens == 512
    assert generation_config.temperature == 0.1
    assert generation_config.extra_kwargs == {"response_format": {"type": "json_object"}}


def test_translation_stage_applies_generation_config_to_llm_translation() -> None:
    pytest.importorskip("nemo_curator")

    stage = translation_step._build_translation_stage(
        {
            "source_language": "en",
            "target_language": "hi",
            "backend": "llm",
            "text_field": "text",
            "server": {
                "model": "translation-model",
                "url": "http://example.invalid/v1",
                "api_key": "dummy",
            },
            "generation_config": {
                "max_tokens": 777,
                "temperature": 0.0,
            },
            "faith_eval": {
                "enabled": False,
            },
        }
    )

    segment_stage = next(item for item in stage.decompose() if item.name == "SegmentTranslationStage")
    assert segment_stage.generation_config.max_tokens == 777
    assert segment_stage.generation_config.temperature == 0.0


def test_translation_stage_applies_faith_generation_config_to_faith_filter() -> None:
    pytest.importorskip("nemo_curator")

    stage = translation_step._build_translation_stage(
        {
            "source_language": "en",
            "target_language": "hi",
            "backend": "llm",
            "text_field": "text",
            "server": {
                "model": "translation-model",
                "url": "http://example.invalid/v1",
                "api_key": "dummy",
            },
            "faith_eval": {
                "enabled": True,
                "filter_enabled": False,
                "model_name": "faith-model",
                "max_concurrent_requests": 3,
                "generation_config": {
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            },
        }
    )

    faith_stage = next(item for item in stage.decompose() if item.name == "FaithEvalFilter")
    assert faith_stage.generation_config.max_tokens == 2048
    assert faith_stage.generation_config.temperature == 0.0
    assert faith_stage.generation_config.extra_kwargs == {"response_format": {"type": "json_object"}}
    assert faith_stage.max_concurrent_requests == 3
