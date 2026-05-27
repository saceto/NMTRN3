"""Tests for generated SDG JSON/JSONL loading."""

from __future__ import annotations

import json

from nemotron.recipes.embed.stage1_data_prep.scripts.convert_to_retriever_data import load_generated_json_files


def _record(name: str) -> dict:
    return {
        "file_name": name,
        "qa_evaluations": {"evaluations": []},
        "deduplicated_qa_pairs": [],
    }


def test_load_generated_json_file_allows_leading_whitespace_pretty_object(tmp_path):
    input_file = tmp_path / "generated.json"
    input_file.write_text("\n  " + json.dumps(_record("doc.txt"), indent=2), encoding="utf-8")

    df = load_generated_json_files(str(input_file))

    assert len(df) == 1
    assert df.iloc[0]["file_name"] == ["doc.txt"]


def test_load_generated_json_file_falls_back_to_jsonl(tmp_path):
    input_file = tmp_path / "generated.jsonl"
    input_file.write_text(
        "\n".join(json.dumps(_record(name)) for name in ["a.txt", "b.txt"]),
        encoding="utf-8",
    )

    df = load_generated_json_files(str(input_file))

    assert len(df) == 2
    assert df["file_name"].tolist() == [["a.txt"], ["b.txt"]]
