# BYOB Guide

## Generation

Prepare source documents as one directory or parquet file per target subject. In the MCQ family, each target
subject maps to few-shot source subjects from the configured Hugging Face benchmark. The generation run:

1. Samples few-shot examples and target document chunks into `seed.parquet`.
2. Generates candidate MCQs with Data Designer.
3. Judges candidate quality.
4. Applies semantic deduplication.
5. Optionally expands distractors from four choices to ten.
6. Optionally measures document coverage.
7. Checks distractor validity and semantic outliers.
8. Applies hallucination and easiness filters.
9. Exports the final MCQ benchmark parquet.

## Translation

The translation run expects an MCQ benchmark parquet with stable question IDs and option lists. It translates
questions and choices, backtranslates the result, computes configured quality metrics, and exports the final
translated benchmark.

## Extending To Another Family

Read [new-family-checklist.md](new-family-checklist.md) first. Answer the schema, source-data,
quality-gate, and validation questions before editing code. Then add a new package under
`runtime/benchmark_families/<family>/`, implement the family-specific schema, prompts, postprocessing,
and export code, and register a `BenchmarkFamilySpec` in `runtime/benchmark_families/registry.py`.
Keep CLI parsing and stage dispatch unchanged.
