from __future__ import annotations

from pathlib import Path
from typing import Any

from .annotation import ANNOTATION_FIELDS
from .io import read_json, read_jsonl
from .schema import CONDITIONS

EXPECTED_GENERATION_FIELDS = {
    "run_id",
    "item_id",
    "dataset",
    "condition",
    "model",
    "prompt",
    "raw_output",
    "parsed_final_answer",
    "parsed_reasoning_steps",
    "parsed_intermediate_claims",
    "declared_origin_tags",
    "gold_label",
    "context",
    "claim_or_question",
    "metadata",
}


def validate_data(data_file: str | Path) -> dict[str, Any]:
    items = read_jsonl(data_file)
    if len(items) != 40:
        raise AssertionError(f"Expected 40 examples, found {len(items)}")
    counts = {
        "scifact": sum(item.get("dataset") == "scifact" for item in items),
        "pubmedqa": sum(item.get("dataset") == "pubmedqa" for item in items),
    }
    if counts != {"scifact": 25, "pubmedqa": 15}:
        raise AssertionError(f"Unexpected dataset counts: {counts}")
    return {"examples": len(items), "dataset_counts": counts}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    generations_path = run_path / "generations.jsonl"
    annotation_path = run_path / "manual_annotation.csv"
    if not generations_path.exists():
        raise AssertionError(f"Missing {generations_path}")
    if not annotation_path.exists():
        raise AssertionError(f"Missing {annotation_path}")
    generations = read_jsonl(generations_path)
    manifest = read_json(run_path / "run_manifest.json")
    expected_generations = manifest.get("num_generations_expected", 200)
    if expected_generations is not None and len(generations) != expected_generations:
        raise AssertionError(f"Expected {expected_generations} generations, found {len(generations)}")
    seen_conditions = {record.get("condition") for record in generations}
    expected_conditions = set(manifest.get("conditions") or CONDITIONS)
    if seen_conditions != expected_conditions:
        raise AssertionError(f"Unexpected conditions: {seen_conditions}")
    for idx, record in enumerate(generations):
        missing = EXPECTED_GENERATION_FIELDS - set(record)
        if missing:
            raise AssertionError(f"Generation {idx} missing fields: {sorted(missing)}")
        if not isinstance(record.get("raw_output"), str):
            raise AssertionError(f"Generation {idx} does not preserve raw_output as a string")
    header = annotation_path.open("r", encoding="utf-8").readline().strip().split(",")
    missing_annotation_fields = set(ANNOTATION_FIELDS) - set(header)
    if missing_annotation_fields:
        raise AssertionError(f"Annotation file missing fields: {sorted(missing_annotation_fields)}")
    return {
        "generations": len(generations),
        "conditions": sorted(seen_conditions),
        "annotation_file": str(annotation_path),
    }
