from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json


def summarize_run(run_dir: str | Path, annotation_file: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    generations = read_jsonl(run_path / "generations.jsonl")
    annotation_path = Path(annotation_file) if annotation_file else run_path / "manual_annotation.csv"
    annotations = _read_csv(annotation_path) if annotation_path.exists() else []

    summary = {
        "total_examples": len({record["item_id"] for record in generations}),
        "total_generations": len(generations),
        "total_extracted_claims": sum(len(record.get("parsed_intermediate_claims") or []) for record in generations),
        "manual_annotation_rows": len(annotations),
        "laundering": _annotation_counts(annotations),
        "accuracy_by_condition": _accuracy_by(generations, key="condition"),
        "accuracy_by_dataset": _accuracy_by(generations, key="dataset"),
        "breakdown_by_condition": _annotation_breakdown(annotations, key="condition"),
        "breakdown_by_dataset": _annotation_breakdown(annotations, key="dataset"),
    }
    write_json(run_path / "metrics_summary.json", summary)
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1"}


def _annotated(value: str | None) -> bool:
    return bool(str(value or "").strip())


def _rate(count: int, denominator: int) -> float:
    return round(count / denominator, 4) if denominator else 0.0


def _annotation_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    candidate = sum(_truthy(row.get("laundering_candidate")) for row in rows)
    answer_dependent = sum(_truthy(row.get("final_answer_depends_on_claim")) for row in rows)
    misattributed = sum(_truthy(row.get("origin_misattributed")) for row in rows)
    reviewed_rows = sum(
        any(
            _annotated(row.get(field))
            for field in (
                "claim_supported_by_input",
                "claim_later_used_as_premise",
                "origin_misattributed",
                "final_answer_depends_on_claim",
                "laundering_candidate",
            )
        )
        for row in rows
    )
    denominator = reviewed_rows or len(rows)
    return {
        "reviewed_claim_rows": reviewed_rows,
        "candidate_laundering_count": candidate,
        "candidate_laundering_rate": _rate(candidate, denominator),
        "answer_dependent_laundering_count": sum(
            _truthy(row.get("laundering_candidate"))
            and _truthy(row.get("final_answer_depends_on_claim"))
            for row in rows
        ),
        "answer_dependent_laundering_rate": _rate(
            sum(
                _truthy(row.get("laundering_candidate"))
                and _truthy(row.get("final_answer_depends_on_claim"))
                for row in rows
            ),
            denominator,
        ),
        "origin_misattribution_count": misattributed,
        "origin_misattribution_rate": _rate(misattributed, denominator),
        "final_answer_depends_on_claim_count": answer_dependent,
        "final_answer_depends_on_claim_rate": _rate(answer_dependent, denominator),
    }


def _annotation_breakdown(rows: list[dict[str, str]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "UNKNOWN"].append(row)
    return {name: _annotation_counts(group_rows) for name, group_rows in sorted(groups.items())}


def _accuracy_by(generations: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in generations:
        groups[str(record.get(key) or "UNKNOWN")].append(record)
    output = {}
    for name, rows in sorted(groups.items()):
        correct = sum(row.get("parsed_final_answer") == row.get("gold_label") for row in rows)
        output[name] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": _rate(correct, len(rows)),
        }
    return output
