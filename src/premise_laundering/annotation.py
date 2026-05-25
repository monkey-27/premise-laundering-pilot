from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import write_csv

ANNOTATION_FIELDS = [
    "run_id",
    "item_id",
    "dataset",
    "condition",
    "gold_label",
    "model_answer",
    "claim_id",
    "claim_text",
    "declared_origin",
    "reasoning_step",
    "claim_supported_by_input",
    "claim_later_used_as_premise",
    "origin_misattributed",
    "final_answer_depends_on_claim",
    "laundering_candidate",
    "notes",
    "source_context_excerpt",
    "raw_output_pointer",
]


def build_annotation_rows(generation_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in generation_records:
        claims = record.get("parsed_intermediate_claims") or []
        if not claims:
            claims = [
                {
                    "claim_id": "no-claim-extracted",
                    "claim_text": "",
                    "declared_origin": "",
                    "reasoning_step": "",
                }
            ]
        for claim in claims:
            rows.append(
                {
                    "run_id": record["run_id"],
                    "item_id": record["item_id"],
                    "dataset": record["dataset"],
                    "condition": record["condition"],
                    "gold_label": record["gold_label"],
                    "model_answer": record.get("parsed_final_answer") or "",
                    "claim_id": claim.get("claim_id") or "",
                    "claim_text": claim.get("claim_text") or "",
                    "declared_origin": claim.get("declared_origin") or "",
                    "reasoning_step": claim.get("reasoning_step") or "",
                    "claim_supported_by_input": "",
                    "claim_later_used_as_premise": "",
                    "origin_misattributed": "",
                    "final_answer_depends_on_claim": "",
                    "laundering_candidate": "",
                    "notes": "",
                    "source_context_excerpt": _excerpt(record.get("context") or ""),
                    "raw_output_pointer": "generations.jsonl",
                }
            )
    return rows


def write_annotation_file(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, ANNOTATION_FIELDS)


def _excerpt(text: str, max_chars: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
