from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 27
CONDITIONS = ("direct", "cot", "reflection", "provenance", "quarantine")
SCIFACT_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
PUBMEDQA_LABELS = ("yes", "no", "maybe")


@dataclass(frozen=True)
class PilotItem:
    item_id: str
    dataset: str
    claim_or_question: str
    context: str
    gold_label: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRecord:
    run_id: str
    item_id: str
    dataset: str
    condition: str
    model: str
    prompt: str
    raw_output: str
    parsed_final_answer: str | None
    parsed_reasoning_steps: list[str]
    parsed_intermediate_claims: list[dict[str, Any]]
    declared_origin_tags: list[str]
    gold_label: str
    context: str
    claim_or_question: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    value = str(label).strip().lower()
    mapping = {
        "support": "SUPPORT",
        "supports": "SUPPORT",
        "supported": "SUPPORT",
        "entailment": "SUPPORT",
        "contradict": "CONTRADICT",
        "contradicts": "CONTRADICT",
        "contradicted": "CONTRADICT",
        "contradiction": "CONTRADICT",
        "noinfo": "NOINFO",
        "no_info": "NOINFO",
        "not enough info": "NOINFO",
        "insufficient information": "NOINFO",
        "insufficient": "NOINFO",
        "unknown": "NOINFO",
        "yes": "yes",
        "no": "no",
        "maybe": "maybe",
    }
    if value in mapping:
        return mapping[value]
    upper = str(label).strip().upper()
    if upper in SCIFACT_LABELS:
        return upper
    return None
