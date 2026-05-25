from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_jsonl, write_json, write_jsonl
from .schema import PUBMEDQA_LABELS, SCIFACT_LABELS, SEED, PilotItem


def prepare_data(output_dir: str | Path, seed: int = SEED, overwrite: bool = False) -> Path:
    output = ensure_dir(output_dir)
    items_path = output / "pilot_items.jsonl"
    manifest_path = output / "manifest.json"
    if items_path.exists() and not overwrite:
        return items_path

    scifact = select_scifact_examples(seed=seed)
    pubmedqa = select_pubmedqa_examples(seed=seed)
    records = [item.to_record() for item in [*scifact, *pubmedqa]]

    if len(records) != 40:
        raise RuntimeError(f"Expected 40 total examples, got {len(records)}")

    write_jsonl(items_path, records)
    write_json(
        manifest_path,
        {
            "seed": seed,
            "total_examples": len(records),
            "datasets": {
                "scifact": len(scifact),
                "pubmedqa": len(pubmedqa),
            },
            "scifact_label_counts": _label_counts(scifact),
            "pubmedqa_label_counts": _label_counts(pubmedqa),
        },
    )
    return items_path


def load_items(path: str | Path) -> list[PilotItem]:
    return [PilotItem(**record) for record in read_jsonl(path)]


def select_scifact_examples(seed: int = SEED, n: int = 25) -> list[PilotItem]:
    from datasets import load_dataset

    claims_ds = load_dataset("allenai/scifact", "claims", trust_remote_code=True)
    corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
    claims_rows = _all_rows(claims_ds)
    corpus_rows = _all_rows(corpus_ds)
    corpus_by_id = {
        str(row.get("doc_id") or row.get("document_id") or row.get("id")): row
        for row in corpus_rows
    }

    buckets: dict[str, list[PilotItem]] = defaultdict(list)
    for row in claims_rows:
        label = _scifact_label(row)
        if label not in SCIFACT_LABELS:
            continue
        doc_id = _first_existing_doc_id(row, corpus_by_id)
        if not doc_id:
            continue
        corpus = corpus_by_id[doc_id]
        context = _format_scifact_context(corpus)
        claim = str(row.get("claim") or "").strip()
        if not claim or not context:
            continue
        item_id = f"scifact-{row.get('id') or row.get('claim_id') or len(buckets[label])}"
        buckets[label].append(
            PilotItem(
                item_id=item_id,
                dataset="scifact",
                claim_or_question=claim,
                context=context,
                gold_label=label,
                metadata={
                    "source_dataset": "allenai/scifact",
                    "source_config": "claims+corpus",
                    "claim_row_id": row.get("id") or row.get("claim_id"),
                    "evidence_doc_id": doc_id,
                    "evidence_sentences": row.get("evidence_sentences"),
                    "split": row.get("_split"),
                },
            )
        )

    return _balanced_sample(buckets, labels=SCIFACT_LABELS, n=n, seed=seed)


def select_pubmedqa_examples(seed: int = SEED, n: int = 15) -> list[PilotItem]:
    from datasets import load_dataset

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)
    rows = _all_rows(dataset)
    buckets: dict[str, list[PilotItem]] = defaultdict(list)
    for row in rows:
        label = str(row.get("final_decision") or row.get("label") or "").strip().lower()
        if label not in PUBMEDQA_LABELS:
            continue
        question = str(row.get("question") or "").strip()
        context = _format_pubmedqa_context(row.get("context"))
        if not question or not context:
            continue
        item_id = f"pubmedqa-{row.get('pubid') or row.get('id') or len(buckets[label])}"
        buckets[label].append(
            PilotItem(
                item_id=item_id,
                dataset="pubmedqa",
                claim_or_question=question,
                context=context,
                gold_label=label,
                metadata={
                    "source_dataset": "qiaojin/PubMedQA",
                    "source_config": "pqa_labeled",
                    "pubid": row.get("pubid"),
                    "split": row.get("_split"),
                },
            )
        )

    # Prefer maybe/no while keeping yes represented when possible.
    selected: list[PilotItem] = []
    rng = random.Random(seed)
    target = {"maybe": 6, "no": 6, "yes": 3}
    for label in ("maybe", "no", "yes"):
        rows_for_label = list(buckets.get(label, []))
        rng.shuffle(rows_for_label)
        selected.extend(rows_for_label[: target[label]])
    if len(selected) < n:
        seen = {item.item_id for item in selected}
        leftovers = [item for label in PUBMEDQA_LABELS for item in buckets.get(label, []) if item.item_id not in seen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    return selected[:n]


def _all_rows(dataset_dict: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split in dataset_dict.items():
        for row in split:
            record = dict(row)
            record["_split"] = split_name
            rows.append(record)
    return rows


def _balanced_sample(
    buckets: dict[str, list[PilotItem]], labels: tuple[str, ...], n: int, seed: int
) -> list[PilotItem]:
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    selected: list[PilotItem] = []
    base = n // len(labels)
    remainder = n % len(labels)
    targets = {label: base + (1 if idx < remainder else 0) for idx, label in enumerate(labels)}
    for label in labels:
        selected.extend(buckets.get(label, [])[: targets[label]])
    if len(selected) < n:
        seen = {item.item_id for item in selected}
        leftovers = [item for label in labels for item in buckets.get(label, []) if item.item_id not in seen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    rng.shuffle(selected)
    if len(selected) != n:
        counts = {label: len(buckets.get(label, [])) for label in labels}
        raise RuntimeError(f"Could not sample {n} examples; available label counts: {counts}")
    return selected


def _scifact_label(row: dict[str, Any]) -> str | None:
    raw = row.get("evidence_label") or row.get("label")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if raw:
        value = str(raw).upper()
        if value in SCIFACT_LABELS:
            return value
    # SciFact claims without evidence labels are treated as insufficient evidence.
    evidence = row.get("evidence_doc_id") or row.get("evidence_sentences")
    if not evidence:
        return "NOINFO"
    return None


def _first_existing_doc_id(row: dict[str, Any], corpus_by_id: dict[str, dict[str, Any]]) -> str | None:
    candidates: list[Any] = []
    for key in ("evidence_doc_id", "cited_doc_ids", "doc_id", "document_id"):
        value = row.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif value is not None:
            candidates.append(value)
    for candidate in candidates:
        candidate_id = str(candidate)
        if candidate_id in corpus_by_id:
            return candidate_id
    return None


def _format_scifact_context(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    abstract = row.get("abstract")
    if isinstance(abstract, list):
        abstract_text = " ".join(str(sentence).strip() for sentence in abstract if str(sentence).strip())
    else:
        abstract_text = str(abstract or "").strip()
    if title:
        return f"Title: {title}\nAbstract: {abstract_text}"
    return abstract_text


def _format_pubmedqa_context(context: Any) -> str:
    if isinstance(context, dict):
        labels = context.get("labels") or []
        contexts = context.get("contexts") or []
        parts = []
        for idx, text in enumerate(contexts):
            label = labels[idx] if idx < len(labels) else f"Context {idx + 1}"
            parts.append(f"{label}: {text}")
        return "\n".join(parts).strip()
    if isinstance(context, list):
        return "\n".join(str(part).strip() for part in context if str(part).strip())
    return str(context or "").strip()


def _label_counts(items: list[PilotItem]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item.gold_label] += 1
    return dict(sorted(counts.items()))
