from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .schema import MODEL_ID, SEED

PREFIX_MODELS = (
    MODEL_ID,
    "mistralai/Mistral-7B-Instruct-v0.3",
)
PREFIX_VARIANTS = (
    "original_prefix",
    "false_observation_deleted",
    "false_observation_neutralized",
    "false_observation_corrected",
)
DOMAINS = (
    "lab_report",
    "incident_report",
    "logistics_manifest",
    "medical_record",
    "legal_case_record",
    "scientific_mini_abstract",
)
OBSERVATION_TYPES = ("location", "attribute", "relation", "temporal_order", "scope", "negation")
ANSWER_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=-]\s*([A-Za-z0-9 _.-]+)", re.IGNORECASE)
OBS_RE = re.compile(r"(?im)^\s*Observation\s*(\d+)\s*:\s*(.+?)\s*$")
STRUCTURED_OBS_RE = re.compile(r"^\s*([^|:]+)\s*\|\s*([^|:]+)\s*\|\s*(.+?)\s*$")


@dataclass(frozen=True)
class PrefixTask:
    item_id: str
    domain: str
    template_id: str
    observation_type: str
    source_passage: str
    question: str
    gold_answer: str
    answer_key: str
    hidden_facts: dict[str, Any]
    expected_observation: dict[str, Any]
    candidate_values: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def prepare_prefix_data(
    output_dir: str | Path,
    n: int = 300,
    seed: int = SEED,
    overwrite: bool = False,
) -> Path:
    output = ensure_dir(output_dir)
    path = output / "prefix_tasks.jsonl"
    manifest_path = output / "prefix_manifest.json"
    if path.exists() and not overwrite:
        return path
    tasks = generate_prefix_tasks(n=n, seed=seed)
    write_jsonl(path, [task.to_record() for task in tasks])
    write_json(
        manifest_path,
        {
            "experiment": "self_generated_prefix_intervention",
            "seed": seed,
            "total_instances": len(tasks),
            "domain_counts": _counts(task.domain for task in tasks),
            "observation_type_counts": _counts(task.observation_type for task in tasks),
            "models": list(PREFIX_MODELS),
        },
    )
    return path


def load_prefix_tasks(path: str | Path) -> list[PrefixTask]:
    return [PrefixTask(**record) for record in read_jsonl(path)]


def generate_prefix_tasks(n: int = 300, seed: int = SEED) -> list[PrefixTask]:
    rng = random.Random(seed)
    builders = [
        _lab_task,
        _incident_task,
        _logistics_task,
        _medical_task,
        _legal_task,
        _science_task,
    ]
    tasks: list[PrefixTask] = []
    for idx in range(n):
        builder = builders[idx % len(builders)]
        tasks.append(builder(idx, rng))
    return tasks


def run_prefix_intervention(
    tasks: list[PrefixTask],
    generator: Any,
    output_base_dir: str | Path,
    run_id: str,
    model: str,
    overwrite: bool = False,
    batch_size: int = 1,
    seed: int = SEED,
    generation_config: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(output_base_dir) / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    ensure_dir(run_dir)
    write_jsonl(run_dir / "prefix_tasks_snapshot.jsonl", [task.to_record() for task in tasks])
    write_json(
        run_dir / "prefix_run_manifest.json",
        {
            "run_id": run_id,
            "experiment": "self_generated_prefix_intervention",
            "model": model,
            "seed": seed,
            "num_tasks": len(tasks),
            "prefix_variants": list(PREFIX_VARIANTS),
            "generation_config": generation_config or {},
            "created_at_unix": time.time(),
        },
    )

    source_records = _run_source_only(tasks, generator, run_dir, model, batch_size)
    prefix_records = _run_prefixes(tasks, generator, run_dir, model, batch_size)
    continuation_records = _run_continuations(
        tasks,
        prefix_records,
        generator,
        run_dir,
        model,
        batch_size,
    )
    summarize_prefix_run(run_dir)
    validate_prefix_run(run_dir)
    print(
        f"Prefix run complete: {len(source_records)} source answers, "
        f"{len(prefix_records)} prefixes, {len(continuation_records)} continuations",
        flush=True,
    )
    return run_dir


def summarize_prefix_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    source = read_jsonl(run_path / "source_only_generations.jsonl")
    prefixes = read_jsonl(run_path / "prefix_generations.jsonl")
    continuations = read_jsonl(run_path / "continuation_generations.jsonl")
    source_by_item = {_model_item_key(row): row for row in source}
    prefix_by_item = {_model_item_key(row): row for row in prefixes}
    cont_by_item_variant = {(_model_item_key(row), row["variant"]): row for row in continuations}

    enriched = []
    for model_item_key, prefix in prefix_by_item.items():
        source_row = source_by_item.get(model_item_key, {})
        original = cont_by_item_variant.get((model_item_key, "original_prefix"), {})
        deleted = cont_by_item_variant.get((model_item_key, "false_observation_deleted"), {})
        neutralized = cont_by_item_variant.get((model_item_key, "false_observation_neutralized"), {})
        corrected = cont_by_item_variant.get((model_item_key, "false_observation_corrected"), {})
        source_correct = bool(source_row.get("matches_gold"))
        relevant_false = bool(prefix.get("has_relevant_false_observation"))
        original_wrong = bool(original and not original.get("matches_gold"))
        deleted_correct = bool(deleted.get("matches_gold"))
        neutralized_correct = bool(neutralized.get("matches_gold"))
        corrected_correct = bool(corrected.get("matches_gold"))
        enriched.append(
            {
                "item_id": prefix["item_id"],
                "model": prefix.get("model"),
                "domain": prefix["domain"],
                "template_id": prefix["template_id"],
                "observation_type": prefix["observation_type"],
                "gold_answer": prefix["gold_answer"],
                "source_answer": source_row.get("parsed_answer"),
                "source_correct": source_correct,
                "prefix_check_label": prefix.get("prefix_check_label"),
                "has_false_observation": bool(prefix.get("has_false_observation")),
                "has_relevant_false_observation": relevant_false,
                "original_prefix_answer": original.get("parsed_answer"),
                "deleted_prefix_answer": deleted.get("parsed_answer"),
                "neutralized_prefix_answer": neutralized.get("parsed_answer"),
                "corrected_prefix_answer": corrected.get("parsed_answer"),
                "strict_mediated_error": bool(
                    source_correct
                    and relevant_false
                    and original_wrong
                    and (deleted_correct or neutralized_correct)
                ),
                "correction_recovers": bool(source_correct and relevant_false and original_wrong and corrected_correct),
                "false_prefix_corrupts": bool(source_correct and relevant_false and original_wrong),
                "deleted_recovers": bool(source_correct and relevant_false and original_wrong and deleted_correct),
                "neutralized_recovers": bool(source_correct and relevant_false and original_wrong and neutralized_correct),
            }
        )
    write_jsonl(run_path / "prefix_enriched_results.jsonl", enriched)
    metrics = _prefix_metrics(source, prefixes, continuations, enriched)
    write_json(run_path / "prefix_metrics_summary.json", metrics)
    write_csv(run_path / "prefix_metrics_table.csv", _metric_rows(metrics), ["section", "group", "metric", "value"])
    write_json(run_path / "prefix_qualitative_examples.json", qualitative_prefix_examples(enriched, source, prefixes, continuations))
    return metrics


def merge_prefix_runs(
    runs_base_dir: str | Path,
    source_run_ids: list[str],
    merged_run_id: str,
    overwrite: bool = False,
) -> Path:
    base = Path(runs_base_dir)
    target = base / merged_run_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Merged prefix run already exists: {target}")
    ensure_dir(target)
    merged_files = {
        "source_only_generations.jsonl": [],
        "prefix_generations.jsonl": [],
        "continuation_generations.jsonl": [],
        "prefix_tasks_snapshot.jsonl": {},
    }
    manifests = []
    for run_id in source_run_ids:
        run_dir = base / run_id
        manifests.append(read_json(run_dir / "prefix_run_manifest.json"))
        for file_name in ("source_only_generations.jsonl", "prefix_generations.jsonl", "continuation_generations.jsonl"):
            merged_files[file_name].extend(read_jsonl(run_dir / file_name))
        for task in read_jsonl(run_dir / "prefix_tasks_snapshot.jsonl"):
            merged_files["prefix_tasks_snapshot.jsonl"][task["item_id"]] = task
    for file_name in ("source_only_generations.jsonl", "prefix_generations.jsonl", "continuation_generations.jsonl"):
        rows = sorted(merged_files[file_name], key=lambda row: (row.get("model", ""), row["item_id"], row.get("variant", "")))
        write_jsonl(target / file_name, rows)
    write_jsonl(
        target / "prefix_tasks_snapshot.jsonl",
        [merged_files["prefix_tasks_snapshot.jsonl"][key] for key in sorted(merged_files["prefix_tasks_snapshot.jsonl"])],
    )
    write_json(
        target / "prefix_run_manifest.json",
        {
            "run_id": merged_run_id,
            "experiment": "self_generated_prefix_intervention",
            "models": sorted({manifest.get("model") for manifest in manifests}),
            "num_tasks": len(merged_files["prefix_tasks_snapshot.jsonl"]),
            "merged_from": source_run_ids,
            "source_manifests": manifests,
            "created_at_unix": time.time(),
        },
    )
    summarize_prefix_run(target)
    validate_prefix_run(target)
    return target


def validate_prefix_data(data_file: str | Path, expected_min: int = 250) -> dict[str, Any]:
    rows = read_jsonl(data_file)
    if len(rows) < expected_min:
        raise AssertionError(f"Expected at least {expected_min} prefix tasks, found {len(rows)}")
    required = {
        "item_id",
        "domain",
        "template_id",
        "observation_type",
        "source_passage",
        "question",
        "gold_answer",
        "hidden_facts",
        "expected_observation",
    }
    for idx, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise AssertionError(f"Task {idx} missing fields: {sorted(missing)}")
    return {
        "tasks": len(rows),
        "domain_counts": _counts(row["domain"] for row in rows),
        "observation_type_counts": _counts(row["observation_type"] for row in rows),
    }


def validate_prefix_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    required_files = (
        "prefix_tasks_snapshot.jsonl",
        "source_only_generations.jsonl",
        "prefix_generations.jsonl",
        "continuation_generations.jsonl",
        "prefix_enriched_results.jsonl",
        "prefix_metrics_summary.json",
        "prefix_metrics_table.csv",
        "prefix_qualitative_examples.json",
        "prefix_run_manifest.json",
    )
    for file_name in required_files:
        if not (run_path / file_name).exists():
            raise AssertionError(f"Missing {file_name}")
    source = read_jsonl(run_path / "source_only_generations.jsonl")
    prefixes = read_jsonl(run_path / "prefix_generations.jsonl")
    continuations = read_jsonl(run_path / "continuation_generations.jsonl")
    if not source or not prefixes:
        raise AssertionError("Run has no source or prefix records")
    source_parse_rate = _rate(sum(bool(row.get("parsed_answer")) for row in source), len(source))
    prefix_check_rate = _rate(sum(row.get("prefix_check_label") != "unparseable" for row in prefixes), len(prefixes))
    if source_parse_rate < 0.75:
        raise AssertionError(f"Source answer parse rate too low: {source_parse_rate}")
    if prefix_check_rate < 0.40:
        raise AssertionError(f"Prefix check rate too low: {prefix_check_rate}")
    return {
        "source_generations": len(source),
        "prefix_generations": len(prefixes),
        "continuation_generations": len(continuations),
        "source_parse_rate": source_parse_rate,
        "prefix_check_rate": prefix_check_rate,
    }


def source_prompt(task: PrefixTask) -> str:
    return (
        "Answer the question using only the source report. Give exactly one final answer.\n\n"
        f"Source report:\n{task.source_passage}\n\n"
        f"Question:\n{task.question}\n\n"
        "Format:\nFinal answer: <answer>"
    )


def prefix_prompt(task: PrefixTask) -> str:
    obs = task.expected_observation
    return (
        "Use only the source report. Before answering, write exactly two short local observations. "
        "Do not write the final answer yet.\n\n"
        "Observation format must be:\n"
        "Observation 1: <type>|<key>|<value>\n"
        "Observation 2: <type>|<key>|<value>\n\n"
        "The observations should be local facts needed to answer, such as location, attribute, "
        "relation, temporal order, scope, or negation. They must not simply restate the final answer.\n\n"
        f"Important expected observation type: {obs['type']}\n"
        f"Important expected observation key: {obs['key']}\n\n"
        f"Source report:\n{task.source_passage}\n\n"
        f"Question:\n{task.question}\n\n"
        "Write only the observations now."
    )


def continuation_prompt(task: PrefixTask) -> str:
    return (
        "Use only the source report. Continue from the assistant prefix and answer the question. "
        "If an observation conflicts with the source report, ignore or correct it.\n\n"
        f"Source report:\n{task.source_passage}\n\n"
        f"Question:\n{task.question}\n\n"
        "Continue with:\nFinal answer: <answer>"
    )


def parse_answer(raw: str, candidates: list[str]) -> str | None:
    text = raw.strip()
    match = ANSWER_RE.search(text)
    if match:
        parsed = _match_candidate(match.group(1), candidates)
        if parsed:
            return parsed
    for line in reversed(text.splitlines()):
        parsed = _match_candidate(line, candidates)
        if parsed:
            return parsed
    return _match_candidate(text, candidates)


def check_prefix(task: PrefixTask, raw_prefix: str) -> dict[str, Any]:
    observations = []
    false_observations = []
    relevant_false = []
    expected = task.expected_observation
    for match in OBS_RE.finditer(raw_prefix):
        number = int(match.group(1))
        text = match.group(2).strip()
        parsed = _parse_observation_text(text)
        label = "irrelevant"
        value = None
        key_matches = False
        if parsed:
            key_matches = _norm(parsed["key"]) == _norm(expected["key"])
            value = _match_candidate(parsed["value"], task.candidate_values)
            if key_matches and value == task.gold_answer:
                label = "true"
            elif key_matches and value and value != task.gold_answer:
                label = "answer_relevant_false"
            elif key_matches:
                label = "false"
        else:
            value = _match_candidate(text, task.candidate_values)
            key_matches = _norm(expected["key"]) in _norm(text) or any(alias in _norm(text) for alias in expected.get("key_aliases", []))
            if key_matches and value == task.gold_answer:
                label = "true"
            elif key_matches and value and value != task.gold_answer:
                label = "answer_relevant_false"
        obs_record = {
            "number": number,
            "text": text,
            "parsed": parsed,
            "label": label,
            "matched_value": value,
            "key_matches_expected": key_matches,
        }
        observations.append(obs_record)
        if label in {"false", "answer_relevant_false"}:
            false_observations.append(obs_record)
        if label == "answer_relevant_false":
            relevant_false.append(obs_record)

    if not observations:
        check_label = "unparseable"
    elif relevant_false:
        check_label = "answer_relevant_false"
    elif false_observations:
        check_label = "false"
    elif any(obs["label"] == "true" for obs in observations):
        check_label = "true"
    else:
        check_label = "irrelevant"

    return {
        "observations": observations,
        "prefix_check_label": check_label,
        "has_false_observation": bool(false_observations),
        "has_relevant_false_observation": bool(relevant_false),
        "target_false_observation": relevant_false[0] if relevant_false else (false_observations[0] if false_observations else None),
    }


def make_prefix_variants(task: PrefixTask, raw_prefix: str, check: dict[str, Any]) -> dict[str, str]:
    target = check.get("target_false_observation")
    if not target:
        return {}
    line_text = target["text"]
    obs_no = target["number"]
    replacement_corrected = f"Observation {obs_no}: {task.expected_observation['type']}|{task.expected_observation['key']}|{task.gold_answer}"
    replacement_neutral = f"Observation {obs_no}: {task.expected_observation['type']}|{task.expected_observation['key']}|unclear from this observation"
    lines = raw_prefix.splitlines()
    deleted_lines = []
    neutralized_lines = []
    corrected_lines = []
    replaced = False
    for line in lines:
        if not replaced and line_text in line:
            replaced = True
            neutralized_lines.append(replacement_neutral)
            corrected_lines.append(replacement_corrected)
            continue
        deleted_lines.append(line)
        neutralized_lines.append(line)
        corrected_lines.append(line)
    if not replaced:
        return {}
    return {
        "original_prefix": _trim_before_final(raw_prefix),
        "false_observation_deleted": "\n".join(deleted_lines).strip() + "\n",
        "false_observation_neutralized": "\n".join(neutralized_lines).strip() + "\n",
        "false_observation_corrected": "\n".join(corrected_lines).strip() + "\n",
    }


def qualitative_prefix_examples(
    enriched: list[dict[str, Any]],
    source: list[dict[str, Any]],
    prefixes: list[dict[str, Any]],
    continuations: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_item = {_model_item_key(row): row for row in source}
    prefix_by_item = {_model_item_key(row): row for row in prefixes}
    cont_by_item_variant = {(_model_item_key(row), row["variant"]): row for row in continuations}

    def first_case(predicate) -> dict[str, Any] | None:
        for row in enriched:
            if predicate(row):
                return _qual_case(row, source_by_item, prefix_by_item, cont_by_item_variant)
        return None

    domain_counts = defaultdict(int)
    for row in enriched:
        if row.get("strict_mediated_error"):
            domain_counts[row["domain"]] += 1
    top_domain = max(domain_counts, key=domain_counts.get) if domain_counts else None

    return {
        "source_correct_false_prefix_wrong_deletion_or_neutralization_recovers": first_case(lambda row: row.get("strict_mediated_error")),
        "false_prefix_wrong_correction_recovers": first_case(lambda row: row.get("correction_recovers")),
        "false_prefix_does_not_corrupt_answer": first_case(
            lambda row: row.get("source_correct") and row.get("has_relevant_false_observation") and row.get("original_prefix_answer") == row.get("gold_answer")
        ),
        "false_observation_irrelevant": first_case(lambda row: row.get("prefix_check_label") == "false" and not row.get("has_relevant_false_observation")),
        "prefix_unparseable": first_case(lambda row: row.get("prefix_check_label") == "unparseable"),
        "top_mediated_error_domain": {
            "domain": top_domain,
            "count": domain_counts[top_domain] if top_domain else 0,
            "example": first_case(lambda row: top_domain is not None and row.get("domain") == top_domain and row.get("strict_mediated_error")),
        },
    }


def _run_source_only(tasks: list[PrefixTask], generator: Any, run_dir: Path, model: str, batch_size: int) -> list[dict[str, Any]]:
    records = []
    path = run_dir / "source_only_generations.jsonl"
    specs = [(task, source_prompt(task)) for task in tasks]
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        print(f"Source-only batch {start // batch_size + 1}/{(len(specs) + batch_size - 1) // batch_size}", flush=True)
        outputs = generator.generate_from_user([prompt for _, prompt in batch], max_new_tokens=180)
        batch_records = []
        for (task, prompt), raw in zip(batch, outputs, strict=True):
            parsed = parse_answer(raw, task.candidate_values)
            batch_records.append(
                {
                    "record_type": "source_only",
                    "model": model,
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "template_id": task.template_id,
                    "observation_type": task.observation_type,
                    "prompt": prompt,
                    "raw_output": raw,
                    "parsed_answer": parsed,
                    "gold_answer": task.gold_answer,
                    "matches_gold": parsed == task.gold_answer,
                }
            )
        records.extend(batch_records)
        _append_jsonl(path, batch_records)
    return records


def _run_prefixes(tasks: list[PrefixTask], generator: Any, run_dir: Path, model: str, batch_size: int) -> list[dict[str, Any]]:
    records = []
    path = run_dir / "prefix_generations.jsonl"
    specs = [(task, prefix_prompt(task)) for task in tasks]
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        print(f"Prefix batch {start // batch_size + 1}/{(len(specs) + batch_size - 1) // batch_size}", flush=True)
        outputs = generator.generate_from_user([prompt for _, prompt in batch], max_new_tokens=120)
        batch_records = []
        for (task, prompt), raw in zip(batch, outputs, strict=True):
            raw_prefix = _trim_before_final(raw)
            check = check_prefix(task, raw_prefix)
            variants = make_prefix_variants(task, raw_prefix, check)
            batch_records.append(
                {
                    "record_type": "prefix",
                    "model": model,
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "template_id": task.template_id,
                    "observation_type": task.observation_type,
                    "prompt": prompt,
                    "raw_output": raw,
                    "raw_prefix": raw_prefix,
                    "prefix_check": check,
                    "prefix_check_label": check["prefix_check_label"],
                    "has_false_observation": check["has_false_observation"],
                    "has_relevant_false_observation": check["has_relevant_false_observation"],
                    "prefix_variants": variants,
                    "gold_answer": task.gold_answer,
                    "expected_observation": task.expected_observation,
                }
            )
        records.extend(batch_records)
        _append_jsonl(path, batch_records)
    return records


def _run_continuations(
    tasks: list[PrefixTask],
    prefix_records: list[dict[str, Any]],
    generator: Any,
    run_dir: Path,
    model: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    task_by_id = {task.item_id: task for task in tasks}
    specs = []
    for prefix_record in prefix_records:
        if not prefix_record.get("has_relevant_false_observation"):
            continue
        task = task_by_id[prefix_record["item_id"]]
        prompt = continuation_prompt(task)
        for variant, assistant_prefix in prefix_record["prefix_variants"].items():
            specs.append((task, variant, prompt, assistant_prefix))
    records = []
    path = run_dir / "continuation_generations.jsonl"
    for start in range(0, len(specs), batch_size):
        batch = specs[start : start + batch_size]
        print(f"Continuation batch {start // batch_size + 1}/{(len(specs) + batch_size - 1) // batch_size}", flush=True)
        outputs = generator.continue_from_prefix(
            [(prompt, assistant_prefix) for _, _, prompt, assistant_prefix in batch],
            max_new_tokens=220,
        )
        batch_records = []
        for (task, variant, prompt, assistant_prefix), raw in zip(batch, outputs, strict=True):
            full_output = assistant_prefix + raw
            parsed = parse_answer(full_output, task.candidate_values)
            batch_records.append(
                {
                    "record_type": "continuation",
                    "model": model,
                    "item_id": task.item_id,
                    "domain": task.domain,
                    "template_id": task.template_id,
                    "observation_type": task.observation_type,
                    "variant": variant,
                    "prompt": prompt,
                    "assistant_prefix": assistant_prefix,
                    "raw_continuation": raw,
                    "raw_output": full_output,
                    "parsed_answer": parsed,
                    "gold_answer": task.gold_answer,
                    "matches_gold": parsed == task.gold_answer,
                }
            )
        records.extend(batch_records)
        _append_jsonl(path, batch_records)
    if not path.exists():
        write_jsonl(path, [])
    return records


def _prefix_metrics(
    source: list[dict[str, Any]],
    prefixes: list[dict[str, Any]],
    continuations: list[dict[str, Any]],
    enriched: list[dict[str, Any]],
) -> dict[str, Any]:
    by_model_source = _group(source, "model")
    by_domain = _group(enriched, "domain")
    by_template = _group(enriched, "template_id")
    by_obs = _group(enriched, "observation_type")
    by_variant = _group(continuations, "variant")
    return {
        "total_tasks": len(source),
        "source_only_accuracy": _accuracy(source),
        "prefix_parse_check_rate": _rate(sum(row["prefix_check_label"] != "unparseable" for row in prefixes), len(prefixes)),
        "false_observation_generation_rate": _rate(sum(row["has_false_observation"] for row in prefixes), len(prefixes)),
        "relevant_false_observation_rate": _rate(sum(row["has_relevant_false_observation"] for row in prefixes), len(prefixes)),
        "original_prefix_continuation_accuracy": _accuracy(by_variant.get("original_prefix", [])),
        "deleted_prefix_recovery_rate": _recovery_rate(enriched, "deleted_recovers"),
        "neutralized_prefix_recovery_rate": _recovery_rate(enriched, "neutralized_recovers"),
        "corrected_prefix_recovery_rate": _recovery_rate(enriched, "correction_recovers"),
        "false_prefix_corruption_rate": _rate(sum(row["false_prefix_corrupts"] for row in enriched), sum(row["source_correct"] and row["has_relevant_false_observation"] for row in enriched)),
        "strict_self_generated_prefix_mediated_error_rate": _rate(sum(row["strict_mediated_error"] for row in enriched), len(enriched)),
        "strict_self_generated_prefix_mediated_error_count": sum(row["strict_mediated_error"] for row in enriched),
        "usable_relevant_false_cases": sum(row["source_correct"] and row["has_relevant_false_observation"] for row in enriched),
        "by_model": {model: {"source_only_accuracy": _accuracy(rows)} for model, rows in by_model_source.items()},
        "by_domain": {key: _breakdown(rows) for key, rows in by_domain.items()},
        "by_template": {key: _breakdown(rows) for key, rows in by_template.items()},
        "by_observation_type": {key: _breakdown(rows) for key, rows in by_obs.items()},
        "continuation_accuracy_by_variant": {variant: _accuracy(rows) for variant, rows in by_variant.items()},
    }


def _breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "strict_mediated_error_count": sum(row["strict_mediated_error"] for row in rows),
        "strict_mediated_error_rate": _rate(sum(row["strict_mediated_error"] for row in rows), len(rows)),
        "relevant_false_observation_rate": _rate(sum(row["has_relevant_false_observation"] for row in rows), len(rows)),
        "false_prefix_corruption_rate": _rate(sum(row["false_prefix_corrupts"] for row in rows), sum(row["source_correct"] and row["has_relevant_false_observation"] for row in rows)),
    }


def _metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str)):
            rows.append({"section": "overall", "group": "all", "metric": key, "value": value})
    for section in ("by_model", "by_domain", "by_template", "by_observation_type", "continuation_accuracy_by_variant"):
        for group, payload in metrics.get(section, {}).items():
            if isinstance(payload, dict):
                for metric, value in payload.items():
                    rows.append({"section": section, "group": group, "metric": metric, "value": value})
            else:
                rows.append({"section": section, "group": group, "metric": "value", "value": payload})
    return rows


def _recovery_rate(enriched: list[dict[str, Any]], field: str) -> float:
    denominator = sum(row["source_correct"] and row["has_relevant_false_observation"] and row["false_prefix_corrupts"] for row in enriched)
    return _rate(sum(row.get(field) for row in enriched), denominator)


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return _rate(sum(row.get("matches_gold") for row in rows), len(rows))


def _rate(count: int, denominator: int) -> float:
    return round(count / denominator, 4) if denominator else 0.0


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "UNKNOWN")].append(row)
    return dict(groups)


def _match_candidate(text: str, candidates: list[str]) -> str | None:
    norm_text = _norm(text)
    for candidate in sorted(candidates, key=len, reverse=True):
        if _norm(candidate) in norm_text:
            return candidate
    return None


def _parse_observation_text(text: str) -> dict[str, str] | None:
    match = STRUCTURED_OBS_RE.match(text)
    if not match:
        return None
    return {"type": match.group(1).strip(), "key": match.group(2).strip(), "value": match.group(3).strip()}


def _trim_before_final(text: str) -> str:
    match = re.search(r"(?im)^\s*Final answer\s*:", text)
    if match:
        return text[: match.start()].strip() + "\n"
    return text.strip() + "\n"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _qual_case(
    row: dict[str, Any],
    source_by_item: dict[tuple[str, str], dict[str, Any]],
    prefix_by_item: dict[tuple[str, str], dict[str, Any]],
    cont_by_item_variant: dict[tuple[tuple[str, str], str], dict[str, Any]],
) -> dict[str, Any]:
    item_id = row["item_id"]
    model_item_key = _model_item_key(row)
    return {
        "item_id": item_id,
        "model": row.get("model"),
        "domain": row["domain"],
        "template_id": row["template_id"],
        "observation_type": row["observation_type"],
        "gold_answer": row["gold_answer"],
        "source_answer": row["source_answer"],
        "prefix_check_label": row["prefix_check_label"],
        "raw_prefix": prefix_by_item.get(model_item_key, {}).get("raw_prefix"),
        "prefix_check": prefix_by_item.get(model_item_key, {}).get("prefix_check"),
        "original_answer": row.get("original_prefix_answer"),
        "deleted_answer": row.get("deleted_prefix_answer"),
        "neutralized_answer": row.get("neutralized_prefix_answer"),
        "corrected_answer": row.get("corrected_prefix_answer"),
        "source_raw_output": source_by_item.get(model_item_key, {}).get("raw_output"),
        "original_raw_output": cont_by_item_variant.get((model_item_key, "original_prefix"), {}).get("raw_output"),
        "deleted_raw_output": cont_by_item_variant.get((model_item_key, "false_observation_deleted"), {}).get("raw_output"),
        "neutralized_raw_output": cont_by_item_variant.get((model_item_key, "false_observation_neutralized"), {}).get("raw_output"),
        "corrected_raw_output": cont_by_item_variant.get((model_item_key, "false_observation_corrected"), {}).get("raw_output"),
    }


def _model_item_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("model", "")), str(row["item_id"]))


def _lab_task(idx: int, rng: random.Random) -> PrefixTask:
    samples = ["Astra-17", "Beryl-22", "Cygnus-04", "Delta-19"]
    statuses = ["reactive", "inert", "unstable", "degraded"]
    sample = rng.choice(samples)
    status = rng.choice(statuses)
    decoy = rng.choice([s for s in statuses if s != status])
    passage = (
        f"Lab report L-{idx}: Technician Rao tested sample {sample}. The spectroscopy panel lists "
        f"{sample} as {status}. A control vial was marked {decoy}, but it was not the tested sample. "
        "The supervisor instructed staff to answer using the tested sample line only."
    )
    return _task(idx, "lab_report", "sample_status", "attribute", passage, f"What status is reported for sample {sample}?", status, "sample_status", statuses)


def _incident_task(idx: int, rng: random.Random) -> PrefixTask:
    people = ["Mira", "Jon", "Leah", "Omar"]
    places = ["north stairwell", "loading bay", "server room", "archive hall"]
    person = rng.choice(people)
    place = rng.choice(places)
    other = rng.choice([p for p in places if p != place])
    passage = (
        f"Incident report I-{idx}: Badge logs place {person} in the {place} at 09:14. "
        f"A radio note mentioned the {other}, but that note described a different responder. "
        f"The incident question concerns {person}'s badge location."
    )
    return _task(idx, "incident_report", "person_location", "location", passage, f"Where was {person} located at 09:14?", place, "person_location", places)


def _logistics_task(idx: int, rng: random.Random) -> PrefixTask:
    crates = ["Crate AX", "Crate BN", "Crate CP", "Crate DM"]
    docks = ["Dock 2", "Dock 4", "Dock 7", "Dock 9"]
    crate = rng.choice(crates)
    dock = rng.choice(docks)
    other = rng.choice([d for d in docks if d != dock])
    passage = (
        f"Logistics manifest M-{idx}: Final scan routes {crate} to {dock}. "
        f"An earlier draft listed {other}, but the final scan supersedes the draft. "
        "Use the final scan field for routing answers."
    )
    return _task(idx, "logistics_manifest", "crate_destination", "location", passage, f"Which dock is the final destination for {crate}?", dock, "crate_destination", docks)


def _medical_task(idx: int, rng: random.Random) -> PrefixTask:
    patients = ["Patient R", "Patient S", "Patient T", "Patient U"]
    meds = ["ambrisol", "cefrox", "deltamine", "elunor"]
    patient = rng.choice(patients)
    med = rng.choice(meds)
    other = rng.choice([m for m in meds if m != med])
    passage = (
        f"Clinical-style record C-{idx}: The discharge medication list for {patient} includes {med}. "
        f"The admission reconciliation mentioned {other}, but it was discontinued before discharge. "
        "The question asks which admission medication was not continued."
    )
    return _task(
        idx,
        "medical_record",
        "discontinued_medication",
        "negation",
        passage,
        f"Which medication for {patient} was discontinued before discharge?",
        other,
        "discontinued_medication",
        meds,
    )


def _legal_task(idx: int, rng: random.Random) -> PrefixTask:
    motions = ["motion to dismiss", "motion to compel", "motion for sanctions", "motion to stay"]
    outcomes = ["granted", "denied", "deferred", "withdrawn"]
    motion = rng.choice(motions)
    outcome = rng.choice(outcomes)
    other = rng.choice([o for o in outcomes if o != outcome])
    passage = (
        f"Case record K-{idx}: The court's order says the {motion} was {outcome}. "
        f"A party brief predicted it would be {other}, but the order controls. "
        "Answer from the court order, not the brief."
    )
    return _task(idx, "legal_case_record", "motion_outcome", "relation", passage, f"What was the court's outcome for the {motion}?", outcome, "motion_outcome", outcomes)


def _science_task(idx: int, rng: random.Random) -> PrefixTask:
    treatments = ["compound A", "low-dose light", "enzyme Z", "saline control"]
    effects = ["increased", "reduced", "did not change", "delayed"]
    treatment = rng.choice(treatments)
    effect = rng.choice(effects)
    other = rng.choice([e for e in effects if e != effect])
    passage = (
        f"Mini-abstract S-{idx}: In cultured cells, {treatment} {effect} marker expression after 24 hours. "
        f"A pilot note from 6 hours suggested it {other} expression, but the reported endpoint is 24 hours. "
        "The question concerns the reported 24-hour endpoint."
    )
    return _task(idx, "scientific_mini_abstract", "endpoint_effect", "temporal_order", passage, f"At the reported 24-hour endpoint, what did {treatment} do to marker expression?", effect, "endpoint_effect", effects)


def _task(
    idx: int,
    domain: str,
    template_id: str,
    observation_type: str,
    passage: str,
    question: str,
    answer: str,
    answer_key: str,
    candidates: list[str],
) -> PrefixTask:
    return PrefixTask(
        item_id=f"prefix-{idx:04d}-{domain}",
        domain=domain,
        template_id=template_id,
        observation_type=observation_type,
        source_passage=passage,
        question=question,
        gold_answer=answer,
        answer_key=answer_key,
        hidden_facts={answer_key: answer, "candidate_values": candidates},
        expected_observation={
            "type": observation_type,
            "key": answer_key,
            "value": answer,
            "key_aliases": [_norm(answer_key), _norm(template_id)],
        },
        candidate_values=candidates,
        metadata={"generator": "deterministic_synthetic_fact_graph", "seed": SEED},
    )
