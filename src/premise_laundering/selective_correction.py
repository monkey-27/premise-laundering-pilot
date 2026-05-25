from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .schema import MODEL_ID, SEED

EXPERIMENT = "selective_correction_propagation"
DOMAINS = ("tool_agent_state", "rule_scope_reasoning")
CORRECTION_TYPES = ("local_only", "local_to_final", "scope_limited", "rule_level")
PROMPT_CONDITIONS = (
    "standard_feedback",
    "step_targeted_feedback",
    "natural_feedback",
    "full_regeneration",
    "correction_scope_map",
)
CORRECTED_PROMPT_CONDITIONS = (
    "standard_feedback",
    "natural_feedback",
    "full_regeneration",
    "correction_scope_map",
)


@dataclass(frozen=True)
class CorrectionTask:
    item_id: str
    domain: str
    correction_type: str
    context: str
    initial_trace: str
    feedback_by_condition: dict[str, str]
    gold_state_before: dict[str, Any]
    gold_state_after: dict[str, Any]
    should_update: list[str]
    should_preserve: list[str]
    probes: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def prepare_selective_data(
    output_dir: str | Path,
    n: int = 320,
    seed: int = SEED,
    overwrite: bool = False,
) -> Path:
    output = ensure_dir(output_dir)
    tasks_path = output / "feedback_tasks.jsonl"
    manifest_path = output / "selective_data_manifest.json"
    if tasks_path.exists() and not overwrite:
        return tasks_path

    per_cell = max(1, n // (len(DOMAINS) * len(CORRECTION_TYPES)))
    rng = random.Random(seed)
    tasks: list[CorrectionTask] = []
    builders = {
        ("tool_agent_state", "local_only"): _tool_local_only,
        ("tool_agent_state", "local_to_final"): _tool_local_to_final,
        ("tool_agent_state", "scope_limited"): _tool_scope_limited,
        ("tool_agent_state", "rule_level"): _tool_rule_level,
        ("rule_scope_reasoning", "local_only"): _rule_local_only,
        ("rule_scope_reasoning", "local_to_final"): _rule_local_to_final,
        ("rule_scope_reasoning", "scope_limited"): _rule_scope_limited,
        ("rule_scope_reasoning", "rule_level"): _rule_rule_level,
    }
    for domain in DOMAINS:
        for correction_type in CORRECTION_TYPES:
            for cell_idx in range(per_cell):
                idx = len(tasks)
                tasks.append(builders[(domain, correction_type)](idx, cell_idx, rng))

    write_jsonl(tasks_path, [task.to_record() for task in tasks])
    write_json(
        manifest_path,
        {
            "experiment": EXPERIMENT,
            "seed": seed,
            "requested_examples": n,
            "examples": len(tasks),
            "domains": list(DOMAINS),
            "correction_types": list(CORRECTION_TYPES),
            "prompt_conditions": list(PROMPT_CONDITIONS),
            "probes_per_example": 5,
            "construction": (
                "Synthetic hidden state/dependency graphs. Feedback is localized; should_update "
                "and should_preserve keys define automatic selective revision scoring."
            ),
        },
    )
    return tasks_path


def prepare_selective_corrected_data(
    output_dir: str | Path,
    n: int = 96,
    seed: int = SEED,
    overwrite: bool = False,
) -> Path:
    output = ensure_dir(output_dir)
    tasks_path = output / "feedback_tasks_corrected.jsonl"
    manifest_path = output / "selective_corrected_data_manifest.json"
    if tasks_path.exists() and not overwrite:
        return tasks_path

    source_dir = output / "_source_uncorrected"
    source_path = prepare_selective_data(source_dir, n=n, seed=seed, overwrite=True)
    tasks = [_corrected_task(task, idx) for idx, task in enumerate(load_selective_tasks(source_path))]
    write_jsonl(tasks_path, [task.to_record() for task in tasks])
    write_json(
        manifest_path,
        {
            "experiment": f"{EXPERIMENT}_corrected_small",
            "seed": seed,
            "requested_examples": n,
            "examples": len(tasks),
            "domains": list(DOMAINS),
            "correction_types": list(CORRECTION_TYPES),
            "prompt_conditions": list(CORRECTED_PROMPT_CONDITIONS),
            "model_plan": ["Qwen/Qwen2.5-7B-Instruct"],
            "probes_per_example": 5,
            "fixes": [
                "aligned value-field probe wording with value golds",
                "removed step_targeted_feedback from corrected pilot conditions",
                "tightened local_only update/preserve labeling",
                "enabled semantic aliases and scoring diagnostics",
            ],
        },
    )
    return tasks_path


def load_selective_tasks(path: str | Path) -> list[CorrectionTask]:
    return [CorrectionTask(**record) for record in read_jsonl(path)]


def _corrected_task(task: CorrectionTask, idx: int) -> CorrectionTask:
    feedback = {condition: task.feedback_by_condition[condition] for condition in CORRECTED_PROMPT_CONDITIONS}
    probes = [dict(probe) for probe in task.probes]
    should_update = list(task.should_update)
    should_preserve = list(task.should_preserve)

    if task.domain == "tool_agent_state" and task.correction_type == "local_only":
        probes[0]["question"] = "What is the corrected notification time?"
        should_update = ["notification_time"]
        should_preserve = ["notification_sent", "upload_success", "archive_copy_ready", "next_action"]
        for probe in probes:
            if probe["state_key"] == "archive_copy_ready":
                probe["probe_type"] = "final_state"
            elif probe["state_key"] in should_preserve:
                probe["probe_type"] = "preserved_boundary" if probe["state_key"] != "next_action" else "future_use"
    if task.domain == "rule_scope_reasoning" and task.correction_type == "local_only":
        should_update = ["current_fee"]
        should_preserve = [
            "application_type",
            "renewal_review_allowed",
            "historical_procedure_valid",
            "next_action",
        ]

    return CorrectionTask(
        item_id=f"selective-corrected-{idx:04d}-{task.domain}-{task.correction_type}",
        domain=task.domain,
        correction_type=task.correction_type,
        context=task.context,
        initial_trace=task.initial_trace,
        feedback_by_condition=feedback,
        gold_state_before=task.gold_state_before,
        gold_state_after=task.gold_state_after,
        should_update=should_update,
        should_preserve=should_preserve,
        probes=probes,
        metadata={**task.metadata, "corrected_pilot": True, "source_item_id": task.item_id},
    )


def build_revision_prompt(task: CorrectionTask, condition: str) -> str:
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown prompt condition: {condition}")
    feedback = task.feedback_by_condition[condition]
    probes = "\n".join(f"- {probe['probe_id']}: {probe['question']}" for probe in task.probes)
    old_trace = "" if condition == "full_regeneration" else f"\nInitial flawed trace/state:\n{task.initial_trace}\n"
    scope_map = ""
    if condition == "correction_scope_map":
        scope_map = (
            "\nBefore revising, explicitly separate what must change from what must stay the same. "
            "Bind feedback to the narrowest target state variable/entity/rule that it affects."
        )
        schema = (
            "{\n"
            '  "feedback_target": "...",\n'
            '  "must_change": ["state_key", "..."],\n'
            '  "must_stay_same": ["state_key", "..."],\n'
            '  "revised_reasoning": "...",\n'
            '  "final_answer": "...",\n'
            '  "probe_answers": {"p1": "...", "p2": "...", "p3": "...", "p4": "...", "p5": "..."}\n'
            "}"
        )
    else:
        schema = (
            "{\n"
            '  "revised_reasoning": "...",\n'
            '  "final_answer": "...",\n'
            '  "probe_answers": {"p1": "...", "p2": "...", "p3": "...", "p4": "...", "p5": "..."}\n'
            "}"
        )
    return (
        "You are revising an agent reasoning trace after localized feedback. The goal is selective "
        "correction: update exactly the affected state and its dependents, while preserving unrelated "
        "state. Do not rewrite unaffected facts just because feedback was given."
        f"{scope_map}\n\n"
        f"Context:\n{task.context}\n"
        f"{old_trace}\n"
        f"Localized feedback:\n{feedback}\n\n"
        f"Answer these probes after revision:\n{probes}\n\n"
        "Return JSON only with this exact shape:\n"
        f"{schema}"
    )


def run_selective_revision(
    tasks: list[CorrectionTask],
    generator: Callable[[list[str]], list[str]],
    output_base_dir: str | Path,
    run_id: str | None = None,
    model: str = MODEL_ID,
    backend: str = "transformers",
    conditions: Iterable[str] = PROMPT_CONDITIONS,
    overwrite: bool = False,
    batch_size: int = 1,
    seed: int = SEED,
    generation_config: dict[str, Any] | None = None,
    tasks_filename: str = "feedback_tasks.jsonl",
) -> Path:
    run_id = run_id or f"selective-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(output_base_dir) / run_id
    generations_path = run_dir / "revision_generations.jsonl"
    resume = run_dir.exists() and not overwrite and generations_path.exists()
    if run_dir.exists() and not overwrite and not resume:
        raise FileExistsError(f"Run directory already exists without resumable generations: {run_dir}")
    ensure_dir(run_dir)

    conditions = tuple(conditions)
    for condition in conditions:
        if condition not in PROMPT_CONDITIONS:
            raise ValueError(f"Unknown prompt condition: {condition}")

    write_jsonl(run_dir / tasks_filename, [task.to_record() for task in tasks])
    if tasks_filename != "feedback_tasks.jsonl":
        write_jsonl(run_dir / "feedback_tasks.jsonl", [task.to_record() for task in tasks])
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "experiment": EXPERIMENT,
            "model": model,
            "backend": backend,
            "seed": seed,
            "conditions": list(conditions),
            "num_examples": len(tasks),
            "num_generations": len(tasks) * len(conditions),
            "resume_enabled": True,
            "resumed_from_existing_generations": resume,
            "generation_config": generation_config or {},
            "tasks_filename": tasks_filename,
            "created_at_unix": time.time(),
        },
    )

    prompt_specs = [(task, condition, build_revision_prompt(task, condition)) for task in tasks for condition in conditions]
    generations = read_jsonl(generations_path) if resume else []
    completed_keys = {(row["item_id"], row["condition"]) for row in generations}
    remaining_specs = [
        spec for spec in prompt_specs if (spec[0].item_id, spec[1]) not in completed_keys
    ]
    total_batches = (len(remaining_specs) + batch_size - 1) // batch_size
    print(
        f"selective correction resume: {len(generations)} complete, "
        f"{len(remaining_specs)} remaining",
        flush=True,
    )
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(remaining_specs))
        outputs = generator([spec[2] for spec in remaining_specs[start:end]])
        for (task, condition, prompt), raw_output in zip(remaining_specs[start:end], outputs):
            parsed = parse_revision_output(raw_output)
            generations.append(_generation_record(run_id, model, task, condition, prompt, raw_output, parsed))
        write_jsonl(generations_path, generations)
        print(f"selective correction batch {batch_idx + 1}/{total_batches}", flush=True)

    summarize_selective_run(run_dir)
    validate_selective_run(run_dir)
    return run_dir


def summarize_selective_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    generations = read_jsonl(run_path / "revision_generations.jsonl")
    probe_rows = []
    enriched = []
    for row in generations:
        rows, summary = score_generation(row)
        probe_rows.extend(rows)
        enriched.append(summary)
    write_jsonl(run_path / "parsed_probe_results.jsonl", probe_rows)
    write_jsonl(run_path / "enriched_results.jsonl", enriched)
    metrics = selective_metrics(enriched, probe_rows)
    write_json(run_path / "metrics_summary.json", metrics)
    write_json(run_path / "scoring_diagnostics.json", scoring_diagnostics(probe_rows))
    _write_metric_tables(run_path, metrics)
    write_json(run_path / "qualitative_examples.json", qualitative_examples(enriched, probe_rows, generations))
    return metrics


def merge_selective_runs(
    runs_base_dir: str | Path,
    source_run_ids: list[str],
    merged_run_id: str,
    overwrite: bool = False,
) -> Path:
    base = Path(runs_base_dir)
    target = base / merged_run_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Merged selective run already exists: {target}")
    ensure_dir(target)
    generations = []
    tasks: dict[str, dict[str, Any]] = {}
    manifests = []
    for run_id in source_run_ids:
        run_dir = base / run_id
        manifests.append(read_json(run_dir / "run_manifest.json"))
        generations.extend(read_jsonl(run_dir / "revision_generations.jsonl"))
        for task in read_jsonl(run_dir / "feedback_tasks.jsonl"):
            tasks[task["item_id"]] = task
    generations = sorted(generations, key=lambda row: (row.get("model", ""), row["item_id"], row["condition"]))
    write_jsonl(target / "feedback_tasks.jsonl", [tasks[key] for key in sorted(tasks)])
    write_jsonl(target / "revision_generations.jsonl", generations)
    write_json(
        target / "run_manifest.json",
        {
            "run_id": merged_run_id,
            "experiment": EXPERIMENT,
            "models": sorted({manifest.get("model") for manifest in manifests}),
            "merged_from": source_run_ids,
            "source_manifests": manifests,
            "num_examples": len(tasks),
            "num_generations": len(generations),
            "created_at_unix": time.time(),
        },
    )
    summarize_selective_run(target)
    validate_selective_run(target)
    return target


def parse_revision_output(raw_output: str) -> dict[str, Any]:
    data = _extract_json(raw_output)
    if not isinstance(data, dict):
        return {
            "parse_error": "no_json_object_found",
            "feedback_target": "",
            "must_change": [],
            "must_stay_same": [],
            "revised_reasoning": "",
            "final_answer": "",
            "probe_answers": {},
        }
    probe_answers = data.get("probe_answers") if isinstance(data.get("probe_answers"), dict) else {}
    return {
        "parse_error": None,
        "feedback_target": _stringify(data.get("feedback_target", "")),
        "must_change": _string_list(data.get("must_change", [])),
        "must_stay_same": _string_list(data.get("must_stay_same", [])),
        "revised_reasoning": _stringify(data.get("revised_reasoning", "")),
        "final_answer": _stringify(data.get("final_answer", "")),
        "probe_answers": {str(key): _stringify(value) for key, value in probe_answers.items()},
    }


def score_generation(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = row["parsed"]
    probes = row["probes"]
    should_update = set(row["should_update"])
    should_preserve = set(row["should_preserve"])
    probe_rows = []
    for probe in probes:
        answer = parsed.get("probe_answers", {}).get(probe["probe_id"], "")
        correct, scoring_method = score_answer(answer, probe["gold_answer"])
        probe_rows.append(
            {
                "run_id": row["run_id"],
                "model": row["model"],
                "item_id": row["item_id"],
                "domain": row["domain"],
                "correction_type": row["correction_type"],
                "condition": row["condition"],
                "probe_id": probe["probe_id"],
                "probe_type": probe["probe_type"],
                "state_key": probe["state_key"],
                "question": probe["question"],
                "gold_answer": probe["gold_answer"],
                "model_answer": answer,
                "is_correct": correct,
                "scoring_method": scoring_method,
                "is_update_probe": probe["state_key"] in should_update,
                "is_preserve_probe": probe["state_key"] in should_preserve,
            }
        )
    update_rows = [probe for probe in probe_rows if probe["is_update_probe"]]
    preserve_rows = [probe for probe in probe_rows if probe["is_preserve_probe"]]
    future_rows = [probe for probe in probe_rows if probe["probe_type"] == "future_use"]
    final_rows = [probe for probe in probe_rows if probe["probe_type"] == "final_state"]
    target_ok = target_binding_correct(row, parsed, probe_rows)
    dependent_recall = _rate(sum(probe["is_correct"] for probe in update_rows), len(update_rows))
    preservation = _rate(sum(probe["is_correct"] for probe in preserve_rows), len(preserve_rows))
    trace_consistency = trace_answer_consistent(parsed, probe_rows, row)
    summary = {
        "run_id": row["run_id"],
        "model": row["model"],
        "item_id": row["item_id"],
        "domain": row["domain"],
        "correction_type": row["correction_type"],
        "condition": row["condition"],
        "parse_error": parsed.get("parse_error"),
        "target_binding_correct": target_ok,
        "dependent_update_recall": dependent_recall,
        "boundary_preservation_precision": preservation,
        "overcorrection_rate": round(1.0 - preservation, 4),
        "under_propagation_rate": round(1.0 - dependent_recall, 4),
        "trace_answer_consistency": trace_consistency,
        "future_use_accuracy": _rate_optional(sum(probe["is_correct"] for probe in future_rows), len(future_rows)),
        "final_state_accuracy": _rate_optional(sum(probe["is_correct"] for probe in final_rows), len(final_rows)),
        "selective_revision_score": round(dependent_recall * preservation, 4),
        "probe_accuracy": _rate(sum(probe["is_correct"] for probe in probe_rows), len(probe_rows)),
        "updated_probe_count": len(update_rows),
        "preserved_probe_count": len(preserve_rows),
        "raw_output": row["raw_output"],
        "parsed": parsed,
        "target_key": row["metadata"].get("target_key"),
        "target_aliases": row["metadata"].get("target_aliases", []),
    }
    return probe_rows, summary


def selective_metrics(enriched: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_generations": len(enriched),
        "total_probe_results": len(probe_rows),
        "overall": _bundle(enriched),
        "by_model": {key: _bundle(group) for key, group in _group(enriched, "model").items()},
        "by_domain": {key: _bundle(group) for key, group in _group(enriched, "domain").items()},
        "by_correction_type": {key: _bundle(group) for key, group in _group(enriched, "correction_type").items()},
        "by_condition": {key: _bundle(group) for key, group in _group(enriched, "condition").items()},
        "by_probe_type": {key: _probe_bundle(group) for key, group in _group(probe_rows, "probe_type").items()},
        "by_state_key": {key: _probe_bundle(group) for key, group in _group(probe_rows, "state_key").items()},
        "scoring_method_counts": _counts(row.get("scoring_method", "unknown") for row in probe_rows),
    }


def scoring_diagnostics(probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_counts = _counts(row.get("scoring_method", "unknown") for row in probe_rows)
    examples_by_method = {}
    for method in method_counts:
        examples_by_method[method] = next(
            (
                {
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "state_key": row["state_key"],
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "model_answer": row["model_answer"],
                    "is_correct": row["is_correct"],
                }
                for row in probe_rows
                if row.get("scoring_method") == method
            ),
            None,
        )
    return {
        "total_probe_results": len(probe_rows),
        "method_counts": method_counts,
        "correct_by_method": _counts(row.get("scoring_method", "unknown") for row in probe_rows if row.get("is_correct")),
        "failed_by_state_key": _counts(row["state_key"] for row in probe_rows if not row.get("is_correct")),
        "examples_by_method": examples_by_method,
    }


def validate_selective_data(
    tasks_file: str | Path,
    expected: int = 320,
    conditions: Iterable[str] = PROMPT_CONDITIONS,
    require_two_update_probes: bool = True,
) -> dict[str, Any]:
    tasks = read_jsonl(tasks_file)
    expected_conditions = set(conditions)
    if len(tasks) != expected:
        raise AssertionError(f"Expected {expected} selective tasks, found {len(tasks)}")
    domain_counts = _counts(row["domain"] for row in tasks)
    type_counts = _counts(row["correction_type"] for row in tasks)
    if set(domain_counts) != set(DOMAINS):
        raise AssertionError(f"Expected domains {DOMAINS}, found {sorted(domain_counts)}")
    expected_domain = expected // len(DOMAINS)
    if any(count != expected_domain for count in domain_counts.values()):
        raise AssertionError(f"Domains are not equally represented: {domain_counts}")
    expected_cell = expected // (len(DOMAINS) * len(CORRECTION_TYPES))
    cell_counts = _counts(f"{row['domain']}::{row['correction_type']}" for row in tasks)
    if any(count != expected_cell for count in cell_counts.values()):
        raise AssertionError(f"Domain/type cells are not balanced: {cell_counts}")
    for row in tasks:
        if len(row.get("probes", [])) != 5:
            raise AssertionError(f"Task {row['item_id']} does not have exactly 5 probes")
        if not row.get("should_update") or not row.get("should_preserve"):
            raise AssertionError(f"Task {row['item_id']} missing should_update/should_preserve")
        if set(row.get("feedback_by_condition", {})) != expected_conditions:
            raise AssertionError(f"Task {row['item_id']} missing feedback conditions")
        if "step_targeted_feedback" in expected_conditions:
            pass
        elif "step_targeted_feedback" in row.get("feedback_by_condition", {}):
            raise AssertionError(f"Corrected task {row['item_id']} unexpectedly includes step_targeted_feedback")
        update_probes = [probe for probe in row["probes"] if probe["state_key"] in row["should_update"]]
        preserve_probes = [probe for probe in row["probes"] if probe["state_key"] in row["should_preserve"]]
        min_update = 2 if require_two_update_probes else 1
        if len(update_probes) < min_update or len(preserve_probes) < 2:
            raise AssertionError(f"Task {row['item_id']} lacks update/preserve probe coverage")
    return {
        "examples": len(tasks),
        "domain_counts": domain_counts,
        "correction_type_counts": type_counts,
        "domain_correction_type_counts": cell_counts,
    }


def validate_selective_corrected_data(tasks_file: str | Path, expected: int = 96) -> dict[str, Any]:
    return validate_selective_data(
        tasks_file,
        expected=expected,
        conditions=CORRECTED_PROMPT_CONDITIONS,
        require_two_update_probes=False,
    )


def validate_selective_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    required = (
        "feedback_tasks.jsonl",
        "revision_generations.jsonl",
        "parsed_probe_results.jsonl",
        "enriched_results.jsonl",
        "metrics_summary.json",
        "metrics_by_condition.csv",
        "metrics_by_domain.csv",
        "metrics_by_correction_type.csv",
        "metrics_by_model.csv",
        "metrics_by_state_key.csv",
        "scoring_diagnostics.json",
        "qualitative_examples.json",
        "run_manifest.json",
    )
    for file_name in required:
        if not (run_path / file_name).exists():
            raise AssertionError(f"Missing {file_name}")
    generations = read_jsonl(run_path / "revision_generations.jsonl")
    probes = read_jsonl(run_path / "parsed_probe_results.jsonl")
    if not generations:
        raise AssertionError("No selective correction generations")
    if len(probes) != len(generations) * 5:
        raise AssertionError(f"Expected 5 probe rows per generation, got {len(probes)} for {len(generations)} generations")
    parse_rate = _rate(sum(row.get("parsed", {}).get("parse_error") is None for row in generations), len(generations))
    if parse_rate < 0.70:
        raise AssertionError(f"Parse rate too low: {parse_rate}")
    if not all("raw_output" in row for row in generations):
        raise AssertionError("Raw model outputs are not preserved")
    return {
        "generations": len(generations),
        "probe_results": len(probes),
        "parse_rate": parse_rate,
        "conditions": _counts(row["condition"] for row in generations),
        "models": _counts(row["model"] for row in generations),
    }


def _generation_record(
    run_id: str,
    model: str,
    task: CorrectionTask,
    condition: str,
    prompt: str,
    raw_output: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "record_type": "selective_revision_generation",
        "model": model,
        "item_id": task.item_id,
        "domain": task.domain,
        "correction_type": task.correction_type,
        "condition": condition,
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed": parsed,
        "context": task.context,
        "initial_trace": task.initial_trace,
        "feedback": task.feedback_by_condition[condition],
        "gold_state_before": task.gold_state_before,
        "gold_state_after": task.gold_state_after,
        "should_update": task.should_update,
        "should_preserve": task.should_preserve,
        "probes": task.probes,
        "metadata": task.metadata,
    }


def target_binding_correct(row: dict[str, Any], parsed: dict[str, Any], probe_rows: list[dict[str, Any]]) -> bool | None:
    target_probe = next((probe for probe in probe_rows if probe["probe_type"] == "target_binding"), None)
    if target_probe:
        return bool(target_probe["is_correct"])
    if row["condition"] != "correction_scope_map":
        return None
    target = normalize_answer(parsed.get("feedback_target", ""))
    aliases = [row["metadata"].get("target_key", ""), *row["metadata"].get("target_aliases", [])]
    return any(normalize_answer(alias) and normalize_answer(alias) in target for alias in aliases)


def trace_answer_consistent(parsed: dict[str, Any], probe_rows: list[dict[str, Any]], row: dict[str, Any]) -> bool | None:
    if parsed.get("parse_error"):
        return None
    final_probe = next((probe for probe in probe_rows if probe["probe_type"] == "final_state"), None)
    if not final_probe:
        return None
    final_answer = normalize_answer(parsed.get("final_answer", ""))
    probe_answer = normalize_answer(final_probe.get("model_answer", ""))
    gold_final = normalize_answer(final_probe.get("gold_answer", ""))
    if not final_answer:
        return None
    if gold_final and gold_final in final_answer and final_probe["is_correct"]:
        return True
    if probe_answer and probe_answer in final_answer:
        return True
    reasoning = normalize_answer(parsed.get("revised_reasoning", ""))
    if gold_final and gold_final in reasoning and final_probe["is_correct"]:
        return True
    return None


def answer_matches(answer: str, gold: str) -> bool:
    return score_answer(answer, gold)[0]


def score_answer(answer: str, gold: str) -> tuple[bool, str]:
    got = normalize_answer(answer)
    expected = normalize_answer(gold)
    if not got:
        return False, "failed_empty"
    if expected in {"yes", "no"}:
        yes_values = {"yes", "true", "succeeded", "success", "successful", "ready", "sent"}
        no_values = {"no", "false", "failed", "failure", "not ready", "not sent"}
        if expected == "yes" and (got in yes_values or got.startswith("yes ")):
            return True, "yes_no"
        if expected == "no" and (got in no_values or got.startswith("no ") or got.startswith("not ")):
            return True, "yes_no"
    if expected == got:
        return True, "exact"
    if _value_match(got, expected):
        return True, "value"
    if expected in got:
        return True, "normalized_contains"
    if _alias_match(got, expected):
        return True, "alias"
    return False, "failed"


def _value_match(got: str, expected: str) -> bool:
    if re.fullmatch(r"\$?\d+(?:\.\d+)?", expected):
        return expected.replace("$", "") in got.replace("$", "")
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(am|pm)", expected):
        return expected in got
    return False


def _alias_match(got: str, expected: str) -> bool:
    alias_groups = {
        "none": (
            "none",
            "no action",
            "no further action",
            "no action required",
            "nothing else is needed",
            "nothing else needed",
            "no next action",
        ),
        "refresh token and retry upload": (
            "refresh token retry upload",
            "renew token retry",
            "reauthenticate upload again",
            "retry upload after fixing token",
            "fix token retry upload",
            "refresh credentials retry upload",
        ),
        "use pre 2022 rule": (
            "legacy validation rule",
            "old validation rule",
            "pre 2022 rule",
            "use pre 2022 rule",
            "legacy rule",
            "historical rule",
        ),
        "apply 2022 rule": (
            "2022 rule",
            "post 2022 rule",
            "new rule",
            "checksum validation rule",
            "apply 2022 rule",
            "checksum validation",
        ),
        "use standard review": (
            "standard review",
            "normal review",
            "regular review",
            "use standard review",
        ),
        "director approval": (
            "director approval",
            "director level approval",
            "approval from director",
            "director approval required",
        ),
        "use renewal review": (
            "renewal review",
            "use renewal review",
            "renewal procedure",
        ),
    }
    aliases = alias_groups.get(expected)
    return bool(aliases and any(alias in got for alias in aliases))


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9:./ ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    mappings = {
        "true": "yes",
        "false": "no",
        "succeed": "yes",
        "succeeded": "yes",
        "success": "yes",
        "failed": "no",
        "failure": "no",
    }
    return mappings.get(text, text)


def qualitative_examples(
    enriched: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> dict[str, Any]:
    gen_by_key = {(row["model"], row["item_id"], row["condition"]): row for row in generations}
    probes_by_key = defaultdict(list)
    for probe in probe_rows:
        probes_by_key[(probe["model"], probe["item_id"], probe["condition"])].append(probe)

    def first(predicate) -> dict[str, Any] | None:
        for row in enriched:
            if predicate(row):
                return _qual_case(row, gen_by_key, probes_by_key)
        return None

    by_item_model = defaultdict(list)
    for row in enriched:
        by_item_model[(row["model"], row["item_id"])].append(row)

    all_success = None
    for rows in by_item_model.values():
        if len(rows) == len(PROMPT_CONDITIONS) and all(row["selective_revision_score"] == 1.0 for row in rows):
            all_success = _qual_case(rows[0], gen_by_key, probes_by_key)
            break

    return {
        "under_propagation": first(lambda row: row["under_propagation_rate"] > 0),
        "overcorrection": first(lambda row: row["overcorrection_rate"] > 0),
        "wrong_scope_update": first(lambda row: row["correction_type"] == "scope_limited" and row["overcorrection_rate"] > 0),
        "trace_answer_mismatch": first(lambda row: row["trace_answer_consistency"] is False),
        "future_use_failure": first(lambda row: row["future_use_accuracy"] < 1.0),
        "correction_scope_map_success": first(lambda row: row["condition"] == "correction_scope_map" and row["selective_revision_score"] == 1.0),
        "full_regeneration_corrupts_unrelated_state": first(lambda row: row["condition"] == "full_regeneration" and row["overcorrection_rate"] > 0),
        "all_conditions_success": all_success,
        "scoring_alias_match": _probe_qual(next((row for row in probe_rows if row.get("scoring_method") == "alias"), None)),
        "scorer_fixed_previous_failure_pattern": _probe_qual(
            next(
                (
                    row
                    for row in probe_rows
                    if row["state_key"] in {"next_action", "job_a_rule", "job_b_rule", "case_a_review", "case_b_review"}
                    and row.get("scoring_method") in {"alias", "value", "yes_no"}
                    and row.get("is_correct")
                ),
                None,
            )
        ),
        "parse_failure": first(lambda row: row.get("parse_error")),
    }


def _probe_qual(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "model": row["model"],
        "item_id": row["item_id"],
        "condition": row["condition"],
        "state_key": row["state_key"],
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "model_answer": row["model_answer"],
        "scoring_method": row.get("scoring_method"),
    }


def _qual_case(
    row: dict[str, Any],
    gen_by_key: dict[tuple[str, str, str], dict[str, Any]],
    probes_by_key: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    key = (row["model"], row["item_id"], row["condition"])
    generation = gen_by_key.get(key, {})
    return {
        "model": row["model"],
        "item_id": row["item_id"],
        "domain": row["domain"],
        "correction_type": row["correction_type"],
        "condition": row["condition"],
        "selective_revision_score": row["selective_revision_score"],
        "dependent_update_recall": row["dependent_update_recall"],
        "boundary_preservation_precision": row["boundary_preservation_precision"],
        "trace_answer_consistency": row["trace_answer_consistency"],
        "feedback": generation.get("feedback"),
        "raw_output": generation.get("raw_output"),
        "parsed": generation.get("parsed"),
        "probe_results": probes_by_key.get(key, []),
    }


def _bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "target_binding_accuracy": _mean_bool(row["target_binding_correct"] for row in rows if row["target_binding_correct"] is not None),
        "dependent_update_recall": _mean(row["dependent_update_recall"] for row in rows),
        "boundary_preservation_precision": _mean(row["boundary_preservation_precision"] for row in rows),
        "overcorrection_rate": _mean(row["overcorrection_rate"] for row in rows),
        "under_propagation_rate": _mean(row["under_propagation_rate"] for row in rows),
        "trace_answer_consistency": _mean_bool(row["trace_answer_consistency"] for row in rows if row["trace_answer_consistency"] is not None),
        "future_use_accuracy": _mean_optional(row["future_use_accuracy"] for row in rows),
        "final_state_accuracy": _mean_optional(row["final_state_accuracy"] for row in rows),
        "selective_revision_score": _mean(row["selective_revision_score"] for row in rows),
        "probe_accuracy": _mean(row["probe_accuracy"] for row in rows),
        "parse_rate": _rate(sum(row.get("parse_error") is None for row in rows), len(rows)),
    }


def _probe_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": _rate(sum(row["is_correct"] for row in rows), len(rows)),
        "update_probe_accuracy": _rate(sum(row["is_correct"] for row in rows if row["is_update_probe"]), sum(row["is_update_probe"] for row in rows)),
        "preserve_probe_accuracy": _rate(sum(row["is_correct"] for row in rows if row["is_preserve_probe"]), sum(row["is_preserve_probe"] for row in rows)),
    }


def _write_metric_tables(run_path: Path, metrics: dict[str, Any]) -> None:
    table_specs = {
        "metrics_by_condition.csv": metrics["by_condition"],
        "metrics_by_domain.csv": metrics["by_domain"],
        "metrics_by_correction_type.csv": metrics["by_correction_type"],
        "metrics_by_model.csv": metrics["by_model"],
        "metrics_by_state_key.csv": metrics["by_state_key"],
    }
    for file_name, grouped in table_specs.items():
        rows = []
        for group, bundle in grouped.items():
            row = {"group": group}
            row.update(bundle)
            rows.append(row)
        fieldnames = ["group", *list(next(iter(grouped.values())).keys())] if grouped else ["group"]
        write_csv(run_path / file_name, rows, fieldnames)


def _extract_json(raw_output: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    return []


def _feedbacks(base: str, step: str, natural: str) -> dict[str, str]:
    return {
        "standard_feedback": base,
        "step_targeted_feedback": f"In {step}, correction: {base}",
        "natural_feedback": natural,
        "full_regeneration": base,
        "correction_scope_map": base,
    }


def _make_task(
    idx: int,
    domain: str,
    correction_type: str,
    context: str,
    initial_trace: str,
    feedback: dict[str, str],
    before: dict[str, Any],
    after: dict[str, Any],
    should_update: list[str],
    should_preserve: list[str],
    probes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> CorrectionTask:
    return CorrectionTask(
        item_id=f"selective-{idx:04d}-{domain}-{correction_type}",
        domain=domain,
        correction_type=correction_type,
        context=context,
        initial_trace=initial_trace,
        feedback_by_condition=feedback,
        gold_state_before=before,
        gold_state_after=after,
        should_update=should_update,
        should_preserve=should_preserve,
        probes=probes,
        metadata=metadata,
    )


def _probe(probe_id: str, question: str, gold: str, key: str, probe_type: str) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "question": question,
        "gold_answer": gold,
        "state_key": key,
        "probe_type": probe_type,
    }


def _tool_local_only(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    report = f"report_{cell_idx}.csv"
    old_time = f"4:{10 + cell_idx % 10:02d} PM"
    new_time = f"4:{20 + cell_idx % 10:02d} PM"
    context = (
        f"An agent processed {report}. Conversion to parquet succeeded, upload to archive succeeded, "
        f"the archive copy is ready, and the analyst notification was sent."
    )
    trace = (
        "Step 1: The CSV was converted to Parquet.\n"
        "Step 2: The Parquet file was uploaded to the archive.\n"
        f"Step 3: The analyst notification was sent at {old_time}.\n"
        "Final answer: The archive copy is ready."
    )
    before = {"conversion_success": "yes", "upload_success": "yes", "archive_copy_ready": "yes", "notification_sent": "yes", "notification_time": old_time, "next_action": "none"}
    after = {**before, "notification_time": new_time}
    probes = [
        _probe("p1", "What is the corrected notification time?", new_time, "notification_time", "updated_dependency"),
        _probe("p2", "Was the analyst notified?", "yes", "notification_sent", "preserved_boundary"),
        _probe("p3", "Did the archive upload succeed?", "yes", "upload_success", "preserved_boundary"),
        _probe("p4", "Is the archive copy ready?", "yes", "archive_copy_ready", "final_state"),
        _probe("p5", "If asked what to do next, what should the agent do?", "none", "next_action", "future_use"),
    ]
    return _make_task(
        idx,
        "tool_agent_state",
        "local_only",
        context,
        trace,
        _feedbacks(
            f"The notification was sent at {new_time}, not {old_time}.",
            "Step 3",
            f"The timestamp in the notification note is off; it should be {new_time}.",
        ),
        before,
        after,
        ["notification_time", "archive_copy_ready"],
        ["notification_sent", "upload_success", "next_action"],
        probes,
        {"target_key": "notification_time", "target_aliases": ["notification time", "timestamp"]},
    )


def _tool_local_to_final(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    token = ["archive token expired", "remote archive rejected credentials", "signed URL expired"][cell_idx % 3]
    context = (
        "An agent converted report.csv to report.parquet. The conversion succeeded. "
        "The analyst notification was sent after processing."
    )
    trace = (
        "Step 1: The CSV was converted to Parquet.\n"
        "Step 2: The Parquet file was uploaded to the archive.\n"
        "Step 3: The analyst was notified that the archive copy is ready.\n"
        "Final answer: The archive copy is ready."
    )
    before = {"conversion_success": "yes", "upload_success": "yes", "archive_copy_ready": "yes", "notification_sent": "yes", "next_action": "none"}
    after = {"conversion_success": "yes", "upload_success": "no", "archive_copy_ready": "no", "notification_sent": "yes", "next_action": "refresh token and retry upload"}
    probes = [
        _probe("p1", "Did the upload succeed?", "no", "upload_success", "updated_dependency"),
        _probe("p2", "Is the archive copy ready?", "no", "archive_copy_ready", "final_state"),
        _probe("p3", "Did conversion succeed?", "yes", "conversion_success", "preserved_boundary"),
        _probe("p4", "Was the analyst notified?", "yes", "notification_sent", "preserved_boundary"),
        _probe("p5", "What should the agent do next?", "refresh token and retry upload", "next_action", "future_use"),
    ]
    return _make_task(
        idx,
        "tool_agent_state",
        "local_to_final",
        context,
        trace,
        _feedbacks(
            f"The upload failed because the {token}.",
            "Step 2",
            f"The remote archive step did not actually complete; the {token}.",
        ),
        before,
        after,
        ["upload_success", "archive_copy_ready", "next_action"],
        ["conversion_success", "notification_sent"],
        probes,
        {"target_key": "upload_success", "target_aliases": ["upload", "archive upload"]},
    )


def _tool_scope_limited(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    files = ["alpha.csv", "beta.csv", "gamma.csv"]
    target = files[cell_idx % len(files)]
    others = [file for file in files if file != target]
    context = "The agent processed alpha.csv, beta.csv, and gamma.csv. All three were converted and marked uploaded."
    trace = (
        "Step 1: alpha.csv uploaded successfully.\n"
        "Step 2: beta.csv uploaded successfully.\n"
        "Step 3: gamma.csv uploaded successfully.\n"
        "Final answer: All remote archive copies are ready."
    )
    before = {f"{file}_upload_success": "yes" for file in files}
    before["all_archives_ready"] = "yes"
    after = dict(before)
    after[f"{target}_upload_success"] = "no"
    after["all_archives_ready"] = "no"
    probes = [
        _probe("p1", f"Did {target} upload successfully?", "no", f"{target}_upload_success", "updated_dependency"),
        _probe("p2", "Are all archive copies ready?", "no", "all_archives_ready", "final_state"),
        _probe("p3", f"Did {others[0]} upload successfully?", "yes", f"{others[0]}_upload_success", "preserved_boundary"),
        _probe("p4", f"Did {others[1]} upload successfully?", "yes", f"{others[1]}_upload_success", "preserved_boundary"),
        _probe("p5", "Which file should be retried?", target, f"{target}_upload_success", "future_use"),
    ]
    return _make_task(
        idx,
        "tool_agent_state",
        "scope_limited",
        context,
        trace,
        _feedbacks(
            f"The upload failure applies only to {target}; the other files uploaded successfully.",
            f"the step for {target}",
            f"One file was overgeneralized: only {target} had the archive failure.",
        ),
        before,
        after,
        [f"{target}_upload_success", "all_archives_ready"],
        [f"{others[0]}_upload_success", f"{others[1]}_upload_success"],
        probes,
        {"target_key": f"{target}_upload_success", "target_aliases": [target, "specific file upload"]},
    )


def _tool_rule_level(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    context = (
        "Archive policy: jobs before 2022 use legacy validation; jobs after Jan 1, 2022 require checksum validation. "
        "Job A was submitted in 2021. Job B was submitted in 2023."
    )
    trace = (
        "Step 1: Apply checksum validation to Job A.\n"
        "Step 2: Apply checksum validation to Job B.\n"
        "Step 3: Both jobs use the same 2022 archive rule.\n"
        "Final answer: Both jobs require checksum validation."
    )
    before = {"job_a_rule": "apply 2022 rule", "job_b_rule": "apply 2022 rule", "legacy_procedure_valid": "no", "job_b_requires_checksum": "yes", "job_a_requires_checksum": "yes"}
    after = {"job_a_rule": "use pre-2022 rule", "job_b_rule": "apply 2022 rule", "legacy_procedure_valid": "yes", "job_b_requires_checksum": "yes", "job_a_requires_checksum": "no"}
    probes = [
        _probe("p1", "Which rule applies to Job A?", "use pre-2022 rule", "job_a_rule", "updated_dependency"),
        _probe("p2", "Does Job A require checksum validation?", "no", "job_a_requires_checksum", "updated_dependency"),
        _probe("p3", "Which rule applies to Job B?", "apply 2022 rule", "job_b_rule", "preserved_boundary"),
        _probe("p4", "Does Job B require checksum validation?", "yes", "job_b_requires_checksum", "preserved_boundary"),
        _probe("p5", "For a future 2023 job, which rule should be used?", "apply 2022 rule", "job_b_rule", "future_use"),
    ]
    return _make_task(
        idx,
        "tool_agent_state",
        "rule_level",
        context,
        trace,
        _feedbacks(
            "The 2022 archive rule applies only to jobs submitted after Jan 1, 2022.",
            "Step 1",
            "The date boundary was missed; the newer archive validation is not retroactive.",
        ),
        before,
        after,
        ["job_a_rule", "job_a_requires_checksum"],
        ["job_b_rule", "job_b_requires_checksum"],
        probes,
        {"target_key": "archive_rule_date_boundary", "target_aliases": ["2022 rule", "date boundary"]},
    )


def _rule_local_only(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    old_fee = f"${100 + cell_idx}"
    new_fee = f"${110 + cell_idx}"
    context = "A historical procedure page and a current fee page describe renewal applications. The application remains eligible for renewal review."
    trace = (
        "Step 1: The request is a renewal application.\n"
        f"Step 2: The current fee is {old_fee}.\n"
        "Step 3: The renewal procedure applies.\n"
        "Final answer: The application can use renewal review."
    )
    before = {"application_type": "renewal", "current_fee": old_fee, "renewal_review_allowed": "yes", "historical_procedure_valid": "yes", "next_action": "use renewal review"}
    after = {**before, "current_fee": new_fee}
    probes = [
        _probe("p1", "What is the corrected current fee?", new_fee, "current_fee", "updated_dependency"),
        _probe("p2", "Is this still a renewal application?", "yes", "application_type", "preserved_boundary"),
        _probe("p3", "Can the application use renewal review?", "yes", "renewal_review_allowed", "final_state"),
        _probe("p4", "Is the historical procedure still valid for procedure steps?", "yes", "historical_procedure_valid", "preserved_boundary"),
        _probe("p5", "What should be used for the next action?", "use renewal review", "next_action", "future_use"),
    ]
    return _make_task(
        idx,
        "rule_scope_reasoning",
        "local_only",
        context,
        trace,
        _feedbacks(
            f"The current fee is {new_fee}, not {old_fee}.",
            "Step 2",
            f"Only the fee number is stale; the procedure classification is unchanged at {new_fee}.",
        ),
        before,
        after,
        ["current_fee", "renewal_review_allowed"],
        ["application_type", "historical_procedure_valid", "next_action"],
        probes,
        {"target_key": "current_fee", "target_aliases": ["fee", "fee number"]},
    )


def _rule_local_to_final(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    context = "A shipment is international, high value, and was initially treated as domestic. Domestic shipments receive a 10 percent discount."
    trace = (
        "Step 1: Treat the shipment as domestic.\n"
        "Step 2: Apply the domestic discount.\n"
        "Step 3: The discounted route is approved.\n"
        "Final answer: Apply the 10 percent domestic discount."
    )
    before = {"shipment_domestic": "yes", "discount_applies": "yes", "route_approved": "yes", "requires_customs_review": "no", "insurance_required": "yes", "high_value": "yes"}
    after = {"shipment_domestic": "no", "discount_applies": "no", "route_approved": "no", "requires_customs_review": "yes", "insurance_required": "yes", "high_value": "yes"}
    probes = [
        _probe("p1", "Is the shipment domestic?", "no", "shipment_domestic", "updated_dependency"),
        _probe("p2", "Does the domestic discount apply?", "no", "discount_applies", "updated_dependency"),
        _probe("p3", "Is insurance still required?", "yes", "insurance_required", "preserved_boundary"),
        _probe("p4", "Is the shipment still high value?", "yes", "high_value", "preserved_boundary"),
        _probe("p5", "Should the discounted route be approved?", "no", "route_approved", "final_state"),
    ]
    return _make_task(
        idx,
        "rule_scope_reasoning",
        "local_to_final",
        context,
        trace,
        _feedbacks(
            "The shipment is international, not domestic.",
            "Step 1",
            "The geographic scope was wrong: this package crosses a border.",
        ),
        before,
        after,
        ["shipment_domestic", "discount_applies", "route_approved"],
        ["insurance_required", "high_value"],
        probes,
        {"target_key": "shipment_domestic", "target_aliases": ["domestic status", "international shipment"]},
    )


def _rule_scope_limited(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    shipments = ["Shipment A", "Shipment B", "Shipment C", "Shipment D"]
    target = shipments[cell_idx % len(shipments)]
    others = [shipment for shipment in shipments if shipment != target]
    context = "Four shipments are being reviewed. An expedited exception can apply to exactly one named shipment if specified."
    trace = (
        "Step 1: Apply expedited exception to Shipment A.\n"
        "Step 2: Apply expedited exception to Shipment B.\n"
        "Step 3: Apply expedited exception to Shipment C and Shipment D.\n"
        "Final answer: All shipments are expedited."
    )
    before = {f"{shipment}_expedited": "yes" for shipment in shipments}
    before["all_shipments_expedited"] = "yes"
    after = {f"{shipment}_expedited": ("yes" if shipment == target else "no") for shipment in shipments}
    after["all_shipments_expedited"] = "no"
    probes = [
        _probe("p1", f"Does the exception apply to {target}?", "yes", f"{target}_expedited", "updated_dependency"),
        _probe("p2", "Are all shipments expedited?", "no", "all_shipments_expedited", "final_state"),
        _probe("p3", f"Does the exception apply to {others[0]}?", "no", f"{others[0]}_expedited", "preserved_boundary"),
        _probe("p4", f"Does the exception apply to {others[1]}?", "no", f"{others[1]}_expedited", "preserved_boundary"),
        _probe("p5", "For a new unrelated shipment, should this exception apply?", "no", "unrelated_future_shipment", "future_use"),
    ]
    return _make_task(
        idx,
        "rule_scope_reasoning",
        "scope_limited",
        context,
        trace,
        _feedbacks(
            f"The expedited exception applies only to {target}.",
            f"the step applying the exception beyond {target}",
            f"The exception was scoped too broadly; only {target} gets it.",
        ),
        before,
        after,
        [f"{target}_expedited", "all_shipments_expedited"],
        [f"{others[0]}_expedited", f"{others[1]}_expedited", "unrelated_future_shipment"],
        probes,
        {"target_key": f"{target}_expedited", "target_aliases": [target, "expedited exception"]},
    )


def _rule_rule_level(idx: int, cell_idx: int, rng: random.Random) -> CorrectionTask:
    del rng
    context = "Approval policy: low-risk cases use standard review. High-risk cases submitted after 2022 require director approval."
    trace = (
        "Step 1: Case A is low-risk and submitted in 2021, so director approval is required.\n"
        "Step 2: Case B is high-risk and submitted in 2023, so director approval is required.\n"
        "Step 3: Apply director approval to all cases.\n"
        "Final answer: Both cases require director approval."
    )
    before = {"case_a_director_approval": "yes", "case_b_director_approval": "yes", "case_a_review": "director approval", "case_b_review": "director approval", "low_risk_rule": "director approval"}
    after = {"case_a_director_approval": "no", "case_b_director_approval": "yes", "case_a_review": "use standard review", "case_b_review": "director approval", "low_risk_rule": "use standard review"}
    probes = [
        _probe("p1", "Does Case A require director approval?", "no", "case_a_director_approval", "updated_dependency"),
        _probe("p2", "What review should Case A use?", "use standard review", "case_a_review", "updated_dependency"),
        _probe("p3", "Does Case B require director approval?", "yes", "case_b_director_approval", "preserved_boundary"),
        _probe("p4", "What review should Case B use?", "director approval", "case_b_review", "preserved_boundary"),
        _probe("p5", "For a future low-risk case, what review should be used?", "use standard review", "low_risk_rule", "future_use"),
    ]
    return _make_task(
        idx,
        "rule_scope_reasoning",
        "rule_level",
        context,
        trace,
        _feedbacks(
            "The director-approval rule applies only to high-risk cases submitted after 2022.",
            "Step 1",
            "The risk/date boundary was missed; low-risk cases stay on standard review.",
        ),
        before,
        after,
        ["case_a_director_approval", "case_a_review", "low_risk_rule"],
        ["case_b_director_approval", "case_b_review"],
        probes,
        {"target_key": "director_approval_scope", "target_aliases": ["high-risk after 2022", "risk/date boundary"]},
    )


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return dict(sorted(grouped.items()))


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else 0.0


def _rate_optional(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 4) if denominator else None


def _mean(values: Iterable[int | float]) -> float:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def _mean_optional(values: Iterable[int | float]) -> float | None:
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 4) if values else None


def _mean_bool(values: Iterable[bool]) -> float | None:
    values = list(values)
    if not values:
        return None
    return _rate(sum(bool(value) for value in values), len(values))
