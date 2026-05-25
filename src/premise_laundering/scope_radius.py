from __future__ import annotations

import itertools
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from .parsing import extract_json_object
from .schema import MODEL_ID, SEED

EXPERIMENT_NAME = "scope_radius_counterfactuals"
SCOPE_CONDITIONS = ("standard_feedback", "full_regeneration", "scope_ledger")
DOMAINS = ("tool_agent_state", "rule_policy_scope", "source_status_rag_scope")
SCOPE_RADII = ("local_entity", "entity_class", "temporal_boundary", "rule_boundary")
CORRECTION_TYPES = (
    "upload_state",
    "archive_status",
    "policy_exception",
    "policy_date",
    "risk_review",
    "source_validity",
    "population_validity",
    "causal_validity",
    "dosage_validity",
)
MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class ScopeTask:
    item_id: str
    base_context_id: str
    domain: str
    scope_radius: str
    correction_type: str
    context: str
    old_trace: str
    feedback_by_condition: dict[str, str]
    state_inventory: list[str]
    gold_direct_target: str
    gold_scope_radius: str
    gold_should_change: list[str]
    gold_should_preserve: list[str]
    gold_scope_boundary: str
    gold_action_consequences: list[str]
    probes: list[dict[str, Any]]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScopeTask":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "base_context_id": self.base_context_id,
            "domain": self.domain,
            "scope_radius": self.scope_radius,
            "correction_type": self.correction_type,
            "context": self.context,
            "old_trace": self.old_trace,
            "feedback_by_condition": self.feedback_by_condition,
            "state_inventory": self.state_inventory,
            "gold_direct_target": self.gold_direct_target,
            "gold_scope_radius": self.gold_scope_radius,
            "gold_should_change": self.gold_should_change,
            "gold_should_preserve": self.gold_should_preserve,
            "gold_scope_boundary": self.gold_scope_boundary,
            "gold_action_consequences": self.gold_action_consequences,
            "probes": self.probes,
            "metadata": self.metadata,
        }


def prepare_scope_radius_data(output_dir: str | Path, n_base_contexts: int = 40, overwrite: bool = False) -> Path:
    output = ensure_dir(output_dir)
    tasks_path = output / "scope_radius_tasks.jsonl"
    manifest_path = output / "scope_radius_data_manifest.json"
    if tasks_path.exists() and not overwrite:
        return tasks_path

    tasks: list[ScopeTask] = []
    variant_counter = 0
    for base_idx in range(n_base_contexts):
        domain = DOMAINS[base_idx % len(DOMAINS)]
        omitted_radius = SCOPE_RADII[base_idx % len(SCOPE_RADII)]
        radii = [radius for radius in SCOPE_RADII if radius != omitted_radius]
        base_id = f"scope-base-{base_idx:03d}-{domain}"
        for variant_idx, radius in enumerate(radii):
            task = _build_task(base_idx, variant_idx, variant_counter, base_id, domain, radius)
            tasks.append(task)
            variant_counter += 1

    write_jsonl(tasks_path, [task.to_dict() for task in tasks])
    write_json(
        manifest_path,
        {
            "experiment": EXPERIMENT_NAME,
            "seed": SEED,
            "base_contexts": n_base_contexts,
            "examples": len(tasks),
            "domains": dict(Counter(task.domain for task in tasks)),
            "scope_radii": dict(Counter(task.scope_radius for task in tasks)),
            "correction_types": dict(Counter(task.correction_type for task in tasks)),
            "conditions": list(SCOPE_CONDITIONS),
            "model": MODEL,
        },
    )
    return tasks_path


def load_scope_tasks(path: str | Path) -> list[ScopeTask]:
    return [ScopeTask.from_dict(record) for record in read_jsonl(path)]


def validate_scope_radius_data(tasks_file: str | Path, expected_bases: int = 40, expected_examples: int = 120) -> dict[str, Any]:
    tasks = load_scope_tasks(tasks_file)
    base_counts = Counter(task.base_context_id for task in tasks)
    errors: list[str] = []
    if len(base_counts) != expected_bases:
        errors.append(f"expected {expected_bases} base contexts, found {len(base_counts)}")
    if len(tasks) != expected_examples:
        errors.append(f"expected {expected_examples} examples, found {len(tasks)}")
    if any(count != 3 for count in base_counts.values()):
        errors.append("each base context must have exactly 3 feedback variants")
    domains = Counter(task.domain for task in tasks)
    if set(domains) != set(DOMAINS):
        errors.append(f"domains mismatch: {sorted(domains)}")
    radii = Counter(task.scope_radius for task in tasks)
    if set(radii) != set(SCOPE_RADII):
        errors.append(f"scope radii mismatch: {sorted(radii)}")
    for task in tasks:
        if set(task.feedback_by_condition) != set(SCOPE_CONDITIONS):
            errors.append(f"{task.item_id} has wrong conditions")
        if len(task.probes) != 5:
            errors.append(f"{task.item_id} does not have 5 probes")
        inventory = set(task.state_inventory)
        change = set(task.gold_should_change)
        preserve = set(task.gold_should_preserve)
        if change & preserve:
            errors.append(f"{task.item_id} has state keys in both change and preserve")
        if not change <= inventory:
            errors.append(f"{task.item_id} has change keys outside state_inventory")
        if not preserve <= inventory:
            errors.append(f"{task.item_id} has preserve keys outside state_inventory")
        for probe in task.probes:
            if "choices" not in probe or probe["gold_answer"] not in probe["choices"]:
                errors.append(f"{task.item_id}/{probe.get('probe_id')} has non-closed gold answer")
            if probe.get("state_key") not in inventory and probe.get("probe_type") not in {
                "update_consequence",
                "delayed_action",
                "stale_trace_conflict",
            }:
                errors.append(f"{task.item_id}/{probe.get('probe_id')} state_key outside state_inventory")
            if probe.get("probe_id") == "p4" and any(token in probe.get("question", "").lower() for token in ("can the", "approved now", "delete all")):
                errors.append(f"{task.item_id}/p4 still uses yes/no-all wording")
        if not task.gold_should_change or not task.gold_should_preserve:
            errors.append(f"{task.item_id} missing change/preserve sets")
        if task.metadata.get("correction_type") == "archive_status" and "gamma_upload" in task.gold_should_change:
            errors.append(f"{task.item_id} archive-status correction changes gamma_upload")
    for base_id, count in base_counts.items():
        variant_change_sets = {
            tuple(sorted(task.gold_should_change))
            for task in tasks
            if task.base_context_id == base_id
        }
        if len(variant_change_sets) < 3:
            errors.append(f"{base_id} variants do not have three distinct gold should_change sets")
    contrast_check = directional_contrast_sanity()
    if contrast_check["perfect"] != 1.0 or contrast_check["wrong"] >= 1.0:
        errors.append("directional contrast sanity check failed")
    if errors:
        raise ValueError("; ".join(errors[:12]))
    return {
        "base_contexts": len(base_counts),
        "examples": len(tasks),
        "domain_counts": dict(domains),
        "scope_radius_counts": dict(radii),
        "correction_type_counts": dict(Counter(task.correction_type for task in tasks)),
        "conditions": list(SCOPE_CONDITIONS),
        "probes_per_example": 5,
        "closed_set_probes": True,
        "directional_contrast_sanity": contrast_check,
    }


def build_scope_prompt(task: ScopeTask, condition: str) -> str:
    if condition not in SCOPE_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    feedback = task.feedback_by_condition[condition]
    probe_lines = "\n".join(
        f"{probe['probe_id']}. {_probe_question_for_condition(probe, condition)}\nChoices: {' | '.join(probe['choices'])}"
        for probe in task.probes
    )
    allowed_keys = sorted(task.state_inventory)
    allowed_key_lines = "\n".join(f"- {key}" for key in allowed_keys)
    shared = (
        "You are revising a reasoning/state trace after localized feedback. "
        "Infer the exact scope/radius of the correction: what changes, what downstream actions change, "
        "what stays unchanged, and where the correction stops applying. "
        "Use only the provided context and feedback. Answer every probe with exactly one listed choice. "
        "For should_change and should_preserve, use only the allowed normalized state keys listed below.\n\n"
        f"Context:\n{task.context}\n\n"
        f"Feedback:\n{feedback}\n\n"
        f"Allowed normalized state keys:\n{allowed_key_lines}\n\n"
        f"Probes:\n{probe_lines}\n\n"
    )
    if condition != "full_regeneration":
        shared += f"Old trace that may contain stale state:\n{task.old_trace}\n\n"

    if condition == "scope_ledger":
        schema = (
            '{\n'
            '  "correction_scope_radius": "local_entity | entity_class | temporal_boundary | rule_boundary",\n'
            '  "direct_target": "...",\n'
            '  "scope_boundary": "...",\n'
            '  "should_change": ["normalized_state_key"],\n'
            '  "should_preserve": ["normalized_state_key"],\n'
            '  "action_consequences": ["..."],\n'
            '  "immediate_answers": {"p1": "...", "p2": "..."},\n'
            '  "delayed_action_answers": {"p3": "...", "p4": "...", "p5": "..."}\n'
            "}"
        )
    else:
        schema = (
            '{\n'
            '  "direct_target": "...",\n'
            '  "should_change": ["normalized_state_key"],\n'
            '  "should_preserve": ["normalized_state_key"],\n'
            '  "immediate_answers": {"p1": "...", "p2": "..."},\n'
            '  "delayed_action_answers": {"p3": "...", "p4": "...", "p5": "..."}\n'
            "}"
        )
    return shared + "Return only compact JSON matching this schema:\n" + schema


def _probe_question_for_condition(probe: dict[str, Any], condition: str) -> str:
    if condition == "full_regeneration" and probe.get("probe_type") == "stale_trace_conflict":
        return probe.get("no_old_trace_question") or probe["question"].replace("Given the prior trace claimed ", "Given the corrected state, ")
    return probe["question"]


def run_scope_radius_inference(
    tasks: list[ScopeTask],
    generator: Callable[[list[str]], list[str]],
    output_base_dir: str | Path,
    run_id: str,
    model: str = MODEL,
    backend: str = "transformers",
    overwrite: bool = False,
    batch_size: int = 1,
    generation_config: dict[str, Any] | None = None,
    conditions: tuple[str, ...] = SCOPE_CONDITIONS,
) -> Path:
    run_dir = ensure_dir(Path(output_base_dir) / run_id)
    generations_path = run_dir / "revision_generations.jsonl"
    if generations_path.exists() and not overwrite:
        raise FileExistsError(f"{run_dir} already exists; pass overwrite=True to replace")

    if overwrite:
        for name in (
            "scope_radius_tasks.jsonl",
            "revision_generations.jsonl",
            "parsed_results.jsonl",
            "enriched_results.jsonl",
            "metrics_summary.json",
            "metrics_by_condition.csv",
            "metrics_by_domain.csv",
            "metrics_by_scope_radius.csv",
            "metrics_by_probe_type.csv",
            "metrics_by_state_key.csv",
            "scope_contrast_metrics.csv",
            "scoring_diagnostics.json",
            "qualitative_examples.json",
            "run_manifest.json",
        ):
            path = run_dir / name
            if path.exists():
                path.unlink()

    write_jsonl(run_dir / "scope_radius_tasks.jsonl", [task.to_dict() for task in tasks])
    prompts: list[tuple[ScopeTask, str, str]] = [
        (task, condition, build_scope_prompt(task, condition))
        for task in tasks
        for condition in conditions
    ]
    records: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        outputs = generator([prompt for _, _, prompt in batch])
        for (task, condition, prompt), raw_output in zip(batch, outputs):
            parsed = parse_scope_output(raw_output)
            score = score_scope_generation(task, condition, parsed)
            generation = {
                "run_id": run_id,
                "item_id": task.item_id,
                "base_context_id": task.base_context_id,
                "domain": task.domain,
                "scope_radius": task.scope_radius,
                "correction_type": task.correction_type,
                "condition": condition,
                "model": model,
                "backend": backend,
                "prompt": prompt,
                "raw_output": raw_output,
                "parsed": parsed,
                "gold_direct_target": task.gold_direct_target,
                "gold_scope_radius": task.gold_scope_radius,
                "gold_should_change": task.gold_should_change,
                "gold_should_preserve": task.gold_should_preserve,
                "gold_scope_boundary": task.gold_scope_boundary,
                "gold_action_consequences": task.gold_action_consequences,
                "metadata": task.metadata,
                "generation_config": generation_config or {},
            }
            enriched = {**generation, **score}
            records.append(generation)
            enriched_rows.append(enriched)
            for probe_score in score["probe_scores"]:
                parsed_rows.append(
                    {
                        "run_id": run_id,
                        "item_id": task.item_id,
                        "base_context_id": task.base_context_id,
                        "domain": task.domain,
                        "scope_radius": task.scope_radius,
                        "correction_type": task.correction_type,
                        "condition": condition,
                        "model": model,
                        **probe_score,
                    }
                )

    write_jsonl(generations_path, records)
    write_jsonl(run_dir / "parsed_results.jsonl", parsed_rows)
    write_jsonl(run_dir / "enriched_results.jsonl", enriched_rows)
    write_json(
        run_dir / "run_manifest.json",
        {
            "experiment": EXPERIMENT_NAME,
            "run_id": run_id,
            "model": model,
            "backend": backend,
            "seed": SEED,
            "conditions": list(conditions),
            "base_contexts": len({task.base_context_id for task in tasks}),
            "examples": len(tasks),
            "generations": len(records),
            "generation_config": generation_config or {},
        },
    )
    summarize_scope_radius_run(run_dir)
    return run_dir


def parse_scope_output(raw_output: str) -> dict[str, Any]:
    data = extract_json_object(raw_output)
    if not isinstance(data, dict):
        return {
            "json": {},
            "parse_error": "no_json_object_found",
            "direct_target": "",
            "scope_radius": "",
            "should_change": [],
            "should_preserve": [],
            "action_consequences": [],
            "probe_answers": {},
        }
    immediate = data.get("immediate_answers") if isinstance(data.get("immediate_answers"), dict) else {}
    delayed = data.get("delayed_action_answers") if isinstance(data.get("delayed_action_answers"), dict) else {}
    probe_answers = {**{str(k): str(v) for k, v in immediate.items()}, **{str(k): str(v) for k, v in delayed.items()}}
    return {
        "json": data,
        "parse_error": None,
        "direct_target": str(data.get("direct_target", "")),
        "scope_radius": normalize_scope_radius(data.get("correction_scope_radius", "")),
        "scope_boundary": str(data.get("scope_boundary", "")),
        "should_change": normalize_state_set(data.get("should_change", [])),
        "should_preserve": normalize_state_set(data.get("should_preserve", [])),
        "action_consequences": [normalize_text(item) for item in _as_list(data.get("action_consequences", []))],
        "probe_answers": {key: normalize_choice(value) for key, value in probe_answers.items()},
    }


def score_scope_generation(task: ScopeTask, condition: str, parsed: dict[str, Any]) -> dict[str, Any]:
    change_gold = set(task.gold_should_change)
    preserve_gold = set(task.gold_should_preserve)
    change_pred = set(parsed["should_change"])
    preserve_pred = set(parsed["should_preserve"])
    update = prf(change_pred, change_gold)
    preserve = prf(preserve_pred, preserve_gold)

    probe_scores: list[dict[str, Any]] = []
    for probe in task.probes:
        answer = parsed["probe_answers"].get(probe["probe_id"], "")
        normalized_gold = normalize_choice(probe["gold_answer"])
        normalized_answer = normalize_closed_answer(answer, probe["choices"])
        aliases = {
            normalize_choice(alias)
            for alias in probe.get("aliases", {}).get(probe["gold_answer"], [])
        }
        correct = normalized_answer == normalized_gold or normalized_answer in aliases
        stale_follow = (
            probe["probe_type"] == "stale_trace_conflict"
            and normalized_answer == normalize_choice(probe.get("old_trace_answer", ""))
            and normalized_answer != normalized_gold
        )
        probe_scores.append(
            {
                "probe_id": probe["probe_id"],
                "probe_type": probe["probe_type"],
                "state_key": probe["state_key"],
                "question": probe["question"],
                "choices": "|".join(probe["choices"]),
                "gold_answer": normalized_gold,
                "model_answer": normalized_answer,
                "raw_model_answer": answer,
                "correct": correct,
                "stale_follow": stale_follow,
            }
        )

    probe_by_id = {row["probe_id"]: row for row in probe_scores}
    direct_correct = bool(probe_by_id.get("p1", {}).get("correct"))
    delayed = [row for row in probe_scores if row["probe_type"] == "delayed_action"]
    stale_probe = probe_by_id.get("p5", {})
    delayed_accuracy = mean_bool(row["correct"] for row in delayed)
    boundary_leak = bool(change_pred & preserve_gold) or any(
        not row["correct"] for row in probe_scores if row["probe_type"] == "boundary_preservation"
    )
    under_scope = bool(change_gold - change_pred) or any(
        not row["correct"] for row in probe_scores if row["probe_type"] == "update_consequence"
    )
    stale_metric_applicable = condition != "full_regeneration"
    exact_old_answer_reuse = bool(direct_correct and stale_probe.get("stale_follow")) if stale_metric_applicable else None
    stale_answer = str(stale_probe.get("model_answer", ""))
    stale_gold = str(stale_probe.get("gold_answer", ""))
    over_conservative = bool(
        stale_metric_applicable
        and direct_correct
        and stale_probe
        and not stale_probe.get("correct")
        and stale_answer in {"none", "no_use", "no"}
        and stale_gold not in {"none", "no_use", "no"}
    )
    under_scoped_after_correction = bool(
        stale_metric_applicable
        and direct_correct
        and stale_probe
        and not stale_probe.get("correct")
        and stale_answer in {"all", "full_use", "yes"}
        and stale_gold not in {"all", "full_use", "yes"}
    )
    wrong_boundary_after_correction = bool(
        stale_metric_applicable
        and direct_correct
        and stale_probe
        and not stale_probe.get("correct")
        and not exact_old_answer_reuse
        and not over_conservative
        and not under_scoped_after_correction
    )
    scope_radius_correct = None
    if condition == "scope_ledger":
        scope_radius_correct = parsed["scope_radius"] == task.gold_scope_radius
    return {
        "parse_ok": parsed.get("parse_error") is None,
        "direct_correction_correct": direct_correct,
        "boundary_probe_correct": bool(probe_by_id.get("p2", {}).get("correct")),
        "update_set_precision": update["precision"],
        "update_set_recall": update["recall"],
        "update_set_f1": update["f1"],
        "preserve_set_precision": preserve["precision"],
        "preserve_set_recall": preserve["recall"],
        "preserve_set_f1": preserve["f1"],
        "boundary_leak": boundary_leak,
        "under_scope": under_scope,
        "scope_radius_correct": scope_radius_correct,
        "delayed_action_accuracy": delayed_accuracy,
        "delayed_action_all_correct": all(row["correct"] for row in delayed),
        "exact_old_answer_reuse": exact_old_answer_reuse,
        "over_conservative_after_correction": over_conservative,
        "under_scoped_after_correction": under_scoped_after_correction,
        "wrong_boundary_after_correction": wrong_boundary_after_correction,
        "stale_conflict_probe_correct": bool(stale_probe.get("correct")),
        "stale_reversion": exact_old_answer_reuse,
        "selective_scope_score": update["f1"] * preserve["f1"] * delayed_accuracy,
        "probe_scores": probe_scores,
    }


def summarize_scope_radius_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    enriched = read_jsonl(run / "enriched_results.jsonl")
    probes = read_jsonl(run / "parsed_results.jsonl")
    contrast_rows = scope_contrast_rows(enriched)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "total_generations": len(enriched),
        "total_probe_results": len(probes),
        "parse_rate": mean_bool(row["parse_ok"] for row in enriched),
        "overall": aggregate_scope_metrics(enriched),
        "by_condition": group_metrics(enriched, "condition"),
        "by_domain": group_metrics(enriched, "domain"),
        "by_scope_radius": group_metrics(enriched, "scope_radius"),
        "by_correction_type": group_metrics(enriched, "correction_type"),
        "scope_contrast": aggregate_contrast(contrast_rows),
    }
    write_json(run / "metrics_summary.json", summary)
    write_metric_csv(run / "metrics_by_condition.csv", summary["by_condition"])
    write_metric_csv(run / "metrics_by_domain.csv", summary["by_domain"])
    write_metric_csv(run / "metrics_by_scope_radius.csv", summary["by_scope_radius"])
    write_metric_csv(run / "metrics_by_correction_type.csv", summary["by_correction_type"])
    write_probe_csv(run / "metrics_by_probe_type.csv", probes, "probe_type")
    write_probe_csv(run / "metrics_by_state_key.csv", probes, "state_key")
    write_csv(
        run / "scope_contrast_metrics.csv",
        contrast_rows,
        [
            "base_context_id",
            "condition",
            "model",
            "pairs",
            "variant_update_set_f1",
            "delta_added_precision",
            "delta_added_recall",
            "delta_added_f1",
            "delta_removed_precision",
            "delta_removed_recall",
            "delta_removed_f1",
            "directional_delta_f1",
        ],
    )
    write_json(run / "scoring_diagnostics.json", scoring_diagnostics(enriched, probes))
    write_json(run / "qualitative_examples.json", qualitative_examples(enriched, contrast_rows))
    return summary


def validate_scope_radius_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    required = [
        "scope_radius_tasks.jsonl",
        "revision_generations.jsonl",
        "parsed_results.jsonl",
        "enriched_results.jsonl",
        "metrics_summary.json",
        "metrics_by_condition.csv",
        "metrics_by_domain.csv",
        "metrics_by_scope_radius.csv",
        "metrics_by_correction_type.csv",
        "metrics_by_probe_type.csv",
        "metrics_by_state_key.csv",
        "scope_contrast_metrics.csv",
        "scoring_diagnostics.json",
        "qualitative_examples.json",
        "run_manifest.json",
    ]
    missing = [name for name in required if not (run / name).exists()]
    if missing:
        raise ValueError(f"missing run files: {missing}")
    tasks = read_jsonl(run / "scope_radius_tasks.jsonl")
    generations = read_jsonl(run / "revision_generations.jsonl")
    probes = read_jsonl(run / "parsed_results.jsonl")
    if len(tasks) != 120:
        raise ValueError(f"expected 120 tasks, found {len(tasks)}")
    if len(generations) != 360:
        raise ValueError(f"expected 360 generations, found {len(generations)}")
    if len(probes) != 1800:
        raise ValueError(f"expected 1800 probe rows, found {len(probes)}")
    if {row["condition"] for row in generations} != set(SCOPE_CONDITIONS):
        raise ValueError("condition mismatch")
    if {row["model"] for row in generations} != {MODEL} and not all("dry-run" in row["model"] for row in generations):
        raise ValueError("scope-radius pilot should use only Qwen")
    if any(not row.get("raw_output") for row in generations):
        raise ValueError("raw outputs are not preserved")
    qualitative = read_json(run / "qualitative_examples.json")
    for key in ("all_conditions_success", "scope_ledger_success"):
        example = qualitative.get(key)
        if example and not (
            example.get("direct_correction_correct")
            and not example.get("boundary_leak")
            and not example.get("under_scope")
            and example.get("delayed_action_accuracy") == 1.0
        ):
            raise ValueError(f"qualitative example {key} does not satisfy success criteria")
    return {
        "tasks": len(tasks),
        "generations": len(generations),
        "probe_results": len(probes),
        "parse_rate": mean_bool(row.get("parse_ok") for row in read_jsonl(run / "enriched_results.jsonl")),
    }


def aggregate_scope_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_correct_rows = [row for row in rows if row["direct_correction_correct"]]
    return {
        "n": len(rows),
        "parse_rate": mean_bool(row["parse_ok"] for row in rows),
        "direct_correction_accuracy": mean_bool(row["direct_correction_correct"] for row in rows),
        "boundary_probe_accuracy": mean_bool(row["boundary_probe_correct"] for row in rows),
        "update_set_f1": mean_float(row["update_set_f1"] for row in rows),
        "preserve_set_f1": mean_float(row["preserve_set_f1"] for row in rows),
        "boundary_leak_rate": mean_bool(row["boundary_leak"] for row in rows),
        "under_scope_rate": mean_bool(row["under_scope"] for row in rows),
        "scope_radius_accuracy": mean_optional_bool(row["scope_radius_correct"] for row in rows),
        "delayed_action_accuracy": mean_float(row["delayed_action_accuracy"] for row in rows),
        "delayed_action_given_direct_correct": mean_float(row["delayed_action_accuracy"] for row in direct_correct_rows),
        "exact_old_answer_reuse_rate": mean_optional_bool(row.get("exact_old_answer_reuse") for row in direct_correct_rows),
        "over_conservative_after_correction_rate": mean_optional_bool(row.get("over_conservative_after_correction") for row in direct_correct_rows),
        "under_scoped_after_correction_rate": mean_optional_bool(row.get("under_scoped_after_correction") for row in direct_correct_rows),
        "wrong_boundary_after_correction_rate": mean_optional_bool(row.get("wrong_boundary_after_correction") for row in direct_correct_rows),
        "stale_conflict_probe_accuracy": mean_bool(row.get("stale_conflict_probe_correct") for row in rows),
        "stale_reversion_rate": mean_optional_bool(row.get("exact_old_answer_reuse") for row in direct_correct_rows),
        "selective_scope_score": mean_float(row["selective_scope_score"] for row in rows),
    }


def group_metrics(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return [{"group": group, **aggregate_scope_metrics(group_rows)} for group, group_rows in sorted(grouped.items())]


def write_metric_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group",
        "n",
        "parse_rate",
        "direct_correction_accuracy",
        "boundary_probe_accuracy",
        "update_set_f1",
        "preserve_set_f1",
        "boundary_leak_rate",
        "under_scope_rate",
        "scope_radius_accuracy",
        "delayed_action_accuracy",
        "delayed_action_given_direct_correct",
        "exact_old_answer_reuse_rate",
        "over_conservative_after_correction_rate",
        "under_scoped_after_correction_rate",
        "wrong_boundary_after_correction_rate",
        "stale_conflict_probe_accuracy",
        "stale_reversion_rate",
        "selective_scope_score",
    ]
    write_csv(path, rows, fieldnames)


def write_probe_csv(path: Path, probes: list[dict[str, Any]], key: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in probes:
        grouped[str(row[key])].append(row)
    rows = [
        {
            "group": group,
            "n": len(items),
            "accuracy": mean_bool(item["correct"] for item in items),
            "stale_follow_rate": mean_bool(item["stale_follow"] for item in items),
        }
        for group, items in sorted(grouped.items())
    ]
    write_csv(path, rows, ["group", "n", "accuracy", "stale_follow_rate"])


def scope_contrast_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["base_context_id"], row["condition"], row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for (base_id, condition, model), items in sorted(grouped.items()):
        pairs = 0
        added_scores: list[dict[str, float]] = []
        removed_scores: list[dict[str, float]] = []
        variant_f1s = [
            prf(set(item["parsed"]["should_change"]), set(item["gold_should_change"]))["f1"]
            for item in items
        ]
        for left, right in itertools.combinations(items, 2):
            pairs += 1
            gold_left = set(left["gold_should_change"])
            gold_right = set(right["gold_should_change"])
            pred_left = set(left["parsed"]["should_change"])
            pred_right = set(right["parsed"]["should_change"])
            added_scores.append(prf(pred_right - pred_left, gold_right - gold_left))
            removed_scores.append(prf(pred_left - pred_right, gold_left - gold_right))
        added_precision = mean_float(score["precision"] for score in added_scores)
        added_recall = mean_float(score["recall"] for score in added_scores)
        added_f1 = mean_float(score["f1"] for score in added_scores)
        removed_precision = mean_float(score["precision"] for score in removed_scores)
        removed_recall = mean_float(score["recall"] for score in removed_scores)
        removed_f1 = mean_float(score["f1"] for score in removed_scores)
        output.append(
            {
                "base_context_id": base_id,
                "condition": condition,
                "model": model,
                "pairs": pairs,
                "variant_update_set_f1": mean_float(variant_f1s),
                "delta_added_precision": added_precision,
                "delta_added_recall": added_recall,
                "delta_added_f1": added_f1,
                "delta_removed_precision": removed_precision,
                "delta_removed_recall": removed_recall,
                "delta_removed_f1": removed_f1,
                "directional_delta_f1": (added_f1 + removed_f1) / 2,
            }
        )
    return output


def aggregate_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    return {
        condition: {
            "base_contexts": len(items),
            "variant_update_set_f1": mean_float(item["variant_update_set_f1"] for item in items),
            "directional_delta_f1": mean_float(item["directional_delta_f1"] for item in items),
            "delta_added_f1": mean_float(item["delta_added_f1"] for item in items),
            "delta_removed_f1": mean_float(item["delta_removed_f1"] for item in items),
        }
        for condition, items in sorted(by_condition.items())
    }


def directional_contrast_sanity() -> dict[str, float]:
    base = {
        "base_context_id": "b",
        "condition": "c",
        "model": "m",
        "domain": "tool_agent_state",
        "scope_radius": "local_entity",
        "correction_type": "upload_state",
    }
    gold_rows = [
        {**base, "gold_should_change": ["a"], "parsed": {"should_change": ["a"]}},
        {**base, "gold_should_change": ["a", "b"], "parsed": {"should_change": ["a", "b"]}},
        {**base, "gold_should_change": ["c"], "parsed": {"should_change": ["c"]}},
    ]
    wrong_rows = [
        {**base, "gold_should_change": ["a"], "parsed": {"should_change": ["x"]}},
        {**base, "gold_should_change": ["a", "b"], "parsed": {"should_change": ["y"]}},
        {**base, "gold_should_change": ["c"], "parsed": {"should_change": ["z"]}},
    ]
    return {
        "perfect": scope_contrast_rows(gold_rows)[0]["directional_delta_f1"],
        "wrong": scope_contrast_rows(wrong_rows)[0]["directional_delta_f1"],
    }


def scoring_diagnostics(enriched: list[dict[str, Any]], probes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parse_errors": sum(1 for row in enriched if not row["parse_ok"]),
        "empty_should_change": sum(1 for row in enriched if not row["parsed"]["should_change"]),
        "empty_should_preserve": sum(1 for row in enriched if not row["parsed"]["should_preserve"]),
        "probe_accuracy_by_choice": {
            choice: mean_bool(row["correct"] for row in rows)
            for choice, rows in _group_by(probes, "gold_answer").items()
        },
        "closed_set_failures": [
            {
                "item_id": row["item_id"],
                "condition": row["condition"],
                "probe_id": row["probe_id"],
                "gold_answer": row["gold_answer"],
                "raw_model_answer": row["raw_model_answer"],
            }
            for row in probes
            if not row["correct"]
        ][:30],
    }


def qualitative_examples(enriched: list[dict[str, Any]], contrast_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def first(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        for row in enriched:
            if predicate(row):
                return compact_example(row)
        return None

    contrast_failure = next((row for row in contrast_rows if row["directional_delta_f1"] < 0.5), None)
    success = lambda row: (
        row["parse_ok"]
        and row["direct_correction_correct"]
        and row["boundary_probe_correct"]
        and row["update_set_f1"] == 1.0
        and row["preserve_set_f1"] == 1.0
        and row["delayed_action_all_correct"]
        and not row["boundary_leak"]
        and not row["under_scope"]
    )
    return {
        "direct_correct_but_delayed_wrong": first(lambda row: row["direct_correction_correct"] and row["delayed_action_accuracy"] < 1.0),
        "exact_old_answer_reuse": first(lambda row: row.get("exact_old_answer_reuse")),
        "over_conservative_after_correction": first(lambda row: row.get("over_conservative_after_correction")),
        "under_scoped_after_correction": first(lambda row: row.get("under_scoped_after_correction")),
        "wrong_boundary_after_correction": first(lambda row: row.get("wrong_boundary_after_correction")),
        "over_scope_boundary_leak": first(lambda row: row["boundary_leak"]),
        "under_scope_missing_update": first(lambda row: row["under_scope"]),
        "wrong_scope_radius": first(lambda row: row["condition"] == "scope_ledger" and row["scope_radius_correct"] is False),
        "full_regeneration_boundary_damage": first(lambda row: row["condition"] == "full_regeneration" and row["boundary_leak"]),
        "scope_ledger_success": first(lambda row: row["condition"] == "scope_ledger" and success(row)),
        "scope_ledger_failure": first(lambda row: row["condition"] == "scope_ledger" and row["selective_scope_score"] < 1.0),
        "scope_contrast_failure": contrast_failure,
        "all_conditions_success": first(success),
        "parse_failure": first(lambda row: not row["parse_ok"]),
    }


def compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "base_context_id": row["base_context_id"],
        "domain": row["domain"],
        "scope_radius": row["scope_radius"],
        "correction_type": row["correction_type"],
        "condition": row["condition"],
        "gold_should_change": row["gold_should_change"],
        "pred_should_change": row["parsed"]["should_change"],
        "gold_should_preserve": row["gold_should_preserve"],
        "pred_should_preserve": row["parsed"]["should_preserve"],
        "direct_correction_correct": row["direct_correction_correct"],
        "delayed_action_accuracy": row["delayed_action_accuracy"],
        "boundary_leak": row["boundary_leak"],
        "under_scope": row["under_scope"],
        "exact_old_answer_reuse": row.get("exact_old_answer_reuse"),
        "stale_conflict_probe_correct": row.get("stale_conflict_probe_correct"),
        "raw_output": row["raw_output"],
    }


def _build_task(base_idx: int, variant_idx: int, global_idx: int, base_id: str, domain: str, radius: str) -> ScopeTask:
    if domain == "tool_agent_state":
        return _tool_task(base_idx, variant_idx, global_idx, base_id, radius)
    if domain == "rule_policy_scope":
        return _rule_task(base_idx, variant_idx, global_idx, base_id, radius)
    return _source_task(base_idx, variant_idx, global_idx, base_id, radius)


def _tool_task(base_idx: int, variant_idx: int, global_idx: int, base_id: str, radius: str) -> ScopeTask:
    files = ("alpha.csv", "beta.csv", "gamma.csv")
    token_by_file = {"alpha.csv": "T7", "beta.csv": "T8", "gamma.csv": "T7"}
    upload_minute_by_file = {"alpha.csv": 15 * 60 + 40, "beta.csv": 16 * 60 + 10, "gamma.csv": 16 * 60 + 25}
    all_upload_keys = [f"{short_name(file)}_upload" for file in files]
    all_archive_keys = [f"{short_name(file)}_archive_ready" for file in files]
    state_inventory = sorted(
        set(all_upload_keys)
        | set(all_archive_keys)
        | {
            "alpha_token",
            "beta_token",
            "gamma_token",
            "local_conversion_all",
            "archive_endpoint",
            "gamma_status_check",
            "notification_sent",
            "retry_queue",
            "local_files_deleted",
        }
    )
    context = (
        f"Batch {base_idx} processed alpha.csv, beta.csv, and gamma.csv. "
        "Each file was converted locally to parquet. alpha.csv used token T7 at 3:40 PM, "
        "beta.csv used token T8 at 4:10 PM, and gamma.csv used token T7 at 4:25 PM. "
        "The archive link should be sent only for files with ready remote archive copies. "
        "Local files may be deleted only after every needed remote archive copy is ready."
    )
    old_trace = (
        "Step 1: alpha.csv, beta.csv, and gamma.csv converted locally.\n"
        "Step 2: All three uploads succeeded.\n"
        "Step 3: All remote archive copies are ready.\n"
        "Step 4: Send the archive link for all files and delete all local files."
    )
    if radius == "local_entity":
        failed = ["gamma.csv"]
        feedback = "Correction: gamma.csv upload failed; alpha.csv and beta.csv upload status was not corrected."
        boundary = "only gamma.csv"
        retry_gold = "gamma"
        direct_key, direct_gold = "gamma_upload", "no"
    elif radius == "entity_class":
        failed = [file for file in files if token_by_file[file] == "T7"]
        feedback = "Correction: all files uploaded with token T7 failed; files with other tokens were not affected."
        boundary = "all and only token T7 files"
        retry_gold = "alpha,gamma"
        direct_key, direct_gold = "gamma_upload", "no"
    elif radius == "temporal_boundary":
        failed = [file for file in files if upload_minute_by_file[file] > 16 * 60]
        feedback = "Correction: uploads after 4 PM failed because the archive endpoint rotated credentials; earlier uploads were unaffected."
        boundary = "files uploaded after 4 PM"
        retry_gold = "beta,gamma"
        direct_key, direct_gold = "gamma_upload", "no"
    else:
        failed = []
        feedback = "Correction: only the archive-status readiness check for gamma.csv was stale. The actual gamma.csv upload succeeded, but gamma.csv archive readiness is unverified."
        boundary = "gamma archive readiness only; upload success is unchanged"
        retry_gold = "none"
        direct_key, direct_gold = "gamma_archive_ready", "no"
    changed_uploads = [f"{short_name(file)}_upload" for file in failed]
    changed_archives = [f"{short_name(file)}_archive_ready" for file in failed]
    should_change = changed_uploads + changed_archives
    if radius == "rule_boundary":
        should_change = ["gamma_archive_ready", "gamma_status_check"]
    preserve = [
        key
        for key in all_upload_keys + all_archive_keys + ["local_conversion_all", "notification_sent"]
        if key not in should_change
    ]
    boundary_file = "beta.csv" if radius == "entity_class" else "alpha.csv"
    boundary_key = f"{short_name(boundary_file)}_upload"
    boundary_gold = "no" if boundary_file in failed else "yes"
    delete_gold = "all" if retry_gold == "none" and radius != "rule_boundary" else ("unaffected_only" if should_change else "none")
    link_gold = "all" if retry_gold == "none" and radius != "rule_boundary" else "unaffected_only"
    return _task_from_parts(
        global_idx,
        base_id,
        "tool_agent_state",
        radius,
        context,
        old_trace,
        feedback,
        direct_key=direct_key,
        state_inventory=state_inventory,
        should_change=should_change,
        should_preserve=preserve,
        boundary=boundary,
        consequences=[f"retry:{retry_gold}", f"delete:{delete_gold}", f"send_link:{link_gold}"],
        probes=[
            probe("p1", "direct_target", direct_key, f"What is the corrected value for {direct_key}?", ["yes", "no"], direct_gold, "yes"),
            probe("p2", "boundary_preservation", boundary_key, f"Did {boundary_file} upload successfully?", ["yes", "no"], boundary_gold, "yes"),
            probe(
                "p3",
                "update_consequence",
                "retry_set",
                "Which files should be retried?",
                ["none", "alpha", "beta", "gamma", "alpha,beta", "alpha,gamma", "beta,gamma", "all"],
                retry_gold,
                "none",
            ),
            probe("p4", "delayed_action", "delete_local_files", "Which local files can be deleted now?", ["all", "unaffected_only", "none"], delete_gold, "all"),
            probe("p5", "stale_trace_conflict", "send_archive_link", "Given the prior trace claimed all remote copies were ready, which archive links may be sent now?", ["all", "unaffected_only", "none"], link_gold, "all", no_old_trace_question="Based on the corrected state, which archive links may be sent now?"),
        ],
    )


def _rule_task(base_idx: int, variant_idx: int, global_idx: int, base_id: str, radius: str) -> ScopeTask:
    del variant_idx
    state_inventory = [
        "case_a_review",
        "case_b_review",
        "case_c_review",
        "case_a_waiver",
        "case_b_waiver",
        "case_c_waiver",
        "case_a_finalized",
        "case_b_finalized",
        "case_c_finalized",
        "domestic_renewal_boundary",
        "submission_date_boundary",
        "risk_level_rule",
    ]
    context = (
        f"Policy queue {base_idx} contains Case A, Case B, and Case C. "
        "Case A is a domestic renewal submitted in 2021 with low risk. "
        "Case B is a domestic renewal submitted in 2023 with high risk. "
        "Case C is an international initial application submitted in 2023 with low risk. "
        "Cases requiring review cannot be finalized until that review is complete."
    )
    old_trace = (
        "Step 1: The waiver was applied broadly.\n"
        "Step 2: All three cases passed without review.\n"
        "Step 3: Finalize all cases."
    )
    if radius == "local_entity":
        feedback = "Correction: Case B's risk was recorded incorrectly; Case B is high-risk and needs director review."
        review = ["B"]
        boundary = "Case B only"
        p1_key, p1_question, p1_gold = "case_b_review", "Does Case B require review?", "yes"
    elif radius == "entity_class":
        feedback = "Correction: the expedited waiver applies only to domestic renewals; other case types still require normal review."
        review = ["C"]
        boundary = "domestic renewals only"
        p1_key, p1_question, p1_gold = "case_c_review", "Does Case C require review?", "yes"
    elif radius == "temporal_boundary":
        feedback = "Correction: the 2022 simplified rule applies only after Jan 1, 2022; pre-2022 cases use the legacy review rule."
        review = ["A"]
        boundary = "submission date after Jan 1, 2022"
        p1_key, p1_question, p1_gold = "case_a_review", "Does Case A require review?", "yes"
    else:
        feedback = "Correction: high-risk cases require director review even when the expedited waiver otherwise applies."
        review = ["B"]
        boundary = "high-risk exception overrides expedited waiver"
        p1_key, p1_question, p1_gold = "risk_level_rule", "Does the high-risk rule require director review?", "yes"
    review_choice = ",".join(review) if len(review) > 1 else review[0]
    should_change = [f"case_{case.lower()}_review" for case in review]
    if radius == "rule_boundary":
        should_change.append("risk_level_rule")
    should_preserve = [
        key
        for key in state_inventory
        if key not in should_change and (key.endswith("_review") or key.endswith("_waiver") or key in {"domestic_renewal_boundary", "submission_date_boundary"})
    ]
    return _task_from_parts(
        global_idx,
        base_id,
        "rule_policy_scope",
        radius,
        context,
        old_trace,
        feedback,
        direct_key=p1_key,
        state_inventory=state_inventory,
        should_change=should_change,
        should_preserve=should_preserve,
        boundary=boundary,
        consequences=[f"review:{review_choice}", "finalize:unaffected_only"],
        probes=[
            probe("p1", "direct_target", p1_key, p1_question, ["yes", "no"], p1_gold, "no"),
            probe(
                "p2",
                "boundary_preservation",
                "case_b_review" if p1_key == "case_a_review" else "case_a_review",
                "Does Case B require review?" if p1_key == "case_a_review" else "Does Case A require review?",
                ["yes", "no"],
                "no",
                "no",
            ),
            probe("p3", "update_consequence", "review_set", "Which cases require review?", ["none", "A", "B", "C", "A,B", "A,C", "B,C", "all"], review_choice, "none"),
            probe("p4", "delayed_action", "finalize_cases", "Which cases can be finalized now?", ["all", "unaffected_only", "none"], "unaffected_only", "all"),
            probe("p5", "stale_trace_conflict", "finalize_after_old_trace", "Given the prior trace claimed all cases passed, which cases should the system finalize now?", ["all", "unaffected_only", "none"], "unaffected_only", "all", no_old_trace_question="Based on the corrected state, which cases should the system finalize now?"),
        ],
    )


def _source_task(base_idx: int, variant_idx: int, global_idx: int, base_id: str, radius: str) -> ScopeTask:
    del variant_idx
    state_inventory = [
        "current_fee_source",
        "historical_procedure_source",
        "adult_population_support",
        "child_population_support",
        "association_claim_support",
        "causal_claim_support",
        "diagnosis_guideline_support",
        "dosage_guideline_support",
        "limited_citation_boundary",
        "newer_fee_source_needed",
        "child_specific_source_needed",
        "causal_evidence_needed",
    ]
    context = (
        f"Evidence packet {base_idx} has one source used for a policy answer. "
        "The source describes historical filing procedure, current fee amount, adult outcomes, child outcomes, "
        "and whether a treatment association is causal. The old answer cited the source for the full answer."
    )
    old_trace = (
        "Step 1: The source supports current fee, historical procedure, population, and causal recommendation.\n"
        "Step 2: Cite it for the full answer.\n"
        "Step 3: Make the recommendation without qualification."
    )
    if radius == "local_entity":
        feedback = "Correction: only the current fee table in the source is outdated; its historical procedure section was not corrected."
        p1_key, p1_question, p1_gold = "current_fee_source", "Can the source be used for current fees?", "no"
        historical_ok, request = "yes", "newer_fee_source"
        boundary = "current fee only"
        changes = ["current_fee_source", "newer_fee_source_needed"]
    elif radius == "entity_class":
        feedback = "Correction: this study applies to adults, not children; adult use is unaffected."
        p1_key, p1_question, p1_gold = "child_population_support", "Can the source be used for children?", "no"
        historical_ok, request = "yes", "child_specific_source"
        boundary = "adult population only"
        changes = ["child_population_support", "child_specific_source_needed"]
    elif radius == "temporal_boundary":
        feedback = "Correction: the guideline was superseded for medication dosage after Jan 1, 2023, but it remains valid for diagnosis criteria."
        p1_key, p1_question, p1_gold = "dosage_guideline_support", "Can the guideline be used for current medication dosage?", "no"
        historical_ok, request = "yes", "newer_dosage_source"
        boundary = "current dosage use after Jan 1, 2023; diagnosis criteria still valid"
        changes = ["dosage_guideline_support"]
    else:
        feedback = "Correction: the paper shows association, not causation; associative claims are still supported."
        p1_key, p1_question, p1_gold = "causal_claim_support", "Can the source support a causal claim?", "no"
        historical_ok, request = "yes", "causal_evidence"
        boundary = "association is supported, causation is not"
        changes = ["causal_claim_support", "causal_evidence_needed"]
    preserve = [
        key
        for key in [
            "historical_procedure_source",
            "adult_population_support",
            "association_claim_support",
            "diagnosis_guideline_support",
            "limited_citation_boundary",
        ]
        if key not in changes
    ]
    return _task_from_parts(
        global_idx,
        base_id,
        "source_status_rag_scope",
        radius,
        context,
        old_trace,
        feedback,
        direct_key=p1_key,
        state_inventory=state_inventory,
        should_change=changes,
        should_preserve=preserve,
        boundary=boundary,
        consequences=[f"request:{request}", "limited_citation"],
        probes=[
            probe("p1", "direct_target", p1_key, p1_question, ["yes", "no"], p1_gold, "yes"),
            probe("p2", "boundary_preservation", "historical_procedure_source", "Can the source be used for historical procedure?", ["yes", "no"], historical_ok, "yes"),
            probe("p3", "update_consequence", "requested_source", "What evidence should be requested for the unsupported use?", ["none", "newer_fee_source", "child_specific_source", "newer_dosage_source", "causal_evidence"], request, "none"),
            probe("p4", "delayed_action", "recommendation_use", "Which citation/use is valid now?", ["full_use", "limited_use", "no_use"], "limited_use", "full_use"),
            probe("p5", "stale_trace_conflict", "full_citation", "Given the prior trace claimed the source supports the full answer, how should it be cited now?", ["full_use", "limited_use", "no_use"], "limited_use", "full_use", no_old_trace_question="Based on the corrected source status, how should the source be cited now?"),
        ],
    )


def _task_from_parts(
    global_idx: int,
    base_id: str,
    domain: str,
    radius: str,
    context: str,
    old_trace: str,
    feedback: str,
    direct_key: str,
    state_inventory: list[str],
    should_change: list[str],
    should_preserve: list[str],
    boundary: str,
    consequences: list[str],
    probes: list[dict[str, Any]],
) -> ScopeTask:
    feedback_by_condition = {
        "standard_feedback": feedback,
        "full_regeneration": feedback,
        "scope_ledger": feedback,
    }
    return ScopeTask(
        item_id=f"scope-radius-{global_idx:04d}-{domain}-{radius}",
        base_context_id=base_id,
        domain=domain,
        scope_radius=radius,
        correction_type=correction_type_for(domain, radius),
        context=context,
        old_trace=old_trace,
        feedback_by_condition=feedback_by_condition,
        state_inventory=sorted(set(state_inventory)),
        gold_direct_target=direct_key,
        gold_scope_radius=radius,
        gold_should_change=sorted(set(should_change)),
        gold_should_preserve=sorted(set(should_preserve) - set(should_change)),
        gold_scope_boundary=boundary,
        gold_action_consequences=consequences,
        probes=probes,
        metadata={"target_key": direct_key, "experiment": EXPERIMENT_NAME, "correction_type": correction_type_for(domain, radius)},
    )


def correction_type_for(domain: str, radius: str) -> str:
    if domain == "tool_agent_state":
        return "archive_status" if radius == "rule_boundary" else "upload_state"
    if domain == "rule_policy_scope":
        if radius == "local_entity":
            return "risk_review"
        if radius == "temporal_boundary":
            return "policy_date"
        return "policy_exception"
    if radius == "local_entity":
        return "source_validity"
    if radius == "entity_class":
        return "population_validity"
    if radius == "temporal_boundary":
        return "dosage_validity"
    return "causal_validity"


def probe(
    probe_id: str,
    probe_type: str,
    state_key: str,
    question: str,
    choices: list[str],
    gold: str,
    old: str,
    aliases: dict[str, list[str]] | None = None,
    no_old_trace_question: str | None = None,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "probe_type": probe_type,
        "state_key": state_key,
        "question": question,
        "choices": choices,
        "gold_answer": gold,
        "old_trace_answer": old,
        "aliases": aliases or {},
        "no_old_trace_question": no_old_trace_question,
    }


def short_name(file: str) -> str:
    return file.split(".", 1)[0]


def normalize_scope_radius(value: Any) -> str:
    text = normalize_text(value)
    for radius in SCOPE_RADII:
        if radius in text:
            return radius
    aliases = {
        "entity": "local_entity",
        "single": "local_entity",
        "class": "entity_class",
        "property": "entity_class",
        "temporal": "temporal_boundary",
        "time": "temporal_boundary",
        "date": "temporal_boundary",
        "boundary": "rule_boundary",
        "rule": "rule_boundary",
    }
    for key, radius in aliases.items():
        if key in text:
            return radius
    return text


def normalize_state_set(value: Any) -> list[str]:
    items = _as_list(value)
    normalized = sorted({normalize_state_key(item) for item in items if normalize_state_key(item)})
    return normalized


def normalize_state_key(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("csv", "").replace("file", "")
    text = re.sub(r"\b(uploaded|upload success|upload_success|upload)\b", "upload", text)
    text = re.sub(r"\b(archive copy|archive ready|archive_ready|ready)\b", "archive_ready", text)
    text = text.strip("_")
    replacements = {
        "alpha upload": "alpha_upload",
        "beta upload": "beta_upload",
        "gamma upload": "gamma_upload",
        "alpha archive_ready": "alpha_archive_ready",
        "beta archive_ready": "beta_archive_ready",
        "gamma archive_ready": "gamma_archive_ready",
        "case a review": "case_a_review",
        "case b review": "case_b_review",
        "case c review": "case_c_review",
        "current fee": "current_fee_source",
        "historical procedure": "historical_procedure_source",
        "causal claim": "causal_claim_support",
    }
    return replacements.get(text, text.replace(" ", "_").replace("-", "_").replace("__", "_"))


def normalize_choice(value: Any) -> str:
    return normalize_text(value).replace(" ", "_")


def normalize_closed_answer(answer: str, choices: list[str]) -> str:
    normalized = normalize_choice(answer)
    normalized_choices = [normalize_choice(choice) for choice in choices]
    if normalized in normalized_choices:
        return normalized
    text = normalize_text(answer)
    for choice, normalized_choice in zip(choices, normalized_choices):
        if re.search(rf"\b{re.escape(choice.lower())}\b", text):
            return normalized_choice
    if "all token t7" in text or "token t7" in text:
        return "all_token_t7"
    if "after 4" in text:
        return "all_after_4pm"
    if "unaffected" in text and "unaffected_only" in normalized_choices:
        return "unaffected_only"
    if "limited" in text and "limited_use" in normalized_choices:
        return "limited_use"
    if ("full" in text or "all" in text) and "full_use" in normalized_choices:
        return "full_use"
    if ("all" in text or "every" in text) and "all" in normalized_choices:
        return "all"
    if ("none" in text or "no " in text) and "none" in normalized_choices:
        return "none"
    if "unaffected" in text and "file" in text:
        return "only_for_unaffected_files"
    if "unaffected" in text and ("local" in text or "delete" in text):
        return "yes_unaffected_only"
    if "unaffected" in text and "case" in text:
        return "only_unaffected_cases"
    if "limited" in text or "partial" in text or "qualified" in text:
        return "only_for_limited_scope" if "cite" in text or "source" in text else "only_with_qualification"
    if "newer" in text and "fee" in text:
        return "newer_fee_source"
    if "causal" in text:
        return "causal_evidence"
    if "adult" in text:
        return "adult_specific_source"
    return normalized


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        pieces = re.split(r"[,;\n]+", value)
        return [piece.strip() for piece in pieces if piece.strip()]
    return []


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_,:.\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prf(predicted: set[str], gold: set[str]) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = len(predicted & gold) / len(predicted) if predicted else 0.0
    recall = len(predicted & gold) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_bool(values: Any) -> float:
    vals = [bool(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def mean_optional_bool(values: Any) -> float | None:
    vals = [bool(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def mean_float(values: Any) -> float:
    vals = [float(value) for value in values if value is not None]
    return statistics.mean(vals) if vals else 0.0


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped
