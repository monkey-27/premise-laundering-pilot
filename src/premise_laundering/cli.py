from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_items, prepare_data
from .metrics import summarize_run
from .schema import MODEL_ID, SEED
from .validate import validate_data, validate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Premise laundering pilot utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data")
    prep.add_argument("--output-dir", default="data")
    prep.add_argument("--overwrite", action="store_true")

    validate_data_parser = subparsers.add_parser("validate-data")
    validate_data_parser.add_argument("--data-file", default="data/pilot_items.jsonl")

    validate_run_parser = subparsers.add_parser("validate-run")
    validate_run_parser.add_argument("--run-dir", required=True)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--run-dir", required=True)
    metrics.add_argument("--annotation-file")

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--data-file", default="data/pilot_items.jsonl")
    dry.add_argument("--output-dir", default="runs")
    dry.add_argument("--run-id", default="dry-run")
    dry.add_argument("--overwrite", action="store_true")

    controlled_prep = subparsers.add_parser("controlled-prepare-data")
    controlled_prep.add_argument("--output-dir", default="data/controlled")
    controlled_prep.add_argument("--n", type=int, default=99)
    controlled_prep.add_argument("--overwrite", action="store_true")

    controlled_validate_data = subparsers.add_parser("controlled-validate-data")
    controlled_validate_data.add_argument(
        "--data-file", default="data/controlled/controlled_items.jsonl"
    )

    controlled_validate_run = subparsers.add_parser("controlled-validate-run")
    controlled_validate_run.add_argument("--run-dir", required=True)

    controlled_metrics = subparsers.add_parser("controlled-metrics")
    controlled_metrics.add_argument("--run-dir", required=True)

    controlled_dry = subparsers.add_parser("controlled-dry-run")
    controlled_dry.add_argument("--data-file", default="data/controlled/controlled_items.jsonl")
    controlled_dry.add_argument("--output-dir", default="runs/controlled")
    controlled_dry.add_argument("--run-id", default="controlled-dry-run")
    controlled_dry.add_argument("--overwrite", action="store_true")

    prefix_prep = subparsers.add_parser("prefix-prepare-data")
    prefix_prep.add_argument("--output-dir", default="data/prefix")
    prefix_prep.add_argument("--n", type=int, default=300)
    prefix_prep.add_argument("--overwrite", action="store_true")

    prefix_validate_data = subparsers.add_parser("prefix-validate-data")
    prefix_validate_data.add_argument("--data-file", default="data/prefix/prefix_tasks.jsonl")

    prefix_validate_run = subparsers.add_parser("prefix-validate-run")
    prefix_validate_run.add_argument("--run-dir", required=True)

    prefix_metrics = subparsers.add_parser("prefix-metrics")
    prefix_metrics.add_argument("--run-dir", required=True)

    prefix_dry = subparsers.add_parser("prefix-dry-run")
    prefix_dry.add_argument("--data-file", default="data/prefix/prefix_tasks.jsonl")
    prefix_dry.add_argument("--output-dir", default="runs/prefix")
    prefix_dry.add_argument("--run-id", default="prefix-dry-run")
    prefix_dry.add_argument("--overwrite", action="store_true")

    diagnostic_prep = subparsers.add_parser("diagnostic-prepare-data")
    diagnostic_prep.add_argument("--output-dir", default="data/diagnostic")
    diagnostic_prep.add_argument("--n", type=int, default=240)
    diagnostic_prep.add_argument("--overwrite", action="store_true")

    diagnostic_validate_data = subparsers.add_parser("diagnostic-validate-data")
    diagnostic_validate_data.add_argument("--data-file", default="data/diagnostic/diagnostic_items.jsonl")

    diagnostic_validate_run = subparsers.add_parser("diagnostic-validate-run")
    diagnostic_validate_run.add_argument("--run-dir", required=True)

    diagnostic_metrics = subparsers.add_parser("diagnostic-metrics")
    diagnostic_metrics.add_argument("--run-dir", required=True)

    diagnostic_dry = subparsers.add_parser("diagnostic-dry-run")
    diagnostic_dry.add_argument("--data-file", default="data/diagnostic/diagnostic_items.jsonl")
    diagnostic_dry.add_argument("--output-dir", default="runs/diagnostic")
    diagnostic_dry.add_argument("--run-id", default="diagnostic-dry-run")
    diagnostic_dry.add_argument("--overwrite", action="store_true")

    selective_prep = subparsers.add_parser("selective-prepare-data")
    selective_prep.add_argument("--output-dir", default="data/selective")
    selective_prep.add_argument("--n", type=int, default=320)
    selective_prep.add_argument("--overwrite", action="store_true")

    selective_validate_data = subparsers.add_parser("selective-validate-data")
    selective_validate_data.add_argument("--data-file", default="data/selective/feedback_tasks.jsonl")

    selective_validate_run = subparsers.add_parser("selective-validate-run")
    selective_validate_run.add_argument("--run-dir", required=True)

    selective_metrics = subparsers.add_parser("selective-metrics")
    selective_metrics.add_argument("--run-dir", required=True)

    selective_dry = subparsers.add_parser("selective-dry-run")
    selective_dry.add_argument("--data-file", default="data/selective/feedback_tasks.jsonl")
    selective_dry.add_argument("--output-dir", default="runs/selective")
    selective_dry.add_argument("--run-id", default="selective-dry-run")
    selective_dry.add_argument("--overwrite", action="store_true")

    selective_corrected_prep = subparsers.add_parser("selective-corrected-prepare-data")
    selective_corrected_prep.add_argument("--output-dir", default="data/selective_corrected")
    selective_corrected_prep.add_argument("--n", type=int, default=96)
    selective_corrected_prep.add_argument("--overwrite", action="store_true")

    selective_corrected_validate_data = subparsers.add_parser("selective-corrected-validate-data")
    selective_corrected_validate_data.add_argument("--data-file", default="data/selective_corrected/feedback_tasks_corrected.jsonl")

    selective_corrected_dry = subparsers.add_parser("selective-corrected-dry-run")
    selective_corrected_dry.add_argument("--data-file", default="data/selective_corrected/feedback_tasks_corrected.jsonl")
    selective_corrected_dry.add_argument("--output-dir", default="runs/selective_corrected")
    selective_corrected_dry.add_argument("--run-id", default="selective-corrected-dry-run")
    selective_corrected_dry.add_argument("--overwrite", action="store_true")

    scope_radius_prep = subparsers.add_parser("scope-radius-prepare-data")
    scope_radius_prep.add_argument("--output-dir", default="data/scope_radius")
    scope_radius_prep.add_argument("--n-base-contexts", type=int, default=40)
    scope_radius_prep.add_argument("--overwrite", action="store_true")

    scope_radius_validate_data = subparsers.add_parser("scope-radius-validate-data")
    scope_radius_validate_data.add_argument("--data-file", default="data/scope_radius/scope_radius_tasks.jsonl")

    scope_radius_validate_run = subparsers.add_parser("scope-radius-validate-run")
    scope_radius_validate_run.add_argument("--run-dir", required=True)

    scope_radius_metrics = subparsers.add_parser("scope-radius-metrics")
    scope_radius_metrics.add_argument("--run-dir", required=True)

    scope_radius_dry = subparsers.add_parser("scope-radius-dry-run")
    scope_radius_dry.add_argument("--data-file", default="data/scope_radius/scope_radius_tasks.jsonl")
    scope_radius_dry.add_argument("--output-dir", default="runs/scope_radius")
    scope_radius_dry.add_argument("--run-id", default="scope-radius-dry-run")
    scope_radius_dry.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-data":
        path = prepare_data(args.output_dir, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_data(path)}, indent=2))
    elif args.command == "validate-data":
        print(json.dumps(validate_data(args.data_file), indent=2))
    elif args.command == "validate-run":
        print(json.dumps(validate_run(args.run_dir), indent=2))
    elif args.command == "metrics":
        print(json.dumps(summarize_run(args.run_dir, args.annotation_file), indent=2))
    elif args.command == "dry-run":
        from .run import run_inference

        items = load_items(Path(args.data_file))

        def fake_generator(prompts: list[str]) -> list[str]:
            del prompts
            return [
                json.dumps(
                    {
                        "reasoning_steps": ["This is a parser and plumbing dry run."],
                        "tagged_claims": [
                            {
                                "claim": "This placeholder output is not model evidence.",
                                "origin_tag": "[ASSUMPTION]",
                                "used_for_final_answer": False,
                            }
                        ],
                        "final_answer": item.gold_label,
                        "confidence": 0.01,
                    }
                )
                for item in items
                for _ in range(5)
            ][: len(items) * 5]

        run_dir = run_inference(
            items=items,
            generator=fake_generator,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL_ID}-dry-run",
            overwrite=args.overwrite,
            seed=SEED,
            generation_config={"engine": "fake_generator"},
            batch_size=len(items) * 5,
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_run(run_dir)}, indent=2))
    elif args.command == "controlled-prepare-data":
        from .controlled import prepare_controlled_data, validate_controlled_data

        path = prepare_controlled_data(args.output_dir, n=args.n, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_controlled_data(path)}, indent=2))
    elif args.command == "controlled-validate-data":
        from .controlled import validate_controlled_data

        print(json.dumps(validate_controlled_data(args.data_file), indent=2))
    elif args.command == "controlled-validate-run":
        from .controlled import validate_controlled_run

        print(json.dumps(validate_controlled_run(args.run_dir), indent=2))
    elif args.command == "controlled-metrics":
        from .controlled import summarize_controlled_run

        print(json.dumps(summarize_controlled_run(args.run_dir), indent=2))
    elif args.command == "controlled-dry-run":
        from .controlled import load_controlled_items, run_controlled_inference, validate_controlled_run

        items = load_controlled_items(Path(args.data_file))

        def fake_generator(prompts: list[str]) -> list[str]:
            outputs = []
            for prompt in prompts:
                if "Prior reasoning artifact" in prompt:
                    label = "CONTRADICT" if "CONTRADICT" in prompt else "SUPPORT"
                else:
                    label = "SUPPORT"
                outputs.append(
                    json.dumps(
                        {
                            "final_answer": label,
                            "evidence_used": "fixture evidence",
                            "artifact_used": "Prior reasoning artifact" in prompt,
                            "rationale": "controlled dry-run output",
                        }
                    )
                )
            return outputs

        run_dir = run_controlled_inference(
            items=items,
            generator=fake_generator,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            overwrite=args.overwrite,
            model=f"{MODEL_ID}-dry-run",
            backend="fake_generator",
            generation_config={"engine": "fake_generator"},
            batch_size=32,
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_controlled_run(run_dir)}, indent=2))
    elif args.command == "prefix-prepare-data":
        from .self_prefix import prepare_prefix_data, validate_prefix_data

        path = prepare_prefix_data(args.output_dir, n=args.n, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_prefix_data(path, expected_min=max(1, int(args.n * 0.8)))}, indent=2))
    elif args.command == "prefix-validate-data":
        from .self_prefix import validate_prefix_data

        print(json.dumps(validate_prefix_data(args.data_file), indent=2))
    elif args.command == "prefix-validate-run":
        from .self_prefix import validate_prefix_run

        print(json.dumps(validate_prefix_run(args.run_dir), indent=2))
    elif args.command == "prefix-metrics":
        from .self_prefix import summarize_prefix_run

        print(json.dumps(summarize_prefix_run(args.run_dir), indent=2))
    elif args.command == "prefix-dry-run":
        from .self_prefix import (
            continuation_prompt,
            load_prefix_tasks,
            prefix_prompt,
            run_prefix_intervention,
            source_prompt,
            validate_prefix_run,
        )

        tasks = load_prefix_tasks(Path(args.data_file))[:12]
        source_by_prompt = {source_prompt(task): task for task in tasks}
        prefix_by_prompt = {prefix_prompt(task): task for task in tasks}
        continuation_by_prompt = {continuation_prompt(task): task for task in tasks}

        class FakePrefixGenerator:
            def generate_from_user(self, prompts: list[str], max_new_tokens: int | None = None) -> list[str]:
                del max_new_tokens
                outputs = []
                for prompt in prompts:
                    if prompt in prefix_by_prompt:
                        task = prefix_by_prompt[prompt]
                        wrong = next(value for value in task.candidate_values if value != task.gold_answer)
                        outputs.append(
                            f"Observation 1: {task.observation_type}|{task.answer_key}|{wrong}\n"
                            "Observation 2: scope|question_scope|source report\n"
                        )
                    else:
                        task = source_by_prompt[prompt]
                        outputs.append(f"Final answer: {task.gold_answer}")
                return outputs

            def continue_from_prefix(
                self,
                prompts_and_prefixes: list[tuple[str, str]],
                max_new_tokens: int | None = None,
            ) -> list[str]:
                del max_new_tokens
                outputs = []
                for prompt, prefix in prompts_and_prefixes:
                    task = continuation_by_prompt[prompt]
                    if "unclear" in prefix:
                        outputs.append(f"Final answer: {task.gold_answer}")
                    else:
                        tail = prefix.strip().split("|")[-1].splitlines()[0].strip()
                        outputs.append(f"Final answer: {tail}")
                return outputs

        run_dir = run_prefix_intervention(
            tasks=tasks,
            generator=FakePrefixGenerator(),
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL_ID}-dry-run",
            overwrite=args.overwrite,
            batch_size=4,
            generation_config={"engine": "fake_prefix_generator"},
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_prefix_run(run_dir)}, indent=2))
    elif args.command == "diagnostic-prepare-data":
        from .diagnostic_compression import prepare_diagnostic_data, validate_diagnostic_data

        path = prepare_diagnostic_data(args.output_dir, n=args.n, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_diagnostic_data(path, expected_min=max(1, int(args.n * 0.8)))}, indent=2))
    elif args.command == "diagnostic-validate-data":
        from .diagnostic_compression import validate_diagnostic_data

        print(json.dumps(validate_diagnostic_data(args.data_file), indent=2))
    elif args.command == "diagnostic-validate-run":
        from .diagnostic_compression import validate_diagnostic_run

        print(json.dumps(validate_diagnostic_run(args.run_dir), indent=2))
    elif args.command == "diagnostic-metrics":
        from .diagnostic_compression import summarize_diagnostic_run

        print(json.dumps(summarize_diagnostic_run(args.run_dir), indent=2))
    elif args.command == "diagnostic-dry-run":
        from .diagnostic_compression import (
            load_trace_variants,
            prepare_diagnostic_data,
            run_diagnostic_verification,
            validate_diagnostic_run,
        )

        data_file = Path(args.data_file)
        if not data_file.exists():
            data_file = prepare_diagnostic_data(data_file.parent, n=24, overwrite=True)
        variants = load_trace_variants(data_file.with_name("diagnostic_trace_variants.jsonl"))

        def fake_verifier(prompts: list[str]) -> list[str]:
            outputs = []
            for prompt in prompts:
                lowered = prompt.lower()
                if "potential issue type:" in lowered:
                    error_type = lowered.split("potential issue type:", 1)[1].split(".", 1)[0].strip()
                    step = lowered.split("first risky step is step ", 1)[1].split(".", 1)[0].strip() if "first risky step is step " in lowered else "unknown"
                    outputs.append(json.dumps({"is_answer_correct": "no", "first_error_step": step, "error_type": error_type, "confidence": 0.7}))
                elif "radio note says" in lowered:
                    outputs.append(json.dumps({"is_answer_correct": "no", "first_error_step": "2", "error_type": "evidence_misread", "confidence": 0.7}))
                elif "strongest visible label" in lowered:
                    outputs.append(json.dumps({"is_answer_correct": "no", "first_error_step": "2", "error_type": "constraint_violation", "confidence": 0.7}))
                elif "yields [0, 1, 2, 3" in lowered or "yields [0, 1, 2, 3, 4" in lowered or "yields [0, 1, 2, 3, 4, 5" in lowered:
                    outputs.append(json.dumps({"is_answer_correct": "no", "first_error_step": "2", "error_type": "off_by_one_error", "confidence": 0.6}))
                else:
                    outputs.append(json.dumps({"is_answer_correct": "yes", "first_error_step": "none", "error_type": "none", "confidence": 0.8}))
            return outputs

        run_dir = run_diagnostic_verification(
            variants=variants,
            generator=fake_verifier,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL_ID}-dry-run",
            backend="fake_generator",
            overwrite=args.overwrite,
            batch_size=16,
            generation_config={"engine": "fake_diagnostic_verifier"},
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_diagnostic_run(run_dir)}, indent=2))
    elif args.command == "selective-prepare-data":
        from .selective_correction import prepare_selective_data, validate_selective_data

        path = prepare_selective_data(args.output_dir, n=args.n, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_selective_data(path, expected=args.n)}, indent=2))
    elif args.command == "selective-validate-data":
        from .selective_correction import validate_selective_data

        print(json.dumps(validate_selective_data(args.data_file), indent=2))
    elif args.command == "selective-validate-run":
        from .selective_correction import validate_selective_run

        print(json.dumps(validate_selective_run(args.run_dir), indent=2))
    elif args.command == "selective-metrics":
        from .selective_correction import summarize_selective_run

        print(json.dumps(summarize_selective_run(args.run_dir), indent=2))
    elif args.command == "selective-dry-run":
        from .selective_correction import (
            PROMPT_CONDITIONS,
            build_revision_prompt,
            load_selective_tasks,
            prepare_selective_data,
            run_selective_revision,
            validate_selective_run,
        )

        data_file = Path(args.data_file)
        if not data_file.exists():
            data_file = prepare_selective_data(data_file.parent, n=320, overwrite=True)
        tasks = load_selective_tasks(data_file)
        prompt_map = {
            build_revision_prompt(task, condition): (task, condition)
            for task in tasks
            for condition in PROMPT_CONDITIONS
        }

        def fake_revision_generator(prompts: list[str]) -> list[str]:
            outputs = []
            for prompt in prompts:
                task, condition = prompt_map[prompt]
                answers = {probe["probe_id"]: probe["gold_answer"] for probe in task.probes}
                if condition == "standard_feedback" and task.correction_type == "local_to_final":
                    for probe in task.probes:
                        if probe["state_key"] in task.should_update:
                            answers[probe["probe_id"]] = str(task.gold_state_before.get(probe["state_key"], "unknown"))
                if condition == "full_regeneration" and task.correction_type == "scope_limited":
                    for probe in task.probes:
                        if probe["state_key"] in task.should_preserve:
                            answers[probe["probe_id"]] = "yes" if probe["gold_answer"] == "no" else "no"
                payload = {
                    "revised_reasoning": "Dry-run revised state using the localized feedback.",
                    "final_answer": answers.get("p2") or answers.get("p5") or "updated",
                    "probe_answers": answers,
                }
                if condition == "correction_scope_map":
                    payload["feedback_target"] = task.metadata.get("target_key", "")
                    payload["must_change"] = task.should_update
                    payload["must_stay_same"] = task.should_preserve
                outputs.append(json.dumps(payload))
            return outputs

        run_dir = run_selective_revision(
            tasks=tasks,
            generator=fake_revision_generator,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL_ID}-dry-run",
            backend="fake_generator",
            overwrite=args.overwrite,
            batch_size=64,
            generation_config={"engine": "fake_selective_revision_generator"},
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_selective_run(run_dir)}, indent=2))
    elif args.command == "selective-corrected-prepare-data":
        from .selective_correction import prepare_selective_corrected_data, validate_selective_corrected_data

        path = prepare_selective_corrected_data(args.output_dir, n=args.n, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_selective_corrected_data(path, expected=args.n)}, indent=2))
    elif args.command == "selective-corrected-validate-data":
        from .selective_correction import validate_selective_corrected_data

        print(json.dumps(validate_selective_corrected_data(args.data_file), indent=2))
    elif args.command == "selective-corrected-dry-run":
        from .selective_correction import (
            CORRECTED_PROMPT_CONDITIONS,
            build_revision_prompt,
            load_selective_tasks,
            prepare_selective_corrected_data,
            run_selective_revision,
            validate_selective_run,
        )

        data_file = Path(args.data_file)
        if not data_file.exists():
            data_file = prepare_selective_corrected_data(data_file.parent, n=96, overwrite=True)
        tasks = load_selective_tasks(data_file)
        prompt_map = {
            build_revision_prompt(task, condition): (task, condition)
            for task in tasks
            for condition in CORRECTED_PROMPT_CONDITIONS
        }

        def fake_corrected_generator(prompts: list[str]) -> list[str]:
            outputs = []
            for prompt in prompts:
                task, condition = prompt_map[prompt]
                answers = {probe["probe_id"]: probe["gold_answer"] for probe in task.probes}
                if condition == "standard_feedback" and task.correction_type == "local_to_final":
                    for probe in task.probes:
                        if probe["state_key"] in task.should_update:
                            answers[probe["probe_id"]] = str(task.gold_state_before.get(probe["state_key"], "unknown"))
                if condition == "full_regeneration" and task.correction_type == "scope_limited":
                    for probe in task.probes:
                        if probe["state_key"] in task.should_preserve:
                            answers[probe["probe_id"]] = "yes" if probe["gold_answer"] == "no" else "no"
                if task.correction_type == "rule_level" and condition == "correction_scope_map":
                    for probe in task.probes:
                        if probe["gold_answer"] == "use pre-2022 rule":
                            answers[probe["probe_id"]] = "legacy validation rule"
                        elif probe["gold_answer"] == "apply 2022 rule":
                            answers[probe["probe_id"]] = "checksum validation rule"
                        elif probe["gold_answer"] == "use standard review":
                            answers[probe["probe_id"]] = "normal review"
                        elif probe["gold_answer"] == "director approval":
                            answers[probe["probe_id"]] = "director-level approval"
                if task.domain == "tool_agent_state" and task.correction_type == "local_to_final":
                    for probe in task.probes:
                        if probe["gold_answer"] == "refresh token and retry upload":
                            answers[probe["probe_id"]] = "renew token and retry"
                payload = {
                    "revised_reasoning": "Dry-run corrected selective revision.",
                    "final_answer": answers.get("p2") or answers.get("p5") or "updated",
                    "probe_answers": answers,
                }
                if condition == "correction_scope_map":
                    payload["feedback_target"] = task.metadata.get("target_key", "")
                    payload["must_change"] = task.should_update
                    payload["must_stay_same"] = task.should_preserve
                outputs.append(json.dumps(payload))
            return outputs

        run_dir = run_selective_revision(
            tasks=tasks,
            generator=fake_corrected_generator,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL_ID}-dry-run",
            backend="fake_generator",
            overwrite=args.overwrite,
            batch_size=64,
            conditions=CORRECTED_PROMPT_CONDITIONS,
            generation_config={"engine": "fake_selective_corrected_generator"},
            tasks_filename="feedback_tasks_corrected.jsonl",
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_selective_run(run_dir)}, indent=2))
    elif args.command == "scope-radius-prepare-data":
        from .scope_radius import prepare_scope_radius_data, validate_scope_radius_data

        path = prepare_scope_radius_data(args.output_dir, n_base_contexts=args.n_base_contexts, overwrite=args.overwrite)
        print(json.dumps({"data_file": str(path), **validate_scope_radius_data(path)}, indent=2))
    elif args.command == "scope-radius-validate-data":
        from .scope_radius import validate_scope_radius_data

        print(json.dumps(validate_scope_radius_data(args.data_file), indent=2))
    elif args.command == "scope-radius-validate-run":
        from .scope_radius import validate_scope_radius_run

        print(json.dumps(validate_scope_radius_run(args.run_dir), indent=2))
    elif args.command == "scope-radius-metrics":
        from .scope_radius import summarize_scope_radius_run

        print(json.dumps(summarize_scope_radius_run(args.run_dir), indent=2))
    elif args.command == "scope-radius-dry-run":
        from .scope_radius import (
            MODEL,
            SCOPE_CONDITIONS,
            build_scope_prompt,
            load_scope_tasks,
            prepare_scope_radius_data,
            run_scope_radius_inference,
            validate_scope_radius_run,
        )

        data_file = Path(args.data_file)
        if not data_file.exists():
            data_file = prepare_scope_radius_data(data_file.parent, n_base_contexts=40, overwrite=True)
        tasks = load_scope_tasks(data_file)
        prompt_map = {
            build_scope_prompt(task, condition): (task, condition)
            for task in tasks
            for condition in SCOPE_CONDITIONS
        }

        def fake_scope_generator(prompts: list[str]) -> list[str]:
            outputs = []
            for prompt in prompts:
                task, condition = prompt_map[prompt]
                answers = {probe["probe_id"]: probe["gold_answer"] for probe in task.probes}
                should_change = list(task.gold_should_change)
                should_preserve = list(task.gold_should_preserve)
                scope_radius = task.gold_scope_radius
                if condition == "standard_feedback" and task.scope_radius in {"entity_class", "rule_boundary"}:
                    should_change = should_change[:1]
                    for probe in task.probes:
                        if probe["probe_id"] in {"p3", "p5"}:
                            answers[probe["probe_id"]] = probe["old_trace_answer"]
                if condition == "full_regeneration":
                    should_preserve = []
                    if task.scope_radius == "local_entity":
                        for probe in task.probes:
                            if probe["probe_type"] == "boundary_preservation":
                                answers[probe["probe_id"]] = "no" if probe["gold_answer"] == "yes" else "yes"
                if condition == "scope_ledger" and task.scope_radius == "source_status":
                    scope_radius = "rule_boundary"
                payload = {
                    "direct_target": task.gold_direct_target,
                    "should_change": should_change,
                    "should_preserve": should_preserve,
                    "immediate_answers": {"p1": answers["p1"], "p2": answers["p2"]},
                    "delayed_action_answers": {"p3": answers["p3"], "p4": answers["p4"], "p5": answers["p5"]},
                }
                if condition == "scope_ledger":
                    payload.update(
                        {
                            "correction_scope_radius": scope_radius,
                            "scope_boundary": task.gold_scope_boundary,
                            "action_consequences": task.gold_action_consequences,
                        }
                    )
                outputs.append(json.dumps(payload))
            return outputs

        run_dir = run_scope_radius_inference(
            tasks=tasks,
            generator=fake_scope_generator,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            model=f"{MODEL}-dry-run",
            backend="fake_generator",
            overwrite=args.overwrite,
            batch_size=64,
            generation_config={"engine": "fake_scope_radius_generator"},
        )
        print(json.dumps({"run_dir": str(run_dir), **validate_scope_radius_run(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
