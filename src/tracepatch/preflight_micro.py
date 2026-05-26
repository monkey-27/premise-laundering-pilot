from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .micro import (
    DOMAINS,
    MODELS,
    SYSTEMS,
    build_initial_prompt,
    build_localize_prompt,
    build_repair_prompt,
    check_output,
    hard_fail_audit,
    parse_localization,
    parse_model_output,
    prepare_micro_data,
    read_jsonl,
    run_data_eval_audit,
    write_csv,
    write_json,
)


def main() -> None:
    data_path = prepare_micro_data("data/tracepatch", overwrite=True)
    examples = read_jsonl(data_path)
    errors: list[str] = []

    if len(examples) != 60:
        errors.append(f"expected 60 examples, found {len(examples)}")
    counts = Counter(ex["domain"] for ex in examples)
    if counts != Counter({"code": 20, "math": 20, "rag": 20}):
        errors.append(f"expected 20/domain, found {dict(counts)}")

    mock = '{"trace":[{"step_id":1,"text":"read problem"},{"step_id":2,"text":"solve"}],"final_answer":"42","code":"def f(x):\\n    return x"}'
    for ex in examples:
        parsed = parse_model_output(mock, ex["domain"])
        if not parsed["parse_ok"]:
            errors.append(f"parser failed mock for {ex['example_id']}")
        gold_parsed = {
            "parse_ok": True,
            "trace": [{"step_id": 1, "text": "gold"}],
            "final_answer": ex["gold_answer"],
            "code": ex["checker_spec"].get("canonical_code", ""),
            "json": {},
        }
        if not check_output(ex, gold_parsed)["passed"]:
            errors.append(f"gold checker failed for {ex['example_id']}")
        initial_prompt = build_initial_prompt(ex)
        if ex["domain"] != "rag" and str(ex["gold_answer"]).lower() in initial_prompt.lower():
            errors.append(f"gold answer leaked into initial prompt for {ex['example_id']}")
        initial = parse_model_output(mock, ex["domain"])
        feedback = "Expected answer differs from output."
        for system in SYSTEMS:
            if system == "tracepatch":
                loc_raw = '{"faulty_span":{"start_step_id":2,"end_step_id":2},"correct_prefix_step_ids":[1],"localization_reason":"mock"}'
                loc = parse_localization(loc_raw, initial, {"feedback": feedback})
                prompt = build_repair_prompt("tracepatch", ex, initial, feedback, loc)
            else:
                prompt = build_repair_prompt(system, ex, initial, feedback)
            if not prompt:
                errors.append(f"empty prompt for {system}/{ex['example_id']}")
        if not build_localize_prompt(ex, initial, feedback):
            errors.append(f"empty localize prompt for {ex['example_id']}")

    audit_rows, audit_summary, manual_subset = run_data_eval_audit(examples)
    out = Path("outputs/tracepatch_micro_v1")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "data_eval_audit.csv", audit_rows, list(audit_rows[0]))
    write_json(out / "data_eval_audit_summary.json", audit_summary)
    write_csv(out / "manual_data_eval_audit_sample.csv", manual_subset, list(manual_subset[0]))

    errors.extend(hard_fail_audit(audit_summary))
    if set(MODELS) != {"Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"}:
        errors.append("model names do not match requested micropilot")

    if errors:
        print(json.dumps({"status": "FAILED", "errors": errors, "audit_summary": audit_summary}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "PASSED", "data_file": str(data_path), "audit_summary": audit_summary}, indent=2))
    print("TRACEPATCH MICRO PREFLIGHT PASSED")


if __name__ == "__main__":
    main()

