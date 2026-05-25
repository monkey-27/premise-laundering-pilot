from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Callable, Iterable

from .annotation import build_annotation_rows, write_annotation_file
from .io import append_jsonl, ensure_dir, write_json
from .parsing import parse_output
from .prompts import build_prompt
from .schema import CONDITIONS, MODEL_ID, SEED, GenerationRecord, PilotItem

Generator = Callable[[list[str]], list[str]]


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def create_run_dir(base_dir: str | Path, run_id: str, overwrite: bool = False) -> Path:
    run_dir = Path(base_dir) / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new --run-id or pass --overwrite."
        )
    ensure_dir(run_dir)
    ensure_dir(run_dir / "raw_outputs")
    return run_dir


def run_inference(
    items: list[PilotItem],
    generator: Generator,
    output_base_dir: str | Path,
    run_id: str | None = None,
    model: str = MODEL_ID,
    conditions: Iterable[str] = CONDITIONS,
    overwrite: bool = False,
    seed: int = SEED,
    batch_size: int = 8,
    generation_config: dict | None = None,
) -> Path:
    run_id = run_id or make_run_id()
    conditions = tuple(conditions)
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")

    run_dir = create_run_dir(output_base_dir, run_id, overwrite=overwrite)
    generations_path = run_dir / "generations.jsonl"

    prompts: list[tuple[PilotItem, str, str]] = []
    for item in items:
        for condition in conditions:
            prompts.append((item, condition, build_prompt(item, condition)))

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "model": model,
            "seed": seed,
            "conditions": list(conditions),
            "num_items": len(items),
            "num_generations_expected": len(prompts),
            "generation_config": generation_config or {},
            "created_at_unix": time.time(),
            "python": platform.python_version(),
        },
    )

    records: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        print(
            f"Generating batch {start // batch_size + 1}/"
            f"{(len(prompts) + batch_size - 1) // batch_size} "
            f"({start + 1}-{min(start + len(batch), len(prompts))}/{len(prompts)})",
            flush=True,
        )
        raw_outputs = generator([prompt for _, _, prompt in batch])
        if len(raw_outputs) != len(batch):
            raise RuntimeError(f"Generator returned {len(raw_outputs)} outputs for {len(batch)} prompts")
        batch_records = [
            _build_generation_record(
                run_id=run_id,
                item=item,
                condition=condition,
                prompt=prompt,
                raw_output=raw,
                model=model,
                seed=seed,
                generation_config=generation_config or {},
            ).to_record()
            for (item, condition, prompt), raw in zip(batch, raw_outputs, strict=True)
        ]
        append_jsonl(generations_path, batch_records)
        records.extend(batch_records)

    annotation_rows = build_annotation_rows(records)
    write_annotation_file(run_dir / "manual_annotation.csv", annotation_rows)
    write_json(run_dir / "manual_annotation_schema.json", manual_annotation_schema())
    return run_dir


def _build_generation_record(
    run_id: str,
    item: PilotItem,
    condition: str,
    prompt: str,
    raw_output: str,
    model: str,
    seed: int,
    generation_config: dict,
) -> GenerationRecord:
    parsed = parse_output(raw_output)
    return GenerationRecord(
        run_id=run_id,
        item_id=item.item_id,
        dataset=item.dataset,
        condition=condition,
        model=model,
        prompt=prompt,
        raw_output=raw_output,
        parsed_final_answer=parsed["parsed_final_answer"],
        parsed_reasoning_steps=parsed["parsed_reasoning_steps"],
        parsed_intermediate_claims=parsed["parsed_intermediate_claims"],
        declared_origin_tags=parsed["declared_origin_tags"],
        gold_label=item.gold_label,
        context=item.context,
        claim_or_question=item.claim_or_question,
        metadata={
            **item.metadata,
            "seed": seed,
            "generation_config": generation_config,
            "parser_json": parsed["json"],
        },
    )


def manual_annotation_schema() -> dict[str, str]:
    return {
        "claim_supported_by_input": "Human judgment: yes/no/partial/unclear.",
        "claim_later_used_as_premise": "Human judgment: yes/no/unclear.",
        "origin_misattributed": "Human judgment: yes/no/unclear.",
        "final_answer_depends_on_claim": "Human judgment: yes/no/unclear.",
        "laundering_candidate": (
            "Human judgment: yes when an unsupported or partially supported claim is later "
            "used as a premise and the final answer depends on it or likely depends on it."
        ),
        "notes": "Free text reviewer notes.",
    }
