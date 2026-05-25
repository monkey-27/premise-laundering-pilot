from __future__ import annotations

import time
from pathlib import Path

from .annotation import build_annotation_rows, write_annotation_file
from .io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .run import manual_annotation_schema
from .schema import CONDITIONS, MODEL_ID, SEED
from .validate import validate_run


def merge_run_dirs(
    runs_base_dir: str | Path,
    source_run_ids: list[str],
    merged_run_id: str,
    overwrite: bool = False,
) -> Path:
    base = Path(runs_base_dir)
    target = base / merged_run_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"Merged run already exists: {target}")
    ensure_dir(target)

    records: list[dict] = []
    source_manifests: list[dict] = []
    for run_id in source_run_ids:
        run_dir = base / run_id
        source_manifests.append(read_json(run_dir / "run_manifest.json"))
        records.extend(read_jsonl(run_dir / "generations.jsonl"))

    records.sort(key=lambda row: (row["item_id"], row["condition"]))
    write_jsonl(target / "generations.jsonl", records)
    write_annotation_file(target / "manual_annotation.csv", build_annotation_rows(records))
    write_json(target / "manual_annotation_schema.json", manual_annotation_schema())
    write_json(
        target / "run_manifest.json",
        {
            "run_id": merged_run_id,
            "model": MODEL_ID,
            "seed": SEED,
            "conditions": list(CONDITIONS),
            "num_items": len({record["item_id"] for record in records}),
            "num_generations_expected": 200,
            "created_at_unix": time.time(),
            "merged_from": source_run_ids,
            "source_manifests": source_manifests,
        },
    )
    validate_run(target)
    return target
