from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP_NAME = "premise-laundering-pilot"
CACHE_DIR = "/cache"
DATA_DIR = "/cache/data"
RUNS_DIR = "/outputs/runs"
CONTROLLED_DATA_DIR = "/cache/controlled_data"
CONTROLLED_RUNS_DIR = "/outputs/controlled_runs"
PREFIX_DATA_DIR = "/cache/prefix_data"
PREFIX_RUNS_DIR = "/outputs/prefix_runs"
DIAGNOSTIC_DATA_DIR = "/cache/diagnostic_data"
DIAGNOSTIC_RUNS_DIR = "/outputs/diagnostic_runs"
SELECTIVE_DATA_DIR = "/cache/selective_data"
SELECTIVE_RUNS_DIR = "/outputs/selective_runs"
SELECTIVE_CORRECTED_DATA_DIR = "/cache/selective_corrected_data"
SELECTIVE_CORRECTED_RUNS_DIR = "/outputs/selective_corrected_runs"
SCOPE_RADIUS_DATA_DIR = "/cache/scope_radius_data"
SCOPE_RADIUS_RUNS_DIR = "/outputs/scope_radius_runs"
TRACEPATCH_DATA_DIR = "/cache/tracepatch_data"
TRACEPATCH_RUNS_DIR = "/outputs/tracepatch_runs"

LOCAL_SRC = Path(__file__).parent / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

cache_volume = modal.Volume.from_name("premise-laundering-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("premise-laundering-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.19.0,<3.0.0",
        "huggingface-hub>=0.23.0",
        "pandas>=2.2.0",
        "pyarrow>=15.0.0",
        "tqdm>=4.66.0",
        "torch>=2.2.0",
        "transformers>=4.41.0,<5.0.0",
        "vllm>=0.5.0",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_DATASETS_CACHE": f"{CACHE_DIR}/huggingface/datasets",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/huggingface/transformers",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    .add_local_python_source("premise_laundering")
    .add_local_python_source("tracepatch")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60,
)
def prepare_data(overwrite: bool = False) -> str:
    from premise_laundering.data import prepare_data as prepare
    from premise_laundering.validate import validate_data

    path = prepare(DATA_DIR, overwrite=overwrite)
    result = validate_data(path)
    cache_volume.commit()
    return f"Prepared {result['examples']} examples at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60,
)
def prepare_controlled_data(overwrite: bool = False, n: int = 99) -> str:
    from premise_laundering.controlled import (
        prepare_controlled_data as prepare,
        validate_controlled_data,
    )

    path = prepare(CONTROLLED_DATA_DIR, n=n, overwrite=overwrite)
    result = validate_controlled_data(path)
    cache_volume.commit()
    return f"Prepared {result['examples']} controlled SciFact examples at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare_prefix_data(overwrite: bool = False, n: int = 300) -> str:
    from premise_laundering.self_prefix import prepare_prefix_data as prepare
    from premise_laundering.self_prefix import validate_prefix_data

    path = prepare(PREFIX_DATA_DIR, n=n, overwrite=overwrite)
    result = validate_prefix_data(path, expected_min=max(1, int(n * 0.8)))
    cache_volume.commit()
    return f"Prepared {result['tasks']} self-prefix synthetic tasks at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare_diagnostic_data(overwrite: bool = False, n: int = 240) -> str:
    from premise_laundering.diagnostic_compression import prepare_diagnostic_data as prepare
    from premise_laundering.diagnostic_compression import validate_diagnostic_data

    path = prepare(DIAGNOSTIC_DATA_DIR, n=n, overwrite=overwrite)
    result = validate_diagnostic_data(path, expected_min=max(1, int(n * 0.8)))
    cache_volume.commit()
    return f"Prepared {result['items']} diagnostic-compression items at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare_selective_data(overwrite: bool = False, n: int = 320) -> str:
    from premise_laundering.selective_correction import prepare_selective_data as prepare
    from premise_laundering.selective_correction import validate_selective_data

    path = prepare(SELECTIVE_DATA_DIR, n=n, overwrite=overwrite)
    result = validate_selective_data(path, expected=n)
    cache_volume.commit()
    return f"Prepared {result['examples']} selective-correction tasks at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare_selective_corrected_data(overwrite: bool = False, n: int = 96) -> str:
    from premise_laundering.selective_correction import prepare_selective_corrected_data as prepare
    from premise_laundering.selective_correction import validate_selective_corrected_data

    path = prepare(SELECTIVE_CORRECTED_DATA_DIR, n=n, overwrite=overwrite)
    result = validate_selective_corrected_data(path, expected=n)
    cache_volume.commit()
    return f"Prepared {result['examples']} corrected selective-correction tasks at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume},
    timeout=30 * 60,
)
def prepare_scope_radius_data(overwrite: bool = False, n_base_contexts: int = 40) -> str:
    from premise_laundering.scope_radius import prepare_scope_radius_data as prepare
    from premise_laundering.scope_radius import validate_scope_radius_data

    path = prepare(SCOPE_RADIUS_DATA_DIR, n_base_contexts=n_base_contexts, overwrite=overwrite)
    result = validate_scope_radius_data(path, expected_bases=n_base_contexts, expected_examples=n_base_contexts * 3)
    cache_volume.commit()
    return f"Prepared {result['examples']} scope-radius counterfactual tasks at {path}"


@app.function(
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=30 * 60,
)
def prepare_tracepatch_micro(overwrite: bool = False) -> str:
    from tracepatch.micro import prepare_micro_data, read_jsonl, run_data_eval_audit, hard_fail_audit

    path = prepare_micro_data(TRACEPATCH_DATA_DIR, overwrite=overwrite)
    examples = read_jsonl(path)
    _, audit_summary, _ = run_data_eval_audit(examples)
    errors = hard_fail_audit(audit_summary)
    if len(examples) != 60:
        errors.append(f"expected 60 examples, found {len(examples)}")
    if errors:
        raise ValueError(f"TracePatch preflight audit failed: {errors}; summary={audit_summary}")
    cache_volume.commit()
    runs_volume.commit()
    return f"Prepared TracePatch micro data at {path}; audit={audit_summary}"


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_pilot(
    run_id: str | None = None,
    overwrite: bool = False,
    batch_size: int = 1,
    backend: str = "transformers",
    conditions_csv: str | None = None,
) -> str:
    from premise_laundering.data import load_items, prepare_data
    from premise_laundering.run import run_inference
    from premise_laundering.validate import validate_run
    if backend == "vllm":
        from premise_laundering.vllm_runner import VLLMGenerator as Generator
        from premise_laundering.vllm_runner import generation_config
    elif backend == "transformers":
        from premise_laundering.transformers_runner import TransformersGenerator as Generator
        from premise_laundering.transformers_runner import generation_config
    else:
        raise ValueError("backend must be one of: transformers, vllm")

    data_file = prepare_data(DATA_DIR, overwrite=False)
    items = load_items(data_file)
    conditions = tuple(
        condition.strip() for condition in conditions_csv.split(",") if condition.strip()
    ) if conditions_csv else None
    generator = Generator()
    run_dir = run_inference(
        items=items,
        generator=generator,
        output_base_dir=RUNS_DIR,
        run_id=run_id,
        overwrite=overwrite,
        batch_size=batch_size,
        conditions=conditions or None,
        generation_config=generation_config(),
    )
    result = validate_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return f"Completed run at {run_dir} with {result['generations']} generations"


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_controlled(
    run_id: str | None = None,
    overwrite: bool = False,
    batch_size: int = 1,
    backend: str = "transformers",
    conditions_csv: str | None = None,
    n: int = 300,
) -> str:
    from premise_laundering.controlled import (
        CONTROLLED_CONDITIONS,
        load_controlled_items,
        prepare_controlled_data,
        run_controlled_inference,
        validate_controlled_run,
    )

    if backend == "vllm":
        from premise_laundering.vllm_runner import VLLMGenerator as Generator
        from premise_laundering.vllm_runner import generation_config
    elif backend == "transformers":
        from premise_laundering.transformers_runner import TransformersGenerator as Generator
        from premise_laundering.transformers_runner import generation_config
    else:
        raise ValueError("backend must be one of: transformers, vllm")

    data_file = prepare_controlled_data(CONTROLLED_DATA_DIR, n=n, overwrite=False)
    items = load_controlled_items(data_file)
    conditions = tuple(
        condition.strip() for condition in conditions_csv.split(",") if condition.strip()
    ) if conditions_csv else CONTROLLED_CONDITIONS
    generator = Generator()
    run_dir = run_controlled_inference(
        items=items,
        generator=generator,
        output_base_dir=CONTROLLED_RUNS_DIR,
        run_id=run_id,
        overwrite=overwrite,
        batch_size=batch_size,
        backend=backend,
        conditions=conditions,
        generation_config=generation_config(),
    )
    result = validate_controlled_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return f"Completed controlled run at {run_dir} with {result['generations']} generations"


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_prefix_intervention(
    run_id: str,
    model: str,
    overwrite: bool = False,
    batch_size: int = 1,
    n: int = 300,
    shard_index: int = 0,
    num_shards: int = 1,
) -> str:
    from premise_laundering.prefix_transformers import PrefixTransformersGenerator
    from premise_laundering.prefix_transformers import prefix_generation_config
    from premise_laundering.self_prefix import load_prefix_tasks, prepare_prefix_data
    from premise_laundering.self_prefix import run_prefix_intervention as run_experiment
    from premise_laundering.self_prefix import validate_prefix_run

    data_file = prepare_prefix_data(PREFIX_DATA_DIR, n=n, overwrite=False)
    tasks = load_prefix_tasks(data_file)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    tasks = [task for idx, task in enumerate(tasks) if idx % num_shards == shard_index]
    generator = PrefixTransformersGenerator(model=model)
    run_dir = run_experiment(
        tasks=tasks,
        generator=generator,
        output_base_dir=PREFIX_RUNS_DIR,
        run_id=run_id,
        model=model,
        overwrite=overwrite,
        batch_size=batch_size,
        generation_config=prefix_generation_config(model),
    )
    result = validate_prefix_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed self-prefix run at {run_dir}; "
        f"source={result['source_generations']} prefixes={result['prefix_generations']} "
        f"continuations={result['continuation_generations']}"
    )


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_diagnostic_compression(
    run_id: str,
    model: str,
    overwrite: bool = False,
    batch_size: int = 1,
    n: int = 240,
    shard_index: int = 0,
    num_shards: int = 1,
) -> str:
    from premise_laundering.diagnostic_compression import (
        load_trace_variants,
        prepare_diagnostic_data,
        run_diagnostic_verification,
        validate_diagnostic_run,
    )
    from premise_laundering.transformers_runner import TransformersGenerator
    from premise_laundering.transformers_runner import generation_config

    data_file = prepare_diagnostic_data(DIAGNOSTIC_DATA_DIR, n=n, overwrite=False)
    variants = load_trace_variants(Path(data_file).with_name("diagnostic_trace_variants.jsonl"))
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    variants = [variant for idx, variant in enumerate(variants) if idx % num_shards == shard_index]
    generator = TransformersGenerator(model=model, max_new_tokens=220, temperature=0.2, top_p=0.9)
    config = generation_config()
    config.update({"model": model, "max_new_tokens": 220, "task": "diagnostic_verifier"})
    run_dir = run_diagnostic_verification(
        variants=variants,
        generator=generator,
        output_base_dir=DIAGNOSTIC_RUNS_DIR,
        run_id=run_id,
        model=model,
        backend="transformers",
        overwrite=overwrite,
        batch_size=batch_size,
        generation_config=config,
    )
    result = validate_diagnostic_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed diagnostic-compression run at {run_dir}; "
        f"generations={result['generations']} parse_rate={result['parse_rate']}"
    )


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_selective_correction(
    run_id: str,
    model: str,
    overwrite: bool = False,
    batch_size: int = 1,
    n: int = 320,
    shard_index: int = 0,
    num_shards: int = 1,
    conditions_csv: str | None = None,
) -> str:
    from premise_laundering.selective_correction import (
        PROMPT_CONDITIONS,
        load_selective_tasks,
        prepare_selective_data,
        run_selective_revision,
        validate_selective_run,
    )
    from premise_laundering.transformers_runner import TransformersGenerator
    from premise_laundering.transformers_runner import generation_config

    data_file = prepare_selective_data(SELECTIVE_DATA_DIR, n=n, overwrite=False)
    tasks = load_selective_tasks(data_file)
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    tasks = [task for idx, task in enumerate(tasks) if idx % num_shards == shard_index]
    conditions = tuple(
        condition.strip() for condition in conditions_csv.split(",") if condition.strip()
    ) if conditions_csv else PROMPT_CONDITIONS
    generator = TransformersGenerator(model=model, max_new_tokens=700, temperature=0.2, top_p=0.9)
    config = generation_config()
    config.update({"model": model, "max_new_tokens": 700, "task": "selective_correction_revision"})
    run_dir = run_selective_revision(
        tasks=tasks,
        generator=generator,
        output_base_dir=SELECTIVE_RUNS_DIR,
        run_id=run_id,
        model=model,
        backend="transformers",
        overwrite=overwrite,
        batch_size=batch_size,
        conditions=conditions,
        generation_config=config,
    )
    result = validate_selective_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed selective-correction run at {run_dir}; "
        f"generations={result['generations']} probe_results={result['probe_results']} "
        f"parse_rate={result['parse_rate']}"
    )


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 4,
)
def run_selective_corrected(
    run_id: str,
    overwrite: bool = False,
    batch_size: int = 1,
    n: int = 96,
) -> str:
    from premise_laundering.selective_correction import (
        CORRECTED_PROMPT_CONDITIONS,
        load_selective_tasks,
        prepare_selective_corrected_data,
        run_selective_revision,
        validate_selective_run,
    )
    from premise_laundering.transformers_runner import TransformersGenerator
    from premise_laundering.transformers_runner import generation_config

    model = "Qwen/Qwen2.5-7B-Instruct"
    data_file = prepare_selective_corrected_data(SELECTIVE_CORRECTED_DATA_DIR, n=n, overwrite=False)
    tasks = load_selective_tasks(data_file)
    generator = TransformersGenerator(model=model, max_new_tokens=700, temperature=0.2, top_p=0.9)
    config = generation_config()
    config.update({"model": model, "max_new_tokens": 700, "task": "selective_correction_corrected_small"})
    run_dir = run_selective_revision(
        tasks=tasks,
        generator=generator,
        output_base_dir=SELECTIVE_CORRECTED_RUNS_DIR,
        run_id=run_id,
        model=model,
        backend="transformers",
        overwrite=overwrite,
        batch_size=batch_size,
        conditions=CORRECTED_PROMPT_CONDITIONS,
        generation_config=config,
        tasks_filename="feedback_tasks_corrected.jsonl",
    )
    result = validate_selective_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed corrected selective-correction run at {run_dir}; "
        f"generations={result['generations']} probe_results={result['probe_results']} "
        f"parse_rate={result['parse_rate']}"
    )


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 4,
)
def run_scope_radius(
    run_id: str,
    overwrite: bool = False,
    batch_size: int = 1,
    n_base_contexts: int = 40,
) -> str:
    from premise_laundering.scope_radius import (
        MODEL,
        SCOPE_CONDITIONS,
        load_scope_tasks,
        prepare_scope_radius_data,
        run_scope_radius_inference,
        validate_scope_radius_run,
    )
    from premise_laundering.transformers_runner import TransformersGenerator
    from premise_laundering.transformers_runner import generation_config

    data_file = prepare_scope_radius_data(SCOPE_RADIUS_DATA_DIR, n_base_contexts=n_base_contexts, overwrite=False)
    tasks = load_scope_tasks(data_file)
    generator = TransformersGenerator(model=MODEL, max_new_tokens=520, temperature=0.2, top_p=0.9)
    config = generation_config()
    config.update({"model": MODEL, "max_new_tokens": 520, "task": "scope_radius_counterfactuals"})
    run_dir = run_scope_radius_inference(
        tasks=tasks,
        generator=generator,
        output_base_dir=SCOPE_RADIUS_RUNS_DIR,
        run_id=run_id,
        model=MODEL,
        backend="transformers",
        overwrite=overwrite,
        batch_size=batch_size,
        conditions=SCOPE_CONDITIONS,
        generation_config=config,
    )
    result = validate_scope_radius_run(run_dir)
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed scope-radius run at {run_dir}; "
        f"generations={result['generations']} probe_results={result['probe_results']} "
        f"parse_rate={result['parse_rate']}"
    )


@app.function(
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume, "/outputs": runs_volume},
    timeout=60 * 60 * 8,
)
def run_tracepatch_micro(
    run_id: str = "tracepatch_micro_v1",
    overwrite: bool = False,
    batch_size: int = 2,
) -> str:
    from tracepatch.micro import prepare_micro_data, read_json, run_micro

    data_file = prepare_micro_data(TRACEPATCH_DATA_DIR, overwrite=False)
    run_dir = Path(TRACEPATCH_RUNS_DIR) / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"{run_dir} exists; pass overwrite=True to replace it")
    run_micro(data_file=data_file, output_dir=run_dir, batch_size=batch_size)
    summary = read_json(run_dir / "summary.json")
    cache_volume.commit()
    runs_volume.commit()
    return (
        f"Completed TracePatch micro run at {run_dir}; "
        f"initial={summary['initial']} verdict={summary['verdict']}"
    )


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_metrics(run_id: str, annotation_file: str | None = None) -> dict:
    from premise_laundering.metrics import summarize_run

    run_dir = Path(RUNS_DIR) / run_id
    annotation_path = Path(annotation_file) if annotation_file else None
    summary = summarize_run(run_dir, annotation_path)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_controlled_metrics(run_id: str) -> dict:
    from premise_laundering.controlled import summarize_controlled_run

    run_dir = Path(CONTROLLED_RUNS_DIR) / run_id
    summary = summarize_controlled_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_prefix_metrics(run_id: str) -> dict:
    from premise_laundering.self_prefix import summarize_prefix_run

    run_dir = Path(PREFIX_RUNS_DIR) / run_id
    summary = summarize_prefix_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_diagnostic_metrics(run_id: str) -> dict:
    from premise_laundering.diagnostic_compression import summarize_diagnostic_run

    run_dir = Path(DIAGNOSTIC_RUNS_DIR) / run_id
    summary = summarize_diagnostic_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_selective_metrics(run_id: str) -> dict:
    from premise_laundering.selective_correction import summarize_selective_run

    run_dir = Path(SELECTIVE_RUNS_DIR) / run_id
    summary = summarize_selective_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_selective_corrected_metrics(run_id: str) -> dict:
    from premise_laundering.selective_correction import summarize_selective_run

    run_dir = Path(SELECTIVE_CORRECTED_RUNS_DIR) / run_id
    summary = summarize_selective_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def summarize_scope_radius_metrics(run_id: str) -> dict:
    from premise_laundering.scope_radius import summarize_scope_radius_run

    run_dir = Path(SCOPE_RADIUS_RUNS_DIR) / run_id
    summary = summarize_scope_radius_run(run_dir)
    runs_volume.commit()
    return summary


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def merge_controlled_runs(
    source_run_ids_csv: str,
    merged_run_id: str,
    overwrite: bool = False,
) -> str:
    from premise_laundering.controlled import merge_controlled_run_dirs, summarize_controlled_run

    source_run_ids = [run_id.strip() for run_id in source_run_ids_csv.split(",") if run_id.strip()]
    run_dir = merge_controlled_run_dirs(
        CONTROLLED_RUNS_DIR,
        source_run_ids,
        merged_run_id,
        overwrite=overwrite,
    )
    summary = summarize_controlled_run(run_dir)
    runs_volume.commit()
    return (
        f"Merged {len(source_run_ids)} controlled runs into {run_dir}; "
        f"total generations={summary['total_generations']}"
    )


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def merge_prefix_runs(
    source_run_ids_csv: str,
    merged_run_id: str,
    overwrite: bool = False,
) -> str:
    from premise_laundering.self_prefix import merge_prefix_runs as merge_prefix_run_dirs
    from premise_laundering.self_prefix import summarize_prefix_run

    source_run_ids = [run_id.strip() for run_id in source_run_ids_csv.split(",") if run_id.strip()]
    run_dir = merge_prefix_run_dirs(PREFIX_RUNS_DIR, source_run_ids, merged_run_id, overwrite=overwrite)
    summary = summarize_prefix_run(run_dir)
    runs_volume.commit()
    return (
        f"Merged {len(source_run_ids)} self-prefix runs into {run_dir}; "
        f"total tasks={summary['total_tasks']}"
    )


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def merge_diagnostic_runs(
    source_run_ids_csv: str,
    merged_run_id: str,
    overwrite: bool = False,
) -> str:
    from premise_laundering.diagnostic_compression import merge_diagnostic_runs as merge_diagnostic_run_dirs
    from premise_laundering.diagnostic_compression import summarize_diagnostic_run

    source_run_ids = [run_id.strip() for run_id in source_run_ids_csv.split(",") if run_id.strip()]
    run_dir = merge_diagnostic_run_dirs(DIAGNOSTIC_RUNS_DIR, source_run_ids, merged_run_id, overwrite=overwrite)
    summary = summarize_diagnostic_run(run_dir)
    runs_volume.commit()
    return (
        f"Merged {len(source_run_ids)} diagnostic-compression runs into {run_dir}; "
        f"total generations={summary['total_verifier_generations']}"
    )


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def merge_selective_runs(
    source_run_ids_csv: str,
    merged_run_id: str,
    overwrite: bool = False,
) -> str:
    from premise_laundering.selective_correction import merge_selective_runs as merge_selective_run_dirs
    from premise_laundering.selective_correction import summarize_selective_run

    source_run_ids = [run_id.strip() for run_id in source_run_ids_csv.split(",") if run_id.strip()]
    run_dir = merge_selective_run_dirs(SELECTIVE_RUNS_DIR, source_run_ids, merged_run_id, overwrite=overwrite)
    summary = summarize_selective_run(run_dir)
    runs_volume.commit()
    return (
        f"Merged {len(source_run_ids)} selective-correction runs into {run_dir}; "
        f"total generations={summary['total_generations']}"
    )


@app.function(
    volumes={"/outputs": runs_volume},
    timeout=15 * 60,
)
def merge_runs(source_run_ids_csv: str, merged_run_id: str, overwrite: bool = False) -> str:
    from premise_laundering.merge import merge_run_dirs
    from premise_laundering.metrics import summarize_run

    source_run_ids = [run_id.strip() for run_id in source_run_ids_csv.split(",") if run_id.strip()]
    run_dir = merge_run_dirs(RUNS_DIR, source_run_ids, merged_run_id, overwrite=overwrite)
    summary = summarize_run(run_dir)
    runs_volume.commit()
    return f"Merged {len(source_run_ids)} runs into {run_dir}; total generations={summary['total_generations']}"


@app.local_entrypoint()
def main(
    action: str = "run",
    run_id: str | None = None,
    overwrite: bool = False,
    batch_size: int = 1,
    backend: str = "transformers",
    conditions: str | None = None,
    annotation_file: str | None = None,
    n: int = 99,
    model: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    """Manual Modal entrypoint. Use action=prepare, run, or metrics."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if action == "prepare":
        print(prepare_data.remote(overwrite=overwrite))
    elif action == "controlled-prepare":
        print(prepare_controlled_data.remote(overwrite=overwrite, n=99 if n == 300 else n))
    elif action == "prefix-prepare":
        print(prepare_prefix_data.remote(overwrite=overwrite, n=n))
    elif action == "diagnostic-prepare":
        print(prepare_diagnostic_data.remote(overwrite=overwrite, n=n))
    elif action == "selective-prepare":
        print(prepare_selective_data.remote(overwrite=overwrite, n=320 if n == 99 else n))
    elif action == "selective-corrected-prepare":
        print(prepare_selective_corrected_data.remote(overwrite=overwrite, n=96 if n == 99 else n))
    elif action == "scope-radius-prepare":
        print(prepare_scope_radius_data.remote(overwrite=overwrite, n_base_contexts=40 if n == 99 else n))
    elif action == "tracepatch-prepare":
        print(prepare_tracepatch_micro.remote(overwrite=overwrite))
    elif action == "run":
        print(
            run_pilot.remote(
                run_id=run_id,
                overwrite=overwrite,
                batch_size=batch_size,
                backend=backend,
                conditions_csv=conditions,
            )
        )
    elif action == "controlled-run":
        print(
            run_controlled.remote(
                run_id=run_id,
                overwrite=overwrite,
                batch_size=batch_size,
                backend=backend,
                conditions_csv=conditions,
                n=99 if n == 300 else n,
            )
        )
    elif action == "prefix-run":
        if not run_id:
            raise ValueError("run_id is required for action=prefix-run")
        if not model:
            raise ValueError("model is required for action=prefix-run")
        print(
            run_prefix_intervention.remote(
                run_id=run_id,
                model=model,
                overwrite=overwrite,
                batch_size=batch_size,
                n=n,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        )
    elif action == "diagnostic-run":
        if not run_id:
            raise ValueError("run_id is required for action=diagnostic-run")
        if not model:
            raise ValueError("model is required for action=diagnostic-run")
        print(
            run_diagnostic_compression.remote(
                run_id=run_id,
                model=model,
                overwrite=overwrite,
                batch_size=batch_size,
                n=n,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        )
    elif action == "selective-run":
        if not run_id:
            raise ValueError("run_id is required for action=selective-run")
        if not model:
            raise ValueError("model is required for action=selective-run")
        print(
            run_selective_correction.remote(
                run_id=run_id,
                model=model,
                overwrite=overwrite,
                batch_size=batch_size,
                n=320 if n == 99 else n,
                shard_index=shard_index,
                num_shards=num_shards,
                conditions_csv=conditions,
            )
        )
    elif action == "selective-corrected-run":
        if not run_id:
            raise ValueError("run_id is required for action=selective-corrected-run")
        print(
            run_selective_corrected.remote(
                run_id=run_id,
                overwrite=overwrite,
                batch_size=batch_size,
                n=96 if n == 99 else n,
            )
        )
    elif action == "scope-radius-run":
        if not run_id:
            raise ValueError("run_id is required for action=scope-radius-run")
        print(
            run_scope_radius.remote(
                run_id=run_id,
                overwrite=overwrite,
                batch_size=batch_size,
                n_base_contexts=40 if n == 99 else n,
            )
        )
    elif action == "tracepatch-run":
        print(
            run_tracepatch_micro.remote(
                run_id=run_id or "tracepatch_micro_v1",
                overwrite=overwrite,
                batch_size=batch_size,
            )
        )
    elif action == "metrics":
        if not run_id:
            raise ValueError("run_id is required for action=metrics")
        print(summarize_metrics.remote(run_id=run_id, annotation_file=annotation_file))
    elif action == "controlled-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=controlled-metrics")
        print(summarize_controlled_metrics.remote(run_id=run_id))
    elif action == "prefix-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=prefix-metrics")
        print(summarize_prefix_metrics.remote(run_id=run_id))
    elif action == "diagnostic-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=diagnostic-metrics")
        print(summarize_diagnostic_metrics.remote(run_id=run_id))
    elif action == "selective-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=selective-metrics")
        print(summarize_selective_metrics.remote(run_id=run_id))
    elif action == "selective-corrected-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=selective-corrected-metrics")
        print(summarize_selective_corrected_metrics.remote(run_id=run_id))
    elif action == "scope-radius-metrics":
        if not run_id:
            raise ValueError("run_id is required for action=scope-radius-metrics")
        print(summarize_scope_radius_metrics.remote(run_id=run_id))
    elif action == "controlled-merge":
        if not run_id:
            raise ValueError("run_id is required as the merged run id for action=controlled-merge")
        if not conditions:
            raise ValueError("conditions must contain comma-separated source run ids for action=controlled-merge")
        print(
            merge_controlled_runs.remote(
                source_run_ids_csv=conditions,
                merged_run_id=run_id,
                overwrite=overwrite,
            )
        )
    elif action == "prefix-merge":
        if not run_id:
            raise ValueError("run_id is required as the merged run id for action=prefix-merge")
        if not conditions:
            raise ValueError("conditions must contain comma-separated source run ids for action=prefix-merge")
        print(
            merge_prefix_runs.remote(
                source_run_ids_csv=conditions,
                merged_run_id=run_id,
                overwrite=overwrite,
            )
        )
    elif action == "diagnostic-merge":
        if not run_id:
            raise ValueError("run_id is required as the merged run id for action=diagnostic-merge")
        if not conditions:
            raise ValueError("conditions must contain comma-separated source run ids for action=diagnostic-merge")
        print(
            merge_diagnostic_runs.remote(
                source_run_ids_csv=conditions,
                merged_run_id=run_id,
                overwrite=overwrite,
            )
        )
    elif action == "selective-merge":
        if not run_id:
            raise ValueError("run_id is required as the merged run id for action=selective-merge")
        if not conditions:
            raise ValueError("conditions must contain comma-separated source run ids for action=selective-merge")
        print(
            merge_selective_runs.remote(
                source_run_ids_csv=conditions,
                merged_run_id=run_id,
                overwrite=overwrite,
            )
        )
    elif action == "merge":
        if not run_id:
            raise ValueError("run_id is required as the merged run id for action=merge")
        if not conditions:
            raise ValueError("conditions must contain comma-separated source run ids for action=merge")
        print(
            merge_runs.remote(
                source_run_ids_csv=conditions,
                merged_run_id=run_id,
                overwrite=overwrite,
            )
        )
    else:
        raise ValueError(
            "action must be one of: prepare, run, metrics, merge, "
            "controlled-prepare, controlled-run, controlled-metrics, controlled-merge, "
            "prefix-prepare, prefix-run, prefix-metrics, prefix-merge, "
            "diagnostic-prepare, diagnostic-run, diagnostic-metrics, diagnostic-merge, "
            "selective-prepare, selective-run, selective-metrics, selective-merge, "
            "selective-corrected-prepare, selective-corrected-run, selective-corrected-metrics, "
            "scope-radius-prepare, scope-radius-run, scope-radius-metrics, "
            "tracepatch-prepare, tracepatch-run"
        )
