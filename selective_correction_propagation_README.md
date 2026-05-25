# Selective Correction Propagation

This is a separate pilot experiment for testing whether a model can apply localized feedback to
exactly the affected reasoning state while preserving unrelated state.

The experiment generates 320 synthetic tasks:

- 2 domains: `tool_agent_state`, `rule_scope_reasoning`
- 4 correction types: `local_only`, `local_to_final`, `scope_limited`, `rule_level`
- 40 examples per domain/correction-type cell
- 5 prompt conditions per example
- 5 probes per example, answered in one structured JSON output

No manual annotation is required. Hidden before/after state, update keys, preserve keys, and probe
gold answers are known by construction.

## Modal Commands

Prepare data:

```bash
python3 -m modal run modal_app.py --action selective-prepare --n 320
```

Run one verifier/revision model:

```bash
python3 -m modal run modal_app.py --action selective-run \
  --run-id selective-qwen-001 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 320 \
  --batch-size 1
```

Run Mistral:

```bash
python3 -m modal run modal_app.py --action selective-run \
  --run-id selective-mistral-001 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --n 320 \
  --batch-size 1
```

Shard a run manually:

```bash
python3 -m modal run modal_app.py --action selective-run \
  --run-id selective-qwen-s0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n 320 \
  --num-shards 4 \
  --shard-index 0
```

Merge completed runs or shards:

```bash
python3 -m modal run modal_app.py --action selective-merge \
  --run-id selective-combined-001 \
  --conditions selective-qwen-001,selective-mistral-001
```

Recompute metrics:

```bash
python3 -m modal run modal_app.py --action selective-metrics --run-id selective-combined-001
```

Outputs are stored on the Modal volume:

```text
/outputs/selective_runs/<run-id>/
```

Download a run:

```bash
python3 -m modal volume get premise-laundering-runs /selective_runs/selective-combined-001 ./outputs/selective-combined-001
```

## Local Checks

Prepare and validate data:

```bash
PYTHONPATH=src python3 -m premise_laundering.cli selective-prepare-data --output-dir data/selective --n 320
PYTHONPATH=src python3 -m premise_laundering.cli selective-validate-data --data-file data/selective/feedback_tasks.jsonl
```

Run the no-GPU dry run:

```bash
PYTHONPATH=src python3 -m premise_laundering.cli selective-dry-run \
  --data-file data/selective/feedback_tasks.jsonl \
  --output-dir runs/selective \
  --run-id selective-dry \
  --overwrite
```

Validate and summarize:

```bash
PYTHONPATH=src python3 -m premise_laundering.cli selective-validate-run --run-dir runs/selective/selective-dry
PYTHONPATH=src python3 -m premise_laundering.cli selective-metrics --run-dir runs/selective/selective-dry
```

## Output Files

Each run directory contains:

- `feedback_tasks.jsonl`: generated contexts, initial traces, feedback, hidden states, probes
- `revision_generations.jsonl`: prompts, raw model outputs, parsed JSON, gold metadata
- `parsed_probe_results.jsonl`: one scored row per probe
- `enriched_results.jsonl`: one scored row per generation
- `metrics_summary.json`: overall and grouped metrics
- `metrics_by_condition.csv`
- `metrics_by_domain.csv`
- `metrics_by_correction_type.csv`
- `metrics_by_model.csv`
- `qualitative_examples.json`
- `run_manifest.json`

## Metrics

The primary metrics are selective-revision metrics, not final answer accuracy:

- `target_binding_accuracy`
- `dependent_update_recall`
- `boundary_preservation_precision`
- `overcorrection_rate`
- `under_propagation_rate`
- `trace_answer_consistency`
- `future_use_accuracy`
- `final_state_accuracy`
- `selective_revision_score`

The pilot is promising if standard or natural feedback shows under-propagation or overcorrection
while `correction_scope_map` improves both update recall and preservation precision.
