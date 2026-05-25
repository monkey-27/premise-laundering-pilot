# Corrected Selective Correction Pilot

This is the smaller corrected pilot for `selective_correction_propagation`.

It fixes probe/scoring issues from the first 320-example run:

- value fields ask for values, not yes/no confirmation
- `none` and no-action paraphrases are accepted
- token-refresh/retry-upload paraphrases are accepted
- rule/review aliases are normalized
- local-only update/preserve labels are corrected
- trace-answer consistency is conservative and can be unknown
- target binding is scored only for `correction_scope_map`

## Design

- 96 examples total
- 2 domains: `tool_agent_state`, `rule_scope_reasoning`
- 4 correction types: `local_only`, `local_to_final`, `scope_limited`, `rule_level`
- 12 examples per domain/type cell
- 1 model: `Qwen/Qwen2.5-7B-Instruct`
- 4 conditions:
  - `standard_feedback`
  - `natural_feedback`
  - `full_regeneration`
  - `correction_scope_map`

Total generations: `96 * 4 = 384`.

## Modal

Prepare corrected data:

```bash
python3 -m modal run modal_app.py --action selective-corrected-prepare --n 96
```

Run the corrected Qwen pilot:

```bash
python3 -m modal run modal_app.py --action selective-corrected-run \
  --run-id selective-corrected-20260524-qwen-small \
  --n 96 \
  --batch-size 1
```

Recompute metrics:

```bash
python3 -m modal run modal_app.py --action selective-corrected-metrics \
  --run-id selective-corrected-20260524-qwen-small
```

Outputs are written under:

```text
/outputs/selective_corrected_runs/<run-id>/
```

Download:

```bash
python3 -m modal volume get premise-laundering-runs \
  /selective_corrected_runs/selective-corrected-20260524-qwen-small \
  ./outputs/selective-corrected-20260524-qwen-small
```

## Local Checks

```bash
PYTHONPATH=src python3 -m premise_laundering.cli selective-corrected-prepare-data \
  --output-dir data/selective_corrected --n 96

PYTHONPATH=src python3 -m premise_laundering.cli selective-corrected-validate-data \
  --data-file data/selective_corrected/feedback_tasks_corrected.jsonl

PYTHONPATH=src python3 -m premise_laundering.cli selective-corrected-dry-run \
  --data-file data/selective_corrected/feedback_tasks_corrected.jsonl \
  --output-dir runs/selective_corrected \
  --run-id dry \
  --overwrite
```

## Outputs

- `feedback_tasks_corrected.jsonl`
- `revision_generations.jsonl`
- `parsed_probe_results.jsonl`
- `enriched_results.jsonl`
- `metrics_summary.json`
- `metrics_by_condition.csv`
- `metrics_by_domain.csv`
- `metrics_by_correction_type.csv`
- `metrics_by_model.csv`
- `metrics_by_state_key.csv`
- `scoring_diagnostics.json`
- `qualitative_examples.json`
- `run_manifest.json`

Use `scoring_diagnostics.json` to confirm exact, yes/no, alias, and value scoring paths are active.
