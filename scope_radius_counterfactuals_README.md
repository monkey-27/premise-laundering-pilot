# Scope Radius Counterfactuals

Cheap pilot for the selective correction project. The experiment tests whether a model infers the correct update radius after localized feedback: direct target, downstream updates, preserved boundary state, delayed action consequences, and stale-trace conflicts.

## Pilot Design

- Experiment: `scope_radius_counterfactuals`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Base contexts: 40
- Feedback variants: 3 per base context
- Examples: 120
- Conditions: `standard_feedback`, `full_regeneration`, `scope_ledger`
- Generations: 360
- Domains: `tool_agent_state`, `rule_policy_scope`, `source_status_rag_scope`
- Scope radii: `local_entity`, `entity_class`, `rule_boundary`, `source_status`

Every example has five closed-set probes:

1. direct target probe
2. boundary preservation probe
3. update-set consequence probe
4. delayed action probe
5. stale-trace conflict probe

## Local Checks

```bash
PYTHONPATH=src python3 -m premise_laundering.cli scope-radius-prepare-data \
  --output-dir data/scope_radius \
  --overwrite

PYTHONPATH=src python3 -m premise_laundering.cli scope-radius-validate-data \
  --data-file data/scope_radius/scope_radius_tasks.jsonl

PYTHONPATH=src python3 -m premise_laundering.cli scope-radius-dry-run \
  --data-file data/scope_radius/scope_radius_tasks.jsonl \
  --output-dir runs/scope_radius \
  --run-id scope-radius-dry-run \
  --overwrite
```

## Modal Run

Prepare cached task data:

```bash
python3 -m modal run modal_app.py --action scope-radius-prepare --n 40
```

Run the Qwen pilot:

```bash
python3 -m modal run modal_app.py \
  --action scope-radius-run \
  --run-id scope-radius-YYYYMMDD-qwen-small \
  --batch-size 1 \
  --n 40
```

Recompute metrics for an existing run:

```bash
python3 -m modal run modal_app.py \
  --action scope-radius-metrics \
  --run-id scope-radius-YYYYMMDD-qwen-small
```

Download outputs:

```bash
python3 -m modal volume get --force premise-laundering-runs \
  /scope_radius_runs/scope-radius-YYYYMMDD-qwen-small \
  outputs/scope-radius-YYYYMMDD-qwen-small
```

## Outputs

- `scope_radius_tasks.jsonl`
- `revision_generations.jsonl`
- `parsed_results.jsonl`
- `enriched_results.jsonl`
- `metrics_summary.json`
- `metrics_by_condition.csv`
- `metrics_by_domain.csv`
- `metrics_by_scope_radius.csv`
- `metrics_by_probe_type.csv`
- `metrics_by_state_key.csv`
- `scope_contrast_metrics.csv`
- `scoring_diagnostics.json`
- `qualitative_examples.json`
- `run_manifest.json`

## Headline Metrics

This is not a final-answer-improvement experiment. Read these first:

- `scope_radius_accuracy`
- `update_set_f1`
- `preserve_set_f1`
- `boundary_leak_rate`
- `under_scope_rate`
- `delayed_action_given_direct_correct`
- `stale_reversion_rate`
- `scope_contrast_sensitivity`
- `selective_scope_score`

Promising signal means direct correction is high, but update radius, delayed action, stale-context, or contrast sensitivity are lower. If all metrics are above 0.90, the task is likely too easy.
