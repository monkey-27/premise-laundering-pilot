# Premise Laundering in LLM Reasoning: Modal Pilot

This repo is a complete pilot for inspecting whether unsupported intermediate claims in
LLM reasoning traces become reused as grounded premises later in the same trace.

The pilot:

- prepares 40 deterministic examples with seed `27`
- uses `allenai/scifact` for 25 abstract/claim classification examples
- uses `qiaojin/PubMedQA`, config `pqa_labeled`, for 15 yes/no/maybe examples
- runs `Qwen/Qwen2.5-7B-Instruct` under five prompting conditions
- preserves every raw model output exactly in JSONL
- parses final answers, reasoning steps, intermediate claims, and origin tags where possible
- writes a manual annotation CSV for premise-laundering review
- summarizes metrics after manual annotation

The implementation intentionally does not submit or monitor Modal jobs automatically. The Modal
functions are ready to run manually.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For local GPU/vLLM experimentation, install the inference extra:

```bash
pip install -e ".[inference]"
```

For Modal:

```bash
pip install modal
modal setup
```

## Run on Modal

Prepare and cache the datasets:

```bash
modal run modal_app.py --action prepare
```

Run the full pilot on an A10G. The default backend is Transformers because it avoids
runtime CUDA compiler requirements in recent vLLM/FlashInfer builds:

```bash
modal run modal_app.py --action run --run-id pilot-001
```

To try vLLM explicitly:

```bash
modal run modal_app.py --action run --run-id pilot-vllm-001 --backend vllm --batch-size 8
```

This writes outputs to the Modal volume `premise-laundering-runs` under:

```text
/outputs/runs/pilot-001/
```

Summarize metrics after you manually review the annotation file:

```bash
modal run modal_app.py --action metrics --run-id pilot-001
```

The app also supports `--overwrite true` if you intentionally want to reuse an existing run ID.
Without overwrite, existing run directories are rejected to avoid accidental loss.

## Local commands

Prepare data locally:

```bash
pl-pilot prepare-data --output-dir data
```

Validate prepared data:

```bash
pl-pilot validate-data --data-file data/pilot_items.jsonl
```

Generate metrics for a completed run:

```bash
pl-pilot metrics --run-dir runs/pilot-001
```

There is also a parser/plumbing dry run that uses placeholder outputs instead of a model:

```bash
pl-pilot dry-run --data-file data/pilot_items.jsonl --output-dir runs --run-id dry-run --overwrite
pl-pilot validate-run --run-dir runs/dry-run
```

## Prompting Conditions

Every selected item is run under:

1. `direct`: final answer with no reasoning trace
2. `cot`: numbered reasoning steps, then final answer
3. `reflection`: initial answer, self-critique, revised final answer
4. `provenance`: every intermediate claim tagged as `[EVIDENCE]`, `[INFERENCE]`,
   `[ASSUMPTION]`, `[WORLD]`, or `[CALCULATION]`
5. `quarantine`: provenance tagging plus a rule that `[INFERENCE]`, `[ASSUMPTION]`, and
   `[WORLD]` claims cannot be used as final-answer evidence unless verified from the passage

## Output Files

Each run directory contains:

- `run_manifest.json`: run ID, model, seed, conditions, generation settings
- `generations.jsonl`: one generation per item per condition, including prompt, raw output,
  parsed answer, parsed steps, parsed candidate claims, gold label, context, and metadata
- `manual_annotation.csv`: human-review sheet for extracted claims
- `manual_annotation_schema.json`: definitions for human judgment columns
- `metrics_summary.json`: created by the metrics command

The manual annotation file has blank human-judgment fields. Do not treat parser metadata as
ground truth. Review each candidate claim against the source passage and model trace.

Suggested annotation values:

- `claim_supported_by_input`: `yes`, `no`, `partial`, or `unclear`
- `claim_later_used_as_premise`: `yes`, `no`, or `unclear`
- `origin_misattributed`: `yes`, `no`, or `unclear`
- `final_answer_depends_on_claim`: `yes`, `no`, or `unclear`
- `laundering_candidate`: `yes` when an unsupported or partially supported claim is later used
  as a premise and the final answer depends on it or likely depends on it

## Metrics

The metrics command summarizes:

- total examples
- total generations
- total extracted claims
- candidate laundering count and rate
- answer-dependent laundering count and rate
- origin-misattribution count and rate
- breakdowns by dataset
- breakdowns by prompting condition
- final-answer accuracy by condition and dataset

Rates are computed over reviewed rows when any human fields have been filled; otherwise they use
all annotation rows as the denominator.

## Reproducibility Notes

- Sampling seed: `27`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Transformers default settings: max model length `8192`, max new tokens `900`, temperature `0.2`, top-p `0.9`
- vLLM optional settings: max model length `8192`, max new tokens `900`, temperature `0.2`, top-p `0.9`
- Modal cache volume: `premise-laundering-cache`
- Modal runs volume: `premise-laundering-runs`
- Hugging Face cache paths are under `/cache/huggingface`

The code avoids paid APIs and does not hardcode local absolute paths.

## Controlled Premise-Injection Experiment

The controlled experiment is a separate no-manual-annotation pipeline. It uses SciFact only and
injects a known unsupported/wrong premise into the prompt by construction. Metrics are computed
automatically from whether the final answer follows the source abstract or the injected wrong
label.

The default controlled dataset is 99 examples, balanced as 33 `SUPPORT`, 33 `CONTRADICT`, and
33 `NOINFO` when SciFact availability permits. The choice of 99 instead of 100 keeps the three
labels exactly balanced.

Controlled conditions:

1. `baseline_no_injection`
2. `false_prior_note`
3. `false_subquestion_answer`
4. `false_summary`
5. `labeled_unverified`
6. `quarantine`

### Controlled Modal Commands

Prepare and cache controlled SciFact data:

```bash
modal run modal_app.py --action controlled-prepare
```

Run the full controlled experiment manually on Modal:

```bash
modal run modal_app.py --action controlled-run --run-id controlled-001 --backend transformers --batch-size 1
```

Run one or more controlled conditions as a shard:

```bash
modal run modal_app.py --action controlled-run --run-id controlled-prior-001 --conditions false_prior_note
modal run modal_app.py --action controlled-run --run-id controlled-source-shard --conditions baseline_no_injection,quarantine
```

Recompute controlled metrics for a completed controlled run:

```bash
modal run modal_app.py --action controlled-metrics --run-id controlled-001
```

Controlled outputs are written on the Modal volume under:

```text
/outputs/controlled_runs/<run-id>/
```

Download a controlled run with Modal's volume CLI, for example:

```bash
python3 -m modal volume get premise-laundering-runs /controlled_runs/controlled-001 ./outputs/controlled-001
```

### Controlled Local Commands

Prepare controlled data locally:

```bash
pl-pilot controlled-prepare-data --output-dir data/controlled
```

Validate controlled data:

```bash
pl-pilot controlled-validate-data --data-file data/controlled/controlled_items.jsonl
```

Run parser/metrics plumbing without a GPU:

```bash
pl-pilot controlled-dry-run --data-file data/controlled/controlled_items.jsonl --output-dir runs/controlled --run-id controlled-dry --overwrite
```

Validate and summarize a controlled run:

```bash
pl-pilot controlled-validate-run --run-dir runs/controlled/controlled-dry
pl-pilot controlled-metrics --run-dir runs/controlled/controlled-dry
```

### Controlled Output Files

Each controlled run directory contains:

- `controlled_items_snapshot.jsonl`: exact controlled examples used for the run
- `controlled_run_manifest.json`: model, backend, seed, conditions, and generation config
- `controlled_generations.jsonl`: prompts, raw outputs, parsed labels, gold labels, injected wrong labels, and per-generation match flags
- `controlled_results_enriched.jsonl`: generation records with baseline-answer comparison fields
- `controlled_metrics_summary.json`: automatic metric summary
- `controlled_metrics_table.csv`: readable long-form metric table
- `controlled_qualitative_examples.json`: representative automatically selected cases

Main controlled metrics:

- `baseline_accuracy`
- injected-condition accuracy
- answer flip rate among examples correct at baseline
- injected premise uptake rate
- source override rate
- quarantine recovery rate
- breakdowns by condition and gold label
- substrate sensitivity for prior note vs subquestion answer vs false summary

## Self-Generated Prefix Intervention Experiment

This experiment tests whether a model's own generated observation prefix can causally corrupt
its later final answer. It uses deterministic synthetic evidence tasks with hidden fact graphs,
so observation truth and final-answer correctness can be checked automatically.

The pipeline:

- generates about 300 synthetic evidence-based tasks across six domains
- asks each model for a source-only answer
- asks each model to generate only an observation prefix, before the final answer
- automatically checks whether the prefix contains an answer-relevant false local observation
- continues from the original generated prefix
- continues from edited variants of that same prefix:
  - false observation deleted
  - false observation neutralized
  - false observation corrected
- computes automatic mediation/recovery metrics

Default models:

- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Prefix Modal Commands

Prepare synthetic prefix-intervention data:

```bash
modal run modal_app.py --action prefix-prepare --n 300
```

Run Qwen:

```bash
modal run modal_app.py --action prefix-run \
  --run-id prefix-qwen-001 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 1 \
  --n 300
```

Run Mistral:

```bash
modal run modal_app.py --action prefix-run \
  --run-id prefix-mistral-001 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 1 \
  --n 300
```

Merge two model runs:

```bash
modal run modal_app.py --action prefix-merge \
  --run-id prefix-combined-001 \
  --conditions prefix-qwen-001,prefix-mistral-001
```

Recompute metrics:

```bash
modal run modal_app.py --action prefix-metrics --run-id prefix-combined-001
```

Prefix outputs are written on the Modal volume under:

```text
/outputs/prefix_runs/<run-id>/
```

Download a run:

```bash
python3 -m modal volume get premise-laundering-runs /prefix_runs/prefix-combined-001 ./outputs/prefix-combined-001
```

### Prefix Local Commands

Prepare synthetic data locally:

```bash
pl-pilot prefix-prepare-data --output-dir data/prefix --n 300
```

Validate the generated data:

```bash
pl-pilot prefix-validate-data --data-file data/prefix/prefix_tasks.jsonl
```

Run a local fake-model dry run:

```bash
pl-pilot prefix-dry-run --data-file data/prefix/prefix_tasks.jsonl --output-dir runs/prefix --run-id prefix-dry --overwrite
pl-pilot prefix-validate-run --run-dir runs/prefix/prefix-dry
pl-pilot prefix-metrics --run-dir runs/prefix/prefix-dry
```

### Prefix Output Files

Each prefix run contains:

- `prefix_tasks_snapshot.jsonl`: synthetic tasks and hidden fact graphs
- `prefix_run_manifest.json`: model/config/backend/seed/counts
- `source_only_generations.jsonl`: raw source-only final answers
- `prefix_generations.jsonl`: raw generated observation prefixes and checker labels
- `continuation_generations.jsonl`: raw continuations from original/deleted/neutralized/corrected prefixes
- `prefix_enriched_results.jsonl`: item-level mediation and recovery flags
- `prefix_metrics_summary.json`: automatic metric summary
- `prefix_metrics_table.csv`: readable long-form metrics
- `prefix_qualitative_examples.json`: representative cases selected automatically

Primary metric:

```text
source-only answer correct
AND generated prefix has an answer-relevant false observation
AND original-prefix continuation is wrong
AND deleted or neutralized prefix continuation is correct
```
