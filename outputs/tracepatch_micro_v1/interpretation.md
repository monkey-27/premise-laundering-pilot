# TracePatch Micro V1 Interpretation

Verdict: **no scale**

## Main Result
| system | final accuracy | repair success on failed | repeated error | prefix damage | parse error |
|---|---:|---:|---:|---:|---:|
| self_correction | 0.817 | 0.185 | 0.704 | 0.000 | 0.148 |
| full_regeneration | 0.850 | 0.333 | 0.481 | 0.000 | 0.185 |
| edit_only_repair | 0.817 | 0.185 | 0.741 | 0.037 | 0.111 |
| tracepatch | 0.825 | 0.222 | 0.481 | 0.111 | 0.148 |

## Model Breakdown
| model | initial accuracy | initial failures | tracepatch final | full regeneration final |
|---|---:|---:|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | 0.850 | 9 | 0.917 | 0.900 |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.700 | 18 | 0.733 | 0.800 |

## Scale Criteria
- TracePatch gain over full regeneration: -0.025
- TracePatch gain over self-correction: 0.008
- TracePatch gain over edit-only repair: 0.008
- Repeated-error drop vs full regeneration: 0.0%
- Repeated-error drop vs self-correction: 31.6%
- Prefix-damage drop vs full regeneration: 0.0%
- Oracle gap on overlapping subset: 0.000 over n=7 oracle repairs

## Failure Modes
- Full regeneration matched or beat TracePatch on aggregate final accuracy.
- TracePatch did not reduce prefix damage relative to full regeneration in the automatic approximation.
- Oracle localization did not beat normal TracePatch on the available checker-derived subset.
- RAG had no initial failures in this micropilot, so repair gains are driven by code and math only.
- Repeated-error and prefix-damage metrics are automatic approximations; inspect the manual audit sample before treating them as paper-grade.

## Data/Evaluator Audit
- n_examples: 60
- invalid/ambiguous: 0 (0.0%)
- gold checker failures: 0 (0.0%)
- wrong-answer false negatives: 0 (0.0%)
- low-locality feedback cases: 0 (0.0%)
- low-confidence prefix metric cases: 0 (0.0%)
- low-confidence repeated-error metric cases: 0 (0.0%)

The data/evaluator audit passes, but the mechanism result does not pass the predeclared scale gate.

## Evaluator Caveat
A post-run negation-aware RAG check flagged one Mistral initial answer (`rag_02`: "The blue route does not stop at Cedar.") that the original substring RAG checker counted as correct. Because the run was failed-only, no repairs were generated for that case. This makes the already-negative TracePatch result slightly optimistic rather than hiding a scale-worthy effect.
