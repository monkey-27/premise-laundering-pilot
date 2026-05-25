# Scope Radius Counterfactuals Audit Note

This note records the audit of `scope_radius_counterfactuals` after the `scope-radius-20260524-qwen-small-v2` run.

## What Was Wrong

- The delayed-action probes mixed yes/no wording with subset answers. For example, asking whether all local files can be deleted while scoring `yes_unaffected_only` as gold made `no` a reasonable model answer but a scored failure.
- Closed-set labels such as `yes_unaffected_only`, `yes_all`, and `only_for_unaffected_cases` created label artifacts.
- `source_status` was used as a scope radius even though it was really a domain/correction-type concept.
- The tool-agent stale-status case incorrectly included `gamma_upload` in `gold_should_change`, even though the feedback said the actual upload succeeded.
- The prompt exposed only `gold_should_change ∪ gold_should_preserve` as allowed state keys, which made update-set prediction too easy and hid distractor sensitivity.
- The contrast metric only checked whether predicted sets changed when gold sets changed. Random changes could count as contrast sensitivity.
- `stale_reversion_rate` only captured exact old-answer reuse, while missing over-conservative, under-scoped, and wrong-boundary stale-conflict failures.
- `full_regeneration` still saw old-trace content in p5, so stale metrics for that condition were not clean.
- Qualitative “success” examples could be selected even when direct correction failed or boundary leak was true.

## What Was Fixed

- Delayed-action probes now ask selection questions:
  - `Which local files can be deleted now?`
  - `Which cases can be finalized now?`
  - `Which citation/use is valid now?`
- Action labels now use simple closed sets such as `all`, `unaffected_only`, `none`, `full_use`, `limited_use`, and `no_use`.
- `scope_radius` is now one of `local_entity`, `entity_class`, `temporal_boundary`, `rule_boundary`.
- `correction_type` is a separate axis, with values such as `upload_state`, `archive_status`, `policy_exception`, `source_validity`, `population_validity`, `causal_validity`, and `dosage_validity`.
- Each task now exposes a broader `state_inventory`, including distractors, rather than just gold update/preserve keys.
- The archive-status stale-check case no longer marks `gamma_upload` as changed.
- Full-regeneration p5 no longer injects old-trace wording; exact stale-reuse metrics are not treated as applicable for that condition.
- Directional contrast now scores added and removed delta sets with precision, recall, and F1.
- Stale-conflict diagnostics now separate:
  - `exact_old_answer_reuse`
  - `over_conservative_after_correction`
  - `under_scoped_after_correction`
  - `wrong_boundary_after_correction`
  - `stale_conflict_probe_accuracy`
- Validation now checks state-inventory consistency, change/preserve overlap, p4 wording, source/status semantic bugs, base-context variant diversity, qualitative success criteria, and a synthetic directional-contrast sanity check.

## Status Of Old V2 Metrics

The old `scope-radius-20260524-qwen-small-v2` metrics are not valid as final evidence. They are useful as a debugging artifact, but they are distorted by misworded delayed-action probes, an entangled scope-radius ontology, incomplete stale metrics, and an invalid contrast metric.

The rerun after these fixes should be treated as the first interpretable version of the pilot.
