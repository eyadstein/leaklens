# AI intelligence benchmark

This benchmark measures the **optional, experimental AI layer** only. It is not an improvement claim over the deterministic scanner, which remains authoritative. All numbers below are measured by `python -m leaklens.bench_ai`.

## Protocol and data provenance

- Model: `logistic_regression_balanced` (leaklens-ai-p7-2026.11), 108 features.
- Training: 223 authored cases from 15 scenario sources, including 40 Phase 7 hard-positive/hard-negative development cases.
- Validation: 40 authored cases from 4 disjoint sources.
- Locked test: 47 authored cases from 4 further disjoint sources.
- Phase 7 holdout: 12 authored cases from 5 further source families, frozen before the final fit and scored once.
- Groups: 176 template groups across the development corpus; grouped folds never split a family.
- Threshold: 0.60 (chosen on validation only).
- train=223 cases/15 sources; validation=40 cases/4 sources; locked test imported only by the benchmark; candidate selection frozen after validation; threshold rule=validation_midpoint; runtime hypotheses also require at least one semantic risk indicator.

Splits are by **scenario source and template family**, not by random row. The locked test is imported only by this benchmark and its tests; training, feature selection, calibration and threshold selection never import it.

### Split integrity

- Shared sources between development and locked test: 0
- Shared groups: 0
- Exact duplicate code across splits: 0
- Maximum token Jaccard overlap: 0.765 (`centered rolling window over all stores` vs `centered rolling window over every reading`)

## Baseline comparison

Phase 4 baseline (recorded on commit `9a4cba5`, reproduced before the Phase 5 rebuild):

| Stage | Accuracy | Precision | Recall | F1 | FPR | TP / FP / FN / TN |
|---|---:|---:|---:|---:|---:|---:|
| Phase 4 grouped CV | 0.736 | 0.833 | 0.698 | 0.759 | 0.207 | 30 / 6 / 13 / 23 |
| Phase 4 external | 0.773 | 0.850 | 0.689 | 0.761 | 0.134 | 51 / 9 / 23 / 58 |
| Phase 5 locked test | 0.979 | 1.000 | 0.960 | 0.980 | 0.000 | 24 / 0 / 1 / 22 |

The comparison is reported as measured. The Phase 4 and Phase 5 test sets are different authored corpora, so the row difference is not a controlled A/B experiment.

## Candidate architectures

Selection uses grouped CV on training cases followed by validation, with the same calibration and threshold rule for every candidate.

| Candidate | CV F1 | Validation F1 | Validation FPR |
|---|---:|---:|---:|
| `gradient_boosting_depth2` | 0.911 | 0.977 | 0.053 |
| `gradient_boosting_depth1` | 0.850 | 0.977 | 0.053 |
| `random_forest_depth6` | 0.845 | 0.930 | 0.105 |
| `random_forest_unbounded` | 0.842 | 0.952 | 0.053 |
| `extra_trees_depth8` | 0.875 | 0.930 | 0.105 |
| `extra_trees_unbounded` | 0.861 | 0.976 | 0.000 |
| `hist_gradient_boosting` | 0.803 | 0.766 | 0.421 |
| `logistic_regression` | 0.931 | 1.000 | 0.000 |
| `logistic_regression_l1` | 0.906 | 1.000 | 0.000 |
| `logistic_regression_balanced` **(selected)** | 0.931 | 1.000 | 0.000 |
| `random_forest_depth4` | 0.798 | 0.864 | 0.210 |

## Development results (not the independent test)

### Grouped out-of-fold training metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.937 |
| Precision | 0.981 |
| Recall | 0.896 |
| F1 | 0.936 |
| Specificity | 0.982 |
| False-positive rate | 0.018 |
| TP / FP / FN / TN | 103 / 2 / 12 / 106 |

### Validation metrics

| Metric | Value |
|---|---:|
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| Specificity | 1.000 |
| False-positive rate | 0.000 |
| TP / FP / FN / TN | 21 / 0 / 0 / 19 |

Validation is used for architecture selection and threshold choice. Because the authored validation sources are cleanly separable, its metrics are optimistic and are not the headline result.

### Calibration on validation

| Metric | Value |
|---|---:|
| `brier` | 0.0017 |
| `expected_calibration_error` | 0.0158 |
| `mean_probability` | 0.5368 |
| `observed_rate` | 0.5250 |

## Locked independent test (evaluated once)

The frozen model and frozen threshold are applied to 47 cases from four scenario sources that never appear in training or validation.

| Metric | Value |
|---|---:|
| Accuracy | 0.979 |
| Precision | 1.000 |
| Recall | 0.960 |
| F1 | 0.980 |
| Specificity | 1.000 |
| False-positive rate | 0.000 |
| TP / FP / FN / TN | 24 / 0 / 1 / 22 |

### Locked-test calibration

| Metric | Value |
|---|---:|
| `brier` | 0.0212 |
| `expected_calibration_error` | 0.0317 |
| `mean_probability` | 0.5293 |
| `observed_rate` | 0.5319 |

### Locked-test metrics by category

| Category | Cases | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| `grouping` | 6 | 1.000 | 0.667 | 0.800 | 0.000 |
| `preprocessing` | 10 | 1.000 | 1.000 | 1.000 | 0.000 |
| `resampling` | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| `split` | 6 | 1.000 | 1.000 | 1.000 | 0.000 |
| `target_derived` | 4 | 1.000 | 1.000 | 1.000 | 0.000 |
| `target_encoding` | 7 | 1.000 | 1.000 | 1.000 | 0.000 |
| `target_leakage` | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| `temporal` | 7 | 1.000 | 1.000 | 1.000 | 0.000 |
| `tuning` | 3 | 1.000 | 1.000 | 1.000 | 0.000 |

### Locked-test metrics by unseen source

| Source | Cases | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| `test:credit` | 12 | 1.000 | 0.857 | 0.923 | 0.000 |
| `test:marketing` | 12 | 1.000 | 1.000 | 1.000 | 0.000 |
| `test:sensor` | 11 | 1.000 | 1.000 | 1.000 | 0.000 |
| `test:support` | 12 | 1.000 | 1.000 | 1.000 | 0.000 |

### Locked-test mistakes

- No false positives on the locked test.
- Missed leak: `random split over repeated applicants` (test:credit, p=0.054; signals: fit_on_train_frame, train_only_fit, safe_boundary_ratio, target_reference)

## Phase 7 frozen holdout (authored, evaluated once)

Authored synthetic pairs, frozen before the final fit; grouped by template pair and source family; never used for training or tuning.

Cases: 12 (6 leaky, 6 legitimate) from 5 source families. Threshold: 0.60, frozen before the final fit.

| Metric | Value |
|---|---:|
| Accuracy | 0.917 |
| Precision | 0.857 |
| Recall | 1.000 |
| F1 | 0.923 |
| Specificity | 0.833 |
| False-positive rate | 0.167 |
| TP / FP / FN / TN | 6 / 1 / 0 / 5 |

### Phase 7 holdout calibration

| Metric | Value |
|---|---:|
| `brier` | 0.0938 |
| `expected_calibration_error` | 0.1339 |
| `mean_probability` | 0.6339 |
| `observed_rate` | 0.5000 |

### Phase 7 holdout by source family

| Family | Cases | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| `holdout:p7-groups` | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| `holdout:p7-order` | 4 | 0.667 | 1.000 | 0.800 | 0.500 |
| `holdout:p7-stats` | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| `holdout:p7-target` | 2 | 1.000 | 1.000 | 1.000 | 0.000 |
| `holdout:p7-temporal` | 2 | 1.000 | 1.000 | 1.000 | 0.000 |

### Phase 7 holdout mistakes

- False positive: `scaler fitted on each training fold` (holdout:p7-order, p=0.973; local signals: fit_before_split (+1.44), unsafe_preprocessing (+1.12), transformer_before_split (+1.12), unsafe_transformer_count_before_split (+1.07))
- No missed leaks on the Phase 7 holdout.

## Runtime hypothesis gate (development)

Hypothesis-level output. A leaky unit with no emitted hint is still scored by the model; it is suppressed because no semantic risk indicator was active.

| Metric | Value |
|---|---:|
| Accuracy | 0.989 |
| Precision | 1.000 |
| Recall | 0.978 |
| F1 | 0.989 |
| Specificity | 1.000 |
| False-positive rate | 0.000 |
| TP / FP / FN / TN | 133 / 0 / 3 / 127 |

- Units with an emitted hint: 133 of 263
- Units scored but suppressed: 130

No hint was emitted on a legitimate development case.

## Local attribution example

Phase 7 authored development pair; identical transformer, only the position relative to the fold boundary differs.

Contributions are signed local log-odds for this unit, not global feature importance and not a causal claim.

| Variant | Label | p | Predicted | Supporting local signals | Opposing local signals |
|---|---|---:|---|---|---|
| `scaler fitted before a manual fold loop` | leaky | 0.996 | True | `plain_kfold` (+3.05), `fit_before_split` (+1.44), `transformer_before_split` (+1.12), `unsafe_preprocessing` (+1.12) | `fold_local_fit` (-3.41) |
| `scaler fitted inside a manual fold loop` | legitimate | 0.070 | False | `plain_kfold` (+3.05), `has_shuffle` (+0.70), `target_reference` (+0.35) | `fold_local_fit` (-3.41), `scaler` (-0.30) |

## External transfer to existing scanner cases (development)

Cases: 143 (labels from the existing scanner benchmark). These cases are not used for fitting, calibration or threshold selection. **They were used for Phase 6 error analysis**, so the current perfect score is a development result, not independent evidence. The frozen real-world holdout, not this set, is the generalization check.

| Metric | Value |
|---|---:|
| Accuracy | 0.993 |
| Precision | 0.987 |
| Recall | 1.000 |
| F1 | 0.993 |
| Specificity | 0.985 |
| False-positive rate | 0.015 |
| TP / FP / FN / TN | 75 / 1 / 0 / 67 |

### Phase 5 external errors and the Phase 6 response

Phase 5 recorded 6 false positives and 10 named misses on this set (the task summary counts 11 misses; only 10 were named in the report). Phase 6 added paired authored cases and explicit indicators for each family:

| Phase 5 error family | Phase 6 response |
|---|---|
| FP: fit inside a KFold loop, scaler fitted per fold | `fold_local_fit`, `train_only_transformer_fit` |
| FP: GridSearchCV over a pipeline with raw `X` | `search_over_pipeline_safe` |
| FP: ColumnTransformer with only one-hot before split | `onehot_only_full_data` |
| FP: KNN imputer fitted on train only | `train_only_transformer_fit` |
| FP: positive target lag before split | `target_lag_safe` |
| FN: `scale()` / `normalize()` / `power_transform()` before split | `functional_scaling_before_split` |
| FN: hand-written divide-by-max / column min-max | `manual_scaling_before_split` |
| FN: full-frame `fillna(mode/mean)` | `full_data_imputation_before_split` |
| FN: `get_dummies` on the full frame | `get_dummies_before_split` |
| FN: CV or repeated scoring on the test partition | `cv_on_test_partition`, `repeated_test_scoring` |
| FN: GroupKFold without `groups` | `group_kfold_without_groups` |
| FN: frequency encoding from all rows | `frequency_encoding_before_split` |

The response is evaluated on the set that motivated it, so it cannot be read as an accuracy claim. The Phase 6 real-world holdout exposes where the new indicators still over-fire.

## Threshold sweep on external cases

This sweep is diagnostic only. The shipped threshold was chosen on validation, not here.

| Threshold | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|
| 0.30 | 0.974 | 1.000 | 0.987 | 0.029 |
| 0.40 | 0.974 | 1.000 | 0.987 | 0.029 |
| 0.50 | 0.987 | 1.000 | 0.993 | 0.015 |
| 0.60 | 0.987 | 1.000 | 0.993 | 0.015 |
| 0.63 | 0.987 | 1.000 | 0.993 | 0.015 |
| 0.70 | 0.987 | 1.000 | 0.993 | 0.015 |
| 0.80 | 0.987 | 1.000 | 0.993 | 0.015 |
| 0.90 | 0.987 | 0.987 | 0.987 | 0.015 |

## Semantic-rule regression set

- Cases: 15 (11 positive, 4 clean)
- Positive-case detection: 1.000
- Expected-concept recall: 1.000
- Unexpected concepts: 0

## Top model signals (global importance)

| Feature | Importance |
|---|---:|
| `group_boundary_ignored` | 0.0364 |
| `target_feature_risk` | 0.0353 |
| `future_feature_risk` | 0.0308 |
| `group_aggregate_before_split` | 0.0288 |
| `identifier_feature_risk` | 0.0274 |
| `target_relabel_safe` | 0.0240 |
| `cv_on_full_data` | 0.0222 |
| `fit_before_split` | 0.0209 |
| `duplicates_after_split` | 0.0203 |
| `manual_full_data_stats` | 0.0199 |
| `group_columns_used` | 0.0195 |
| `frequency_encoding_before_split` | 0.0193 |

Global importance is not a causal explanation for any single prediction.

## External false-positive analysis

- Clean case flagged: `derived copy and func fitted after split on train` (p=0.998)

No leaky external cases were missed at the selected threshold.

## M6 false-positive reduction

Four deterministic extraction guards replace shape-only signals with boundary-aware ones. They
change emitted features, not the model class or the feature catalogue (108 model features).

| Family | Guard | Development hard negative | H1 FPs before → after |
|---|---|---|---:|
| Safe CV | `plain_cv_safe` emits `cv_full_data_safe` and drops `cv_on_full_data`, `plain_kfold`, `has_shuffle` only when nothing is fitted outside the folds, no time/group/identifier feature is used, and no group-aware splitter is present | `plain fold local cross validation`; `plain cross validation inside an objective` | 2 → 0 |
| Temporal split | `sort_values(time)` + `.iloc[:cut]` / `.iloc[cut:]` is a split boundary; label carriers after it are exempt | `positional time split into catboost pools` | 2 → 0 |
| CatBoost `Pool` / label carriers | `label=`, target-vector arguments and `Pool(...)` targets are carriers, exempt only when no boundary exists or they are built after it | existing safe pool cases; `positional time split into catboost pools` | 1 → 0 |
| EDA-only | A full-frame fit with no split, CV or held-out scoring that only feeds a plot is `eda_summary_only` | `scaling for a diagnostic scatter` | 1 → 0 |

Two development snippets (`cross validation across time`, `kfold over accounts`) were completed
with the time and account columns their labels already depended on; their labels did not change.
The development corpus grew from 259 to 263 cases (127 legitimate, 176 groups); the validation
slice is unchanged at TP/FP/FN/TN 21/0/0/19. The runtime hypothesis gate on the development set
moved from 131/0/5/123 to 133/0/3/127, recovering `kfold over accounts` (identifier used as a CV
feature) and `borderline smote before cross validation` (prefixed resampler name); it still emits
no hint on a legitimate development case.

The H1 diagnostic moved from TP/FP/FN/TN 22/6/2/18 to 22/0/2/24: precision 0.7857 → 1.0000,
recall 0.9167 → 0.9167, F1 0.8462 → 0.9565, accuracy 0.8333 → 0.9583. The locked test is
unchanged at 24/0/1/22 (accuracy 0.9787, precision 1.000) and the Phase 7 holdout at 6/1/0/5.
The unchanged validation-only rule now selects threshold **0.60** (it was 0.53); the rule itself
was not edited.

Known limitation: `xgboost_early_stopping_on_test` previously scored as leaky only because
`label=y_train` was misread as a target feature. The carrier guard removes that spurious signal,
leaving a genuine false negative with no existing early-stopping indicator. Recall on H1 is
unchanged at 0.9167. Adding such an indicator is deferred because M6 must not tune on H1.

## Selective reporting

- Decision: applied at p >= 0.923.
- Coverage: 95.3% of emitted hypotheses (122 of 128); 46.4% of all development units.
- Empirical selective risk: 0.008 (cluster-bootstrap 95% interval 0.000-0.026).
- Unselective error rate: 0.053.
- Smallest threshold whose Wilson 95% upper bound on selective risk is at or below the unselective error rate, requiring at least 50% coverage of emitted units; fitted on grouped out-of-fold development probabilities.
- Measured on grouped out-of-fold development probabilities from the authored corpus, not on an independent corpus.
- Repository-level shift violates the exchangeability assumption behind selective-risk bounds; no distribution-free guarantee is claimed.
- Hypotheses below the selective threshold stay visible and are labelled insufficient (ranking only); they are not suppressed.

## Overfitting checks

- Grouped out-of-fold train metrics are lower than validation and locked-test metrics.
- Every development and locked-test case belongs to exactly one source and one template group; sources never overlap.
- Locked-test accuracy 0.979 is reported once at the frozen threshold, with no post-hoc tuning.
- Candidate architectures are compared on grouped CV and validation; the locked test never participates in selection.

## Limitations

- Trained on an authored synthetic corpus: Phase 5/6 families plus 40 Phase 7 hard-positive/hard-negative cases and 34 Phase 6 idiom cases.
- The Phase 5 locked test is a small authored corpus of 47 cases, not real repositories.
- The Phase 7 holdout is 12 authored synthetic units frozen before the final fit; it is a regression check, not production evidence.
- The Phase 6 real-world holdout is 48 reviewed units from 12 public repositories; 44 of them are reconstructions, not verbatim upstream files.
- Probabilities are calibrated on 40 validation cases; treat them as approximate.
- English identifiers and Python control-flow cues may not transfer to other styles.
- External scanner transfer and the real-world holdout are checks, not improvement claims.
- The corpus is authored and small; it is a regression corpus, not production evidence.
- The Phase 7 holdout is 12 authored synthetic units frozen before the final fit; it is a regression check, not production evidence.
- The semantic rules are name and control-flow heuristics, not a proof of safety.
- External transfer is measured on a synthetic/scanner benchmark, not production data.
- AI hypotheses never change deterministic findings, severity or the trust score.
