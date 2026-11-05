# AI intelligence benchmark

This benchmark measures the **optional, experimental AI layer** only. It is not an improvement claim over the deterministic scanner, which remains authoritative. All numbers below are measured by `python -m leaklens.bench_ai`.

## Protocol and data provenance

- Model: `logistic_regression` (leaklens-ai-p5-2026.11), 94 features.
- Training: 145 authored cases from 9 scenario sources.
- Validation: 40 authored cases from 4 disjoint sources.
- Locked test: 47 authored cases from 4 further disjoint sources.
- Groups: 137 template groups across the development corpus; grouped folds never split a family.
- Threshold: 0.64 (chosen on validation only).
- train=145 cases/9 sources; validation=40 cases/4 sources; locked test imported only by the benchmark; threshold rule=validation_midpoint.

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
| `gradient_boosting_depth2` | 0.880 | 1.000 | 0.000 |
| `gradient_boosting_depth1` | 0.830 | 0.930 | 0.105 |
| `random_forest_depth6` | 0.817 | 0.933 | 0.158 |
| `random_forest_unbounded` | 0.822 | 0.930 | 0.105 |
| `extra_trees_depth8` | 0.854 | 0.955 | 0.105 |
| `extra_trees_unbounded` | 0.833 | 1.000 | 0.000 |
| `hist_gradient_boosting` | 0.740 | 0.792 | 0.421 |
| `logistic_regression` **(selected)** | 0.895 | 1.000 | 0.000 |

## Development results (not the independent test)

### Grouped out-of-fold training metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.897 |
| Precision | 0.931 |
| Recall | 0.872 |
| F1 | 0.901 |
| Specificity | 0.925 |
| False-positive rate | 0.075 |
| TP / FP / FN / TN | 68 / 5 / 10 / 62 |

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
| `brier` | 0.0098 |
| `expected_calibration_error` | 0.0580 |
| `mean_probability` | 0.5575 |
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
| `brier` | 0.0229 |
| `expected_calibration_error` | 0.0356 |
| `mean_probability` | 0.5283 |
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
- Missed leak: `random split over repeated applicants` (test:credit, p=0.084; signals: log_split_position, fit_on_train_frame, target_reference, has_fit)

## External transfer to existing scanner cases

Cases: 141 (labels from the existing scanner benchmark). These cases are not used for training, calibration or threshold selection; they are a separate transfer check on different code styles.

| Metric | Value |
|---|---:|
| Accuracy | 0.879 |
| Precision | 0.913 |
| Recall | 0.851 |
| F1 | 0.881 |
| Specificity | 0.910 |
| False-positive rate | 0.090 |
| TP / FP / FN / TN | 63 / 6 / 11 / 61 |

## Threshold sweep on external cases

This sweep is diagnostic only. The shipped threshold was chosen on validation, not here.

| Threshold | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|
| 0.30 | 0.886 | 0.946 | 0.915 | 0.134 |
| 0.40 | 0.887 | 0.851 | 0.869 | 0.119 |
| 0.50 | 0.887 | 0.851 | 0.869 | 0.119 |
| 0.60 | 0.913 | 0.851 | 0.881 | 0.090 |
| 0.63 | 0.913 | 0.851 | 0.881 | 0.090 |
| 0.70 | 0.913 | 0.851 | 0.881 | 0.090 |
| 0.80 | 0.938 | 0.811 | 0.870 | 0.060 |
| 0.90 | 0.967 | 0.797 | 0.874 | 0.030 |

## Semantic-rule regression set

- Cases: 15 (11 positive, 4 clean)
- Positive-case detection: 1.000
- Expected-concept recall: 1.000
- Unexpected concepts: 0

## Top model signals (global importance)

| Feature | Importance |
|---|---:|
| `target_feature_risk` | 0.0416 |
| `group_boundary_ignored` | 0.0363 |
| `group_aggregate_before_split` | 0.0335 |
| `group_columns_used` | 0.0323 |
| `identifier_feature_risk` | 0.0318 |
| `future_feature_risk` | 0.0307 |
| `dedupe_before_split` | 0.0302 |
| `duplicates_after_split` | 0.0266 |
| `shuffle_false` | 0.0254 |
| `target_relabel_safe` | 0.0232 |
| `manual_full_data_stats` | 0.0222 |
| `unsafe_resampling` | 0.0218 |

Global importance is not a causal explanation for any single prediction.

## External false-positive analysis

- Clean case flagged: `fit inside KFold loop` (p=0.999)
- Clean case flagged: `GridSearchCV over a pipeline with raw X` (p=0.852)
- Clean case flagged: `ColumnTransformer with only one-hot before split` (p=0.734)
- Clean case flagged: `scaler fitted per fold inside the loop` (p=0.779)
- Clean case flagged: `knn imputer train only` (p=0.825)
- Clean case flagged: `positive target lag before split` (p=0.961)

Leaky cases missed:
- `scale() before split` (p=0.336)
- `divide by max before split` (p=0.336)
- `column max scaling before split` (p=0.336)
- `power_transform() on all data before split` (p=0.336)
- `dataframe mode fill before split` (p=0.184)
- `get_dummies on full data before split` (p=0.336)
- `cross-validation on the test partition` (p=0.018)
- `group CV without groups` (p=0.346)
- `repeated scoring on the test partition` (p=0.053)
- `frequency encoding from all rows before split` (p=0.008)

## Overfitting checks

- Grouped out-of-fold train metrics are lower than validation and locked-test metrics.
- Every development and locked-test case belongs to exactly one source and one template group; sources never overlap.
- Locked-test accuracy 0.979 is reported once at the frozen threshold, with no post-hoc tuning.
- Candidate architectures are compared on grouped CV and validation; the locked test never participates in selection.

## Limitations

- Trained on an authored synthetic corpus; it is not production-code evidence.
- The locked test is a small authored corpus of 47 cases, not real repositories.
- Probabilities are calibrated on 40 validation cases; treat them as approximate.
- English identifiers and Python control-flow cues may not transfer to other styles.
- External scanner transfer is a separate check, not an improvement claim.
- The corpus is authored and small; it is a regression corpus, not production evidence.
- The semantic rules are name and control-flow heuristics, not a proof of safety.
- External transfer is measured on a synthetic/scanner benchmark, not production data.
- AI hypotheses never change deterministic findings, severity or the trust score.
