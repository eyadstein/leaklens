# AI intelligence benchmark

This benchmark measures the **optional, experimental AI layer** only. It is not an improvement claim over the deterministic scanner, which remains authoritative. All numbers below are measured by `python -m leaklens.bench_ai`.

## Learned detector: grouped cross-validation

| Metric | Value |
|---|---:|
| Corpus cases | 72 |
| Leaky / legitimate | 43 / 29 |
| Template groups | 26 |
| Folds | 5 |
| Fixed threshold | 0.65 |
| Precision | 0.833 |
| Recall | 0.698 |
| F1 | 0.759 |
| False-positive rate | 0.207 |
| True positives / false positives | 30 / 6 |
| False negatives / true negatives | 13 / 23 |

Template families are grouped, so no family appears in both training and validation folds. The threshold is fixed at 0.65 (chosen once on grouped out-of-fold predictions) so shipped behavior stays stable across supported scikit-learn versions. Metrics can vary slightly between scikit-learn releases because the underlying tree ensemble is not bit-identical.

## External transfer to existing scanner cases

Cases: 141 (labels from the existing scanner benchmark).

| Metric | Value |
|---|---:|
| Precision | 0.850 |
| Recall | 0.689 |
| F1 | 0.761 |
| False-positive rate | 0.134 |
| Accuracy | 0.773 |

External cases are **not** used for training or threshold selection. They are a transfer check, and the measured false-positive rate is a real limitation.

## Threshold sweep on external cases

| Threshold | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|
| 0.3 | 0.556 | 0.932 | 0.697 | 0.821 |
| 0.4 | 0.547 | 0.784 | 0.644 | 0.716 |
| 0.5 | 0.576 | 0.770 | 0.659 | 0.627 |
| 0.6 | 0.850 | 0.689 | 0.761 | 0.134 |
| 0.7 | 0.848 | 0.676 | 0.752 | 0.134 |
| 0.8 | 0.848 | 0.676 | 0.752 | 0.134 |
| 0.9 | 0.860 | 0.662 | 0.748 | 0.119 |

## Semantic-rule regression set

- Cases: 15 (11 positive, 4 clean)
- Positive-case detection: 1.000
- Expected-concept recall: 1.000
- Unexpected concepts: 0

## Top model signals (global importance)

| Feature | Importance |
|---|---:|
| `train_only_fit` | 0.3301 |
| `log_split_position` | 0.3231 |
| `shuffle_false` | 0.1305 |
| `pipeline_boundary` | 0.0528 |
| `pipeline` | 0.0382 |
| `has_shuffle` | 0.0372 |
| `group_kfold` | 0.0204 |
| `has_split` | 0.0139 |
| `group_boundary_ignored` | 0.0137 |
| `describe` | 0.0105 |
| `split_count` | 0.0097 |
| `identifier_feature_risk` | 0.0092 |

Global importance is not a causal explanation for any single prediction.

## False-positive analysis

- Clean case flagged: `label encoder before split (benign)` (p=0.902)
- Clean case flagged: `one-hot before split (benign)` (p=0.902)
- Clean case flagged: `groupby agg summary before split (EDA, not a feature)` (p=0.902)
- Clean case flagged: `statistics from a different frame` (p=0.902)
- Clean case flagged: `ColumnTransformer with only one-hot before split` (p=0.902)
- Clean case flagged: `iterative imputer train only` (p=0.902)
- Clean case flagged: `target relabelled before split is harmless` (p=0.902)
- Clean case flagged: `frequency summary not used as a feature` (p=0.838)
- Clean case flagged: `sorting an unrelated column before split` (p=0.902)

Leaky cases missed:
- `scaler fitted on test data` (p=0.380)
- `scaler before KFold` (p=0.380)
- `fillna with full-data mean before split` (p=0.574)
- `fillna with full-data median before split` (p=0.574)
- `fit then transform before cross-validation` (p=0.349)
- `derived copy fitted after split` (p=0.577)
- `scaler fitted on all data then cross_val_score` (p=0.380)
- `SMOTE on all data then cross_val_score` (p=0.380)
- `scaler on all data then GridSearchCV.fit` (p=0.349)
- `GridSearchCV fitted on test data` (p=0.349)

## Limitations

- Trained on a small synthetic corpus of Python patterns, not production code.
- Learns surface syntax and English identifiers; other languages and naming styles may fail.
- The probability is uncalibrated and should not be read as a false-positive guarantee.
- External transfer is measured on the existing scanner benchmark, not claimed as an improvement.
- Applies to Python code only; dataset metadata is handled by deterministic semantic rules.
- The semantic rules are name and control-flow heuristics, not a proof of safety.
- External transfer is measured on a synthetic/scanner benchmark, not production data.
- AI hypotheses never change deterministic findings, severity or the trust score.
