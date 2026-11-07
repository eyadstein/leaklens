# Missingness benchmark

Synthetic, deterministic cases for the split-aware missingness audit. Times are wall-clock on the machine that generated this file.

## Metrics

| Metric | Value |
|---|---:|
| Cases | 10 |
| Leaky / clean | 6 / 4 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| False-positive rate | 0.00 |
| True positives | 6 |
| False positives | 0 |
| True negatives | 4 |
| False negatives | 0 |
| Total runtime | 14 ms |
| Median case | 1 ms |
| Slowest case | 4 ms |

## Cases

| Case | Category | Leaky | Flagged | Findings | ms |
|---|---|:---:|:---:|---|---:|
| binary target association | target | yes | yes | missingness_target_signal | 4.3 |
| string target association | target | yes | yes | missingness_target_signal | 1.4 |
| numeric target association | target | yes | yes | missingness_target_signal | 0.9 |
| train and test missingness differ | split | yes | yes | missingness_split_shift | 1.5 |
| feature entirely absent from train | split | yes | yes | missingness_split_shift | 1.4 |
| group-associated missingness | group | yes | yes | missingness_group_signal | 1.6 |
| ordinary random missingness | clean | no | no | - | 1.5 |
| small sample safeguard | clean | no | no | - | 0.2 |
| constant and entirely missing columns | clean | no | no | - | 0.8 |
| shift without test metadata | clean | no | no | - | 0.8 |

## Misclassified cases

- none

## Caveats

These are hand-written synthetic controls with deliberately strong signals; they do not estimate real-world precision or recall. The detector reports patterns that merit investigation, not confirmed leaks, and it never modifies the data or the split.
