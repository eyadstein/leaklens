# Dataset detector benchmark

Synthetic, deterministic cases for the dataset-level checks. Times are wall-clock on the machine that generated this file.

## Metrics

| Metric | Value |
|---|---:|
| Cases | 12 |
| Leaky / clean | 9 / 3 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 9 |
| False positives | 0 |
| True negatives | 3 |
| False negatives | 0 |
| Total runtime | 174 ms |
| Median case | 3 ms |
| Slowest case | 46 ms |

## Cases

| Case | Category | Leaky | Flagged | Findings | ms |
|---|---|:---:|:---:|---|---:|
| clean numeric frame | clean | no | no | - | 46.2 |
| target copied into a feature | target | yes | yes | target_leakage | 40.3 |
| identifier column | identifier | yes | yes | id_like | 38.9 |
| high-cardinality categorical target encoding | categorical | yes | yes | categorical_target_leakage | 31.2 |
| identical feature rows with conflicting labels | duplicates | yes | yes | feature_overlap | 8.3 |
| case/whitespace near duplicates | duplicates | yes | yes | near_duplicate_overlap | 4.6 |
| exact rows in both splits | duplicates | yes | yes | split_overlap | 1.0 |
| disjoint splits | clean | no | no | - | 2.0 |
| entity repeated across rows | group | yes | yes | group_leakage_risk | 0.5 |
| entity in both splits | group | yes | yes | group_overlap | 0.2 |
| future rows before earlier rows | temporal | yes | yes | temporal_leakage | 0.6 |
| chronological split | temporal | no | no | - | 0.3 |

## Misclassified cases

- none

## Caveats

These are hand-written synthetic controls, not a sample of real datasets. They guard against regressions and make runtime visible; they do not establish real-world precision or recall.
