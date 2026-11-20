# Scanner benchmark

The benchmark contains the original regression baseline and an expanded set of categorized cases.

## Combined benchmark

| Metric | Value |
|---|---:|
| Cases | 117 |
| Leaky / clean | 63 / 54 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 63 |
| False positives | 0 |
| True negatives | 54 |
| False negatives | 0 |

## Original regression baseline

| Metric | Value |
|---|---:|
| Cases | 73 |
| Leaky / clean | 40 / 33 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 40 |
| False positives | 0 |
| True negatives | 33 |
| False negatives | 0 |

## Additional categorized cases

| Metric | Value |
|---|---:|
| Cases | 44 |
| Leaky / clean | 23 / 21 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 23 |
| False positives | 0 |
| True negatives | 21 |
| False negatives | 0 |

## Advanced detection cases

These cases cover detection rules added after the original regression baseline.

| Metric | Value |
|---|---:|
| Cases | 32 |
| Leaky / clean | 16 / 16 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 16 |
| False positives | 0 |
| True negatives | 16 |
| False negatives | 0 |

## Category breakdown

| Category | Cases | Leaky | Clean | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| encoding | 1 | 1 | 0 | 1.00 | 1.00 | 1.00 |
| future_target | 10 | 6 | 4 | 1.00 | 1.00 | 1.00 |
| imputation | 15 | 7 | 8 | 1.00 | 1.00 | 1.00 |
| scaling | 18 | 9 | 9 | 1.00 | 1.00 | 1.00 |

## Advanced category breakdown

| Category | Cases | Leaky | Clean | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| encoding | 9 | 4 | 5 | 1.00 | 1.00 | 1.00 |
| feature_selection | 5 | 3 | 2 | 1.00 | 1.00 | 1.00 |
| group_cv | 2 | 1 | 1 | 1.00 | 1.00 | 1.00 |
| temporal | 4 | 2 | 2 | 1.00 | 1.00 | 1.00 |
| tuning | 12 | 6 | 6 | 1.00 | 1.00 | 1.00 |

## Reproducibility checks

Low-severity reproducibility cases (issue #13): 19/19 correct; 0 misclassified.

## Misclassified cases

- none

## Caveats

These cases are hand-written regression tests, not a representative sample of real-world notebooks. Some examples are deliberately shaped around supported scanner rules. A perfect score therefore does not establish real-world precision or recall. Unsupported patterns should be tracked separately as known detection gaps rather than counted as successful positive cases.
