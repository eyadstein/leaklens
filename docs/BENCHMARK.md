# Scanner benchmark

The benchmark contains the original regression baseline and an expanded set of categorized cases.

## Combined benchmark

| Metric | Value |
|---|---:|
| Cases | 105 |
| Leaky / clean | 55 / 50 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 55 |
| False positives | 0 |
| True negatives | 50 |
| False negatives | 0 |

## Original regression baseline

| Metric | Value |
|---|---:|
| Cases | 69 |
| Leaky / clean | 37 / 32 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 37 |
| False positives | 0 |
| True negatives | 32 |
| False negatives | 0 |

## Additional categorized cases

| Metric | Value |
|---|---:|
| Cases | 36 |
| Leaky / clean | 18 / 18 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True positives | 18 |
| False positives | 0 |
| True negatives | 18 |
| False negatives | 0 |

## Category breakdown

| Category | Cases | Leaky | Clean | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| encoding | 1 | 1 | 0 | 1.00 | 1.00 | 1.00 |
| future_target | 4 | 2 | 2 | 1.00 | 1.00 | 1.00 |
| imputation | 15 | 7 | 8 | 1.00 | 1.00 | 1.00 |
| scaling | 16 | 8 | 8 | 1.00 | 1.00 | 1.00 |

## Reproducibility checks

Low-severity reproducibility cases (issue #13): 8/8 correct; 0 misclassified.

## Misclassified cases

- none

## Caveats

These cases are hand-written regression tests, not a representative sample of real-world notebooks. Some examples are deliberately shaped around supported scanner rules. A perfect score therefore does not establish real-world precision or recall. Unsupported patterns should be tracked separately as known detection gaps rather than counted as successful positive cases.
