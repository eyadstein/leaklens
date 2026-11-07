# Drift benchmark

Deterministic synthetic train/test pairs for the opt-in drift audit. Times are wall-clock on the machine that generated this file.

## Metrics

| Metric | Value |
|---|---:|
| Cases | 10 |
| Leaky / clean | 4 / 6 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| False-positive rate | 0.00 |
| True positives | 4 |
| False positives | 0 |
| True negatives | 6 |
| False negatives | 0 |
| Total runtime | 12 ms |
| Median case | 1 ms |
| Slowest case | 3 ms |

## Cases

| Case | Category | Leaky | Flagged | Status | Findings | ms |
|---|---|:---:|:---:|---|---|---:|
| numeric shift | numeric | yes | yes | ok | distribution_shift | 2.8 |
| categorical shift | categorical | yes | yes | ok | distribution_shift | 2.3 |
| missingness shift | missingness | yes | yes | ok | distribution_shift | 0.9 |
| high cardinality | categorical | yes | yes | ok | distribution_shift | 2.1 |
| identical distributions | clean | no | no | ok | - | 1.2 |
| small sample safeguard | clean | no | no | insufficient_data | - | 0.3 |
| constant column | clean | no | no | ok | - | 0.5 |
| all-missing column | clean | no | no | ok | - | 0.6 |
| target column ignored | clean | no | no | ok | - | 0.8 |
| without test metadata | clean | no | no | skipped | - | 0.1 |

## Misclassified cases

- none

## Caveats

These are hand-picked synthetic controls with strong effects; they do not estimate real-world precision or recall. Drift is a diagnostic, not a leakage finding: it is low severity, excluded from the trust score and the default exit code, and never modifies the data.
