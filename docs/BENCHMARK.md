# Scanner benchmark

69 hand-written labeled snippets (37 leaky, 32 clean). A snippet counts as flagged when the scanner reports a high or medium finding.

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True / false positives | 37 / 0 |
| True negatives / false negatives | 32 / 0 |

Low-severity cases (reproducibility and pre-split category detection): 11/11 correct, 0 misclassified.

Misclassified:
- none

## Caveats
The set is small, hand-written and partly shaped by the scanner's own rules, so it guards against regressions; it does not measure real-world accuracy. For prevalence on real notebooks, see [STUDY.md](STUDY.md) (also a heuristic).
