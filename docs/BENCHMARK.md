# Scanner benchmark

52 hand-written labeled snippets (26 leaky, 26 clean). A snippet counts as flagged when the scanner reports a high or medium finding.

| Metric | Value |
|---|---|
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| True / false positives | 26 / 0 |
| True negatives / false negatives | 26 / 0 |

Low-severity reproducibility cases (issue #13): 8/8 correct, 0 misclassified.

Misclassified:
- none

## Caveats
The set is small, hand-written and partly shaped by the scanner's own rules, so it guards against regressions; it does not measure real-world accuracy. For prevalence on real notebooks, see [STUDY.md](STUDY.md) (also a heuristic).
