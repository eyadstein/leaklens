# Real-world notebook validation

Evaluated 60 public notebooks from `realworld/` against hand-reviewed labels (`realworld/labels.json`). Only 8 labels are scored (see below); the other 52 are reported but excluded from precision/recall/F1.

High/medium findings are the prediction; low findings (no_random_state, parse_error, get_dummies_before_split) are noise and not counted.

| Metric | Value |
|---|---:|
| Notebooks evaluated | 60 |
| Definitive labels scored | 8 (5 confirmed_leakage + 3 legitimate) |
| Notebooks evaluated but excluded from scoring | 52 (2 potential_leakage + 50 uncertain) |
| True positives | 5 |
| False positives | 0 |
| False negatives | 0 |
| True negatives | 3 |
| Precision (on the 8 scored labels) | 1.00 |
| Recall (on the 8 scored labels) | 1.00 |
| F1 (on the 8 scored labels) | 1.00 |

Only 8 notebooks have definitive reviewed labels: 5 `confirmed_leakage` and 3 `legitimate`. The remaining 52 (2 `potential_leakage` + 50 `uncertain`) are excluded from precision/recall/F1 — they are not ground truth, and counting them would inflate or deflate the score. The 8 reviewed labels are a very small sample; treat the resulting metrics as a smoke check, not a population accuracy estimate.

## Reviewed-label breakdown (scored)

| Verdict | Notebooks | Flagged with high/medium | Role in scoring |
|---|---:|---:|---|
| confirmed_leakage | 5 | 5 | positive ground truth |
| potential_leakage | 2 | 0 | excluded (not scored) |
| legitimate | 3 | 0 | negative ground truth |
| uncertain | 50 | 1 | excluded (not scored) |

## Excluded notebooks (reported, not scored)

2 `potential_leakage` and 50 `uncertain` notebooks are listed in the
table above so the full 60-notebook corpus stays transparent. One
`uncertain` notebook does carry a high/medium finding, but it is **not**
counted as a false positive — without a hand-confirmed verdict it
cannot be scored. `potential_leakage` notebooks are plausible but
unproven (commented-out code, separate train/test files, unused
transforms) and are likewise excluded.

## By verdict class (full corpus, for context)

| Verdict | Notebooks | Flagged with high/medium |
|---|---:|---:|
| confirmed_leakage | 5 | 5 |
| potential_leakage | 2 | 0 |
| legitimate | 3 | 0 |
| uncertain | 50 | 1 |

## Methodology and caveats

- Notebooks come from GitHub Code Search with the same queries as `scripts/study.py` plus groupby/shift/rolling queries. The sample is not random, so results are not a population prevalence estimate.
- Verdicts were hand-reviewed from static inspection of notebook code; no notebook was executed. Uncertain cases are kept separate and are not counted against the scanner.
- This real-world corpus is **kept separate** from the synthetic `leaklens/bench.py` benchmark (different directory, different scoring).
- Scanner findings are high/medium only; low-severity noise (unseeded splits, parse errors, get_dummies) is excluded from precision/recall.
- Regenerate this file with `python -m leaklens.realworld_eval --write docs/REALWORLD.md`
  (do not edit the numbers by hand — edit `realworld/labels.json` instead).
