# Distribution drift diagnostics

Distribution drift compares a training frame with a held-out frame and reports
columns whose distribution changed. It is **opt-in**, **low severity**,
**diagnostic** and **never part of the leakage score or the default exit
code**:

```sh
leaklens train.csv --target label --test test.csv --drift
leaklens train.csv --target label --test test.csv --drift --json
```

Drift is a reason to investigate, **not proof of leakage**. A shifted column
can come from a later time window, a different sample, a filter, a broken join
or a genuine covariate shift. LeakLens never modifies the data or the split.

## Algorithms

Implemented with pandas/numpy only; no SciPy and no new dependency.

| Signal | Method | Notes |
|---|---|---|
| Numeric columns | Two-sample Kolmogorov-Smirnov statistic | Max distance between empirical CDFs; sensitive to location, scale and shape. |
| Categorical columns | Total-variation distance | `0.5 * sum(abs(p_train - p_test))` over the top-k categories plus an aggregate "other" bucket. |
| Every column | Absolute missingness-rate difference | Catches collection or join differences. |

One `distribution_shift` finding is emitted per column, using the largest of
the three effects. The finding is `low`, marked `diagnostic: true`, and carries
aggregate evidence only (`ks=0.62`, `tvd=0.55`, `missingness=0.41`) - never raw
values, category labels or rows.

## Defaults and safeguards

| Setting | Default | Meaning |
|---|---:|---|
| `enabled` | `false` | `--drift` or `[drift].enabled`. |
| `min_rows` | `200` | Minimum rows in each split before any test runs. |
| `min_effect` | `0.20` | Minimum KS/TVD/missingness effect to report. |
| `max_rows` | `100000` | Evenly spaced deterministic subsample cap per column. |
| `max_columns` | `50` | Columns checked per audit. |
| `max_categories` | `20` | Top-k categories; the tail is aggregated into "other". |

Additional guards:

- constant columns (zero range on both sides) and fully missing columns are
  skipped;
- a few non-numeric columns that cannot be counted are skipped as
  `unsupported`;
- the target column is never compared, even when it is present in both frames;
- if there is no test frame the report is `skipped`; below `min_rows` it is
  `insufficient_data`; both emit no findings;
- calculations are deterministic: sorted samples, deterministic top-k
  tie-breaking and evenly spaced subsampling (no RNG).

## Score and exit isolation

`distribution_shift` findings have `diagnostic=True`. `leakage.score()` ignores
diagnostics, so adding `--drift` cannot lower the trust score. The default exit
gate is unchanged (a high-severity *active* finding or `--fail-under`). JSON
includes the additive `diagnostic: true` field; SARIF exposes
`properties.leaklens.diagnostic`; HTML renders a drift section only when the
audit ran.

## Configuration

```toml
[drift]
enabled = true
min_rows = 200
min_effect = 0.20
max_rows = 100000
max_columns = 50
max_categories = 20
```

API: `POST /audit` accepts an optional `test` upload and `drift=true`. The
response gains an additive `drift` report; the trust score is identical to a
run without drift.

## Interpretation

A finding is a hypothesis, not a verdict:

- **Large KS with a tiny sample** is unreliable; LeakLens requires
  `min_rows` on both sides.
- **KS grows with sample size** for any non-identical distribution. With very
  large frames, treat a small effect as weak evidence even above the floor.
- **TVD on high-cardinality columns** depends on the top-k cap; unseen
  categories are folded into "other", which deliberately hides rare labels.
- **No multiple-comparison correction** is applied. Checking 50 columns at an
  effect floor can still produce occasional noise; the effect floors keep the
  default false-positive rate low on the labeled controls.
- Drift does not identify *which* rows or categories moved, and it does not
  distinguish train-to-test drift from a train-time sampling artifact.
- Time-series and per-group drift are out of scope in this release.

See [`DRIFT_BENCHMARK.md`](DRIFT_BENCHMARK.md) for the deterministic labeled
controls and their measured numbers. They are strong synthetic cases, not a
real-world accuracy estimate.
