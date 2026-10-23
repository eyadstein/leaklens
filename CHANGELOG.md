# Changelog

## Unreleased
- Excel, JSON and Feather input alongside CSV and Parquet, via the new `excel` and `feather` extras for the optional formats

## 0.8.0
- Dataflow tracking: names derived from the dataset passed to train_test_split are followed, so a transformer fitted on the full data after the split is reported (fit_on_full_data, medium)
- Detects chained calls such as StandardScaler().fit_transform(X)
- SARIF 2.1.0 output (`--sarif FILE`) for GitHub code scanning
- Benchmark grows to 35 cases

## 0.7.1
- Labeled scanner benchmark (`python -m leaklens.bench`, results in docs/BENCHMARK.md), run in CI
- Scanner now recognizes SelectKBest / SelectPercentile / SelectFromModel fitted before the split
- SECURITY.md, CODE_OF_CONDUCT.md, Dependabot, PR template

## 0.7.0
- `--explain`: suggested fix under each finding

## 0.6.1
- Detects function-style `scale()`, `minmax_scale()`, `power_transform()` before the split
- `ColumnTransformer` before the split reported as medium severity

## 0.6.0
- Parquet input (thanks @jay23606) and `--fail-under` CI gate (thanks @slopggobblercobbler81)
- Encoders before the split downgraded to low severity

## 0.5.0
- Shareable single-file HTML report (`--html`)

## 0.4.0
- PyPI packaging (`leaklens-ml`), release workflow, pre-commit hook, multi-file `--scan`, group/entity leakage (`--group`)

## 0.3.0
- FastAPI service, Streamlit UI, Docker, GitHub Action, trust-score badge

## 0.2.0
- Model arena with MLflow tracking

## 0.1.0
- Data leakage detectors, static code scanner, CLI