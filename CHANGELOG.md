# Changelog

## Unreleased
- Excel, JSON and Feather input alongside CSV and Parquet, via the new `excel` and `feather` extras for the optional formats
- Store prompt templates as files and support user overrides via CLI, environment variable, or .leaklens/prompt.txt (#256)

## 0.12.0
- Add opt-in LLM explanations for findings (#30, thanks @Vamsi-2581)

## 0.11.0
- Dataflow now follows transformed data into later splits, even after an earlier split (re-run notebook cells): correct kind and severity instead of "fit_on_full_data"
- `ColumnTransformer` is classified by its steps (one-hot only is low severity)
- `permutation_test_score` covered; repeated cross-validation calls are grouped per transform
- Hand-checked 5 real notebooks: docs/VALIDATION.md; benchmark grows to 47 cases

## 0.10.0
- New detector: preprocessing or resampling applied to all rows before `cross_val_score` / `cross_validate` / `GridSearchCV` (`preprocess_outside_cv`, `resample_outside_cv`)
- README: fixed curly quotes that broke copy-pasting `pip install` lines

## 0.9.0
- Excel, JSON and Feather input via the `excel` / `feather` extras (thanks @tyza66)
- Low-severity `no_random_state` finding for unseeded splits (thanks @rahul05ranjan); not raised for `shuffle=False`; ShuffleSplit family covered

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