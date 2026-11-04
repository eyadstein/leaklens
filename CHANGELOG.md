# Changelog

## Unreleased

## 0.13.0
- `leaklens doctor` checks Python/LeakLens versions, install location, core and optional dependencies, and TOML config validity (text or `--json`; no network calls); Python 3.10 uses `tomli`
- Upload protection for the API and Streamlit UI: 8 MiB / 50,000-row limits, ZIP/gzip decompressed-size limit (256 MiB), ZIP path-traversal rejection, and nested-gzip cap
- CI: `ruff check` / `ruff format --check` lint job; `ruff` added to the `dev` extra
- Security automation: CodeQL `security-extended` workflow; CycloneDX 1.5 SBOM generation (`scripts/generate_sbom.py`) kept outside `dist/` and uploaded with release packages; CI also runs `pip-licenses`
- Windows PowerShell installer (`scripts/install.ps1`) that creates/reuses a venv, installs LeakLens, and runs `leaklens doctor`
- Release workflow fixes: `contents: read` for checkout, wheel/sdist artifact upload, and SBOM generated after the build
- Docs: document `leaklens doctor`, upload/archive limits, `/predict`'s `./models/` restriction, SBOM/security automation, and cross-platform install notes
- `leaklens init` writes a commented starter `.leaklens.toml` and prints the pre-commit snippet
- `leaklens images TRAIN_DIR TEST_DIR` finds near-duplicate images across splits with a 64-bit dHash (`--max-distance`, default 6); needs the new `images` extra (Pillow)
- New detector: hyperparameter search (`GridSearchCV` / `RandomizedSearchCV` / `HalvingGridSearchCV` / …) fitted on test data (`tuning_on_test_data`, high) — 4 new benchmark cases
- New detector: high-cardinality categorical columns that encode the target via entropy reduction (`categorical_target_leakage`, high/medium) — complements the existing `id_like` detector
- Security: CI `bandit` + `pip-audit` steps, expanded `SECURITY.md` threat model, optional TaintWatch guard for `--llm` (`LEAKLENS_TAINTWATCH=1`, `taintwatch` extra)
- Real-world notebook validation: 60 public notebooks in `realworld/` with hand-reviewed labels (`realworld/labels.json`); only the 8 definitive labels (5 `confirmed_leakage` + 3 `legitimate`) contribute to precision/recall/F1 — the other 52 are reported but excluded (see [docs/REALWORLD.md](docs/REALWORLD.md)). Corpus is kept separate from the synthetic benchmark
- Tests: direct unit coverage for `leaklens/intelligence/` (`analyze_findings`, `analyze_dataset_impact`, `assess_confidence`) and `cvcheck.py`

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