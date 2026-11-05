# Changelog

## Unreleased

### Experimental AI intelligence (0.14.0 phase 4)

- Added the optional, offline `leaklens.ai` package. It is opt-in (`--ai`, `--ai-llm`, `include_ai=true`, Python `leaklens.ai`) and never changes deterministic findings, severity, the trust score or the exit code.
- Added deterministic semantic rules (`leaklens/ai/semantics.py`) over code, notebooks and dataset metadata: concat-before-split, target-derived features, future fills, centered windows, negative shifts, runtime clocks, target-proxy and future-information names, identifiers and shared train/test keys. Code is parsed, never executed.
- Added a small in-memory CPU gradient-boosted detector (`leaklens/ai/learned.py`) over 61 interpretable indicators, trained on a 72-case/26-group labeled corpus (`leaklens/ai/corpus.py`). Groups never cross folds; the model is never serialized or loaded. The shipped threshold is fixed at 0.65 so behavior is stable across scikit-learn releases.
- Added strict evidence-bound schemas (`leaklens/ai/schemas.py`): confidence ≤ 0.95, severity hints capped at medium, unique evidence ids, validated group/hypothesis/evidence cross-references, control-character rejection, and an explicit `changes_deterministic_findings: false`.
- Added root-cause grouping, remediation actions, code hints and verification steps (`leaklens/ai/reasoning.py`). Optional `--ai-llm` rephrasing receives sanitized group metadata only and falls back deterministically on provider failure, malformed JSON, invented group ids or unsupported safety claims.
- Prompt-injection defense: instruction-like comments and column names are reported as an `info` `untrusted_instruction` hypothesis and never followed. Dataset analysis receives metadata only, never rows or values.
- Integrated with the CLI (`--ai`, `--ai-llm`, JSON `ai` object, text summary, safe fallback warning), the HTML report (escaped, separate experimental section), the public live-scan API (`include_ai` on `/audit` and `/scan`, AI capability declaration) and the website live page (opt-in checkbox, separate `textContent` rendering).
- Added `leaklens/bench_ai.py` + `docs/AI_BENCHMARK.md` with grouped cross-validation precision/recall/F1/FPR, external transfer to the existing scanner cases, a threshold sweep, semantic regression results, top signals, explicit false positives/misses and limitations. CI runs it. It is explicitly **not** an improvement claim.
- Added `docs/AI_INTELLIGENCE.md`, README and docs-site sections, and tests for schemas, semantics, learning discipline, reasoning/fallback, orchestration, CLI/report integration, the public API, the site and the benchmark.

### Deployment readiness (0.14.0 phase 3.1)

- Validated `render.yaml` against Render's published Blueprint JSON schema and replaced the deprecated `autoDeploy: true` with `autoDeployTrigger: checksPass`, so a push to `main` redeploys the backend only after GitHub Actions passes. Added a `buildFilter` so docs-only or site-only commits do not rebuild the API.
- Documented the exact owner-run deployment path in [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md): the Render Blueprint shortcut, free-plan review, `/health` and `/capabilities` checks, `scripts/smoke_public_api.py --url`, the `?api=` test, the `live-config.js` switch and the rollback to disconnected.
- `scripts/build_site.py` and `tests/test_site.py` now accept an empty `apiBase` or a clean HTTPS origin in `docs/assets/live-config.js`; credentials, paths, queries and plain HTTP are rejected. The default remains empty and the deployed site remains disconnected.
- Built and ran `Dockerfile.api` locally (Linux image, non-root uid 10001, healthy, one worker). The real-HTTP smoke test and additional end-to-end checks passed against the container: real findings and score arithmetic, clean datasets and clean code, report generation and HTML escaping, CORS allow/deny, malformed and mismatched uploads, traversal, oversized/row-limit rejection, rate limiting, generic errors, and no filesystem changes after uploads. `docker stop` exited 0 in under a second.
- Drove the actual `docs/live.html` frontend against the container in headless Chrome: `?api=` connection, real `/inspect`, target selection, real `/audit`, rendered score/findings, and an HTML report download all worked with no console errors. That check exposed a null-guard bug in `docs/assets/site.js` on pages with an empty audit container; it is fixed, and `check_site_layout.mjs` now fails on uncaught page errors.
- **No Render account credentials were available in this environment, so no public backend is deployed and the GitHub Pages site is still disconnected.** The repository cannot deploy itself.

### Live scanning platform (0.14.0 phase 3)

- Added `leaklens.public_api`: a separate hardened FastAPI app for untrusted uploads, with `GET /health`, `GET /capabilities`, `POST /inspect`, `POST /audit` (optional held-out file, group and time columns) and `POST /scan`. It has no `/predict`, no model deserialisation, no accounts and no persistent storage.
- Added `leaklens/upload_guard.py`: filename sanitisation, extension allowlist, real content sniffing for CSV/JSON/XLSX/Parquet/Feather, ZIP traversal/encryption/expansion checks, and row/column/cell limits applied before and after parsing. Uploads are parsed from an in-memory buffer and are never written to disk.
- Added request hardening: declared and chunked body-size limits, bounded per-client sliding-window rate limiting (`leaklens/ratelimit.py`), a single-worker scan runner with a wall-clock timeout and `503 scan_busy`/`504 scan_timeout` envelopes, generic error bodies, conservative security headers, CORS allowlisting and stable request IDs.
- Added `docs/live.html` and `docs/assets/live.js` + `live-engine.mjs`: a real drag-and-drop live scan page with backend connection, client-side validation, column inspection, target/group/time selectors, threshold, real elapsed progress, cancellation, retry/offline states, severity/kind/search filters, audit and measured split-impact panels, and downloadable HTML/JSON reports. It never replays prerecorded fixtures and explicitly says **No backend connected** when no URL is configured.
- Deployment: `Dockerfile.api` (non-root, one worker, proxy headers, healthcheck), `render.yaml` free Render Blueprint, and [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md) with verified free limits, cold starts, environment variables and the security model. No public backend is deployed by this repository; GitHub Pages remains static and disconnected until a URL is supplied.
- Added `api` extra (`fastapi`, `uvicorn`, `python-multipart`), `scripts/smoke_public_api.py` (real HTTP smoke test), `tests/test_upload_guard.py` (31 tests), `tests/test_public_api.py` (45 tests) and `tests/site/live-engine.test.mjs` (18 tests). CI now has an `api` job and the `site` job runs both Node suites.
- Updated the site navigation, landing hero, docs, report link, 404 page and README, and extended `build_site.py`, `test_site.py` and `check_site_layout.mjs` to validate the live page and reject a hard-coded backend URL.

### Website and demo (0.14.0 phase 2)

- Rebuilt the GitHub Pages site in `docs/` as a real product site: landing page, interactive demo, documentation, example report and a 404, all hand-written HTML/CSS/ES modules with no bundler, external font or third-party script.
- Added `docs/demo.html`, which replays verified scanner output for seven fixtures (leaky and safe code, leaky and clean transaction tables, exact and normalised split overlap). Every result is labelled **prerecorded**; the page states that GitHub Pages cannot execute Python and shows the live CLI command for each fixture.
- Added `scripts/build_site_fixtures.py` (deterministic fixtures from `scan_source`, `leakage.scan/score`, the overlap checks and `fixes.advice`) and `scripts/build_site.py` (regenerate + validate links, fragments, assets, landmarks, version and score arithmetic). CI fails when the recorded output drifts.
- Added a dependency-free canvas 3-D field on the landing page built from real rows in the split fixture, with drag/keyboard rotation, duplicate highlighting and reduced-motion/offscreen pausing.
- Added `examples/safe_pipeline.py`, the corrected pipeline used as the clean code fixture.
- Added frontend tests: `tests/test_site.py` (fixtures, links, accessibility landmarks, JS syntax) and `tests/site/demo-engine.test.mjs` (Node's built-in runner). A new `site` CI job runs `build_site.py --check` and the Node tests; `scripts/check_site_layout.mjs` performs local desktop/tablet/mobile checks in headless Chrome or Edge.
- README now points at the new site, demo and documentation, and explains how to regenerate the fixtures.

### Detection engine (0.14.0 phase 1)

- New `leaklens/partition.py`: infers train/test/validation roles from `train_test_split` unpacking, name hints (`test`, `holdout`, `te`) and cheap aliases. Tuning on an arbitrarily named second return value is now caught; a validation split is deliberately treated as legitimate.
- Hyperparameter tuning and test-set reuse (`leaklens/cvcheck.py`):
  - `tuning_on_test_data` (high) now also covers hand-rolled `for C in [...]` loops fitted on the test partition.
  - `tuning_before_split` (medium): a search or CV call on the full dataset before the explicit split.
  - `tuning_on_full_data` (medium): a search fitted on the original frame after the split exists.
  - `cv_on_test_data` (medium): `cross_val_score`/`cross_validate` over the held-out partition.
  - `test_set_reuse` (low): the test partition is scored three or more times, i.e. used for model selection.
  - `group_cv_missing_groups` (medium): `GroupKFold`/group splitter used without `groups=`, including inside `GridSearchCV`.
  - Nested CV (`cross_val_score(GridSearchCV(...), X_train, ...)`) and tuning on a validation split stay clean.
- High-cardinality / target-derived encoding (`leaklens/scanner.py`, `flow.py`):
  - `target_encode_before_split` (high): target means/aggregates pasted into a feature via `map`/`replace`/`merge`, including inline `groupby`, or a `TargetEncoder`/`CatBoostEncoder`/`WOEEncoder` fitted before the split.
  - `frequency_encode_before_split` (low): `value_counts` mapped into a column before the split; the label is not involved, so this is reported as a suspicious pattern rather than confirmed leakage.
  - `feature_selection_before_split` (high): `SelectKBest`/`SelectPercentile`/`SelectFromModel`/`RFE`/`SequentialFeatureSelector` fitted before the split (previously misclassified as generic preprocessing).
- Other gaps:
  - `temporal_split_shuffle` (high when `shuffle=True`, medium otherwise): `train_test_split` over a frame sorted by a time column with shuffling.
  - `notebook_out_of_order_execution` (low): non-monotonic `execution_count` in `.ipynb`; also emitted when another cell fails to parse.
  - `feature_overlap` (high/medium) and `near_duplicate_overlap` (medium) in `leaklens/leakage.py`; the CLI runs exact → feature-vector → normalised near-duplicate checks with bounded sampling (50k/20k rows).
- Every new finding kind has an `--explain` remediation hint (`leaklens/fixes.py`).
- Benchmarks:
  - `leaklens/bench_advanced.py`: 32 new labeled cases (tuning 12, encoding 9, feature selection 5, group CV 2, temporal 4), reported separately so historical 109-case counts stay stable.
  - `leaklens/bench_data.py`: 12 synthetic dataset cases with precision/recall and per-case runtime (`python -m leaklens.bench_data`).
  - `docs/DATA_BENCHMARK.md` and the advanced section of `docs/BENCHMARK.md` are generated by those commands; both benchmarks run in CI.
- Tests: `test_partition.py`, `test_tuning_advanced.py`, `test_encoding_advanced.py`, `test_temporal_group.py`, `test_notebook_order.py`, `test_duplicate_overlap.py`, `test_bench_advanced.py`, `test_bench_data.py` (68 new tests).

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