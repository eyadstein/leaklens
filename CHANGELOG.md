# Changelog

## Unreleased

- **The backend runs without Docker.** `python -m leaklens.public_api` starts
  the hardened live-scan app and `python -m leaklens.api` the developer app —
  the same `app` objects the images serve — bound to `127.0.0.1:8000` by
  default. `LEAKLENS_HOST`, `LEAKLENS_PORT` and `LEAKLENS_LOG_LEVEL` configure
  it; invalid values exit 2 with a message. Docker and Docker Compose remain
  supported. A core-only install still does not pull FastAPI, Uvicorn or
  Starlette, and a missing extra now names `leaklens-ml[api]`.
- **Safer deployment defaults.** Docker Compose publishes the API and UI on
  `127.0.0.1` only, and `Dockerfile.api` pins the same base image as
  `Dockerfile`. Rate limiting behind a proxy keys on the nearest
  `X-Forwarded-For` hop, so a forged prefix cannot buy a fresh bucket. The
  `/audit` request cap covers two uploads plus a graph baseline, and the docs
  now state that an upload above 1 MiB uses a per-request temporary file.
  `LIVE_SCANNING.md` documents authentication duties, HTTPS, trusted headers and
  the process-local limits; the API still has no client authentication.

## 0.14.0

Changes in the `0.14.0` release. The package is published on PyPI; release notes
are in [docs/releases/0.14.0.md](docs/releases/0.14.0.md).

Optional capabilities name the extra they require (`[sql]`, `[yaml]`,
`[images]`, `[app]`, `[api]`). Diagnostic engines never
change `leakage.score()`, an existing finding's severity, or the default exit
gate.

### Added

#### Detection

- **Partition inference** (`leaklens/partition.py`) identifies train, test and
  validation roles from `train_test_split` unpacking, name hints (`test`,
  `holdout`, `te`) and cheap aliases, so tuning on an arbitrarily named second
  return value is caught. A validation split stays legitimate.
- **Tuning and test-set reuse** (`leaklens/cvcheck.py`): `tuning_on_test_data`
  now also covers hand-rolled `for C in [...]` loops fitted on the test
  partition; new `tuning_before_split`, `tuning_on_full_data`,
  `cv_on_test_data`, `test_set_reuse` and `group_cv_missing_groups`. Nested CV
  and tuning on a validation split stay clean.
- **Encoding and feature selection**: `target_encode_before_split`,
  `frequency_encode_before_split` and `feature_selection_before_split`
  (`SelectKBest`, `SelectPercentile`, `SelectFromModel`, `RFE`,
  `SequentialFeatureSelector`).
- **Temporal and notebook checks**: `temporal_split_shuffle`,
  `notebook_out_of_order_execution`, and `future_target_feature_before_split`
  for negative target shifts and centered target rolling windows built before
  the split.
- **Split overlap**: `feature_overlap` and `near_duplicate_overlap`, run with
  bounded sampling (50k/20k rows).
- **`no_stratify`** low-severity advisory for `train_test_split(X, y, ...)`
  without `stratify=`. Not flagged: splits that pass `stratify=`,
  `shuffle=False` ordered splits, single-array splits without separate labels,
  and `**kwargs` expansions that may carry `stratify`.
- **Derived-variable dataflow** (#19): the scanner follows multi-hop variables
  derived from the full dataset (`X2 = X.copy(); X3 = f(X2)` and column
  selections) and seeds split tracking from cross-validation splitters, so
  full-data fits behind derived names are reported.
- Every new kind has an `--explain` remediation hint, benchmark cases and a
  detector-manifest entry (`docs/DETECTORS.md`).

#### Evidence, baselines and policy

- **Evidence** (`leaklens/evidence.py`): immutable, bounded, JSON-safe records
  (`source`, `kind`, `description`, optional `location`/`observed`/`expected`)
  with control-character rejection and a 16-record cap. Evidence explains why a
  detector fired and never carries raw rows, credentials, image bytes or full
  source. `target_leakage`, `split_overlap`, `temporal_leakage`, `group_overlap`,
  `near_duplicate_overlap` and `image_overlap` emit it.
- **Fingerprints** (`leaklens/fingerprint.py`): deterministic, domain-separated
  SHA-256 over a versioned canonical form. Line numbers, notebook cells, detail
  text, timestamps and observed counts are excluded, so moved lines keep their
  identity. Repeated same-kind code findings get stable keys from the offending
  construct's structural AST shape and enclosing scope.
- **Detector manifest** (`leaklens/manifest.py`, `docs/DETECTORS.md`): every
  emitted kind records its surface, default severity, advice key, benchmark case
  or explicit reason, documentation anchor and introduced version.
- **Baselines** (`leaklens/baseline.py`): a versioned, deterministic JSON
  baseline that records stable fingerprints plus kind, severity and column.
  Loading is strict, atomic writes refuse directories/symlinks/missing parents,
  and comparison classifies `new`, `unchanged`, `changed` and `resolved`.
- **Suppressions** (`leaklens/suppression.py`): exact-fingerprint rules with a
  required reason and optional ISO expiry. No wildcards; suppression only labels
  a finding and never changes severity, evidence, fingerprint or kind.
- **Configuration** (`leaklens/config.py`): a strict `.leaklens.toml` schema for
  the existing flags plus `[baseline]`, `[[suppressions]]`, `[missingness]`,
  `[sql]`, `[drift]`, `[graph]` and `[pipeline]` tables. Precedence is defaults <
  configuration < explicit CLI flags; unknown keys, wrong types, invalid values
  and unsafe paths exit 2 with the offending location.
- **Shared policy** (`leaklens/policy.py`): baselines and suppressions are
  applied once for the CLI and API, without duplicating or mutating findings.
- **Report schema** (`leaklens/schemas/report-1.schema.json`): JSON Schema
  2020-12 for the `--json` document and the `text`, `dl` and `images` surfaces.
  Every JSON report carries `$schema`, and `leaklens schema` prints the same
  document. Finding and evidence keys are closed so a typo fails validation,
  while top-level sections stay open to additive growth. Additive: existing
  fields, scores and exit codes are unchanged. `jsonschema` is a test-only
  dependency (`[dev]`), not a runtime one; the contract is documented in
  [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md).
- **Audit manifest** (`leaklens/manifest.py`): `--manifest PATH` writes a
  deterministic, unsigned digest manifest; `leaklens verify MANIFEST` recomputes
  it offline (exit 0 clean, 1 mismatch, 2 unreadable/unsupported). It detects
  content changes, not who wrote the manifest. See
  [docs/AUDIT_MANIFEST.md](docs/AUDIT_MANIFEST.md).
- **Graph snapshots** (`leaklens/intelligence/graph_baseline.py`): validated,
  deterministic evidence-graph snapshots and `compare_graph_snapshots()`,
  exposed through `--graph-baseline`, `--graph-snapshot`,
  `--update-graph-snapshot` and the opt-in `--fail-on-graph-new` gate.

#### Evidence graph and intelligence

- **Evidence graph** (`leaklens/intelligence/provenance.py`): findings grouped
  into inspectable `EvidenceCluster`s and `EvidenceLink`s over existing evidence;
  detector evidence preserved as stable `scanner-000000-e01`-style records;
  deterministic cluster ordering and a 400-link cap. Additive `clusters` and
  `links` in `IntelligenceReport`.
- **Graph exploration and export**: `evidence_by_id`, `cluster_for`,
  `neighbors`, `relation_counts`, `summary`, plus `to_mermaid()`, `to_dot()` and
  `to_dict()`. `--graph-out` writes `.mmd`/`.md`, `.dot`/`.gv` or `.json` and
  implies `--intelligence`; `--graph-format` overrides inference. HTML reports
  gain an evidence-links table.
- **Unified baseline intelligence**: location-neutral evidence identity, a
  completeness model (`summary.completeness`), and conservative `uncertain`
  comparisons when either side is incomplete or the analysis context differs.
- **Investigation plans** (`leaklens/intelligence/investigation.py`): an ordered,
  deterministic plan under `--intelligence` with `inspect_evidence`,
  `suggested_test`, `missing_evidence` and `narration_slot` steps. Suggested
  tests name a registered detector kind and question; `missing_evidence` records
  every skip with a reason and never changes the score.
- **Monitoring report intake** (`--monitoring-report PATH`): reads a
  caller-supplied drift/skew JSON report read-only and size-capped. Each
  breached metric becomes a low, diagnostic `monitoring_drift_report` or
  `monitoring_skew_report` finding. Missing, malformed, oversized or out-of-root
  files are counted skips and never change the trust score or the exit gate.
  See [docs/INVESTIGATION.md](docs/INVESTIGATION.md).
- **Governance pack** (`leaklens/governance/`): opt-in `--governance` labels
  every field `asserted`, `measured` or `unknown`, lists its non-claims, and
  renders to `.json` or `.md` with `--governance-out`. It never adds or rescales
  a finding. See [docs/GOVERNANCE.md](docs/GOVERNANCE.md).

#### SQL, drift and missingness

- **SQL lineage** (`leaklens/sqlscan.py`, `[sql]` → `sqlglot>=25`, evaluated at
  30.17.0): offline parsing only, no execution or database connection. Rules
  `sql_target_as_feature` (high), `sql_future_window` (medium),
  `sql_split_predicate_missing` (low) and diagnostic `sql_unresolved_column`.
  Explicit dialects `ansi`, `postgres`, `bigquery`, `snowflake` with no silent
  fallback; bounded to 20 statements and 200 projections. See
  [docs/SQL_LINEAGE.md](docs/SQL_LINEAGE.md).
- **Drift diagnostics** (`leaklens/drift.py`, opt-in `--drift`): two-sample KS
  for numeric columns, total-variation distance for categorical columns and
  missingness-rate deltas, one low `distribution_shift` finding per column,
  pandas/numpy only. See [docs/DRIFT.md](docs/DRIFT.md).
- **Missingness** (`leaklens/missingness.py`): conservative target-, split- and
  group-aware signals (`missingness_target_signal` and
  `missingness_split_shift` capped at medium, `missingness_group_signal` low),
  aggregate-only evidence and no imputation. See
  [docs/MISSINGNESS.md](docs/MISSINGNESS.md).
- **Pipeline and MLOps intelligence** (`leaklens/pipeline.py`, opt-in
  `--pipeline`, `[yaml]` for DVC files): read-only audit of `dvc.yaml`,
  `dvc.lock`, `*.dvc`, `*.dag.json`, notebook cell order and requirement pins.
  Kinds `pipeline_split_overlap` (high), `pipeline_unresolved_lineage`,
  `pipeline_param_drift`, `pipeline_missing_digest` and
  `pipeline_unpinned_dependency`. It never runs a stage or imports DVC/MLflow.
  See [docs/PIPELINE.md](docs/PIPELINE.md).
- **Text and LLM contamination** (`leaklens text TRAIN EVAL`,
  `leaklens/analysis/text/`): exact and near-duplicate overlap,
  hash-only contamination against user-supplied eval fingerprints, static RAG
  wiring checks and optional user-supplied embedding similarity. Kinds
  `text_exact_duplicate`, `text_near_duplicate`, `text_eval_contamination`,
  `rag_index_leakage` and `text_embedding_similarity` are diagnostics. See
  [docs/TEXT.md](docs/TEXT.md).
- **Deep-learning and artifact audit** (`leaklens dl DIR`,
  `leaklens/analysis/dl/`): static inspection of training code, framework
  configs, checkpoints and requirement pins. Kinds `dl_split_overlap` (high),
  `dl_training_protocol_mismatch`, `artifact_unsafe_format`,
  reproducible-seed and dependency-pin advisories. It never imports
  torch/TensorFlow/Keras/ONNX, unpickles a checkpoint or executes scanned code.
  See [docs/DL_AUDIT.md](docs/DL_AUDIT.md).

#### AI assistance (optional, experimental)

- **Offline AI layer** (`leaklens.ai`, opt-in `--ai`, `--ai-llm`,
  `include_ai=true`): deterministic semantic rules, a small CPU-only learned
  detector, evidence-bound hypotheses capped at `medium`/`0.95`, root-cause
  grouping and deterministic fallback. It never changes deterministic findings,
  severity, the trust score or the exit code.
- **Model artifact** (`leaklens/ai/model.json`): a checked-in, plain-JSON
  artifact holding the fitted scaler, coefficients, calibrator, threshold,
  importances and model card. A matching SHA-256, corpus, feature-catalogue,
  version and scikit-learn check loads it; any mismatch falls back to an
  in-memory fit. JSON cannot execute code and no pickle is used.
- **Local explanations**: signed local log-odds contributions, opposing signals,
  a margin and an uncertainty band; low-margin units are marked `[needs review]`.
  Global importances remain benchmark-only.
- **Runtime hypothesis gate**: a learned hint is emitted only when a
  context-aware risk indicator is active, unless a deterministic finding
  corroborates the unit.
- **Investigation narration** (`--intelligence --ai-llm`): fills an
  investigation plan's `narration_slot` only with evidence ids the plan already
  carries; malformed or unsupported replies fall back deterministically.
- **Prompt-injection hardening**: a versioned 47-case synthetic corpus, pinned
  digests, an offline evaluator, and prompt builders that flatten control
  characters and reject invented safety/severity verdicts. See
  [docs/PROMPT_INJECTION.md](docs/PROMPT_INJECTION.md).
- **Customizable LLM prompt templates** (#256): the default template lives in
  `leaklens/prompts/explain.txt`; `--llm-prompt-template PATH`,
  `LEAKLENS_LLM_PROMPT_TEMPLATE` and `.leaklens/prompt.txt` override it.

#### Reporting, API and workspace

- **JSON Schema** for reports plus `leaklens schema` and `$schema` on every JSON
  document (see Evidence, baselines and policy).
- **SARIF metadata**: additive `properties.leaklens` evidence, fingerprints,
  diagnostic flags, policy status and optional graph metadata; output without
  those options keeps its previous shape.
- **HTML reports**: evidence-links and evidence-cluster tables, drift and policy
  sections, location moves and incomplete-scan reasons, all escaped.
- **Interactive workspace** (`leaklens workspace [DIR]`): a local, read-only
  FastAPI page with live per-file progress, cancellation, filters, evidence
  clusters, source preview, one HTML report and raw JSON. Later phases add
  dataset audits, baseline/suppression visibility, evidence-graph navigation,
  history, comparisons and bounded server-sent events. It binds `127.0.0.1`,
  confines paths to the root, runs one scan at a time and never writes baselines.
  See [docs/WORKSPACE.md](docs/WORKSPACE.md).
- **Public live-scan API** (`leaklens.public_api`, `[api]`): a separate hardened
  FastAPI app for untrusted uploads with `GET /health`, `GET /capabilities`,
  `POST /inspect`, `POST /audit` and `POST /scan`; no `/predict`, no model
  deserialisation, no accounts and no persistent storage.
- **Upload guard** (`leaklens/upload_guard.py`): filename sanitisation,
  extension allowlist, content sniffing, ZIP traversal/encryption/expansion
  checks, and row/column/cell limits applied before and after parsing. Uploads
  are parsed from memory and never written to disk.
- **Request hardening**: body-size limits, bounded per-client rate limiting,
  a single-worker scan runner with a wall-clock timeout, generic error bodies,
  conservative security headers, CORS allowlisting and stable request IDs.
- **Live page** (`docs/live.html`): a real drag-and-drop scan page that connects
  to a configured backend, validates inputs, inspects columns and produces
  reports; it says **No backend connected** when no URL is configured.
- **Deployment assets**: `Dockerfile.api` (non-root, one worker, healthcheck),
  `render.yaml` Blueprint with `autoDeployTrigger: checksPass`, and
  [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md). No public backend is deployed
  by this repository.

#### Datasets, splits and vision

- **Image clustering** (`leaklens images`): byte-identical exact groups and
  perceptual near-duplicate connected-component clusters, with an additive
  `clustering` JSON payload, `train:`/`test:` namespaced members and the shared
  `near_duplicate.cluster_key`. `image_overlap()` keeps its signature, cost and
  finding.
- **Image decode budgets**: bounded encoded bytes, pixels per side, total
  megapixels, animation frames and a shared pixel budget; refusals use stable
  reasons and a corrupt image is counted as unreadable instead of crashing. See
  [docs/IMAGE_DECODE_BUDGETS.md](docs/IMAGE_DECODE_BUDGETS.md).
- **Declared-split audit** (`leaklens splits DATASET`): reads COCO, YOLO and
  Pascal VOC split declarations and reports image content appearing in more than
  one partition as a `high` `image_split_overlap` finding, using SHA-256 over
  file bytes and never decoding an image. Duplicates entirely inside one
  partition are counts, not findings. See [docs/VISION_AUDIT.md](docs/VISION_AUDIT.md).

#### Site and demo

- **Product site** (`docs/`): landing page, interactive demo, documentation,
  example report and 404, all hand-written HTML/CSS/ES modules with no bundler,
  external font or third-party script.
- **Prerecorded demo** (`docs/demo.html`): replays verified scanner output for
  seven fixtures; every result is labelled **prerecorded** and shows the live
  CLI command.
- **Deterministic fixtures**: `scripts/build_site_fixtures.py` and
  `scripts/build_site.py` regenerate and validate links, fragments, assets,
  landmarks, version and score arithmetic; CI fails when recorded output drifts.
- **Accessible 3-D field** on the landing page, built from real split rows, with
  drag/keyboard rotation, duplicate highlighting and reduced-motion/offscreen
  pausing.
- `examples/safe_pipeline.py`, the corrected pipeline used as the clean code
  fixture.

### Changed

- **Detector precision** (Phase 19C Wave 2):
  - `no_stratify` only claims an "imbalanced classification target" when the file
    also shows classification evidence (a classifier or a classification
    metric); otherwise the same low finding explains that the advice applies
    only to categorical targets.
  - `id_like` requires an identifier-like column name for numeric columns; a
    unique integer counter, float measurement or epoch timestamp is no longer
    reported, while `customer_id` stored as an integer still is. String columns
    keep the uniqueness rule, and the detail now states evidence rather than
    concluding identity.
  - `temporal_split_shuffle` no longer treats exact cyclic calendar attributes
    (`day_of_week`, `month`, `hour`, …) as a time axis, so chart-prep sorts
    before a split are clean. `date`, `timestamp`, `datetime`, `year`, `week`,
    derived names and time-like receivers still count.
  - Kind, severity, fingerprint and score are unchanged where the finding
    remains. Known limitations: a numeric identifier with an uninformative name
    is still caught only by `single_feature_leaks`, and a genuine time column
    literally named `month` or `hour` is no longer treated as a time axis. See
    [docs/DETECTORS.md](docs/DETECTORS.md).
- **Detector precision** (Phase 19C Wave 1): `manual_scale_before_split`,
  `fillna_before_split`, `group_stat_before_split`,
  `frequency_encode_before_split` and `target_encode_before_split` now require
  the receiver to be bound in the file and disjoint from the frame the split
  divides. Receivers that are not bound locally (module globals, function
  parameters) keep their findings. A known limit remains: a leak on a
  helper-returned frame the dataflow cannot trace is missed when the name is
  also locally bound. Existing baselines will show the removed findings as
  `resolved`.
- **Temporal detection accuracy**: the interim `future_looking_*` kinds (never
  released) are consolidated into `future_target_feature_before_split`, with
  target-like renamed labels recognised (`target`, `labels`, `outcome`, …).
- **Baseline identity** is location-neutral; a finding that only moved lines is
  `unchanged` and reports `changes: ["location"]`. Missing evidence is
  `uncertain`, never `resolved`, when either side is incomplete.
- **AI quality** (Phase 6/7): the feature extractor grew from 98 to 112
  indicators, 40 hard-positive/hard-negative development pairs and a frozen
  12-case Phase 7 holdout were added, 11 candidates were compared, and
  class-balanced logistic regression was selected on validation. The Phase 5
  locked test and Phase 6 real-world corpus were not re-tuned. Measured
  results: the Phase 5 locked test scored accuracy 0.979, precision 1.000,
  recall 0.960, F1 0.980 and FPR 0.000; the frozen real-world holdout (48
  reviewed units from 12 repositories) scored accuracy 0.833, precision 0.767,
  recall 0.958, F1 0.852 and FPR 0.292; the Phase 7 holdout scored accuracy
  0.917, precision 0.857, recall 1.000, F1 0.923 and FPR 0.167. These are
  measured numbers, not a production-accuracy claim. See
  [docs/AI_BENCHMARK.md](docs/AI_BENCHMARK.md) and
  [docs/AI_REALWORLD.md](docs/AI_REALWORLD.md).
- **AI real-notebook smoke check**: 8 public notebooks with definitive hand labels
  from the earlier scanner corpus score 5/5 leaky and 2/3 legitimate (accuracy
  0.875, precision 0.833, recall 1.000). The sample is tiny and non-random, and
  the report says so.
- **AI root-cause coverage** now classifies SQL lineage, unresolved SQL,
  missingness and distribution-shift findings as well as tuning on test/full
  data, CV on the test partition, test-set reuse, `GroupKFold` without groups,
  preprocessing/resampling outside the CV loop, shuffled temporal splits and
  out-of-order notebook execution. Unknown kinds still fall back to a generic
  review.
- **AI model serving** uses the checked-in artifact when it validates, otherwise
  an in-memory fit; predictions and local contributions are bit-identical.
- **`leaklens init`** now prints a pre-commit snippet pinned to the latest
  published release tag (`v0.13.0`) instead of the mutable `main` branch, and the
  site documentation pins the same revision.
- **`leaklens doctor`** validates `.leaklens.toml` (including an explicit
  `--config PATH`), and `init` writes a starter that documents the policy,
  missingness, SQL, drift, graph and pipeline tables.
- **Website and deployment**: `render.yaml` uses
  `autoDeployTrigger: checksPass` and a `buildFilter` so docs-only commits do not
  rebuild the API; the deployed site remains static and disconnected.
- **Feature-importance reporting** reads the fitted pipeline rather than the
  unfitted spec.

### Fixed

- SQL lineage now decodes quoted identifiers across dialects and analyses every
  `UNION`/`UNION ALL`/`INTERSECT`/`EXCEPT` branch, including nested and
  parenthesised combinations, while repeated statements still collapse to one
  fingerprint-stable finding.
- Detector-evidence links now point at the primary record of the finding the
  detector evidence was attached to, not the cluster representative; unknown
  finding indices produce no link.
- Temporal split checks no longer raise `TypeError` on exports that mix
  timezone-aware and naive values or spell the same instant with different
  offsets; naive values are read as UTC and unparseable/all-NaT columns produce
  no finding.
- A corrupt image no longer crashes `leaklens images`; decoder exceptions are
  counted as unreadable.
- `Baseline.load` reports non-UTF-8 files as `BaselineError` instead of leaking a
  decode error, and the workspace keeps display labels relative to its root.
- The live page no longer throws in `docs/assets/site.js` on pages with an empty
  audit container; `check_site_layout.mjs` now fails on uncaught page errors.
- Release metadata validation now checks the version in `pyproject.toml`, the
  runtime package, `CITATION.cff` and the changelog heading, and fails when they
  disagree.
- Benchmark counts and documentation stay consistent with the consolidated
  detector set.

### Security

- **Path containment** (`leaklens/paths.py`): `resolve_within` normalises with
  `os.path.realpath`, accepts only paths inside the root prefix, and rejects
  embedded NUL bytes before touching the filesystem, so siblings, symlinks and
  junctions cannot escape.
- **File selection**: directory scans keep only regular files whose real path
  stays inside the scanned directory, so a FIFO named `*.py` cannot block a scan
  and a symlink pointing outside is skipped.
- **Baseline loading** refuses non-regular files before reading, so a FIFO or
  device cannot hang or exhaust the process.
- **Upload and live-API hardening**: content sniffing, archive traversal and
  expansion limits, body-size limits, rate limiting, a single-worker timeout,
  generic errors and conservative CORS/security headers.
- **Prompt-injection defense**: untrusted finding fields are flattened before
  prompt rendering, invented severity/safety claims are dropped, and a versioned
  corpus plus offline evaluator guard the behaviour.
- **Release pipeline hardening**:
  - the release validates tag format, tag/version match, main ancestry and the
    dispatch ref before any tag content runs;
  - wheel and sdist membership is derived from the tagged tree and validated
    byte for byte;
  - the SBOM defaults to `sbom/sbom.json`, is refused inside `dist/`, and is
    uploaded separately;
  - the PyPI publisher is pinned to the verified `v1.14.3` commit and publishes
    last under the `pypi` environment with OIDC.
- **CI hardening**: read-only workflow permissions, PR-only concurrency
  cancellation, bounded job timeouts, no persisted checkout credentials, a
  wheel rebuilt and validated from the sdist, and owner review required when a
  PR touches the Action, `MANIFEST.in`, pre-commit configuration or `scripts/`.
- `pip-audit`, Bandit, CodeQL `security-extended` and `pip-licenses` remain in
  CI; action references are SHA-pinned, including the composite `action.yml`.

### Performance

Measured locally on Windows with Python 3.14.3; these are local measurements,
not general guarantees.

- **Scanner parsing** (`scan_source`): each input is parsed and walked once and
  the AST, node tuple and stable structural keys are shared across the core,
  flow and CV passes. A bounded, content-addressed, thread-safe source cache
  (128 entries / 2,000,000 characters, inputs over 1,000,000 characters skipped)
  avoids reparsing. Cold `scan_path("leaklens")` over 65 files fell from 1.70 s
  to ~0.68 s (~60%); a repeated scan in the same process fell to ~0.19 s (~89%).
- **Scanner traversal** (`_node_index`): passes iterate a type-grouped index
  instead of re-checking every node with `isinstance`. Repeated warm scans fell
  from 0.192–0.203 s to 0.150–0.158 s (~22%) with unchanged finding order and
  fingerprints. The index is transient per file, not stored in the cache.
- **AI artifact cold load**: loading the checked-in `model.json` drops from a
  1.339 s fit to 1.69 ms while producing bit-identical predictions. See
  [docs/AI_PERFORMANCE.md](docs/AI_PERFORMANCE.md) and
  [docs/PERFORMANCE.md](docs/PERFORMANCE.md).
- Parallel scanning was measured and rejected: for the same 65 files a 4-worker
  process pool took 2.84 s and a 4-worker thread pool 1.00 s versus 0.54 s
  sequential.

### Developer Experience

- **Spell-check** (#532): `.pre-commit-config.yaml` runs codespell (v2.4.3)
  with a documented skip/ignore list, and a CI `spelling` job enforces it.
- **Python 3.14** (#543): the test matrix runs the suite, the self-scan and all
  benchmarks on 3.14 alongside 3.10–3.12. 3.13 remains outside the matrix and is
  not claimed as verified.
- **Initialization**: `leaklens init` writes a commented starter config, prints
  a pre-commit snippet pinned to the published tag, and refuses to overwrite
  without `--force`; `doctor` validates the config.
- **Benchmarks**: `bench_vision`, `bench_sql`, `bench_drift`, `bench_missing`,
  `bench_text`, `bench_dl`, `bench_pipeline`, `bench_investigation` and
  `bench_realworld` provide deterministic, labelled regression guards; each
  writes documentation only when asked. `bench_advanced` adds 32 labelled
  cases and `bench_data` 12 synthetic dataset cases, reported separately so the
  historical scanner counts stay stable.
- **Contributor corpus tooling**: `scripts/collect_expansion.py` can resume a
  candidate collection for the real-world expansion corpus; candidates never
  feed `leakage.score()` or any detector and are reviewed separately. See
  [docs/REALWORLD_EXPANSION.md](docs/REALWORLD_EXPANSION.md).

### Documentation

- Redesigned the English README as a project front page with a banner, table of
  contents, detection matrix, architecture, audit lifecycle and evidence model,
  and added complete Arabic (`docs/README.ar.md`) and Japanese
  (`docs/README.ja.md`) counterparts with the same facts and Mermaid diagrams.
- Added `docs/assets/banner.svg` with accessible title/description, plus
  [docs/INTELLIGENCE.md](docs/INTELLIGENCE.md),
  [docs/INVESTIGATION.md](docs/INVESTIGATION.md),
  [docs/GOVERNANCE.md](docs/GOVERNANCE.md),
  [docs/VISION_AUDIT.md](docs/VISION_AUDIT.md),
  [docs/AUDIT_MANIFEST.md](docs/AUDIT_MANIFEST.md),
  [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md),
  [docs/WORKSPACE.md](docs/WORKSPACE.md),
  [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md),
  [docs/SQL_LINEAGE.md](docs/SQL_LINEAGE.md),
  [docs/DRIFT.md](docs/DRIFT.md),
  [docs/MISSINGNESS.md](docs/MISSINGNESS.md),
  [docs/PIPELINE.md](docs/PIPELINE.md),
  [docs/TEXT.md](docs/TEXT.md),
  [docs/DL_AUDIT.md](docs/DL_AUDIT.md) and
  [docs/AI_INTELLIGENCE.md](docs/AI_INTELLIGENCE.md).
- Added benchmark methodology and results: `docs/VISION_BENCHMARK.md`,
  `docs/SQL_BENCHMARK.md`, `docs/DRIFT_BENCHMARK.md`,
  `docs/MISSINGNESS_BENCHMARK.md`, `docs/TEXT_BENCHMARK.md`,
  `docs/DL_BENCHMARK.md`, `docs/PIPELINE_BENCHMARK.md`,
  `docs/AI_BENCHMARK.md`, `docs/AI_REALWORLD.md`, `docs/AI_PERFORMANCE.md` and
  `docs/DATA_BENCHMARK.md`.
- Labelled the advanced detection cases in `docs/BENCHMARK.md` as
  **unreleased on main** rather than `0.14.0`, and corrected stale command
  references on the site.
- Documented real-world corpus provenance and limits in
  `docs/REALWORLD_CORPUS.md` and the AI expansion roadmap in
  `docs/AI_ML_EXPANSION_ROADMAP.md`.

### Internal

- Added a deterministic report schema, detector manifest, version-consistency
  validator, wheel/sdist completeness validator, SBOM generator with a `dist/`
  guard, and action-contract tests; release validation now runs in CI before a
  tag can publish.
- Added the `package` CI job that builds and validates both distributions, and
  the `api`, `sql`, `site`, `spelling` and `security` jobs that exercise the
  optional extras and the HTTP surfaces.
- Added extensive regression coverage across evidence, fingerprints, baselines,
  suppressions, policy, configuration, missingness, SQL, drift, vision,
  workspace, public API, upload guard, AI schemas/reasoning/corpus, prompt
  injection, release tags, action contract and workflow contracts.
- Added corpus infrastructure and acceptance records:
  `docs/PHASE13_ACCEPTANCE.md`, `docs/PHASE14_ACCEPTANCE.md`,
  `docs/PHASE15_ACCEPTANCE.md`, `docs/PHASE16_ACCEPTANCE.md` and
  `docs/PHASE18_ACCEPTANCE.md`.
- Kept every new engine additive: no network calls in normal scanning, no new
  mandatory dependency, no code execution of scanned input, and no change to
  `leakage.score()` or the default exit gate.

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