# LeakLens

[العربية](docs/README.ar.md) | [日本語](docs/README.ja.md)

[![CI](https://github.com/eyadstein/leaklens/actions/workflows/ci.yml/badge.svg)](https://github.com/eyadstein/leaklens/actions/workflows/ci.yml)
[![Leakage trust](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eyadstein/leaklens/main/badge.json)](#trust-score-and-findings)
[![PyPI](https://img.shields.io/pypi/v/leaklens-ml)](https://pypi.org/project/leaklens-ml/) [![Python](https://img.shields.io/pypi/pyversions/leaklens-ml)](https://pypi.org/project/leaklens-ml/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Find data leakage before it inflates your metrics.** LeakLens audits datasets, train/test splits and ML code for the mistakes that make a model look accurate and then fail in production, and reports what is left when the leak is removed. Detection is deterministic and local: the scanner never executes the code it reads, and nothing is uploaded.

[Documentation](https://eyadstein.github.io/leaklens/docs.html) · [Website and demo](https://eyadstein.github.io/leaklens/) · [Changelog](CHANGELOG.md) · [Issues](https://github.com/eyadstein/leaklens/issues)

## Contents

- [Quick start](#quick-start)
- [What it detects](#what-it-detects)
- [Built-in vs optional](#built-in-vs-optional)
- [Installation](#installation)
- [Command line](#command-line)
- [GitHub Action](#github-action)
- [Trust score and findings](#trust-score-and-findings)
- [Reports and the evidence graph](#reports-and-the-evidence-graph)
- [Python API](#python-api)
- [Configuration, baselines and suppressions](#configuration-baselines-and-suppressions)
- [Optional engines](#optional-engines)
- [Services: API, workspace, Docker](#services-api-workspace-docker)
- [Benchmarks and real-world checks](#benchmarks-and-real-world-checks)
- [Limitations](#limitations)
- [Security and privacy](#security-and-privacy)
- [Contributing and development](#contributing-and-development)
- [Roadmap](#roadmap)
- [License and citation](#license-and-citation)

## Quick start

Install the CLI (Python 3.10 or newer, no optional extras required), then scan a codebase or notebook folder for pipeline leaks:

```sh
python -m pip install --upgrade leaklens-ml
leaklens --version
leaklens --scan .
```

Each finding names the file and line, the rule it matched and the call that confirms it. Add `--explain` for a suggested fix under every finding.

Audit a dataset, optionally against a held-out split:

```sh
leaklens train.csv --target label
leaklens train.csv --test test.csv --target label --time date
```

A clean audit prints `LeakLens trust score: 100/100`. A high-severity finding makes the command exit with code 1, so the same command gates CI; `--fail-under 80` also fails a score below 80.

### See it catch a leak

The repository includes `examples/make_demo.py`, which writes `demo.csv`, a synthetic fraud dataset with one planted leak:

```sh
git clone https://github.com/eyadstein/leaklens.git
cd leaklens
python examples/make_demo.py
leaklens demo.csv --target is_fraud
```

```text
LeakLens trust score: 50/100
[HIGH] target_leakage :: chargeback_flag :: alone predicts target (roc_auc=1.000)
[MEDIUM] id_like :: txn_id :: 100.0% unique values; one value per row
1 high, 1 medium
```

`chargeback_flag` is the label copied into a feature, so accuracy looks perfect until the column is removed. `python examples/make_demo.py` regenerates the file, and the [walkthrough notebook](examples/leak_walkthrough.ipynb) plants three leaks and fixes them step by step.

## What it detects

| Surface | Checks |
|---|---|
| Datasets (CSV, JSON) | single-feature target leakage, high-cardinality categorical target leakage, identifier-like columns, group/entity risk (`--group`) |
| Splits (`--test`) | exact row overlap, identical feature vectors with conflicting labels, case/whitespace near-duplicates, temporal overlap, entity overlap |
| Missingness | missing values associated with the target, the split or the group, only when the association is strong enough to carry information ([details](docs/MISSINGNESS.md)) |
| Code (`.py`, `.ipynb`) | preprocessing, SMOTE, target/frequency encoding and feature selection before the split; fit on test data; tuning on test data or before the split; CV on test rows or without `groups=`; shuffled time-series splits; out-of-order notebook cells; future-looking target shifts and rolling windows |
| Images (optional) | near-duplicate images across train/test folders, exact and perceptual clusters, declared COCO/YOLO/VOC split disjointness (`leaklens splits`) |
| SQL (optional) | target used as a feature through confirmed lineage, future window frames, queries with no split predicate |
| Text (optional) | train/eval duplicates, evaluation contamination, RAG index wiring, caller-supplied embedding similarity |
| Deep learning (optional) | split overlap, training-protocol mismatches, unsafe checkpoint formats, missing seeds |
| Pipeline metadata (optional) | DVC and serialized pipeline lineage, parameter drift, missing digests, unpinned dependencies |

Findings state what was observed, not what must be true: a confirmed rule (a search fitted on `X_test`) is high severity, while a suspicious but legitimate-possible pattern (frequency encoding before the split) is low or medium and says why it was flagged. Every kind, its severity and its scored status are listed in [docs/DETECTORS.md](docs/DETECTORS.md).

## Built-in vs optional

The base install covers CSV/JSON datasets, code scanning, split checks and missingness. Everything else is opt-in and never required for the basic workflow.

| Capability | Command or flag | Needs | Gate behavior |
|---|---|---|---|
| Dataset audit, code scan, split checks, missingness | `leaklens`, `leaklens --scan`, `leaklens splits` | base install | exit 1 on a high-severity finding |
| Drift, pipeline, text, deep learning, AI | `--drift`, `--pipeline`, `leaklens text`, `leaklens dl`, `--ai` | base (`[yaml]` for DVC/YAML) | diagnostic: score and exit unchanged (`--fail-on-findings` for text and dl) |
| Image near-duplicates | `leaklens images` | `[images]` | exit 1 on a high-severity finding |
| SQL lineage | `--sql` | `[sql]` | exit 1 on a high-severity finding |
| Model arena | `--arena` | `[arena]` | writes `models/best.joblib` |
| API, UI, workspace | `uvicorn`, `streamlit`, `leaklens workspace` | `[app]` | not a scan gate |
| Public live-scan API | `uvicorn leaklens.public_api:app` | `[api]` | not a scan gate |

## Installation

The PyPI package is `leaklens-ml`; the installed command is `leaklens`. Supported interpreters are Python 3.10 and newer; CI tests 3.10, 3.11, 3.12 and 3.14 (3.13 is not in the matrix and is not claimed as verified). Windows, Linux and macOS are supported. On Windows use `py -3 -m pip ...` if `python` is not on PATH; if pip reports `externally-managed-environment`, create a virtual environment first.

Optional extras:

| Extra | Adds |
|---|---|
| `arena` | model arena: drops flagged columns, trains three model families in leak-safe pipelines, tracked in MLflow (`--arena`) |
| `parquet`, `feather` | `.parquet` and `.feather` input through `pyarrow` |
| `excel` | `.xlsx` input through `openpyxl` |
| `images` | near-duplicate image check (`leaklens images`) through Pillow |
| `app` | FastAPI service and Streamlit UI |
| `api` | hardened public live-scan API (no Streamlit) |
| `sql` | SQL lineage analysis through SQLGlot |
| `yaml` | YAML pipeline and training configs |
| `taintwatch` | optional information-flow guard for `--llm` |

```sh
python -m pip install "leaklens-ml[parquet,images]"
```

`leaklens doctor` checks the interpreter, install location, entry point, core dependencies, extras and configuration validity without network access; pass `--json` for machine-readable output. Unsupported suffixes fall back to CSV; optional formats raise an `ImportError` naming the extra that provides them.

## Command line

| Command | Purpose |
|---|---|
| `leaklens FILE --target COL` | audit a dataset (CSV/JSON built in) |
| `leaklens --scan PATH...` | statically scan `.py`/`.ipynb` files or folders |
| `leaklens splits DATASET` | audit declared COCO, YOLO or VOC split files |
| `leaklens images TRAIN TEST` | find near-duplicate images across splits |
| `leaklens text TRAIN TEST` | compare text corpora for contamination |
| `leaklens dl DIR` | audit deep-learning code, configs and artifacts |
| `leaklens workspace [DIR]` | serve the local interactive workspace |
| `leaklens verify MANIFEST` | verify an audit manifest offline |
| `leaklens init` | write a starter `.leaklens.toml` and print the pre-commit snippet |
| `leaklens doctor`, `leaklens schema` | check the install, print the report JSON Schema |

```sh
leaklens train.csv --target label --test test.csv --json
leaklens --scan . --sarif leaklens.sarif
leaklens --scan . --intelligence --graph-out evidence.mmd
```

`leaklens --help` lists every flag; [docs/CONFIGURATION.md](docs/CONFIGURATION.md) lists the same options as configuration keys.

## GitHub Action

The Action installs the CLI and runs `leaklens --scan` on a file or folder. It does not run a dataset audit; use a `run:` step for that.

| Input | Required | Default | Purpose |
|---|---|---|---|
| `path` | optional | `.` | file or folder to scan |
| `fail-under` | optional | unset | minimum trust score (0-100); when set, the step also fails on a score below it |

```yaml
name: leakage
on: [push, pull_request]
jobs:
  leaklens:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: eyadstein/leaklens@main
        with:
          path: .
          fail-under: 80
```

The step fails when the scan exits 1: a high-severity finding, or a trust score below `fail-under` when that input is set. Diagnostic findings never fail the step. `@main` tracks the default branch; pin a tag or a full commit SHA for reproducible CI. Release `v0.13.0` predates the `fail-under` input, so use `@main` (or a later tag) when you set it.

## Trust score and findings

LeakLens starts at 100 and deducts points for each active finding, floored at 0:

| Severity | Penalty |
|---|---:|
| high | 35 |
| medium | 15 |
| low | 5 |

Exit codes: `0` for a clean result or diagnostic-only findings, `1` for a high-severity finding, a score below `--fail-under`, or new findings under the opt-in `--fail-on-new`, and `2` for invalid configuration or arguments.

Every finding carries a stable fingerprint and, for the highest-value detectors, a bounded `evidence` list with provenance. Fingerprints survive line moves and message edits, so baselines and suppressions keep their identity; changing the detector, the affected feature or a detector-supplied stable key changes it. JSON reports carry `$schema`, and `leaklens schema` prints the contract; see [docs/REPORT_SCHEMA.md](docs/REPORT_SCHEMA.md) and [docs/DETECTORS.md](docs/DETECTORS.md).

## Reports and the evidence graph

| Flag | Output |
|---|---|
| `--json` | machine-readable report on stdout |
| `--sarif FILE` | SARIF 2.1.0 with evidence and fingerprints |
| `--html FILE` | single self-contained report with severity legend |
| `--badge FILE` | shields.io endpoint JSON with the trust score |
| `--manifest FILE` | unsigned, offline-verifiable audit manifest (`leaklens verify`) |
| `--intelligence` | deterministic confidence, evidence clusters and root-cause analysis |
| `--graph-out FILE` | evidence graph as Mermaid (`.mmd`), DOT (`.dot`) or JSON |
| `--graph-snapshot` / `--graph-baseline` | save and compare evidence across scans |

The evidence graph groups existing findings by an inspectable anchor and links detector evidence; it adds no findings and never changes the score or exit code. Comparisons report `new`, `unchanged`, `changed`, `resolved` and `uncertain`, and `--fail-on-graph-new` is the only opt-in graph gate. See [docs/INTELLIGENCE.md](docs/INTELLIGENCE.md) and [docs/AUDIT_MANIFEST.md](docs/AUDIT_MANIFEST.md).

For GitHub code scanning:

```yaml
permissions:
  contents: read
  security-events: write
steps:
  - uses: actions/checkout@v7
  - run: pip install leaklens-ml
  - run: leaklens --scan . --sarif leaklens.sarif || true
  - uses: github/codeql-action/upload-sarif@v4
    with:
      sarif_file: leaklens.sarif
```

To publish a trust badge, run `leaklens --scan . --badge badge.json`, commit the file, and point a shields.io endpoint badge at its raw URL, as at the top of this file.

## Python API

The library exposes the same engines the CLI uses:

```python
import pandas as pd

from leaklens.leakage import scan, score
from leaklens.scanner import scan_path

findings = scan(pd.read_csv("train.csv"), target="label")
print(score(findings), [(f.severity, f.kind, f.column) for f in findings])

for finding in scan_path("src"):
    print(finding.kind, finding.column, finding.detail)
```

`leaklens.audit.audit(df, target)` returns a dataset profile (rows, columns, cardinality, null rates, target distribution). The JSON report schema, not Python signatures, is the stable contract for downstream integrations.

## Configuration, baselines and suppressions

`leaklens init` writes a commented `.leaklens.toml`; LeakLens reads `./.leaklens.toml` automatically or the file passed with `--config`. Precedence is defaults, then config, then explicit CLI flags. Unknown keys, wrong types and unsafe output paths exit 2 instead of silently changing a scan.

```toml
target = "label"
fail_under = 80

[baseline]
path = "leaklens-baseline.json"
fail_on_new = true

[[suppressions]]
fingerprint = "64-lowercase-hex-characters"
reason = "accepted: the column is recorded before prediction"
```

A baseline records reviewed fingerprints and labels later runs `new`, `unchanged`, `changed` or `resolved`. A suppression accepts one exact fingerprint, requires a reason and never hides future unrelated findings; suppressed findings keep their original severity and evidence but stop counting toward the score and the default gate. Baselines are never overwritten implicitly. Details: [docs/CONFIGURATION.md](docs/CONFIGURATION.md), [docs/BASELINES.md](docs/BASELINES.md).

## Optional engines

Each engine is opt-in. Unless noted, its findings are diagnostic and leave the trust score and exit code unchanged.

- **SQL lineage** (`--sql`, extra `sql`): resolves column lineage through CTEs, joins, unions and aggregate aliases and never executes SQL or opens a connection. A confirmed target-derived projection is `high` and scored; bare projections, renames and filters are clean, and unresolved lineage is a low diagnostic rather than a target finding. See [docs/SQL_LINEAGE.md](docs/SQL_LINEAGE.md).
- **Distribution drift** (`--drift`, needs `--test`): numeric KS, categorical total-variation distance and missingness-rate change, low severity and diagnostic only. See [docs/DRIFT.md](docs/DRIFT.md).
- **Text corpora** (`leaklens text TRAIN TEST`): exact and near duplicates, hash-only evaluation contamination, RAG index wiring and caller-supplied embedding similarity. Evidence carries counts, digests and paths, never document text; no model is downloaded or loaded. See [docs/TEXT.md](docs/TEXT.md).
- **Deep learning** (`leaklens dl DIR`): split overlap, early-stopping and optimizer mismatches, missing seeds and pickle-based artifacts. It parses Python with `ast`, never imports a framework, never unpickles and never extracts an archive. See [docs/DL_AUDIT.md](docs/DL_AUDIT.md).
- **Pipeline metadata** (`--pipeline`): reads DVC stages and locks, serialized DAGs, notebook order and dependency files, and asks whether evaluation is circular. Nothing is executed. See [docs/PIPELINE.md](docs/PIPELINE.md).
- **Experimental AI** (`--ai`): offline, evidence-bound hypotheses and a remediation plan on top of the deterministic findings, capped at medium severity. It never changes a finding, the score or the exit code, and dataset analysis receives column metadata only. Measured results and limits: [docs/AI_BENCHMARK.md](docs/AI_BENCHMARK.md), [docs/AI_INTELLIGENCE.md](docs/AI_INTELLIGENCE.md).
- **LLM explanations** (`--llm`): disabled by default. Only finding metadata is sent (severity, kind, file, line, diagnostic detail), never full files or dataset rows; the default provider is local Ollama. Custom templates use `--llm-prompt-template` or `.leaklens/prompt.txt`; see [docs/PROMPT_INJECTION.md](docs/PROMPT_INJECTION.md) for the trust model.

## Services: API, workspace, Docker

```sh
python -m pip install "leaklens-ml[app]"
uvicorn leaklens.api:app --port 8000    # interactive docs at /docs
streamlit run leaklens/ui.py
leaklens workspace                      # local read-only workspace on 127.0.0.1:8765
```

Endpoints are `GET /health`, `POST /audit`, `POST /scan` and `POST /predict`. `POST /predict` loads one operator-chosen artifact (`LEAKLENS_MODEL_PATH`, default `models/best.joblib`) and rejects any model path in the request; `joblib` uses pickle, so only load models you trust and do not expose the API to untrusted networks. Uploads are capped at 8 MiB and 50,000 rows, with a 256 MiB decompressed container limit and ZIP path validation. See [SECURITY.md](SECURITY.md) and [docs/WORKSPACE.md](docs/WORKSPACE.md).

The public live-scan API (`leaklens.public_api`, extra `api`) is a separate app for untrusted uploads with no `/predict`, no model loading and no disk writes. This repository does not deploy a backend; the static site's live page starts disconnected. Deployment, limits and the security model are in [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md).

```sh
docker compose up --build   # API on :8000, UI on :8501
```

The Docker files are documented but were not executed in the development environment; run them where the daemon is healthy.

## Benchmarks and real-world checks

- The code scanner is checked against 117 hand-written regression cases (63 leaky, 54 clean) with 0 false positives and 0 false negatives: `python -m leaklens.bench`, results in [docs/BENCHMARK.md](docs/BENCHMARK.md).
- Dataset detectors have 12 synthetic cases (`python -m leaklens.bench_data`, [docs/DATA_BENCHMARK.md](docs/DATA_BENCHMARK.md)), and missingness, drift, pipeline, text, deep learning, vision, SQL and AI each have their own labeled benchmarks under `docs/`.
- A static sample of 1,725 public notebooks that split their data found that 38% fit a preprocessing step before the split or on test data ([docs/STUDY.md](docs/STUDY.md)).
- The real-world corpus holds 60 public notebooks, but only 8 have reviewed labels; on those 8 the scanner scores precision/recall/F1 1.00 ([docs/REALWORLD.md](docs/REALWORLD.md)). That is a smoke check on a tiny, non-random sample, not a population accuracy estimate.
- The independent AI check measures accuracy 0.833, precision 0.767, recall 0.958 and FPR 0.292 on 48 reviewed units, with every mistake listed ([docs/AI_REALWORLD.md](docs/AI_REALWORLD.md)); the shipped model has no second independent holdout, and the missing H2 corpus is published as a shortfall rather than a number ([docs/AI_REALWORLD_H2.md](docs/AI_REALWORLD_H2.md)).

All benchmarks are hand-written regression guards shaped around supported rules. They are not real-world accuracy claims, and no benchmark number here is a production guarantee. Scanner runtime measurements and the source-cache decision are in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Limitations

- Detection is heuristic and static. LeakLens does not execute scanned code, run notebooks, connect to a database or perform complete interprocedural dataflow analysis; obfuscated or dynamically constructed patterns can be missed.
- A finding is evidence to inspect, not proof of leakage. An identical feature vector or a group overlap can occur naturally; read the detail and the suggested test before acting.
- Every kind is scoped to the surface that emits it: a code rule does not audit a CSV, and a dataset detector does not read your notebook. Supported formats differ per command.
- Feature-overlap and near-duplicate checks sample at most 50,000 and 20,000 rows, so very large files are bounded rather than exhaustive.
- Diagnostic engines (drift, pipeline, text, deep learning, AI, monitoring intake) never change the trust score or exit code, and AI hypotheses are capped at medium severity and remain experimental.
- Missingness findings cap at medium severity and never modify data or splits.
- Image hashing has documented blind spots (colour-only differences, crops beyond 5%, letterboxing), and declared-split checks compare file bytes only, not perceptual similarity.
- Python 3.13 is outside the CI matrix and is not claimed as verified.

## Security and privacy

- Scanning is local and offline by default. Dataset contents stay on your machine; nothing is uploaded and no service is contacted.
- The code scanner, deep-learning audit and pipeline audit parse files without executing them. They never import torch/TensorFlow, never unpickle a checkpoint and never extract an archive.
- `--llm` is opt-in. It sends finding metadata only, and a remote OpenAI-compatible provider receives that metadata only after you configure it. Provider URLs must use HTTP(S), redirects are refused, and API keys are not sent over plain HTTP to non-local hosts.
- `POST /predict` trusts an operator-chosen pickle; treat the FastAPI app as local and trusted. The public API is the hardened surface for untrusted uploads.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). CI runs bandit, pip-audit, pip-licenses and CodeQL; the release workflow publishes a CycloneDX SBOM outside `dist/` (`python scripts/generate_sbom.py` writes `sbom/sbom.json`).

## Contributing and development

```sh
git clone https://github.com/eyadstein/leaklens.git
cd leaklens
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"
```

Every pull request must pass the same checks CI runs; the short version:

```sh
pytest -q
ruff check leaklens tests scripts
ruff format --check leaklens tests scripts
leaklens --scan leaklens
python -m leaklens.bench
python -m leaklens.bench_data
```

Add a test for every new detector or fix, and keep a new detector in `leaklens/leakage.py` (data) or `leaklens/scanner.py` (code). See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Issues labeled **good first issue** are the easiest way in.

Pre-commit hook:

```yaml
repos:
  - repo: https://github.com/eyadstein/leaklens
    rev: main
    hooks:
      - id: leaklens
```

`leaklens init` prints this snippet. Pin a released tag or commit instead of `main` when you want the hook to be reproducible.

## Roadmap

The published release is **0.13.0**; changes merged on `main` but not yet released are listed under "Unreleased" in [CHANGELOG.md](CHANGELOG.md). Release history lives there, not in this file.

Implemented: dataset and code detectors, model arena, API, workspace, HTML/SARIF/JSON reports, baselines and suppressions, the evidence graph and investigation plans, the governance pack, group leakage, image clustering and declared-split audits, and the opt-in SQL, text, deep-learning, drift, pipeline and AI engines.

Planned work and open design questions are tracked in [ROADMAP.md](ROADMAP.md), the longer-term [AI/ML expansion roadmap](docs/AI_ML_EXPANSION_ROADMAP.md), and [Issues](https://github.com/eyadstein/leaklens/issues). Proposed items there are not commitments.

## License and citation

MIT licensed; see [LICENSE](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).

Contributors: thanks to @jay23606 (Parquet input), @slopggobblercobbler81 (`--fail-under`), @tyza66 (Excel, JSON and Feather input), @rahul05ranjan (`random_state` check), @have-you-tried-restarting (severity summary), @Rashmi-R444 (trust score docs), @Thecoolss (benchmark snippets), @ElDShell (Arabic README), @Aaqibhafeezkhan (colored severity labels and `CITATION.cff`), @Cami0111 (leak walkthrough notebook), @rahul-jain-dev (`--version`) and @Vamsi-2581 (opt-in LLM explanations).
