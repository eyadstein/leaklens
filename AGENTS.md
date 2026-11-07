# LeakLens — agent notes

Single-package Python CLI (`leaklens-ml` on PyPI, command `leaklens`). No monorepo, no build step.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"   # dev + app (FastAPI/Streamlit) for tests & UI
```

Python `>=3.10`; CI tests 3.10–3.12. Optional-format extras are separate (`arena`, `parquet`, `feather`, `excel`, `images`, `app`, `sql`) — don't assume `pyarrow`/`mlflow`/`pillow`/`openpyxl`/`sqlglot` are present unless installed.

## Verify (CI order — `.github/workflows/ci.yml`)

```sh
pytest --cov=leaklens --cov-report=term-missing -q   # or: pytest -q
leaklens --scan leaklens                            # self-scan must report no leaks
python -m leaklens.bench                            # labeled scanner regression benchmark
python -m leaklens.bench_data                       # dataset-detector benchmark
python -m leaklens.bench_missing                    # labeled missingness benchmark
python -m leaklens.bench_drift                      # drift diagnostics benchmark
python -m leaklens.bench_ai                         # optional AI layer benchmark
python -m leaklens.bench_sql                        # optional SQL benchmark (needs .[sql])
```

Single test: `pytest tests/test_scanner.py -q`. Every PR must pass all three (see `CONTRIBUTING.md`). Release = create a GitHub release (`.github/workflows/release.yml` publishes to PyPI); version lives in `pyproject.toml` + `leaklens/__init__.py` and must stay in sync.

## Layout — where things go

| Concern | File |
|---|---|
| CLI wiring, exit code, all flags | `leaklens/cli.py` (`main`, plus `init`/`images` subcommands) |
| Data detectors | `leaklens/leakage.py` — returns `Finding(severity, kind, column, detail)` |
| Code scanner (`.py`/`.ipynb`) | `leaklens/scanner.py` (`scan_path`); pairs: `leaklens/bench.py` + `bench_cases.py` |
| Optional SQL lineage | `leaklens/sqlscan.py` (lazy SQLGlot); benchmark `leaklens/bench_sql.py` + `docs/SQL_BENCHMARK.md`; docs `docs/SQL_LINEAGE.md` |
| Opt-in drift diagnostics | `leaklens/drift.py`; benchmark `leaklens/bench_drift.py` + `docs/DRIFT_BENCHMARK.md`; docs `docs/DRIFT.md` |
| Dataset input dispatch | `leaklens/readers.py` — suffix → pandas reader |
| Model arena (MLflow) | `leaklens/arena.py`; API `leaklens/api.py`; UI `leaklens/ui.py` |
| Deterministic intelligence layer | `leaklens/intelligence/` (`orchestrator.py`, `schemas.py`) |
| Reports / exports | `leaklens/report.py` (HTML), `leaklens/sarif.py` (SARIF), `leaklens/fixes.py` (`--explain`) |
| Dataset audit + CV check + LLM | `leaklens/audit.py` (class balance etc.), `leaklens/cvcheck.py` (CV misuse, used by scanner), `leaklens/llm.py` (`--llm` explanations, off by default) |
| Baselines / suppressions / config | `leaklens/baseline.py`, `leaklens/suppression.py`, `leaklens/config.py`; shared application in `leaklens/policy.py` |
| Missingness audit | `leaklens/missingness.py`; benchmark `leaklens/bench_missing.py` + `docs/MISSINGNESS_BENCHMARK.md` |
| Experimental AI layer | `leaklens/ai/` (`semantics.py`, `features.py`, `corpus.py`, `learned.py`, `reasoning.py`, `schemas.py`, `analysis.py`); benchmark `leaklens/bench_ai.py` + `docs/AI_BENCHMARK.md`; design `docs/AI_INTELLIGENCE.md` |

New detectors: data → `leakage.py`, code → `scanner.py`, SQL → `sqlscan.py`. Add a test in `tests/`. `Finding.severity` is one of `high|medium|low`; trust score (100 − 35/15/5 per finding) is computed centrally in `leakage.score()` — don't duplicate it. `Finding.diagnostic=True` marks opt-in diagnostics that `score()` and the default exit gate ignore.

## Non-obvious behavior

- **Exit code**: CLI exits 1 if any `high` finding OR trust score `< --fail-under`. `leaklens --scan` in CI/pre-commit relies on this.
- **Readers**: unknown/extensionless suffix silently falls back to CSV (`readers.py`); optional formats raise `ImportError` naming the missing extra.
- **`.leaklens.toml`** is read from CWD (or `--config`) when present. Precedence: defaults < config < explicit CLI flags. Invalid keys/types/paths exit 2. Output paths must stay inside the config directory. `leaklens init` still refuses to overwrite without `--force`.
- **Baselines/suppressions**: baseline entries key on the stable fingerprint; comparison statuses are `new|unchanged|changed|resolved`. Suppressions are exact-fingerprint only, require a reason, and only label findings — they never change severity/evidence/fingerprint, and suppressed findings stop counting toward `leakage.score()` and the default exit gate. `--fail-on-new` is the only opt-in new failure mode.
- **Missingness** findings (`missingness_target_signal`, `missingness_split_shift` max `medium`; `missingness_group_signal` low) are additive, deterministic and never modify data or splits.
- **SQL** is optional (`--sql`, `--sql-dialect`, `[sql]`; extra `sql = ["sqlglot>=25"]`). Never execute SQL or connect to a database. Dialects are explicit with no fallback. Only confirmed lineage may produce a high `sql_target_as_feature`; unresolved/ambiguous lineage is a low `sql_unresolved_column` diagnostic. Bare projections, renames and filters are clean.
- **Drift** is opt-in (`--drift`, `[drift].enabled`), low severity and always `diagnostic=True`; it must never change `leakage.score()`, the exit code, or the data/splits. No SciPy.
- **LLM explanations** are off by default; only opt-in via `--llm`. Never send dataset rows or full source — findings only (provider: local Ollama by default; `LEAKLENS_LLM_API_KEY` for remote).
- **AI layer** is off by default (`--ai`, `--ai-llm`, `include_ai`). It must stay additive: never touch `Finding` severity, `leakage.score()`, the exit code or raw rows; hypotheses are capped at `medium`, confidence at `0.95`, and evidence is mandatory.
- **Benchmark**: `python -m leaklens.bench` output goes to `docs/BENCHMARK.md`; treat as regression guard, not accuracy claim. The AI benchmark is `python -m leaklens.bench_ai` → `docs/AI_BENCHMARK.md`; report its measured numbers, never an improvement claim.
- Live demo: `python examples/make_demo.py && leaklens demo.csv --target is_fraud --arena`. API: `uvicorn leaklens.api:app --port 8000`; UI: `streamlit run leaklens/ui.py`; Docker: `docker compose up --build` (API :8000, UI :8501).
