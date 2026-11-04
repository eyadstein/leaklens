# LeakLens — agent notes

Single-package Python CLI (`leaklens-ml` on PyPI, command `leaklens`). No monorepo, no build step.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"   # dev + app (FastAPI/Streamlit) for tests & UI
```

Python `>=3.10`; CI tests 3.10–3.12. Optional-format extras are separate (`arena`, `parquet`, `feather`, `excel`, `images`, `app`) — don't assume `pyarrow`/`mlflow`/`pillow`/`openpyxl` are present unless installed.

## Verify (CI order — `.github/workflows/ci.yml`)

```sh
pytest --cov=leaklens --cov-report=term-missing -q   # or: pytest -q
leaklens --scan leaklens                            # self-scan must report no leaks
python -m leaklens.bench                            # labeled scanner regression benchmark
```

Single test: `pytest tests/test_scanner.py -q`. Every PR must pass all three (see `CONTRIBUTING.md`). Release = create a GitHub release (`.github/workflows/release.yml` publishes to PyPI); version lives in `pyproject.toml` + `leaklens/__init__.py` and must stay in sync.

## Layout — where things go

| Concern | File |
|---|---|
| CLI wiring, exit code, all flags | `leaklens/cli.py` (`main`, plus `init`/`images` subcommands) |
| Data detectors | `leaklens/leakage.py` — returns `Finding(severity, kind, column, detail)` |
| Code scanner (`.py`/`.ipynb`) | `leaklens/scanner.py` (`scan_path`); pairs: `leaklens/bench.py` + `bench_cases.py` |
| Dataset input dispatch | `leaklens/readers.py` — suffix → pandas reader |
| Model arena (MLflow) | `leaklens/arena.py`; API `leaklens/api.py`; UI `leaklens/ui.py` |
| Deterministic intelligence layer | `leaklens/intelligence/` (`orchestrator.py`, `schemas.py`) |
| Reports / exports | `leaklens/report.py` (HTML), `leaklens/sarif.py` (SARIF), `leaklens/fixes.py` (`--explain`) |
| Dataset audit + CV check + LLM | `leaklens/audit.py` (class balance etc.), `leaklens/cvcheck.py` (CV misuse, used by scanner), `leaklens/llm.py` (`--llm` explanations, off by default) |

New detectors: data → `leakage.py`, code → `scanner.py`. Add a test in `tests/`. `Finding.severity` is one of `high|medium|low`; trust score (100 − 35/15/5 per finding) is computed centrally in `leakage.score()` — don't duplicate it.

## Non-obvious behavior

- **Exit code**: CLI exits 1 if any `high` finding OR trust score `< --fail-under`. `leaklens --scan` in CI/pre-commit relies on this.
- **Readers**: unknown/extensionless suffix silently falls back to CSV (`readers.py`); optional formats raise `ImportError` naming the missing extra.
- **`.leaklens.toml`** (`leaklens init` output) is **not read yet** — it's documentation-only; flags on the CLI win.
- **LLM explanations** are off by default; only opt-in via `--llm`. Never send dataset rows or full source — findings only (provider: local Ollama by default; `LEAKLENS_LLM_API_KEY` for remote).
- **Benchmark**: `python -m leaklens.bench` output goes to `docs/BENCHMARK.md`; treat as regression guard, not accuracy claim.
- Live demo: `python examples/make_demo.py && leaklens demo.csv --target is_fraud --arena`. API: `uvicorn leaklens.api:app --port 8000`; UI: `streamlit run leaklens/ui.py`; Docker: `docker compose up --build` (API :8000, UI :8501).
