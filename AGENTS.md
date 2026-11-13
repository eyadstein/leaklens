# LeakLens — agent notes

Single-package Python CLI (`leaklens-ml` on PyPI, command `leaklens`). No monorepo, no build step.

## Setup

```sh
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev,app]"   # dev + app (FastAPI/Streamlit) for tests & UI
```

Python `>=3.10`; CI tests 3.10–3.12. Optional-format extras are separate (`arena`, `parquet`, `feather`, `excel`, `images`, `app`, `sql`, `yaml`) — don't assume `pyarrow`/`mlflow`/`pillow`/`openpyxl`/`sqlglot`/`PyYAML` are present unless installed.

## Verify (CI order — `.github/workflows/ci.yml`)

```sh
pytest --cov=leaklens --cov-report=term-missing -q   # or: pytest -q
leaklens --scan leaklens                            # self-scan must report no leaks
python -m leaklens.bench                            # labeled scanner regression benchmark
python -m leaklens.bench_data                       # dataset-detector benchmark
python -m leaklens.bench_missing                    # labeled missingness benchmark
python -m leaklens.bench_drift                      # drift diagnostics benchmark
python -m leaklens.bench_pipeline                   # pipeline/MLOps benchmark (needs .[yaml])
python -m leaklens.bench_text                       # text/LLM contamination benchmark
python -m leaklens.bench_dl                         # deep-learning and artifact benchmark
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
| Shared near-duplicate primitive | `leaklens/near_duplicate.py` (`bucket → threshold → stable cluster key`, ADR-022); consumed by text dedup, Phase 17 vision next |
| Text/LLM engine | `leaklens/analysis/text/` (`normalize.py`, `dedup.py`, `contamination.py`, `rag.py`, `embeddings.py`); local HF snapshot reader `leaklens/integrations/hf.py`; benchmark `leaklens/bench_text.py` + `docs/TEXT_BENCHMARK.md`; docs `docs/TEXT.md`; surface `leaklens text` in `cli.py` |
| Deep-learning and artifact engine | `leaklens/analysis/dl/` (`training.py`, `config.py`, `checkpoints.py`, `__init__.py` = `audit_dl`); artifact intake `leaklens/analysis/artifacts/` (`formats.py`, `safetensors.py`, `pickle_guard.py`, `onnx.py`); benchmark `leaklens/bench_dl.py` + `docs/DL_BENCHMARK.md`; docs `docs/DL_AUDIT.md`; surface `leaklens dl` in `cli.py` |
| Opt-in drift diagnostics | `leaklens/drift.py`; benchmark `leaklens/bench_drift.py` + `docs/DRIFT_BENCHMARK.md`; docs `docs/DRIFT.md` |
| Dataset input dispatch | `leaklens/readers.py` — suffix → pandas reader |
| Model arena (MLflow) | `leaklens/arena.py`; API `leaklens/api.py`; UI `leaklens/ui.py` |
| Interactive scan workspace | `leaklens/workspace.py` (FastAPI app + scan session), `leaklens/workspace_ui.py` (page), `leaklens/cli.py` (`workspace` subcommand) |
| Deterministic intelligence layer | `leaklens/intelligence/` (`orchestrator.py`, `schemas.py`) |
| Reports / exports | `leaklens/report.py` (HTML), `leaklens/sarif.py` (SARIF), `leaklens/fixes.py` (`--explain`) |
| Dataset audit + CV check + LLM | `leaklens/audit.py` (class balance etc.), `leaklens/cvcheck.py` (CV misuse, used by scanner), `leaklens/llm.py` (`--llm` explanations, off by default) |
| Baselines / suppressions / config | `leaklens/baseline.py`, `leaklens/suppression.py`, `leaklens/config.py`; shared application in `leaklens/policy.py` |
| Shared payload sections (findings/counts/policy) | `leaklens/assemble.py` — one policy application for CLI/API/live scan; never re-serialize findings per surface |
| Shared scanner vocabulary/AST helpers | `leaklens/source.py` — `cvcheck.py`/`flow.py` import from here, `scanner.py` re-exports; no cycles |
| Optional-engine availability | `leaklens/availability.py` (`probe`, `EngineUnavailable`); `sqlscan`/`images` override the error class |
| Path containment | `leaklens/paths.py` (`resolve_within`, `contained`); refuses foreign drives/UNC anchors before `resolve()` |
| Missingness audit | `leaklens/missingness.py`; benchmark `leaklens/bench_missing.py` + `docs/MISSINGNESS_BENCHMARK.md` |
| Experimental AI layer | `leaklens/ai/` (`semantics.py`, `features.py`, `corpus.py`, `learned.py`, `reasoning.py`, `schemas.py`, `analysis.py`); benchmark `leaklens/bench_ai.py` + `docs/AI_BENCHMARK.md`; design `docs/AI_INTELLIGENCE.md` |
| Prompt-injection corpus (M-S6) | `leaklens/ai/prompt_injection.py` + `prompt_injection_corpus.json`; `tests/test_prompt_injection.py`; `python -m leaklens.ai.prompt_injection --check`; docs `docs/PROMPT_INJECTION.md` |

New detectors: data → `leakage.py`, code → `scanner.py`, SQL → `sqlscan.py`. Add a test in `tests/`. `Finding.severity` is one of `high|medium|low`; trust score (100 − 35/15/5 per finding) is computed centrally in `leakage.score()` — don't duplicate it. `Finding.diagnostic=True` marks opt-in diagnostics that `score()` and the default exit gate ignore.

## Non-obvious behavior

- **Exit code**: CLI exits 1 if any `high` finding OR trust score `< --fail-under`; findings with `diagnostic=True` are ignored by both gates (and by `--fail-on-new`). `leaklens --scan` in CI/pre-commit relies on this.
- **Readers**: unknown/extensionless suffix silently falls back to CSV (`readers.py`); optional formats raise `ImportError` naming the missing extra.
- **`.leaklens.toml`** is read from CWD (or `--config`) when present. Precedence: defaults < config < explicit CLI flags. Invalid keys/types/paths exit 2. Output paths must stay inside the config directory. `leaklens init` still refuses to overwrite without `--force`.
- **Baselines/suppressions**: baseline entries key on the stable fingerprint; comparison statuses are `new|unchanged|changed|resolved`. Suppressions are exact-fingerprint only, require a reason, and only label findings — they never change severity/evidence/fingerprint, and suppressed findings stop counting toward `leakage.score()` and the default exit gate. `--fail-on-new` is the only opt-in new failure mode.
- **Missingness** findings (`missingness_target_signal`, `missingness_split_shift` max `medium`; `missingness_group_signal` low) are additive, deterministic and never modify data or splits.
- **SQL** is optional (`--sql`, `--sql-dialect`, `[sql]`; extra `sql = ["sqlglot>=25"]`). Never execute SQL or connect to a database. Dialects are explicit with no fallback. Only confirmed lineage may produce a high `sql_target_as_feature`; unresolved/ambiguous lineage is a low `sql_unresolved_column` diagnostic. Bare projections, renames and filters are clean.
- **Text** is opt-in via `leaklens text TRAIN TEST`. All five kinds are `diagnostic=True`/`scored=False` and never change `leakage.score()` or an existing exit gate; `--fail-on-findings` is the only opt-in failure. Evidence carries counts, 16-hex digests, relative paths and call names, never document text. Contamination is hash-only against user-supplied eval fingerprints (no corpus ships). RAG analysis is AST wiring plus literal path containment, not dataflow. Embeddings are caller-supplied `.npy` vectors and never load a model.
- **Drift** is opt-in (`--drift`, `[drift].enabled`), low severity and always `diagnostic=True`; it must never change `leakage.score()`, the exit code, or the data/splits. No SciPy.
- **Deep learning** is opt-in via `leaklens dl DIR`. All findings are `diagnostic=True`/`scored=False` and never change `leakage.score()` or an existing exit gate; `--fail-on-findings` is the only opt-in failure. The audit never imports torch/TensorFlow/Keras/ONNX (AST-gated by `tests/_meta/test_no_torch_load.py`), never unpickles, never extracts an archive and never executes scanned code. Pickle magic is checked before the extension; archives are typed skips; safetensors/ONNX/JSON parse under byte/depth caps. Split provenance pairs bindings inside one file and never guesses unresolved `Subset` indices. M4 reuses `no_random_state` and `pipeline_unpinned_dependency`; this phase defines no new reproducibility family.
- **LLM explanations** are off by default; only opt-in via `--llm`. Never send dataset rows or full source — findings only (provider: local Ollama by default; `LEAKLENS_LLM_API_KEY` for remote).
- **AI layer** is off by default (`--ai`, `--ai-llm`, `include_ai`). It must stay additive: never touch `Finding` severity, `leakage.score()`, the exit code or raw rows; hypotheses are capped at `medium`, confidence at `0.95`, and evidence is mandatory.
- **Workspace dataset audits** reuse `public_api.run_audit_job` / `run_inspect_job` under `upload_guard.guard_upload` limits; never re-implement parsing, target checks or detectors in `workspace.py`. A source scan and a dataset audit share one activity slot (409 on overlap), and an audit cannot be cancelled once the engine call is running. Both scan types apply the existing `policy.apply_policy()` to their serialized findings; the workspace loads baselines read-only (relative to the root or `[baseline].path`), never writes them, and shows only statuses the policy engine returned.
- **`POST /predict`** never accepts a request-selected model. The server loads `LEAKLENS_MODEL_PATH` (default `models/best.joblib`) and returns 422 for any other `model_path` before touching the filesystem. Pickle artifacts are trusted operator input only.
- **Paths from requests** go through `paths.resolve_within`; it refuses foreign drives/anchor (UNC, device paths) before `Path.resolve()`, so a rejected path cannot trigger network resolution.
- **Benchmark**: `python -m leaklens.bench` output goes to `docs/BENCHMARK.md`; treat as regression guard, not accuracy claim. The AI benchmark is `python -m leaklens.bench_ai` → `docs/AI_BENCHMARK.md`; report its measured numbers, never an improvement claim.
- Live demo: `python examples/make_demo.py && leaklens demo.csv --target is_fraud --arena`. API: `uvicorn leaklens.api:app --port 8000`; UI: `streamlit run leaklens/ui.py`; Docker: `docker compose up --build` (API :8000, UI :8501).
