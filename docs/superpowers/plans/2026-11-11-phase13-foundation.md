# Phase 13 — shared engine foundation (implementation plan)

**Status:** draft, awaiting Ponytail review before any code lands.
**Starting commit:** `4dd4f6f595296397b53054fbca79e36925862b2f` (`main` == `origin/main`).
**Scope:** Phase 13 M1–M9 only. No Phase 14+ engine, no new dependency, no version bump,
no release.

## 0. Non-negotiable invariants (checked at every milestone)

1. `leakage.score()` stays the only scoring path; `assemble.py` calls it, never reimplements it.
2. Findings stay deterministic and byte-identical for the same input.
3. AI / intelligence output stays additive and distinguishable; no AI value enters `Finding`.
4. Confidence and selective-reporting semantics unchanged.
5. Evidence/provenance unchanged (same fields, same bounds).
6. Offline, CPU-first default; optional extras stay optional.
7. Existing API/CLI/workspace/report payload fields, key names, engine ids and exit codes
   unchanged except where a milestone explicitly changes them: M9 (diagnostic gate) and M7
   (`/predict` model selection). The M7 change is a documented security break, not a silent one.
8. No alert suppression, no CodeQL config change, no unverified "fixed" claim.
9. `.phase5-bench.json` and `graphify-out/` are never staged or modified. Graphify output goes
   to `%TEMP%\leaklens-graph`.

## 1. Verified milestone dependency audit

Source of truth: `docs/AI_ML_EXPANSION_ROADMAP.md` §2 (lines 71–143) and
`docs/ARCHITECTURE_DECISIONS.md` ADR-005/019/020/023/024.

| Id | State on `main` | Prerequisites | Eligible now? |
|---|---|---|---|
| M1 catalogue/engine/availability | absent | none | **yes** — pure data + stdlib module, no attack surface |
| M2 `assemble.py` | absent | M1 (engine ids/modalities for payload metadata) | **yes** after M1 |
| M3 `source.py` | absent | none (ADR-019) | **yes** |
| M4 break `ai/artifact` ↔ `ai/learned` | mutual, function-local; `artifact.load_artifact` is a shipped public helper | none | **deferred** — cannot be done without either changing `tests/test_ai_artifact.py` (forbidden by phase acceptance) or a new shared model module; see §10 |
| M5 import-matrix test | absent | M3 (needs the final graph) | **yes** after M3 |
| M6 freeze surfaces/engines/fingerprints | absent | M1 (ENGINES) | **yes** after M1 |
| M7 security gate + `paths.py` | absent; 20 alerts open | none; must precede new intake | **yes, highest priority after M1/M9** |
| M8 release allowlist + SHA pins | absent | none | **yes** |
| M9 diagnostic-aware gate | absent (`cli.py:1148`) | M1 (`scored` documents the rule) | **yes** |

**Verdict on the roadmap's sequence.** "M1 → M2 → security gate" is valid but not optimal:
M1 adds no runtime surface; M9 and M7 are independent of M2 and M7 is blocking, so run
**M1 → M9 → M7/M-S0 → M2 → M3 → M5 → M6 → M8**. M2 is still the structural change every
later phase depends on, and it lands before any Phase 14 work. M1 and M9 can proceed before
M-S0 because they introduce no new intake, model loading, integration or agent surface. M4 is
deferred with a written reason (§10).

**Security status (verified 2026-11-11 via `gh api`):** 16 `py/path-injection`
(`api.py:314,320`; `baseline.py:236`; `scanner.py:1019,1023`; `workspace.py:228,1107,1152,1154×2,1304,1306,1313,1928,1930,1932`),
3 `py/stack-trace-exposure` (`workspace.py:1680,1712,1716`), 1 `py/unsafe-deserialization`
(`api.py:325`). Unchanged since the architecture report. M-S0 is **not** done.

---

## 2. M1 — catalogue, `ENGINES`, `availability`

### 2.1 `leaklens/availability.py` (new, stdlib only)

```python
class EngineUnavailable(ImportError):
    """An optional dependency an engine requires is not installed."""

def probe(
    module: str,
    extra: str,
    *,
    purpose: str,
    label: str | None = None,
    error: type[EngineUnavailable] = EngineUnavailable,
) -> ModuleType:
    """Import ``module`` or raise ``error`` naming the missing extra."""
```

- `importlib.import_module(module)` inside `try/except ImportError`; re-raise
  `error(f'{purpose} need {label or module}: pip install "leaklens-ml[{extra}]"') from None`; return
  the module on success.
- Must not import any engine (ADR-023). Only `importlib`.

### 2.2 `leaklens/manifest.py`

- `MODALITIES = frozenset({"tabular", "code", "sql", "image", "text", "artifact"})` beside
  `SURFACES` (ADR-020). `video` is deliberately absent (Phase 17).
- Extend the frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class Detector:
    kind: str
    surface: str
    severity: str
    advice: str
    benchmark: str | None
    no_bench_reason: str | None
    docs_anchor: str
    introduced: str
    scored: bool = False
    requires: str | None = None      # optional extra name, e.g. "images"
    modalities: frozenset[str] = frozenset()
```

- `_entry(..., *, scored: bool, requires: str | None = None, modalities: tuple[str, ...])`
  with **no default for `scored` or `modalities`**, so every call site is explicit and the registry
  cannot be half-populated. Existing kinds keep `kind/surface/severity/advice/benchmark/...`
  values verbatim.
- Backfill rule:
  - `distribution_shift`, `sql_unresolved_column` → `scored=False` (they are
    `diagnostic=True` today). Every other existing kind → `scored=True`.
  - `image_overlap` → `requires="images"`, `modalities=("image",)`.
  - `sql_target_as_feature`, `sql_split_predicate_missing`, `sql_unresolved_column`,
    `sql_future_window` → `requires="sql"`, `modalities=("sql",)`.
  - `surface="code"` non-SQL kinds → `modalities=("code",)`.
  - `surface="notebook"` → `modalities=("code",)`.
  - `surface="dataset"` and `"split"` → `modalities=("tabular",)`.
- `Detector.to_dict()` gains `scored`, `requires`, `modalities` (sorted list for JSON); return
  type widens to `dict[str, object]`.
- `Engine` frozen dataclass + `ENGINES: dict[str, Engine]` keyed by literal id (ADR-005):

```python
@dataclass(frozen=True, slots=True)
class Engine:
    id: str
    surface: str
    requires: str | None
    kinds: frozenset[str]

ENGINES = {
    "leaklens.python": Engine("leaklens.python", "dataset", None, frozenset({...dataset+split kinds...})),
    "leaklens.scanner.scan_source": Engine(..., "code", None, frozenset({...code+notebook kinds...})),
    "leaklens.ai.experimental": Engine(..., "dataset", None, frozenset()),
    "leaklens.intelligence.deterministic": Engine(..., "dataset", None, frozenset()),
}
```

  Ids are literals, never `__module__` (ADR-005). `ai.experimental` / `intelligence.deterministic`
  emit hypotheses, not finding kinds, so `kinds=frozenset()` is honest.
- `validate_manifest()` adds: `scored` is bool, `requires` is `None` or a known extra
  (`{"images","sql"}` for now), `modalities <= MODALITIES` and non-empty, every engine id
  non-empty, every `Engine.kinds` subset of `BY_KIND`, and every engine `requires` known.
  **Do not** require engine kinds to cover every kind: `image_overlap` has no frozen engine id
  today, and inventing a fifth id would change the public contract (ADR-005).
- `render_docs()` adds columns `Scored`, `Requires`, `Modalities` to the table and the per-kind
  section. Regenerate `docs/DETECTORS.md` with `python -m leaklens.manifest`.

### 2.3 `leaklens/sqlscan.py`

`class SqlUnavailable(EngineUnavailable)`; keep the class name and message. `SqlUnavailable` stays
an `ImportError` through inheritance.

### 2.4 `leaklens/images.py`

```python
class ImageUnavailable(EngineUnavailable):
    """Pillow is not installed; image checks cannot run."""

def _pil():
    return probe(
        "PIL.Image", "images", purpose="image checks", label="Pillow", error=ImageUnavailable
    )
```

`probe` imports `PIL.Image` (the module `dhash`/`image_overlap` call `Image.open` on) and raises
`ImageUnavailable` with `image checks need Pillow: pip install "leaklens-ml[images]"`.
so the helper has one production caller and one message format. `ImportError` behaviour is preserved;
callers/tests matching `pip install` keep passing.

### 2.5 TDD steps

1. `tests/test_availability.py`: probe success (stdlib module), missing module raises
   `EngineUnavailable` with the extra name, `SqlUnavailable`/`ImageUnavailable` are
   `EngineUnavailable` and `ImportError` subclasses.
2. Extend `tests/_meta/test_detector_contract.py`:
   - every emitted kind has `scored` and at least one modality;
   - `requires` is `images`/`sql`/`None` and the expected kinds carry the expected extra;
   - `scored is False` exactly for the two diagnostic kinds.
3. New `tests/test_public_contract.py` (M1 part):
   - `MODALITIES == {tabular, code, sql, image, text, artifact}`;
   - `ENGINES` ids exactly the four frozen ids;
   - a test-local literal `{"data": "dataset", "split": "split", "code": "code",
     "notebook": "notebook", "image": "image"}` maps every `evidence.SOURCES` member to a real
     `SURFACES` member, proving the legacy `data` spelling has an explicit catalogue equivalent;
   - every `Engine.kinds` is a subset of `BY_KIND`.
4. Regenerate docs and run `python -m leaklens.manifest --check` (no CI edit: 
   `tests/_meta/test_detector_contract.py::test_docs_match_the_manifest` already enforces it).

---

## 3. M9 — diagnostic-aware failure predicates

**Files:** `leaklens/cli.py`.

- Gate (line 1148): `any(f.severity == "high" and not f.diagnostic for f in active)`.
- `--fail-on-new` (line 1151): `any(not f.diagnostic for f in outcome.diff.new)`.
- `--fail-on-graph-new` is unaffected (graph diff carries no `Finding`).
- `leakage.score()` already skips diagnostics; do not touch it.

**TDD:** `tests/test_cli_gate.py`
1. A synthetic high `diagnostic=True` finding → exit 0 (via `main([...])` with a fixture or by
   unit-testing the extracted predicate; prefer extracting `_gate(...)` so the test is cheap).
2. Same finding with `diagnostic=False` → exit 1.
3. `--fail-on-new` with only a new diagnostic → exit 0; with a new normal finding → exit 1.
4. Existing CLI tests unchanged.

---

## 4. M7 / M-S0 — security gate and `paths.py`

### 4.1 `leaklens/paths.py` (new, stdlib only)

```python
class PathOutsideRoot(ValueError):
    """A supplied path resolves outside its containment root."""

def contained(root: Path, target: Path) -> bool:
    """True when target is root or a descendant (both already resolved)."""

def resolve_within(root: Path, value: str) -> Path:
    """Normalise, refuse UNC/device escapes, resolve, then enforce containment."""
```

Rules:
- `root = Path(root).resolve()` once by the caller; `value` is stripped, non-empty.
- Parse with the same `\` → `/` normalisation as today (kept for the Windows-separator tests).
- **Before** `resolve()`, refuse any parsed path whose drive/anchor cannot be the root's:
  ```python
  if candidate.drive and candidate.drive.lower() != root.drive.lower():
      raise PathOutsideRoot(...)          # C:\, C:foo, \\host\share, \\?\C:, \\.\...
  if candidate.anchor and candidate.anchor.lower() != root.anchor.lower():
      raise PathOutsideRoot(...)          # UNC/device anchor differing from a local root
  ```
  This covers UNC, `\\?\`, `\\.\`, absolute drives and drive-relative `C:foo` in one place, before
  any filesystem resolution (so no SMB/NTLM probe). A UNC workspace root still works when the
  candidate shares its anchor; a relative path has no drive/anchor and falls through.
- Resolve, then `if target != root and not contained(root, target): raise`.
- Symlinks are handled because `resolve()` follows them before the containment check.

`WorkspaceManager.resolve` becomes:
```python
try:
    return resolve_within(self.root, value)
except PathOutsideRoot as exc:
    raise WorkspacePathError("path is outside the workspace root") from None
```
Keep `WorkspacePathError` (public error type) and the existing messages so the 422 payload and tests
do not change. Route `workspace._policy`'s containment check (line 1129) through `contained()`.
Route `api.predict` through `resolve_within(models_root, req.model_path)` (after the G1 change below).
Leave `config._output_path` semantics alone (it validates operator config, reports `ConfigError`);
it may call `contained()` but must not gain a new message.

### 4.2 G1 — no request-time deserialization of a request-selected file

**File:** `leaklens/api.py`.

Chosen option: **model fixed at startup** (ADR-016, security strategy §2.2 G1).

- Add `MODEL_PATH = Path(os.environ.get("LEAKLENS_MODEL_PATH", "models/best.joblib"))` as a
  helper read **inside** the loader, never at import time.
- `/predict` ignores `req.model_path` for selection; a request naming any other value gets a
  generic `422` ("model_path is fixed by the server via LEAKLENS_MODEL_PATH"). Absolute,
  traversal and arbitrary values are all simply "not the fixed path".
- `_load_model(path: Path)` resolves the path once, then loads and caches **keyed by the resolved
  path** (`lru_cache(maxsize=4)`), so a test that changes CWD/env cannot get a warm model from a
  previous test. Expose a `cache_clear()`-style hook (or a module dict) and clear it in the test
  fixture. Existence failure → `404` with a generic message (no path echo).
- Keep the deserialization to an operator-controlled, startup-fixed file; document in the docstring
  that `./models/` must be operator-trusted and that pickle is never accepted from a request.
- `PredictReq.model_path` stays in the schema for compatibility (default `models/best.joblib`) but
  no longer selects a file.

**This is a deliberate breaking change**, and it edits three shipped tests (they currently exercise
request-selected loading):
- `tests/test_api.py::test_predict_missing_model` → expects 404 for `models/x.joblib`; rewrite to
  assert 422 "fixed by the server" and prove `joblib.load` was not called.
- `tests/test_api.py::test_predict_roundtrip` → keep 200 by setting the fixed model (default
  `models/best.joblib`) before the request; assert predictions.
- `tests/test_upload_container_limits.py::test_predict_invalid_model_rejected` → rewrite to a
  *fixed-path* invalid artifact (404/400 from the fixed loader, never from a request choice).
- Document the break in `README.md` (API section) and note it in the final report as an explicit
  exception to phase criterion 1, justified by M7.

**TDD (`tests/test_api_predict.py`):**
1. Request with `model_path="../../evil.joblib"` → 422, and the file is never opened
   (monkeypatch `joblib.load` to fail the test if called).
2. Request with a path under `./models/` that is **not** the configured model → 422.
3. Configured model (monkeypatched loader returning a stub with `.predict`) → 200 and the stub is
   loaded exactly once across two requests.
4. Error body never contains a traceback string or the configured absolute path.

### 4.3 G4 — generic config error body

**File:** `leaklens/workspace.py` (`/api/config`, ~1680).

- Payload `error` becomes a fixed string, e.g. `"configuration could not be loaded"`.
- Log `repr(exc)` through the module logger (server-side only).
- `tests/test_workspace.py`: malformed `.leaklens.toml` → `error` is the generic string and the
  raw path/message is absent from the response body.
- Confirm G5 (`workspace.py:1712,1716`) returns only fixed strings; add an assertion to the same
  test that `/api/compare/*` 404 bodies are `"unknown scan session"` / `"unknown dataset session"`.

### 4.4 Regression suite

`tests/test_paths.py`:
- `..`, `..\`, `../..`, Windows separators resolving inside, absolute drive path, UNC path outside,
  device path outside, symlink inside → outside, root itself, normal nested path.
- Drive/UNC/device cases are guarded by `pytest.mark.skipif(os.name != "nt")` where they need
  Windows semantics; the Windows-separator normalisation test runs on POSIX too.
- `resolve_within` on a UNC root with a UNC child does not raise (Windows only).
- `tests/test_workspace.py` keeps its existing traversal assertions (no message change).

### 4.5 Dispositions and alert outcome

Write the final dispositions (G1 confirmed → fixed; G2 containment sanitizer + pre-resolution UNC
refusal → fixed by `paths.py`; G3 informational; G4 fixed; G5 confirmed false positive with a test)
into `docs/ARTIFACT_SECURITY_STRATEGY.md` §2.2 as a short "Remediation status (Phase 13 M-S0)"
paragraph.

**CodeQL does not model `resolve()` + `is_relative_to` as a sanitizer**, so the 16 path alerts may
remain open after a genuinely correct fix. After pushing, re-run
`gh api repos/eyadstein/leaklens/code-scanning/alerts?state=open` and record the exact remaining
count and per-alert reason (fixed / false positive / awaiting re-analysis) in the report and in the
§2.2 status paragraph. Do **not** call the gate closed while an alert is open and unexplained; do
**not** dismiss or suppress any alert, and do not change `codeql.yml`.

---

## 5. M2 — `assemble.py` payload parity

### 5.1 Golden capture first (TDD)

Generate goldens **at the M2 base commit (after M7)**, before touching payload code:

- `tests/_meta/golden/cli_dataset.json` — `leaklens <csv> --target y --json`, no baseline.
- `tests/_meta/golden/cli_policy.json` — same with a baseline containing one unchanged and one
  resolved entry plus a suppression.
- `tests/_meta/golden/api_audit.json` — `POST /audit` payload for the same CSV.
- `tests/_meta/golden/public_scan.json` — live-scan payload for a tiny `.py`.
- `tests/_meta/golden/workspace_scan.json` — `_finish()` payload minus `scan_id`, `state`,
  `duration_ms`.

Normalise before compare: drop `version`, any absolute path (replace with a placeholder), and the
workspace envelope keys. Compare `json.dumps(..., sort_keys=True)` so key order is not a contract.
The `leaklens images` JSON path (`cli.py:268-272`) is a fifth payload: add it to the goldens or list
it as deliberately excluded in the test docstring.

### 5.2 `leaklens/assemble.py` (new)

```python
@dataclass(frozen=True, slots=True)
class Sections:
    outcome: PolicyOutcome
    trust_score: int
    findings: list[dict[str, Any]]
    counts: dict[str, int]
    policy: dict[str, Any]
    resolved: list[dict[str, Any]]

def build(
    findings: Sequence[Finding],
    *,
    baseline: Baseline | None = None,
    rules: tuple[SuppressionRule, ...] = (),
    today: date | None = None,
) -> Sections:
    ...
```

- One `policy.apply_policy()` call; `outcome` is returned so the CLI gate reuses
  `outcome.active`/`outcome.diff.new` instead of applying policy a second time with a different
  `today`; `trust_score = score(outcome.active)`;
  `findings = [outcome.finding_payload(f) for f in outcome.findings]`.
- `counts` moves `public_api._severity_counts` here (`{"high","medium","low","total"}`);
  `public_api._severity_counts` stays as a one-line delegation so existing imports/monkeypatches
  keep working.
- `policy`/`resolved` are the existing `_policy_payload` shape.
- No `root` argument, no intelligence, no workspace envelope: the workspace serializer
  (`to_dict(root=...)` + file/line/cell/index/advice) and its `session.counts` are different
  contracts and stay in `workspace.py`.
- No new scoring, no new status words, no reordering: preserve input order everywhere.

### 5.3 Surface wiring (thin delegations)

- `cli.py`: replace the payload block (~745–755) with `sections = assemble.build(...)`; keep the
  surrounding `rep[...]` keys (`live`, `summary`, `audit`, `missingness`, `drift`, `arena`, …) in
  the same order.
- `api.py`: `_finding_payloads`/`_policy_payload` delegate to `sections.findings`/`sections.policy`
  + `sections.resolved`.
- `public_api.py`: payload builders (~656, ~739, ~976) use `build()` for findings/counts/trust; the
  `fix` field stays a local `payload["fix"] = ...` addition.
- `workspace.py`: **unchanged payload path.** `_apply_workspace_policy` already delegates to
  `policy.apply_policy`, and `_finding_payload`/`session.counts`/`analyze_findings` are
  workspace-specific. Add a parity test that the workspace findings and policy statuses/fingerprints
  match the CLI for the same input; exclude the envelope by name.
- `ui.py` / `report.py` consume payloads; do not add a builder there.

**Risk control:** keep monkeypatch targets (`public_api.run_code_job`) as thin delegations; do not
rename serialization helpers used by tests.

---

## 6. M3 — `source.py`

**Files:** `leaklens/source.py` (new), `leaklens/scanner.py`, `leaklens/cvcheck.py`,
`leaklens/flow.py`.

- Move exactly the ADR-019 names: `BENIGN`, `FUNCS`, `HINTS`, `RESAMPLE`, `TARGET_ENCODERS`,
  `FEATURE_SELECTION`, `SKIP`, `_all_names`, `_finding_keys`, `_name`, `_node_index`, `_sev`,
  `_tnames`. Nothing else.
- `scanner.py` re-exports them with `from .source import (...)` and keeps its own public imports
  working (`from leaklens.scanner import SKIP, FUNCS`).
- `cvcheck.py` / `flow.py` import from `.source`.
- `source.py` imports only stdlib + `ast`; never `ai`.

**TDD:** existing scanner/cvcheck/flow tests unchanged; add an architecture assertion (M5) that no
module imports `scanner` from `cvcheck`/`flow` and no import cycle exists.

---

## 7. M5 — allowed-import matrix test

**File:** `tests/test_architecture.py` (new).

- Build a static import graph with `ast` over `leaklens/**/*.py`, including function-local imports.
- Layers: `core` (`leakage`, `evidence`, `fingerprint`, `manifest`, `policy`, `config`,
  `suppression`, `baseline`, `fixes`, `paths`, `availability`, `source`), `engines` (`scanner`, `cvcheck`,
  `flow`, `sqlscan`, `images`, `drift`, `missingness`, `readers`, `upload_guard`, `audit`,
  `partition`, `near_duplicate` when it exists), `ai`, `intelligence`, `surfaces` (`cli`, `api`,
  `public_api`, `workspace`, `workspace_ui`, `ui`, `report`, `sarif`), `infra`/`scripts`.
- Assertions:
  - no `core` → `engines|ai|intelligence|surfaces`;
  - no `engines` → `surfaces`, no `engines` → `ai|intelligence`;
  - no `intelligence` → `ai`; no `ai` module imports `intelligence` except documented
    `ai.analysis` (allowlisted with removal phase 18);
  - no cycles among non-surface modules.
- An explicit `ALLOWED` dict of current exceptions, each `(importer, imported, removal_phase,
  reason)`. The test fails if a new edge appears, and a deliberately wrong fixture import proves the
  failure message is readable.
- Run once before writing assertions to dump the real graph; encode only what exists.

---

## 8. M6 — freeze public identities

**File:** `tests/test_public_contract.py` (extend M1 version).

- `SURFACES` exact set, `MODALITIES` exact set, `ENGINES` exact id set.
- Golden fingerprints: build one `Finding` per surface and assert exact SHA-256 from
  `fingerprint_for`:
  - dataset: `target_leakage` / `feature` / path `data.csv` / no key;
  - split: `split_overlap` / `*` / `train.csv` / key `abc`;
  - code: `parse_error` / `a.py` / no key;
  - notebook: `notebook_out_of_order_execution` / `nb.ipynb` / key `cell-2`;
  - image: `image_overlap` / `*` / `images/a.png` / key `dup`.
- Engine ids and surfaces are already asserted by `tests/test_public_api.py:520`; do not duplicate
  the payload assertions, only the identity strings.

---

## 9. M8 — release allowlist + SHA-pinned actions

### 9.1 `scripts/validate_release.py`

- Add an allowlist that matches only what a wheel actually contains: `leaklens/**` and
  `leaklens_ml-<version>.dist-info/**`. Do **not** list `pyproject.toml`/`LICENSE`/`README` at the
  archive root (they are not in the wheel; licence files live under `*.dist-info/licenses/`).
  Version comes from the wheel filename, so dist-info is matched by pattern, not hardcoded.
- `validate_wheel(names: list[str], version: str) -> None`; unexpected file → `ValueError` naming
  it. Optional `--wheel <path>` mode reads the zip name list (`zipfile`), independent of `--tag`.
  Call it from `release.yml` after build, before publish; keep `--tag` validation unchanged.
- Tests: extend `tests/test_release_tag.py` (do not create a new file) — a fake wheel list passes;
  an injected `tests/secret.csv` fails; a different-version dist-info passes; tag/version tests
  unchanged.

### 9.2 Workflows

- Replace every `uses: owner/action@vN` with `owner/action@<40-char sha> # vN`.
- Resolve pins with `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` (annotated tags: peel to the
  commit via `git/tags/<sha>`); `pypa/gh-action-pypi-publish@release/v1` is a branch ref — pin the
  commit it currently points to and comment `# release/v1`.
- Add a policy check (in `tests/test_release_tag.py`) that every `uses:` line matches
  `@[0-9a-f]{40}`. Dependabot is not configured in this repo today; note that in the report rather
  than adding a bot config (out of scope).

---

## 10. M4 — deferred with a written reason

**Not implemented in this session.** The cycle is `artifact.load_artifact` → `learned` (constructs
`LearnedModel`) and `learned.get_model` → `artifact.load_artifact`. Breaking it without changing
`tests/test_ai_artifact.py`/`test_ai_selective.py` (which import and monkeypatch
`artifact.load_artifact`) requires moving the model dataclasses into a shared leaf module
(`ai/model.py`) or changing a shipped public helper. The roadmap's Phase 13 acceptance says every
existing test passes unmodified, so both routes are out of scope for this phase.

Recorded options for Phase 18, in order of preference:
1. Move `_PlattCalibrator`, `_CalibratedModel`, `LearnedModel`, `ModelEvaluation`,
   `MODEL_VERSION` to `ai/model.py`; both `learned` and `artifact` import it; re-export from
   `learned` for compatibility. Requires checking moved classes have no `learned`-only helpers.
2. Move `_from_payload` + `main()` into `learned` and make `artifact.load_artifact` a documented
   compatibility shim (still an edge; not a real fix).
Option 1 is the only honest fix and is deferred, not forgotten.

**M5 consequence:** the `artifact → learned` edge stays in the `ALLOWED` matrix with
`removal_phase=18` and the reason above.

---

## 11. AI intelligence audit (no Phase 14+ code)

Read: `ai/analysis.py`, `ai/reasoning.py`, `ai/learned.py`, `ai/features.py`, `ai/artifact.py`,
`ai/semantics.py`, `ai/selective` (inside learned), `intelligence/orchestrator.py`,
`intelligence/schemas.py`, and the AI benchmarks.

Deliverable is the audit in the final report, covering per specialist area: inputs/formats,
detection capabilities, evidence requirements, dependencies, CPU/memory budget, optional packages,
outputs, benchmarks/labelled data, false-positive risks, security boundary, acceptance criteria —
and an explicit `implemented / designed-only` label. **No capability is claimed unless a test and a
benchmark exist.** Only additive, evidence-safe changes are allowed; any change that touches
confidence semantics, schemas or model artifacts is deferred to Phase 18 with a written evaluation
protocol.

---

## 12. Verification (after each milestone, and full at the end)

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check leaklens tests scripts
.\.venv\Scripts\python.exe -m ruff format --check leaklens tests scripts
git diff --check
.\.venv\Scripts\python.exe -m leaklens.manifest --check
.\.venv\Scripts\python.exe -m leaklens --scan leaklens
.\.venv\Scripts\python.exe -m leaklens.bench
.\.venv\Scripts\python.exe -m leaklens.bench_data
.\.venv\Scripts\python.exe -m leaklens.bench_missing
.\.venv\Scripts\python.exe -m leaklens.bench_drift
.\.venv\Scripts\python.exe -m leaklens.bench_ai
.\.venv\Scripts\python.exe -m leaklens.bench_sql
.\.venv\Scripts\python.exe scripts\build_site.py --check   # only if docs/site changed
```

Graphify: run before and after with `--output %TEMP%\leaklens-graph` (never `graphify-out/`);
compare cycles/coupling. BLA: only `validate_release.py` / wheel-content auditing for M8.

Benchmarks must keep their digests; if a digest moves, stop and diagnose before committing.

## 13. Commit plan

One commit per milestone, in order: `feat(manifest): ...` (M1), `fix(cli): ...` (M9),
`fix(security): ...` (M7), `refactor: ...` (M2), `refactor: ...` (M3), `test(arch): ...` (M5),
`test(contract): ...` (M6), `ci: ...` (M8). M4 is deferred and gets no commit.
Never `git add .`, never stage the two protected untracked paths, never force-push.

## 14. Definition of done for this session

- M1, M9, M7/M-S0, M2 are complete with evidence.
- M3, M5, M6, M8 complete if verification stays green.
- M4 is reported as deferred, with the reason and the Phase 18 options above.
- `AGENTS.md` layout table gains `assemble.py`, `source.py`, `paths.py`, `availability.py`;
  `README.md` documents the `/predict` fixed-model change and `LEAKLENS_MODEL_PATH`.
- No Phase 13 completion claim unless all eight shipped milestones and the phase acceptance criteria
  pass. Phase criterion 1 has one documented exception: the three `/predict` tests changed by M7.
