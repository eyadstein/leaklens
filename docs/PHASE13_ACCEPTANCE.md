# Phase 13 acceptance audit (2026-11-11)

Independent audit of the Phase 13 foundation at `9b301bd`, against the criteria in
[AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) §2. "Pass" means the evidence exists in the
repository and was re-run for this audit, not that the previous report said so.

## Phase criteria

| # | Criterion | Evidence | Result | Exception |
|---:|---|---|---|---|
| 1 | Every existing test passes without modification (except added tests) | 4 existing tests were edited by M7: `test_predict_missing_model`, `test_predict_roundtrip`, `test_predict_invalid_model_rejected` and one `/api/config` assertion in `test_workspace.py` | **Exception** | The edited tests encoded the old request-selected model contract and the old config error body. Keeping them would keep the G1 deserialization sink and the G4 echo, both blocking M7 requirements. The roadmap does not explicitly permit this; accepting it is a deliberate phase-owner decision. Consequence: `/predict` is a breaking API change (documented in `README.md`). |
| 2 | CLI, public API and workspace produce the same payload for the same input | `tests/test_response_parity.py` runs the same dataset through CLI `--json`, `POST /audit`, `run_audit_job` and the workspace results endpoint and compares finding identity, trust, counts, policy statuses and resolved entries | **Pass after this audit** | Before this audit the public API and workspace silently skipped the missingness audit, so the same dataset produced 1 finding instead of 2. Fixed in `public_api.run_audit_job`; the parity test would fail on the old code. |
| 3 | `docs/DETECTORS.md` is generated and CI-checked | `python -m leaklens.manifest --check` matches; `tests/_meta/test_detector_contract.py` compares rendered text | Pass | — |
| 4 | No public identifier changed: finding kinds, surfaces, engine ids, fingerprint format, JSON fields, SARIF properties | `tests/test_public_contract.py` freezes surfaces, modalities, engine ids and five per-surface fingerprints; SARIF untouched | Pass | `model_path` remains a request field, but its contract changed from "choose the file" to "must equal the server's configured path". The field name is unchanged. |
| 5 | No new dependency | `pyproject.toml` unchanged in Phase 13 | Pass | — |
| 6 | M7 closed: every alert disposition is written, `paths.py` is the only containment helper, regression tests pass | 16 open alerts dispositioned one by one in `ARTIFACT_SECURITY_STRATEGY.md` §2.2; `config._output_path` now calls `paths.contained`; `tests/test_paths.py`, `tests/test_api_predict.py`, `tests/test_response_parity.py` | Pass | Closure here follows the criterion's own definition (dispositions + one helper + tests), not "zero open alerts". 14 path alerts and 2 stack-trace alerts remained at `9b301bd`; the 2 stack-trace alerts were remediated in this audit. |
| 7 | M8 closed: wheel allowlist fails on an unexpected file and every action is SHA-pinned | `tests/test_release_tag.py`; the 0.13.0 wheel validated with 79 allowlisted entries | Pass | — |
| 8 | M9 closed: a high diagnostic does not fail the gate, a high non-diagnostic still does | `tests/test_cli_gate.py` | Pass | — |

## Milestone acceptance deviations

| Milestone | Roadmap wording | What shipped | Disposition |
|---|---|---|---|
| M2 | "byte-identical to today (golden JSON)" | Explicit field-level parity (`tests/test_assemble.py`) plus complete-response identity tests (`tests/test_response_parity.py`) | Accepted deviation. A byte golden of the full envelope would pin `duration_ms`, timestamps and the workspace's root-relative paths, i.e. exactly the nondeterministic fields the roadmap tells later phases to ignore. Every finding identity, trust, count and policy field is pinned instead. |
| M4 | Remove the `ai/artifact` → `ai/learned` edge | Not implemented; recorded in `tests/test_architecture.py` `ALLOWED` with removal phase 18 | Deferred with a written reason (shared `ai/model.py`; Phase 18 option 1). No new cycle appeared. |
| M7 | Triage and remediate 20 alerts | 5 alerts fixed; 16 open with written dispositions; no dismissal, suppression or query filter | Accepted under the criterion's definition; residual risk recorded. |

## Architecture state

`tests/test_architecture.py` still reports exactly two cycles outside the surfaces:

- `ai.artifact ↔ ai.learned` — Phase 18 (M4 deferred).
- `fingerprint ↔ leakage` — Phase 18 (`Finding` lives in `leakage`; moving it is the same leaf-module
  work as M4).

Every allowance still exists (a stale allowance fails the test), and a deliberately wrong edge fails
with both module names in the message.

## Decision

Phase 13 is **accepted with the recorded `/predict` test exception and the M2 golden deviation**, so
Phase 14 is eligible under the roadmap dependency graph. The security gate is closed under its own
definition: every open alert has a source-to-sink disposition and regression coverage, `paths.py` is
the only containment helper, and no suppression or query filter was added.
