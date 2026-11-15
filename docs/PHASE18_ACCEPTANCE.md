# Phase 18 acceptance record

Phase 18 (investigation, observability and governance) is complete. This record
maps every milestone to its acceptance criteria, implementation, tests and
commit. The plan is [PHASE18_IMPLEMENTATION_PLAN.md](PHASE18_IMPLEMENTATION_PLAN.md);
the roadmap scope is [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) §7.

## Milestones

| Milestone | State | Implementation | Tests |
|---|---|---|---|
| M1 deterministic investigation plan | Complete | `leaklens/intelligence/investigation.py`, plan types in `leaklens/intelligence/schemas.py`, re-exported by `leaklens/ai/schemas.py` | `tests/test_investigation.py` |
| M2 test selection | Complete | `SuggestedTest`, `OmittedTest` and `_test_kinds` in `leaklens/intelligence/investigation.py` | `tests/test_test_selection.py` |
| M3 missing evidence | Complete | `leaklens/missing_evidence.py`, attached in `leaklens/cli.py` | `tests/test_missing_evidence.py` |
| M4 optional narration | Complete | `enrich_plan` / `narrate_investigation` in `leaklens/ai/reasoning.py` | `tests/test_narration.py`, `leaklens/bench_investigation.py` |
| M5 monitoring inputs | Complete | `leaklens/monitoring.py`, `--monitoring-report` in `leaklens/cli.py` | `tests/test_monitoring.py` |
| M6 manifest and verifier | Already satisfied by M-S3 | `leaklens/audit_manifest.py`, `leaklens verify` | `tests/test_audit_manifest.py` |
| M7 governance pack | Complete | `leaklens/governance/cards.py`, `leaklens/governance/export.py`, `--governance` / `--governance-out` | `tests/test_governance.py` |

## Acceptance evidence

### M1 - deterministic investigation plan

Criteria: the plan builds without importing `leaklens.ai`; every step cites ids
that already exist; no step is ungrounded; `from leaklens.ai.schemas import
ReasoningPlan` still works.

- **Implementation:** `build_investigation()` builds `InspectEvidence`,
  `SuggestedTest`, `MissingEvidence` and `NarrationSlot` steps from the evidence
  graph; `_ground()` rejects an id outside the payload before serialization.
  `intelligence` imports no `ai` module (enforced by `tests/test_architecture.py`).
- **Tests:** `test_plan_builds_with_the_ai_layer_unimportable`,
  `test_every_step_cites_existing_ids`,
  `test_grounding_rejects_unknown_evidence_and_finding_ids`,
  `test_ai_schemas_still_reexports_the_plan_type`,
  `test_plan_is_deterministic_and_ordered`, `test_step_count_is_capped`.
- **Commit:** `1bd1291`, with the Python 3.10/3.11 dataclass fix in `6422159`.
- **Deliberate decision:** `ai.reasoning.build_plan` remains the AI layer's own
  `ReasoningPlan` builder (used by `ai/analysis.py` and the prompt-injection
  corpus); it is not a second investigation builder. The deterministic plan is
  `intelligence`-owned, per ADR-025.

### M2 - test selection

Criteria: every proposed test maps to a registered detector, none is invented,
and the order is deterministic.

- **Implementation:** a `suggested_test` step carries `detector` and `question`;
  `SuggestedTest.__post_init__` rejects a kind absent from `manifest.BY_KIND`,
  and runtime builds omit an unknown kind into `omitted_tests` with reason
  `unregistered_detector` instead of inventing a check.
- **Tests:** `tests/test_test_selection.py` (registered kinds only, unknown kind
  omitted with a reason, deterministic order, payload surfaces the omission).
- **Commit:** `e16fbf8`.

### M3 - missing evidence

Criteria: skips and unavailable engines are enumerated with reasons, not
summarised away.

- **Implementation:** `missing_evidence.build()` emits one record per
  area/reason/source with a count and effect, collects engine `skipped` maps,
  split `unresolved` counters and limitations, text document skips and omitted
  plan tests, and reports an `entry_cap` record when over `MAX_ENTRIES` (200).
  The CLI adds the key only when something was skipped.
- **Tests:** `tests/test_missing_evidence.py` (per-reason counts, absent extra,
  malformed and unsupported inputs, entry cap visible, score and gate
  unchanged).
- **Commit:** `e16fbf8`.

### M4 - optional narration

Criteria: the injection corpus cannot change findings or the score; failure
yields `ai_unavailable`; the plan is byte-identical with narration removed.

- **Implementation:** `narrate_investigation()` adapts the plan to
  `enrich_plan` through a structural `NarrationTarget` protocol; the response
  allowlist is exactly the plan's own evidence ids. Any rejection yields
  `deterministic_fallback`, an unreachable provider yields `ai_unavailable`, and
  `without_narration()` restores the AI-free plan.
- **Tests:** `tests/test_narration.py` (47 injection-corpus cases replayed,
  byte-identical removal, unknown id fallback, malformed responses, unchanged
  steps/findings/order, CLI unchanged with no LLM).
- **Benchmark:** `python -m leaklens.bench_investigation` compares narrated and
  narration-free plans across fixed cases; it prints and never writes docs.
- **Commit:** `0942fa9`, merged as `3aa3040`.

### M5 - monitoring inputs

Criteria: diagnostic only; never changes the score or the gate; malformed input
becomes a skip.

- **Implementation:** `--monitoring-report` reads a caller-supplied JSON drift or
  skew report read-only, path-contained and size-capped; each breached metric is
  a `low`, `diagnostic` `monitoring_drift_report`/`monitoring_skew_report`
  finding. Failures become counted skips (`oversized`, `too_many_metrics`,
  `malformed`, `unsupported_kind`, `outside_root`).
- **Tests:** `tests/test_monitoring.py` (breached metrics, explicit `breached`
  flag, missing signal, malformed/oversized/traversal skips, score ignores the
  findings, exit gate unchanged).
- **Commit:** `1bd1291`.

### M6 - manifest and verifier

Criteria: manifest digest is stable, tampering is detected, and verification is
standard-library only. Acceptance was delegated to security milestone M-S3, so
Phase 18 adds no second manifest format.

- **Implementation:** `leaklens/audit_manifest.py` consumes `assemble.Sections`;
  `leaklens verify` verifies offline with the standard library.
- **Tests:** `tests/test_audit_manifest.py` (digest stability, changed
  score/evidence/detector version/gate, modified inputs, traversal and symlink
  containment, malformed and oversized manifests, CLI exit codes).
- **Commit:** `534165c`.

### M7 - governance pack

Criteria: no compliance claim; unknowns explicit; JSON and Markdown share one
payload; every field labelled.

- **Implementation:** `leaklens/governance/cards.py` owns the field contracts,
  the `asserted`/`measured`/`unknown` vocabulary and the non-claims;
  `leaklens/governance/export.py` renders the single payload as JSON or
  Markdown and writes it path-contained and atomically. `--governance` adds the
  block to the payload; `--governance-out PATH` writes `.json` or `.md`.
- **Tests:** `tests/test_governance.py` (every field labelled and sourced,
  unknowns explicit and `null`, absent optional sections are `unknown` not zero,
  measured counts match `assemble.severity_counts`, JSON/Markdown parity,
  escaped Markdown cells, no compliance language in any field, deterministic
  output, containment and suffix refusal, atomic write leaves no temporary
  files, `ai` import boundary, CLI adds the pack without changing the audit).
  `tests/test_architecture.py` adds the one-way `governance` layer rules.
- **Documentation:** [GOVERNANCE.md](GOVERNANCE.md), README, CHANGELOG.
- **Commit:** this milestone's own commit (`git log -- docs/GOVERNANCE.md`).

## Phase-wide criteria

| Criterion | Evidence |
|---|---|
| With the LLM removed, a complete plan still exists | `test_plan_builds_with_the_ai_layer_unimportable`, `test_cli_intelligence_plan_is_unchanged_without_a_llm` |
| No investigation or governance output can create, raise or score a finding | Investigation sections never touch `Finding.severity`; `diagnostic=True`/`scored=False`; `tests/test_monitoring.py`, `tests/test_missing_evidence.py`, `tests/test_cli_gate.py` |
| The manifest verifies offline and detects modification | `tests/test_audit_manifest.py` |
| Every governance field is provenance-labelled | `tests/test_governance.py` |

## Known limitations

- **M1/M4 narration is additive only.** Narration can rephrase a cluster but
  cannot add evidence, change a step, a finding, the score or the gate.
- **M2 selection is kind-based.** When a cluster carries several finding kinds,
  the first registered kind becomes the suggested test; the rest are not
  proposed. This is deliberate: one step, one detector, no invented checks.
- **M3 caps at 200 entries.** Overflow is visible as an `entry_cap` record
  naming how many entries were dropped.
- **M5 defines a minimal report contract.** `kind` plus metric objects; there is
  no attempt to understand vendor-specific monitoring schemas.
- **M7 card fields are a fixed stub set.** Model, licence, collection and
  evaluation fields are always `unknown`: LeakLens does not read a trained model
  or a dataset card, so it labels the gap instead of guessing. A clean pack is
  explicitly not a proof of safety or compliance.
