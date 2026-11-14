# Phase 18 implementation plan

Status: **plan only, not implemented.** Prepared at `4314051ea33c770f635e4c53faa7ac2195bac2da`.
Scope is [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) section 7 and the architecture
constraints in [FUTURE_ARCHITECTURE.md](FUTURE_ARCHITECTURE.md) section 9 plus
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) ADR-025.

## 1. Purpose and scope

Phase 18 moves LeakLens from a list of findings to a reviewable investigation:

- a deterministic investigation plan derived from findings and the evidence graph;
- suggested deterministic tests, each mapped to a registered detector;
- an explicit missing-evidence report for what could not be checked and why;
- optional LLM narration that never changes the plan, a finding, the score or the gate;
- monitoring data (drift/skew reports) admitted as diagnostic evidence only;
- an audit manifest that verifies offline (already shipped as M-S3);
- a provenance-labelled governance pack.

The investigator is not an AI module. It must produce a complete plan with the AI layer absent.

## 2. Roadmap references and dependencies

Roadmap milestones: **M1** plan builder, **M2** test selection, **M3** missing-evidence report,
**M4** optional narration, **M5** monitoring inputs, **M6** audit manifest + offline verifier,
**M7** governance pack.

Declared prerequisites and their verified state:

| Prerequisite | State at `4314051` | Evidence |
|---|---|---|
| Phase 12 confidence semantics | Shipped | `leaklens/intelligence/confidence.py` (`assess_confidence`, `assess_cluster_confidence`) |
| Phase 13 M2 shared assembler | Shipped | `leaklens/assemble.py` (`Sections`, `build`, `from_outcome`) |
| Phases 14-16 | Accepted | `docs/PHASE14_ACCEPTANCE.md`, `docs/PHASE15_ACCEPTANCE.md`, `docs/PHASE16_ACCEPTANCE.md` |
| Phase 17 M1-M3 (cross-domain evidence) | Shipped | image decode budgets (M-S5), image clustering (`leaklens/images.py`), declared splits (`leaklens/image_splits.py`) |
| Evidence graph | Shipped | `leaklens/intelligence/provenance.py`, `orchestrator.py`, `impact.py`, documented in `docs/INTELLIGENCE.md` |
| Manifest + offline verifier (M6) | Already shipped | `leaklens/audit_manifest.py`, `leaklens verify` in `cli.py`, commit `534165c`, `tests/test_audit_manifest.py` |
| Prompt-injection corpus | Shipped | `leaklens/ai/prompt_injection.py`, 47 cases, `docs/PROMPT_INJECTION.md` |

The roadmap's `P17 --> P18` dependency is satisfied by Phase 17 M1-M3. Phases 17 M4-M7 remain
unstarted and are **not** Phase 18 prerequisites.

## 3. Current implementation baseline

Existing, reusable:

- `intelligence/` owns the evidence graph, anchors, links, confidence, impact and provenance.
- `ai/reasoning.py` already has `build_plan` and `enrich_plan`, with a deterministic fallback.
- `ai/schemas.py` owns `ReasoningPlan`, `RootCauseGroup`, `Hypothesis` and `Remediation` today.
- `assemble.py` is the single finding/counts/policy section builder for CLI, API and workspace.
- `audit_manifest.py` builds a deterministic digest manifest from `assemble.Sections` and verifies
  it with the standard library.
- `availability.py` defines `EngineUnavailable`; engines raise subclasses with the missing extra.
- `policy.py` keeps diagnostics out of the score and the default gate.

Absent:

- `leaklens/intelligence/investigation.py` and an investigation plan type.
- A test-selection contract mapped to `manifest.DETECTORS`.
- A canonical missing-evidence collection in the assembled payload.
- Monitoring report-file intake.
- `leaklens/governance/`.

Known baseline discrepancy: `AGENTS.md` says `assemble.py` catches `EngineUnavailable`, but
`assemble.py` does not catch it. Engine-specific catches live in the CLI (`PipelineUnavailable`) and
in each engine. M3 must not assume a central skip record exists; it must create one.

## 4. Milestone-by-milestone readiness

| Milestone | Classification | Why |
|---|---|---|
| M1 deterministic investigation plan | **READY WITH CONDITIONS** | Evidence graph, assembler and confidence are shipped. Plan types currently live in `ai/schemas.py`, so this is the ADR-025 refactor plus a new builder. Condition: preserve `from leaklens.ai.schemas import ReasoningPlan`, keep `intelligence` free of `ai` imports, and keep the plan deterministic. |
| M2 test selection | **READY WITH CONDITIONS** | `manifest.DETECTORS` is the registry to map against. Condition: M1's plan step type must exist first. |
| M3 missing-evidence report | **READY WITH CONDITIONS** | `EngineUnavailable` and per-engine `skipped` maps exist, but there is no central collection. Condition: define the skip vocabulary and attach entries to the M1 plan. |
| M4 optional narration | **READY WITH CONDITIONS** | `enrich_plan`, schema validation, deterministic fallback and the injection corpus are shipped. Condition: re-point narration at the new plan after M1; do not build a second narration path. |
| M5 monitoring inputs | **READY** | Diagnostic findings (`Finding.diagnostic=True`) and the drift module exist; this is a new read-only intake with no score changes. It has no M1 dependency and can start in parallel. |
| M6 manifest + verifier | **ALREADY SATISFIED** | M-S3 shipped `--manifest`, `leaklens verify`, digest stability and a tamper matrix. Do not rebuild. |
| M7 governance pack | **READY WITH CONDITIONS** | Assembler and provenance vocabulary exist; the package is absent. Condition: M1 and M3 must define plan/evidence ids and explicit unknowns before fields can be labelled. |

One wording clarification: the roadmap says M1 "runs with the `ai` extra absent", but the AI layer is
part of the base install (`pyproject.toml` defines no `ai` extra). Treat the requirement as "runs
without importing `leaklens.ai`" and test that directly.

## 5. Recommended execution order

| Wave | Milestones | Reason |
|---|---|---|
| 0 | Documentation/status only | Record verified Phase 17 M1-M3 and this plan; no code. |
| 1 | **M1**, **M5** (parallel) | M1 unblocks the plan-dependent work. M5 is independent and low risk. |
| 2 | **M2**, **M3** (parallel) | Both extend the plan; they touch separate step types and separate tests. |
| 3 | **M4** | Narration must consume the finished plan type, not the legacy one. |
| 4 | **M7** | Governance needs the plan/evidence ids (M1) and the explicit unknown list (M3). |
| — | **M6** | Verification pass only; add a regression test if a gap appears. Mark satisfied in the acceptance record. |

Hard dependencies: `M1 -> {M2, M3, M4}`; `M1 + M3 -> M7`.
Soft dependency: `M2 -> M7` (governance may cite proposed checks, but does not require them).
Independent: M5, M6.

## 6. Dependency graph

```text
Phase 12 confidence ─┐
Phase 13 M2 assemble ─┼─> M1 ──> M2 ──┐
Phase 14-17 evidence ─┘   │           ├─> M7
                          ├──> M3 ────┘
                          └──> M4
M5 (independent) ──────────────> assemble payload (diagnostic section)
M6 (already shipped) ──────────> verify/regression only
```

## 7. Architecture and design contracts

1. **One score path.** `leakage.score()` remains the only trust-score computation. No Phase 18
   module may add, raise or rescore a finding.
2. **Diagnostics stay diagnostic.** Every Phase 18 finding kind is `diagnostic=True` and
   `scored=False` in `manifest.DETECTORS`; the exit gate ignores it.
3. **Evidence or silence.** Every plan statement cites finding/evidence ids that already exist in
   the payload. Ungrounded statements are rejected before serialization.
4. **Intelligence owns the plan.** Per ADR-025, `intelligence/schemas.py` owns the plan types and
   `intelligence/investigation.py` owns the builder. `ai` may only narrate. `intelligence` never
   imports `ai`.
5. **Additive payloads.** New JSON keys are additive; existing keys and default output are
   unchanged when the feature is off.
6. **No new mandatory dependency.** M1-M5 use the standard library plus existing dependencies.
   `governance/` uses the standard library only.
7. **Read, never execute.** Monitoring reports are parsed as data; no SQL, no pickle, no code.
8. **File boundaries.** Any input path goes through `leaklens/paths.py` (`resolve_within`), with
   byte/depth caps and no network fetch.

## 8. Detailed implementation boundaries

### M1 - deterministic investigation plan

- Add `leaklens/intelligence/investigation.py`:
  - `build_investigation(findings, graph, context)` returns an ordered plan.
  - Plan step types: `InspectEvidence`, `SuggestedTest`, `MissingEvidence`, `NarrationSlot`.
  - Every step carries `evidence_ids`, `finding_ids`, a rationale and a stable ordinal.
- Move `ReasoningPlan`/`RootCauseGroup`/`Hypothesis`/`Remediation` contracts into
  `intelligence/schemas.py`; import and re-export them from `ai/schemas.py` unchanged.
- Keep `ai.reasoning.build_plan` as a thin compatibility wrapper or deprecate it in favour of the
  intelligence builder; do not keep two builders.
- Wire the plan into `--intelligence`/`--ai` payloads additively. No new CLI flag is required unless
  the payload needs an opt-in; prefer reusing the existing intelligence flag.
- Do not move `intelligence` into a new package or create folders.

### M2 - test selection

- A proposed test is a reference to a registered detector kind, never free text.
- Validate at build time against `manifest.DETECTORS`; unknown kinds are a validation error in
  development and are omitted (with a skip reason) at runtime.
- Selection inputs: finding kind, anchor type (`feature`, `file`, `split`, `dataset`, `image`),
  severity and existing evidence. No LLM.
- Output: ordered `SuggestedTest` steps with the detector kind, the question it answers and the
  evidence ids it would discriminate.

### M3 - missing evidence

- Create one collection in the assembled payload (for example `missing_evidence`) sourced from:
  - `EngineUnavailable` raises and their extras;
  - engine-specific `skipped` maps (text, dl, pipeline, splits, drift);
  - parse limitations and unsupported formats already reported by engines;
  - plan steps that could not be grounded.
- Each entry has `area`, `reason`, `effect`, `source` and, where applicable, `finding_ids`.
- Enumerate reasons and counts; never collapse them into "some checks were skipped".
- Do not change exit codes or the score.

### M4 - optional narration

- Use `ai/reasoning.enrich_plan` as the only narration entry point; adapt its input to the
  investigation plan.
- Validate the model response against a schema; accept only references to evidence ids already in
  the plan.
- On any failure, return `ai_unavailable` or `deterministic_fallback` and the unmodified plan.
- Reuse `leaklens/ai/prompt_injection.py` (47 cases); do not add a second corpus.
- The plan with narration removed must serialize byte-identically to the plan built with narration
  disabled.

### M5 - monitoring inputs

- Add a read-only intake for caller-supplied drift/skew report files (JSON first; no YAML
  requirement). Files are size-capped, parsed with the standard library, and path-contained.
- Emit diagnostic findings (for example `monitoring_drift_report` / `monitoring_skew_report`) whose
  evidence names the report path (relative), the metric and the observed value.
- Missing, malformed or oversized reports become skips, not tracebacks.
- No network, no database, no monitoring service. Never changes the score or gate.

### M6 - manifest and verifier

- No new implementation. Verify:
  - `audit_manifest.build` consumes `assemble.Sections`;
  - digest stability, changed score/evidence/detector version and modified-input detection are
    covered by `tests/test_audit_manifest.py`;
  - verification is stdlib-only.
- Record "already satisfied by M-S3" in the Phase 18 acceptance record. Add a regression test only
  if a Phase 18 payload change would alter the manifest contract.

### M7 - governance pack

- Create `leaklens/governance/` with `cards.py` (field contracts) and `export.py` (renderers).
- Label every field `asserted`, `measured` or `unknown`, with its provenance source.
- One payload renders both JSON and Markdown; the two outputs share the payload.
- Unknowns are explicit. No compliance claim and no certification language.
- `governance` may import `assemble` and `intelligence`, never `ai`.
- Output paths go through `resolve_within`; writes are atomic.

## 9. Testing and validation strategy

| Area | Required tests |
|---|---|
| Plan determinism | Same inputs produce byte-identical golden JSON; ordering is stable |
| Evidence anchoring | Every step cites ids present in the payload; no ungrounded step |
| AI independence | Building the plan succeeds with `leaklens.ai` imports blocked |
| Compatibility | `from leaklens.ai.schemas import ReasoningPlan` still works |
| Test selection | Every proposed kind exists in `manifest.DETECTORS`; no invented checks |
| Missing evidence | Absent optional extra, malformed input and unsupported format each appear with a reason |
| Narration | Injection corpus cannot change findings/score; fallback yields the deterministic plan |
| Monitoring | Malformed/oversized/traversing report is refused; score and gate unchanged |
| Governance | Every field has provenance; unknowns explicit; JSON/Markdown parity |
| Manifest | Existing tamper/stability matrix still passes |
| Regression | Existing suite, self-scan and benchmarks unchanged |

Add `leaklens/bench_investigation.py` with a fixed set of audited repositories and published
misses, plus an LLM-on vs LLM-off equality check. Record measured numbers; do not claim accuracy.

## 10. Security and privacy considerations

- Findings metadata only to any LLM; never source text or dataset rows.
- Report files are untrusted: containment, byte caps, JSON parse limits, no fetch.
- No execution of scanned code, no pickle, no archive extraction.
- The manifest is unsigned by design; it detects change, not authorship.
- The governance pack must not imply certification, compliance or a guarantee.
- New diagnostic kinds must be registered with `requires`, `modalities`, `scored=False` and a docs
  anchor, or `python -m leaklens.manifest --check` fails.

## 11. Dependency and packaging decisions

- No new mandatory dependency.
- Monitoring intake uses `json`/`pathlib` from the standard library.
- Governance uses the standard library only.
- Do not add an `ai` extra; the AI layer stays in the base install and optional at runtime.
- Do not add a scheduler, database, server or hosted component.

## 12. Documentation requirements

- `docs/INVESTIGATION.md`: plan schema, step types, id contract, limits, AI-absent behaviour.
- `docs/GOVERNANCE.md`: provenance vocabulary, field labels, non-claims, export formats.
- README: one section for investigation and one for governance, with the opt-in flags.
- CHANGELOG entry per milestone.
- `docs/ARCHITECTURE_DECISIONS.md`: mark ADR-025 accepted when M1 lands.
- `docs/DETECTORS.md`: regenerate, never hand-edit.
- This plan and the Phase 18 acceptance record.

## 13. CI and release requirements

Every PR: the existing CI order in `AGENTS.md` (tests, self-scan, benchmarks, manifest check), plus
`python -m leaklens.bench_investigation` once it exists. Documentation-only changes still run
`git diff --check`, `python -m leaklens.manifest --check` and
`python scripts/build_site.py --check`. No version bump or release is part of Phase 18
implementation.

## 14. Acceptance criteria per milestone

| Milestone | Acceptance |
|---|---|
| M1 | Plan builds without importing `leaklens.ai`; every step cites existing ids; no ungrounded step; `ReasoningPlan` re-export works |
| M2 | Every proposed test maps to a registered detector; none invented; deterministic order |
| M3 | Skips and unavailable engines are enumerated with reasons, not summarized away |
| M4 | Injection corpus cannot change findings/score; failure yields `ai_unavailable`; plan byte-identical with narration removed |
| M5 | Diagnostic only; never changes score or gate; malformed input becomes a skip |
| M6 | Manifest digest stable; tampering detected; verification stdlib-only (already demonstrated) |
| M7 | No compliance claim; unknowns explicit; JSON and Markdown share one payload; every field labelled |

Phase-wide: with the LLM removed, a complete plan still exists; no investigation output can create,
raise or score a finding; the manifest verifies offline; every governance field is provenance-
labelled.

## 15. Risks, deferred decisions and blockers

| Risk | Mitigation |
|---|---|
| Plan-type move breaks `ai` imports | Re-export from `ai/schemas.py`; add an import-compatibility test |
| Ungrounded LLM or plan claims | Evidence-id allowlist; validation before serialization; corpus test |
| Missing-evidence report becomes a summary | Require per-reason counts and explicit entries |
| Monitoring files become a deserialization sink | JSON only, containment, byte caps, no network |
| Governance implies certification | Explicit non-claims and `unknown` labels |
| Duplicate manifest work | Treat M6 as satisfied; no second manifest format |
| Diagnostics leak into the gate | `diagnostic=True`/`scored=False` plus score/gate regression tests |

Deferred: choosing a monitoring report schema beyond the minimal JSON contract; whether governance
Markdown should be a single file or one file per section; whether M5 ships a CLI flag or only a
config key. These are implementation decisions inside the milestones, not scope changes.

No external blocker. No roadmap clarification is required beyond the M1 "ai extra" wording noted in
section 4.

## 16. Explicit out-of-scope work

- Phase 19 remediation, patches, apply-on-copy, CI action and Semgrep integration.
- Any autonomous agent with repository write access.
- Compliance certification, case management, hosted monitoring, a database or a service.
- Phase 17 M4-M7 (annotation validation, video, real-world photo validation, colour-aware hashing).
- Changing the M2 6-bit threshold or the pinned vision benchmark digest.
- Reopening the Phase 17 M2/M3 contracts.
- New mandatory dependencies, a new scoring path, or a second finding serialization.

## 17. Phase 18 completion criteria

Phase 18 is complete when:

1. M1-M5 and M7 are implemented and accepted; M6 is recorded as already satisfied by M-S3.
2. `docs/INVESTIGATION.md`, `docs/GOVERNANCE.md`, README and CHANGELOG are updated.
3. `leaklens/bench_investigation.py` publishes measured investigation quality with misses.
4. The plan works with `leaklens.ai` unimportable; narration is optional and falls back.
5. Diagnostic findings never change the score, severity, exit code or gate, proven by tests.
6. The manifest still verifies offline and detects tampering.
7. Every governance field is labelled `asserted`, `measured` or `unknown`, with no compliance claim.
8. CI is green on the supported Python versions and the self-scan reports no leaks.
