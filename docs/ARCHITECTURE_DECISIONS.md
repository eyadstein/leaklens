# Architecture decisions

ADR-style log for the future-architecture mission (2026-11-10). Each entry is short by design.

**Status vocabulary:** `shipped` (in the code today), `accepted` (decided, not implemented),
`proposed` (recommended, needs sign-off), `deferred` (deliberately not decided), `open`
(needs an experiment before it can be decided).

The existing feature-catalogue ADR convention lives in
[`docs/adr/0001-feature-catalogue.md`](adr/0001-feature-catalogue.md).

---

## ADR-001 — One deterministic scoring path

**Decision.** `leakage.score()` remains the only trust score. No engine, AI component, workspace
view or report may compute or adjust a score.
**Context.** Every domain that joins must produce findings that are scored consistently; the
ecosystem review found competing tools each shipping their own verdict logic.
**Alternatives.** Per-domain scores aggregated later; a weighted composite; no score at all.
**Rationale.** One path is auditable, testable and stable. Aggregation would make the score depend
on which engines are installed — a silent semantic change.
**Consequences.** New engines must accept the existing severity arithmetic; the score can look
harsh for a noisy engine, which is why ADR-006 exists.
**Status.** shipped.

## ADR-002 — AI is additive, never evidentiary

**Decision.** AI produces hypotheses only. It cannot create findings, add evidence, change
severity, change the score or change the exit code.
**Context.** Phase 12 established this; the ecosystem shows LLM-judged evaluation both popular and
irreproducible.
**Alternatives.** LLM-as-a-detector; LLM raises severity when confident.
**Rationale.** An auditor whose conclusions are non-reproducible is not an auditor.
**Consequences.** Hypothesis severity cap `medium`, confidence cap `0.95`, mandatory evidence
references, schema-validated output, deterministic fallback.
**Status.** shipped.

## ADR-003 — No `core/` package; primitives stay flat

**Decision.** Keep `leakage.py`, `evidence.py`, `fingerprint.py`, `manifest.py`, `partition.py` at
package root as the shared core. Do not introduce `leaklens/core/`.
**Context.** A four-ring design originally proposed `core/`, but the modules that constitute the
core are stable, widely imported and covered by tests; moving them would break imports, fixtures
and monkeypatch targets for no user-visible benefit.
**Alternatives.** `core/` package with re-export façades; full layering.
**Rationale.** Deletion beats addition. The boundary that matters is *engine vs interpretation*,
not a new directory. A rename is not an architecture.
**Consequences.** `leakage.py` stays a large module (Finding + score + tabular detectors). Accepted
as long as new detectors go to engines, not into it.
**Status.** accepted.

## ADR-004 — One payload assembly function

**Decision.** Extract the payload built today in `cli.py`, `public_api.py` and `workspace.py` into
one `leaklens/assemble.py` function that all surfaces call.
**Context.** Three assemblies of the same report dictionary means every new engine must be wired
three times, and they can disagree.
**Alternatives.** Leave it (growth cost ×3 per engine); introduce a typed `AuditResult` and a
renderer refactor at the same time.
**Rationale.** One assembly is required by every later phase. A typed result is not: `report.render`
already accepts a dict and has one shape. Build the smallest thing that removes the triplication.
**Consequences.** `public_api.py` is reclassified as an HTTP surface (it imports FastAPI at module
level), not the engine boundary. Golden payload tests guard output identity.
**Status.** accepted (Phase 13).

## ADR-005 — `manifest` is the single detector catalogue

**Decision.** Extend `manifest.Detector` with `scored`, `requires` and `modalities`, and add a frozen
`ENGINES` registry beside the kind catalogue instead of creating a second registry module.
**Context.** `manifest.py` already holds kind, severity, advice, benchmark coverage,
`surface_for()`, `validate_kinds()`, a `--check` mode and the generated `docs/DETECTORS.md`. A
per-kind `engine_id` cannot describe a kind emitted by two engines (the reproducibility family in
Phases 14 and 16).
**Alternatives.** A new `registry.py`; per-engine self-registration; entry-point discovery; a
per-kind `engine_id` with a "canonical owner" rule.
**Rationale.** One catalogue for kinds and one registry for engines keeps both honest: rows are
keyed by what they describe. Self-registration makes output depend on import order and the
installed set.
**Consequences.** Registry is static and in-process; `engine_id` is a literal, never derived from
`__module__` (tests and committed site fixtures assert those strings). Result attribution uses the
engine that ran, not a catalogue lookup.
**Status.** accepted (Phase 13).

## ADR-006 — New engines ship `scored=False` and graduate with evidence

**Decision.** A new engine's findings carry `diagnostic=True` until its labelled benchmark meets a
published precision floor and a volume check; graduation is a catalogue change, not a scoring change.
**Context.** `score()` already ignores `diagnostic=True` findings, and adding a noisy modality
directly to the score would make the gate unusable.
**Alternatives.** Score everything from day one; never score new engines; per-engine weights.
**Rationale.** Preserves the score's meaning (ADR-001), keeps the exit gate stable, and makes
graduation an evidence-based review rather than a design accident.
**Consequences.** `score()` ignores diagnostics today. The exit gate does not, although `AGENTS.md`
already documents that it should, and the drift docs rely on that reading (drift is `low`, so it
never trips the gate either way). Phase 13 M9 makes the code match the documented contract: every
failure predicate ignores `diagnostic=True`. Until M9, a high-severity diagnostic kind cannot ship;
new engines start at `medium` or lower. The workspace must show the scored/diagnostic distinction.
**Status.** accepted.

## ADR-007 — Learned components never become `scored`

**Decision.** A learned model may not drive scoring. It influences the gate only if re-expressed as
deterministic features evaluated by a deterministic detector.
**Context.** ADR-006 could otherwise be read as allowing a learned detector to graduate into the
score, contradicting ADR-002's spirit and Phase 12's confidence semantics.
**Alternatives.** Allow learned graduation with a calibration record.
**Rationale.** Calibration does not transfer across corpora (the H1 record shows this explicitly),
so a learned gate would be unstable in user environments.
**Consequences.** Any "AI finding" stays a hypothesis forever; deterministic re-expression is the
only promotion path.
**Status.** accepted.

## ADR-008 — Small dependencies as extras, heavy ones as integrations

**Decision.** `pyproject.toml` extras may only name small, pure-Python or small-wheel libraries.
Framework runtimes (torch, TensorFlow, spaCy models, vector stores, MongoDB) are never extras: they
are user-installed integrations behind a documented interface.
**Context.** The ecosystem review found heavyweight stacks (TF for TFDV/DataProfiler, Spark for
Deequ, MongoDB for FiftyOne) that would break the CPU-only, offline install promise.
**Alternatives.** One `all` extra; caps on install size; bundling a small vision stack.
**Rationale.** An extra is a promise. Breaking it once destroys the base-install guarantee.
**Consequences.** Vision/DL/text-embedding users do a manual install and see an `unavailable` skip
with the exact instruction if they have not.
**Status.** accepted.

## ADR-009 — Skips are first-class result data

**Decision.** Every refused, skipped or unavailable check is a typed entry in the result with a
reason, not a log line and not silence.
**Context.** Today only missingness has a structured `skipped` dict; SQL/Pillow absence raises or
hides; the workspace cannot show what did not run.
**Alternatives.** Warnings only; treating absence as a pass.
**Rationale.** "We checked everything" must be falsifiable, and a security refusal is information
(see ARTIFACT_SECURITY_STRATEGY §3).
**Consequences.** Assembly maps engine exceptions and counters into one convention; the workspace
renders status rows; reports list what was not checked.
**Status.** accepted (Phase 13).

## ADR-010 — No services, databases, queues or plugin ABI until triggers are met

**Decision.** Plugins remain internal Python packages. Distribution-level plugins (entry points,
versioned ABI) are deferred until (a) two engines need incompatible dependency versions, (b) an
external maintainer ships a detector, or (c) release cadence must diverge.
**Context.** The ambitious alternative (C) is technically coherent but would add multi-package
release machinery and a public ABI with no external consumers.
**Alternatives.** Entry-point discovery now; a plugin package format; a gRPC engine boundary.
**Rationale.** Premature extensibility is a permanent tax paid for a hypothetical benefit. The
contract shape is kept so the option stays open.
**Consequences.** If a trigger fires, the work is a migration of the internal contract, not a
rewrite.
**Status.** accepted; triggers recorded.

## ADR-011 — Adopt OpenLineage vocabulary; do not run Marquez

**Decision.** Pipeline lineage uses the OpenLineage event vocabulary (`Dataset`/`Job`/`Run`, facets,
JSONL). Consume events; do not implement an emitter or depend on Marquez.
**Context.** OpenLineage is Apache-2.0, its client can write to files/stdout, and its facets are the
established extension point. Marquez is a JVM + Postgres service with a stale release cadence.
**Alternatives.** A bespoke lineage schema; DVC-only; MLMD protos.
**Rationale.** Reuse an existing vocabulary instead of inventing one; avoid operating a service.
**Consequences.** Column-level lineage is available only where an emitter provided it; row/split
lineage remains LeakLens's own contribution. ADR-021 fixes the *internal* contract
(`leaklens/lineage.LineageEdge`); OpenLineage names apply at the event boundary, not to
`EvidenceLink`, which remains an evidence-to-evidence type.
**Status.** proposed (Phase 14).

## ADR-012 — Integrations are read-only and produce inputs, never findings

**Decision.** `integrations/` adapters read external metadata (DVC, MLflow, OpenLineage, Feast,
serialized DAGs, HF snapshots) and produce audit inputs or evidence. Detection logic stays in
engines; no adapter may call a detector.
**Context.** Otherwise integrations become a second detector layer with its own conventions.
**Alternatives.** Integrations as plugins that emit findings.
**Rationale.** Keeps one detection vocabulary and lets a modality engine be tested without any
external system.
**Consequences.** An unparseable metadata file degrades to a skip; no integration executes the
external tool.
**Status.** accepted.

## ADR-013 — No parallelism in the deterministic path

**Decision.** Detection runs sequentially. Any future concurrent engine must sort its output
deterministically and be measured before acceptance.
**Context.** Findings order, evidence order and fingerprints must not depend on scheduling; the
existing scanner already declined parallel scanning after measurement.
**Alternatives.** Thread pool with ordered results; process pool per file.
**Rationale.** Determinism outranks throughput for an auditor; laptop-scale inputs are seconds.
**Consequences.** Very large corpora are slower; sampling and caching are the accepted mitigations.
**Status.** accepted.

## ADR-014 — Workspace stays server-rendered, no framework, no bundler

**Decision.** Every future workspace view must be reachable with server-rendered HTML plus small
vanilla-JS modules and the existing SSE stream.
**Context.** The shipped workspace is a thin local layer over the engine; adding React/Vite would
add a build step, a dependency tree and a client state store to an offline tool.
**Alternatives.** A SPA; a separate frontend package.
**Rationale.** The interface's job is evidence review, not application complexity.
**Consequences.** Some rich interactions (large graph canvases, dense image grids) need server-side
paging and bounded payloads; if a view cannot fit, it waits or it is cut.
**Status.** accepted.

## ADR-015 — One untrusted-intake boundary

**Decision.** All untrusted bytes pass a single intake layer with magic-byte validation, size/
entry/pixel budgets, a format allowlist and no writes to the scanned tree.
**Context.** `upload_guard` already implements this for workbooks (zip-slip, bombs, encryption,
entry limits); new image/archive/artifact inputs must not re-implement it.
**Alternatives.** Per-engine validation; trusting local files.
**Rationale.** Duplicated validation drifts; one boundary is testable with a single hostile corpus.
**Consequences.** New engines declare limits and receive validated references; refusals become
skips (ADR-009). `images.py` remains the image intake and gains the budgets; there is no
`analysis/vision/intake.py`.
**Status.** accepted (generalisation in Phases 16–17).

## ADR-016 — Model artifacts are parsed, never executed

**Decision.** JSON, `safetensors`, ONNX and numpy formats are parsed metadata-first; pickle and
`torch.load` are refused with a finding; no unfickler ships by default.
**Context.** An auditor pointed at a hostile model must not become the exploit.
**Alternatives.** Sandboxed unpickling by default; trusting Hugging Face provenance.
**Rationale.** Refusing is a legitimate, useful audit result; executing is an RCE class.
**Consequences.** Some legitimate pickle-only projects get a refusal instead of an inspection; the
finding explains how to export a safe format.
**Status.** accepted (Phase 16).

## ADR-017 — Phase numbering and the closure of Phase 12

**Decision.** Phase 12 is closed at `717398c` with no extension milestones. Future work is
Phases 13–19 in the order documented in [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md).
**Context.** Phase 12's largest open item is the missing H2 corpus, documented as a shortfall; more
AI work before that evidence exists would repeat the failure.
**Alternatives.** Continue Phase 12 with M9+; renumber everything.
**Rationale.** A phase boundary is also a claim boundary: closing 12 keeps its shortfall honest.
**Consequences.** learned/LLM work moves to Phase 18 under the same invariants.
**Status.** accepted.

## ADR-018 — No generalization or accuracy claims beyond measurement

**Decision.** Every measured number is published with its corpus, method, threshold and misses; no
claim of production accuracy, calibrated probability or distribution-free guarantee.
**Context.** H1 is 48 units at 0.833 accuracy; H2 does not exist; selective-risk bounds do not
survive repository-level shift.
**Alternatives.** Report point estimates without context.
**Rationale.** The project's credibility is its only moat against better-funded tools.
**Consequences.** Every new engine must ship a labelled benchmark or an explicit `no_bench_reason`.
**Status.** shipped (policy).

## ADR-019 — Extract `source.py`; do not move `cvcheck`/`flow`

**Decision.** Break the `scanner` ↔ `cvcheck`/`flow` mutual dependency by extracting the shared
scanner vocabulary and AST helpers into a new `leaklens/source.py`, re-exported by `scanner.py`:
`BENIGN`, `FUNCS`, `HINTS`, `RESAMPLE`, `TARGET_ENCODERS`, `FEATURE_SELECTION`, `SKIP`,
`_all_names`, `_finding_keys`, `_name`, `_node_index`, `_sev`, `_tnames`. Do not relocate the
analyzers into an `analysis/code/` package, and do not invent a `CodeSource` type here.
**Context.** Both analyzers import those names from the scanner while the scanner calls them
(function-locally), so the direction is mutual and invisible. `CodeSource` is not one of them: it
belongs to the AI layer (`ai/semantics.py`) and is unrelated to this cycle. A code-engine package
would fix the diagram but break `from leaklens.cvcheck import ...` imports, monkeypatch targets and
the documented module layout for no functional gain.
**Alternatives.** `analysis/code/` package with façades; leave the coupling; pass spans/flows
instead of the shared parser helpers.
**Rationale.** The smallest change that makes the direction one-way is a shared leaf module
containing what the analyzers actually import.
**Consequences.** `source.py` must stay dependency-free (stdlib only) and may not import `ai`.
Existing `from leaklens.scanner import SKIP, FUNCS, ...` keeps working through the re-export. If
Phase 16 or 17 needs a narrower interface, that is a separate decision with its own measurement.
**Status.** accepted (Phase 13).

## ADR-020 — One canonical modality vocabulary, owned by the catalogue

**Decision.** `manifest.py` owns a single `MODALITIES` frozenset beside `SURFACES`. Every
`Detector.modalities` entry references its members; engines, report and workspace read the
catalogue. Initial members: `tabular`, `code`, `sql`, `image`, `text`, `artifact`. `video` is a
candidate member decided by the Phase 17 experiment, not by an engine.
**Context.** Four modules already describe data shapes in local terms, and each future engine would
add a fifth dialect. A per-engine list cannot be validated, documented or rendered consistently.
**Alternatives.** Free-form strings per detector; a `Modality` enum per package; infer modality from
the file suffix.
**Rationale.** The catalogue already validates `kind` and `surface`; modality belongs to the same
contract and the same `--check`. Adding a modality is then a visible catalogue change with a
regenerated document, not a silent string.
**Consequences.** `tests/test_public_contract.py` freezes the members. Adding one is additive and
requires a one-line ADR amendment. A detector with no meaningful modality declares none rather than
inventing one.
**Status.** accepted (Phase 13, M1).

## ADR-021 — A new `LineageEdge` contract; `EvidenceLink` is not reused

**Decision.** Phase 14 introduces a dependency-free `leaklens/lineage.py` with
`LineageEdge(stage, column, transform, locator, confidence)` and
`confidence ∈ {confirmed, ambiguous, unresolved}`. The existing
`intelligence.schemas.EvidenceLink(source_id, target_id, relation, rationale)` keeps its meaning:
it links *evidence to evidence*. Neither type is converted into the other.
**Context.** ADR-011 adopts OpenLineage vocabulary at the event boundary, and a previous draft
assumed a lineage contract already existed in `intelligence/provenance`. It does not. `EvidenceLink`
supports an argument about findings; it cannot assert that a column flowed through a stage, and
overloading `relation` to do so would make both contracts weaker. The type also cannot live under
`intelligence/`: `analysis/pipeline` must construct edges, and engines may not import intelligence.
**Alternatives.** Reuse `EvidenceLink` with new relations; put `LineageEdge` in
`intelligence/lineage.py`; adopt the full OpenLineage RunEvent internally; defer lineage to a graph
database.
**Rationale.** Two small, honest types beat one type that means two things. A root primitive is the
only location both an engine and `intelligence` may import. OpenLineage field names are used where
they map, but LeakLens does not store RunEvents or run an emitter.
**Consequences.** `scan_sql` exports its resolved column map once (M2) instead of being re-parsed.
Only confirmed lineage can produce a high finding; ambiguous and unresolved lineage are diagnostics.
Row- and split-level lineage remains LeakLens's own contribution.
**Status.** proposed (Phase 14, M2).

## ADR-022 — One shared near-duplicate primitive for text and images

**Decision.** `leaklens/near_duplicate.py` owns `bucket → threshold → stable cluster key` and is
used by `analysis/text/dedup.py` and by the vision work. `images.py` keeps dHash and
`image_overlap`; it is extended, never copied. No embeddings in the primitive.
**Context.** The text and vision plans independently described the same clustering loop. `images.py`
already ships a hashing stack, so a second one under `analysis/vision/` would be a duplicate
implementation with two behaviours.
**Alternatives.** Duplicate the loop per domain; put it in a new `analysis/cluster/` package;
require an embedding backend.
**Rationale.** Clustering is arithmetic, not domain logic. One implementation is testable once,
deterministic by construction, and small enough to stay a single module.
**Consequences.** Cluster ordering is sorted and stable; thresholds are parameters with published
defaults; domain modules supply hash/distance functions only. An embedding path may *feed* the
primitive but never replaces its deterministic core.
**Status.** proposed (Phase 15, M3; consumed by Phase 17).

## ADR-023 — One `EngineUnavailable` base; the assembly maps it, nothing else

**Decision.** `leaklens/availability.py` defines `EngineUnavailable(ImportError)` plus one
module/extra probe. Every engine's unavailable error subclasses it (`SqlUnavailable` becomes a
subclass, keeping its public name). `assemble.py` catches `EngineUnavailable` and records
`(engine_id, reason)` as a skip; any other exception is an error, not a skip.
**Context.** The architecture doc described both "an exception type per optional engine" and "one
unavailable contract"; they cannot both be true. Engines also each grew a private `_available`
helper.
**Alternatives.** Keep per-engine exceptions with no base; introduce a status enum; let assembly
catch all exceptions.
**Rationale.** A base class preserves existing callers and tests while giving assembly one honest
catch. Catching every exception would hide engine bugs as "skipped", which is the failure mode this
decision exists to prevent.
**Consequences.** `availability.py` is stdlib-only and may not import engines. A new engine that
does not subclass the base fails review. Existing private probes are left alone until their engine
is next touched; Phase 13 M1 converts `images._pil` to raise an `EngineUnavailable` subclass
(`ImageUnavailable`) while keeping the `ImportError` behaviour callers already see.
**Status.** accepted (Phase 13, M1).

## ADR-024 — Security remediation gates new attack surface

**Decision.** The 20 open CodeQL alerts on `main` are triaged in
[ARTIFACT_SECURITY_STRATEGY.md](ARTIFACT_SECURITY_STRATEGY.md) §2.2 and remediated as milestone
M-S0 before Phase 14 adds pipeline, integration or agent intake. No alert is dismissed without a
written source-to-sink analysis and a passing regression test; the CodeQL configuration is not
weakened.
**Context.** One alert (`api.py:325`, `joblib.load` on a request-derived path) is a deserialization
hazard in the documented threat model; ten share one resolve+containment pattern CodeQL does not
model as a sanitizer (all ten in `WorkspaceManager.resolve`); six are operator-path false positives
or the containment gate itself; three expose exception text. Phases 14-19 add file decoders,
metadata parsers, an agent narration path and a remediation writer — every one a new intake or
output boundary.
**Alternatives.** Fix opportunistically during feature work; suppress the containment alerts;
disable the queries.
**Rationale.** Remediating shared root causes before adding surface is cheaper and keeps the alert
list meaningful. Suppression would trade a real future signal for a clean dashboard.
**Consequences.** M-S0 blocks Phase 14. The remediation rules are testable: containment cases,
refusal cases, no traceback text in responses. The release allowlist and action pinning (M-S4) land
in Phase 13 alongside the gate.
**Status.** accepted (Phase 13).
**Update (2026-11-13).** The containment alerts are closed by changing the
primitive, not by suppressing them. `paths.resolve_within` now normalises both
the root and the candidate with `os.path.realpath` and accepts the result only
inside a `(root + separator)` prefix, with the checked variable also being the
returned one. That is the shape CodeQL models as a path sanitizer, so every
caller that funnels request paths through `WorkspaceScanManager.resolve` stops
being an alert, and the same check refuses symlink and junction escapes.
Non-regular files are now refused before they are opened: directory scans keep
only regular files, and `Baseline.load` requires a regular file. Residual risk:
a check-then-open race between `resolve_within` and a later `open`/`stat` is not
eliminated; it needs local write access to the workspace to exploit.

## ADR-025 — Deterministic plans are intelligence-owned; `ai` narrates

**Decision.** The investigation plan and its dataclasses live in `intelligence/` (`schemas.py` for
the types, `investigation.py` for the builder). `ai/reasoning.py` consumes a plan through
`enrich_plan` and may only add validated narration. `intelligence` never imports `ai`; `ai.schemas`
re-exports the plan types for backward compatibility.
**Context.** Phase 18's first draft said `intelligence/investigation.py` "extends
`ai/reasoning.build_plan`", which the import matrix forbids. It also made a deterministic capability
depend on an optional extra, contradicting the rule that the AI layer is additive.
**Alternatives.** Keep the builder in `ai` and let surfaces call it (breaks "works with AI absent");
duplicate the plan type in both packages (two schemas drift immediately).
**Rationale.** The plan is evidence analysis, not a model output. Its owner should be the layer that
may read the evidence graph; the model layer should only phrase it.
**Consequences.** Phase 18 M1 may move code, so `ai.schemas` keeps re-exports and a test asserts
`from leaklens.ai.schemas import ReasoningPlan` still works. With the `ai` extra absent, the plan is
complete and the panel shows narration as unavailable.
**Status.** proposed (Phase 18, M1).

---

## ADR-026 - Distribution completeness is proven from the tagged tree

**Context.** `scripts/validate_release.py` rejected unexpected wheel members but never required an
expected one, so a wheel that omitted 46 modules passed the release check unchanged. The local
`dist/` artifacts are exactly that failure mode. A deny-list alone cannot prove completeness, and a
hand-maintained file list drifts from the source it describes.

**Decision.** Expected distribution membership is derived from the checked-out tree plus the
authoritative packaging configuration: `leaklens/**/*.py`, the
`[tool.setuptools.package-data]` globs in `pyproject.toml`, and the `MANIFEST.in` include rules that
point inside the package. Both the wheel and the sdist must contain every derived member; the wheel
must still carry only `leaklens/**` and its dist-info, and the sdist must contain no build, cache or
local-artifact path. A dedicated CI job builds both distributions on every pull request and runs the
same validator the release workflow runs.

**Alternatives.** A committed expected-files list (drifts immediately; the 46-module drift is the
proof); validating only in the release workflow (too late — a broken packaging config merges first);
an archive-content hash allowlist (no link to the tag being published).

**Rationale.** The tag is the source of truth for what must ship. Deriving membership means adding a
module or a package resource needs no validator change, and the failure message names the exact
missing path.

**Consequences.** `leaklens/prompts/explain.txt` is a required wheel member because `llm.py` loads it
by default; a build that drops it now fails before publishing. Archives are inspected as data and
never imported. Python 3.10 keeps working through the existing `tomli` fallback.

**Status.** accepted (release integrity wave P).

---

## Open and deferred decisions

| Id | Question | Why it is not decided | How it will be decided |
|---|---|---|---|
| ADR-U1 | Typed `AuditResult` vs the payload dict | One consumer today; a type would be a fourth representation | Decide when a second consumer or external API needs it |
| ADR-U2 | `DatasetRef` / `SplitSpec` input contract | Paths + `partition.infer_roles` cover today's modalities | Decide with the first modality whose splits cannot be expressed as paths |
| ADR-U3 | Value of perceptual hashing for CV dedup | No measurement yet | M2 of Phase 17: recall on generated duplicate families (crop, rotate, recompress, resize, watermark) |
| ADR-U4 | Shingle/MinHash vs normalised exact text hashing | False-positive rate unknown at laptop scale | M3 of Phase 15 benchmark |
| ADR-U5 | Numeric precision floor for engine graduation | Needs real-world volume and FPR data | Set after the first graduated engine's benchmark; publish the number |
| ADR-U6 | Image/document cluster rendering in the no-bundler workspace | Server-rendered approach plausible but unproven at scale | Spike during Phase 15/17 before committing the view |
| ADR-U7 | Distribution-level plugin ABI | Premature (ADR-010 triggers not met) | Revisit only if a trigger fires |
| ADR-U8 | Credential/secret-adjacent advisories for scanned code | Out of scope for ML integrity; belongs to SAST | Keep as an optional Semgrep path (Phase 19), never a native detector |
