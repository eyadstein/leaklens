# Phase 12.1 AI roadmap: research-backed proposal

**Status:** partly landed. **M1** (complete root-cause maps for every detector kind) shipped in
`leaklens/ai/reasoning.py` and `leaklens/intelligence/orchestrator.py`. **M2** (reproducible
model artifact and provenance) shipped in `leaklens/ai/artifact.py`, `leaklens/ai/model.json`
and `leaklens/ai/learned.py`. **M3–M8 remain proposals** and none of them is implemented.

This is an audit of the AI layer as shipped (`leaklens-ai-p7-2026.11`) plus an ordered
roadmap derived from its measured gaps. It exists so the next phase starts from evidence
instead of ambition. The architecture, safety invariants and limitations of the current
layer are documented in [AI_INTELLIGENCE.md](AI_INTELLIGENCE.md); measured results live in
[AI_BENCHMARK.md](AI_BENCHMARK.md), [AI_REALWORLD.md](AI_REALWORLD.md) and
[AI_PERFORMANCE.md](AI_PERFORMANCE.md).

## How this document was produced

- Read the shipped code paths end to end: `leaklens/ai/*`, `leaklens/cli.py`,
  `leaklens/public_api.py`, `leaklens/report.py`, `leaklens/workspace.py`,
  `leaklens/intelligence/*`, `leaklens/fingerprint.py`.
- Audited the docs listed above for claims that the code does not support.
- Reviewed external work by fetching arXiv abstract pages and the GitHub REST API on
  2026-11-09. Quoted numbers are the numbers those abstracts and repository metadata state;
  no full paper was read, and nothing external was installed, vendored or copied.
- `websearch` was unavailable in this session (HTTP 402), so discovery went through arXiv
  search pages and direct repository metadata. Searches were narrow, so this is a
  purposeful sample of related work, not a systematic literature review.

Claim provenance is marked where it matters: **verified** means observed in this repository
with a file reference, **hypothesis** means a prediction about behaviour that nothing has
measured yet, and **not adopted** means considered and rejected with the reason.

---

## 1. What exists today

### 1.1 Runtime shape

| Aspect | Verified state |
|---|---|
| Entry points | `--ai`, `--ai-llm` on the CLI (`cli.py`), `include_ai=` on `public_api.run_audit_job`, `analyze()`/`analyze_dataset()` in Python |
| Default | Off. Every entry point is opt-in; nothing runs unless asked |
| Model | Class-balanced logistic regression over 108 model-eligible features; `MODEL_VERSION = "leaklens-ai-p7-2026.11"` (`ai/learned.py:56`) |
| Training | Fitted at request time from the authored corpus, grouped CV, calibration from out-of-fold probabilities, threshold from validation only (`ai/learned.py`) |
| Threshold | 0.53, `validation_midpoint` rule |
| Serialization | None. The fitted model lives in a process-local cache; there is no artifact to load, inspect or pin |
| Deterministic counterpart | `ai/features.py` indicator extraction, `ai/semantics.py` name/flow rules, `ai/reasoning.py` grouping and remediation |
| LLM | Optional, local Ollama by default, findings-only prompt, schema-validated output, deterministic fallback with a recorded reason |
| Outputs | `AIReport` (hypotheses, plan, model cards, limitations) → JSON, HTML "AI intelligence (experimental)" section, console summary |

### 1.2 Contracts the layer must keep

These are enforced in code and tests today, and every milestone below inherits them:

1. `Finding` objects, `leakage.score()`, the exit code, baselines, suppressions and
   `policy.apply_policy()` are untouched by AI. The report echoes the deterministic score.
2. `Hypothesis` severity hints are capped at `medium`; confidence at `0.95`; evidence is
   mandatory and must reference a known source kind (`ai/schemas.py`).
3. Evidence ids are schema-valid and unique; the LLM cannot add groups, evidence, findings
   or severity, and any mismatch discards its output.
4. Fingerprints, provenance and the deterministic evidence graph are byte-stable with
   `--ai` off.
5. Offline and CPU-only: no network, no downloaded weights, no pickle loading.
6. AI failure degrades to "AI unavailable", never to a failed scan (broad `except` in
   `cli.py` and `public_api.py`).

### 1.3 Measured behaviour

| Measurement | Value | Source |
|---|---|---|
| Locked test (47 authored cases) | accuracy 0.979, precision 1.000, recall 0.960, F1 0.980, FPR 0.000 | AI_BENCHMARK.md |
| Phase 7 authored holdout (12 units) | 6/6 leaks, 1 FP on 6 legitimate | AI_BENCHMARK.md |
| Real-world holdout H1 (48 units, 12 repos) | accuracy 0.833, precision 0.767, recall 0.958, F1 0.852, FPR 0.292 | AI_REALWORLD.md |
| Calibration on H1 | Brier 0.1149, ECE 0.1264, mean probability 0.6232 vs observed rate 0.5000 | AI_REALWORLD.md |
| Cold fit | 1.339 s, 0.83 MiB peak | AI_PERFORMANCE.md |
| Feature extraction | 2.67 ms/unit median, 3.68 ms p95 | AI_PERFORMANCE.md |
| Whole-project scan | 49 files / 713 KB → 3.386 s feature extraction | AI_PERFORMANCE.md |

The H1 numbers were produced with the Phase 6 model at threshold 0.50. The shipped Phase 7
model (threshold 0.53, changed corpus, extraction guards, local evidence, runtime gate) has
**no independent real-world re-score**. That single fact is the largest credibility gap in
this document.

---

## 2. Capability assessment

### What the layer does well (verified)

- **Additive by construction.** Every AI surface is a separate object; no deterministic
  field is recomputed. Tests assert byte-identical findings and exit codes with and without
  AI (`tests/test_ai_integration.py`).
- **Honest failure.** A missing provider, malformed JSON or invented group id produces a
  deterministic plan plus a recorded reason, not a partial AI result.
- **Cheap on CPU.** Sub-second steady-state cost per unit and a 1.3 s cold fit on a
  machine with no GPU. Memory is under 1 MiB.
- **Evidence-bound.** A hypothesis without evidence fails validation; the LLM cannot invent
  ids.
- **Evaluated with a real protocol.** Grouped splits by source and template family, a
  locked test that training never imports, an external holdout, published per-repository
  and per-family breakdowns.

### Where it is weak (verified, unless marked)

| # | Weakness | Evidence |
|---|---|---|
| W1 | Two deterministic reasoning maps were never extended for the Phase 10 detector kinds, so `sql_*`, `missingness_*` and `distribution_shift` findings land in a generic bucket | `ai/reasoning.py:299-328`, `intelligence/orchestrator.py:56-112` and the `unclassified_leakage` fallback at `orchestrator.py:283-291` |
| W2 | AI "confidence" mixes hand-assigned rule weights (0.58–0.72 in `ai/semantics.py`) with derived values (`0.35+0.55p` in `ai/learned.py`) and is rendered as an unlabelled number | `semantics.py:297,330,354,378,399,417`; `report.py:437,450` |
| W3 | AI evidence ids and the deterministic evidence graph use different identity systems, so a hypothesis cannot point at a graph node or cluster | `ai/schemas.py` `EvidenceSpan` vs `intelligence/schemas.py` `evidence_id`/`cluster_id`/`anchor` |
| W4 | The model is unversioned at the artifact level: no serialized file, no corpus digest in `ModelCard`, no way to reproduce the exact serving model without re-running the fit | `ai/learned.py` `get_model`, `ModelCard` |
| W5 | The shipped model's real-world behaviour is unmeasured; the only external number is from the previous model at the previous threshold | AI_REALWORLD.md freeze note |
| W6 | Precision on real repositories is 0.767, with four recurring false-positive families (safe CV, temporal split, CatBoost `Pool`, EDA-only) | AI_REALWORLD.md limitations |
| W7 | Probabilities are not probabilities off-corpus: overconfident by ≈0.12 on H1 (mean 0.6232 vs observed 0.5000) and worse under shift | AI_REALWORLD.md calibration table; matches Ovadia et al. 2019 |
| W8 | The workspace executes no AI at all; a dataset audit calls `run_audit_job` without `include_ai`, and the report's `ai` key is always `None` for workspace scans | `workspace.py:1553`, `workspace.py` import list |
| W9 | The layer is Python/notebook-only: `collect_code_sources` reads `.py`/`.ipynb` under a 200-file / 512 KB cap, so SQL and non-Python pipelines are invisible to the learned detector | `ai/analysis.py:24-25` |
| W10 | The external holdout is thin and single-reviewed: 48 units, 12 repositories, 44 reviewed reconstructions, one labeller, 7 of 23 cited paths already drifted | AI_INTELLIGENCE.md limitations |

W9 and W10 are scope and evidence limits, not bugs. They are listed so nobody spends a
phase "fixing" a deliberate boundary.

---

## 3. External work review

Each entry records what was verified, what it supports here, and what it does **not**
justify. Nothing was installed or copied.

### 3.1 Leakage in ML systems

- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science"**
  ([arXiv:2207.07048](https://arxiv.org/abs/2207.07048)). Eight leakage types across 17
  fields and 329 papers; proposes model info sheets recording split and preprocessing
  decisions. *Supports:* W1 — a taxonomy is only useful if every detector kind maps into it;
  and the "record what the model saw" idea behind W4/W5. *Does not justify:* importing
  their taxonomy wholesale; LeakLens's finding kinds are the contract.
- **Yang, Brower-Sinning, Lewis, Kästner, "Data Leakage in Notebooks"**
  ([arXiv:2209.03345](https://arxiv.org/abs/2209.03345)). Static detection over 100k+
  public notebooks; reports that false positives are the practical cost of static rules.
  *Supports:* W2/W6 — precision work beats recall work at this stage. *Does not justify:*
  their rule set; this repo already ships an AST-based scanner.
- **LeakageDetector 2.0** ([arXiv:2509.15971](https://arxiv.org/abs/2509.15971)). VS Code
  extension with LLM-generated fix guidance. *Verified:* no maintained public artifact was
  found via GitHub search (results were unrelated or stale). *Supports:* the LLM-as-guidance
  (never LLM-as-verdict) shape already used here. *Not adopted:* no license-verifiable code
  to learn from, and no reason to depend on a closed tool.

### 3.2 AI-assisted code analysis

- **Li, Dutta, Naik, "IRIS: LLM-Assisted Static Analysis for Detecting Security
  Vulnerabilities"** ([arXiv:2405.17238](https://arxiv.org/abs/2405.17238)). Neuro-symbolic:
  the LLM proposes taint specifications, CodeQL validates them; 55 of 120 CVEs found vs 27
  for CodeQL alone. *Supports:* the gate pattern in `ai/reasoning.py` — model output must be
  checked by a deterministic validator, and validation is where the value is. *Does not
  justify:* adding CodeQL-class infrastructure; that is a different product.
- Repository metadata checked for adjacent tooling: **semgrep** (LGPL-2.1, very active),
  **joern** (Apache-2.0, very active), **meta/pyre-check** (MIT, **archived**).
  *Not adopted:* none of these detect leakage; adopting one would replace the scanner.
  pyre is archived and therefore excluded on maintenance grounds, not licence.

### 3.3 Explainability and evidence grounding

- **Turpin et al., "Language Models Don't Always Say What They Think"**
  ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)). Chain-of-thought text can omit the
  real reasons for an answer. *Supports:* never treat LLM prose as evidence — exactly the
  current contract, restated with a citation.
- **Zheng et al., "Judging LLM-as-a-Judge"** ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).
  Judges show position, verbosity and self-enhancement bias. *Supports:* if the LLM
  rephrasing is ever scored, it needs a rubric and reference answers, not a preference
  comparison. *Does not justify:* building an LLM judge — out of scope.
- **Greshake et al., "Not what you've signed up for"**
  ([arXiv:2302.12173](https://arxiv.org/abs/2302.12173)). Indirect prompt injection through
  untrusted content. *Supports:* the existing untrusted-instruction detection in
  `ai/semantics.py`; any future prompt change must keep it.

### 3.4 Uncertainty and calibration

- **Guo et al., "On Calibration of Modern Neural Networks"**
  ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Temperature/Platt scaling recovers
  calibrated probabilities. *Supports:* a calibrated output is a modelling choice with a
  known method; W7 is fixable in principle.
- **Ovadia et al., "Can You Trust Your Model's Uncertainty?"**
  ([arXiv:1906.02530](https://arxiv.org/abs/1906.02530)). Calibration degrades under dataset
  shift across methods. *Supports:* W7 — do not promise transferable probabilities on unseen
  repositories; report a ranking score with an explicit basis instead.
- **Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction"**
  ([arXiv:2107.07511](https://arxiv.org/abs/2107.07511)). Distribution-free intervals under
  exchangeability. *Supports:* M7's selective reporting. *Does not justify:* claiming a
  guarantee on real repositories — exchangeability is violated by repository-level shift,
  so any number must be labelled empirical.
- **Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks"**
  ([arXiv:1705.08500](https://arxiv.org/abs/1705.08500)). Trading coverage for risk with a
  reject option. *Supports:* M7 — "insufficient evidence" is a better output than a
  confident guess.

### 3.5 Documentation and metadata practice

- **Mitchell et al., "Model Cards for Model Reporting"**
  ([arXiv:1810.03993](https://arxiv.org/abs/1810.03993)). *Supports:* W4 — the existing
  `ModelCard` should carry corpus and feature-catalogue digests plus the library version.
- **Gebru et al., "Datasheets for Datasets"** ([arXiv:1803.09010](https://arxiv.org/abs/1803.09010)).
  *Supports:* documenting the authored corpus and both holdouts the same way.
- **Akhtar et al., "Croissant"** ([arXiv:2403.19546](https://arxiv.org/abs/2403.19546)).
  Metadata format for ML-ready datasets. *Not adopted:* LeakLens reads datasets, it does not
  publish them; adding a metadata export would be a new product surface with no detector
  benefit.

### 3.6 Lightweight local inference and data-quality tooling

Repository metadata checked (licence, activity):

| Project | Licence | State | Decision |
|---|---|---|---|
| `microsoft/onnxruntime` | MIT | very active | not adopted — a 108-coefficient linear model needs no runtime |
| `dmlc/treelite` | Apache-2.0 | active | not adopted — same reason; revisit only if the model class changes |
| `MinishLab/model2vec` | MIT | active | not adopted — static embeddings would add a model download, breaking offline-by-default |
| `cleanlab/cleanlab` | Apache-2.0 | active | reference only — label-noise tooling, solves a different problem |
| `deepchecks/deepchecks` | NOASSERTION | last push 2025-12-28 | not adopted — licence is not clean enough for a dependency, and the scope overlaps |
| `fivetran/great_expectations` | Apache-2.0 | active | not adopted — expectation suites are not leakage detection |

**Verified conclusion:** no external dependency is justified for the AI layer at its current
size. Every milestone below is implementable with the standard library plus the already
present `scikit-learn`/`numpy`/`pandas`.

---

## 4. Gap analysis, ranked

Ranking is by (user-visible wrongness × likelihood) ÷ cost.

| Rank | Gap | Why it matters | Cost |
|---|---|---|---|
| 1 | W1 reasoning maps miss Phase 10 kinds | A user with only SQL or missingness findings gets a plan that says "unclassified risk", which reads as a tool defect | small |
| 2 | W4 no artifact or provenance | Cannot answer "which model produced this report", cannot diff model behaviour across releases, pays 1.3 s per process for nothing | small |
| 3 | W2 confidence semantics | A heuristic weight rendered next to "Confidence" is read as a probability; that is the one thing this layer promised not to do | small |
| 4 | W5 shipped model unmeasured | All external credibility rests on a superseded model | medium (data, not code) |
| 5 | W3 evidence identity split | Hypotheses and the evidence graph cannot be navigated together, so the two intelligence surfaces contradict each other's vocabulary | medium |
| 6 | W6 real-world precision 0.767 | Seven false positives on twenty-four legitimate units is the dominant user complaint risk | medium |
| 7 | W7 probability not transferable | Users will calibrate trust on a number that means something else off-corpus | medium |
| 8 | W8 no workspace surface | The interactive product cannot show the layer at all | medium |

---

## 5. Milestones

Ordering rule: correctness and honesty first, then evidence, then modelling, then product
surface. Each milestone is independently shippable and independently revertible.

Every milestone inherits the invariants in §1.2. None may change `Finding`, `leakage.score()`,
fingerprints, baselines, suppressions, policy, or the AI-off output of any command.

### M1 — Complete the root-cause maps for every detector kind

- **Objective:** no deterministic finding kind may resolve to a generic bucket.
- **User capability:** a report containing SQL-lineage, missingness or drift findings gets a
  named root cause, explanation and remediation instead of "unclassified risk".
- **Affected modules:** `leaklens/ai/reasoning.py` (`_FINDING_CATEGORY`, `_CONCEPT_CATEGORY`),
  `leaklens/intelligence/orchestrator.py` (`_ROOT_CAUSES`, `_RECOMMENDATIONS`).
- **New modules:** none.
- **Integration:** the maps are pure lookups consumed by `build_plan` and `analyze_findings`;
  no call-site changes.
- **Compatibility:** new keys only. Existing mapping entries and their strings stay
  byte-identical, so existing plans, groups and tests are unchanged.
- **Evaluation and acceptance:** a table-driven test over the eight currently-unmapped kinds
  (`distribution_shift`, `missingness_target_signal`, `missingness_split_shift`,
  `missingness_group_signal`, `sql_target_as_feature`, `sql_split_predicate_missing`,
  `sql_future_window`, `sql_unresolved_column`) asserts a specific category and asserts that
  `analyze_findings` produces no `unclassified_leakage` group. Existing plan tests and the
  HTML report output must be unchanged.
- **Tests:** extend `tests/test_ai_reasoning.py` and the intelligence tests; one new
  parameterized case per kind.
- **Risks:** category names become a public vocabulary once reports ship. Mitigation: choose
  names that match the deterministic docs (`SQL_LINEAGE.md`, `MISSINGNESS_BENCHMARK.md`,
  `DRIFT.md`).
- **Dependencies:** none.
- **Performance budget:** zero — dictionary lookups.
- **Why first:** smallest complete fix for the most visible wrongness, with no modelling risk.

### M2 — Reproducible model artifact and provenance

- **Objective:** the serving model becomes a versioned, inspectable JSON artifact whose
  provenance is recorded, with an in-memory fit as the fallback.
- **User capability:** a report can state exactly which model, corpus digest, feature
  catalogue and library version produced it, and two installs of the same release produce
  identical scores without a 1.3 s fit.
- **Affected modules:** `leaklens/ai/learned.py` (`get_model`, `ModelCard`).
- **New modules:** `leaklens/ai/artifact.py`, justified because serialization, digesting and
  version negotiation are a separate concern from fitting and would otherwise bloat
  `learned.py`, which is already 1300 lines.
- **Integration:** `get_model()` loads the artifact when present and digest-valid, otherwise
  fits as today. `ModelCard` gains `artifact_digest`, `corpus_digest`,
  `feature_catalogue_digest`, `library_version`.
- **Compatibility:** `ModelCard` is additive; older payloads still validate. No pickle, no
  executable format — plain JSON with a schema version, so a malicious artifact cannot
  execute code. Artifact mismatch or corruption falls back to fitting, never to a wrong
  model.
- **Evaluation and acceptance:** for all 48 H1 units and all 47 locked-test cases, artifact
  and fit predictions agree to 1e-9; `AI_PERFORMANCE.md` records cold-load time; the
  artifact is under 64 KB; digest mismatch, truncation and a hand-edited coefficient each
  fall back to fit and record a reason.
- **Tests:** parity test, corruption matrix, fallback-with-reason test, `ModelCard` round
  trip.
- **Risks:** a stale checked-in artifact silently shadowing a code change. Mitigation: the
  feature-catalogue and corpus digests are part of the validation key, and a benchmark test
  fails when the corpus digest changes without a regenerated artifact.
- **Dependencies:** none.
- **Performance budget:** load ≤ 50 ms, ≤ 64 KB on disk, no measurable change to per-unit
  prediction.
- **Why here:** it makes every later measurement reproducible and cuts cold-start cost.

### M3 — Explicit confidence semantics

- **Objective:** stop presenting three different quantities as one unlabelled number.
- **User capability:** a user can tell whether a hypothesis confidence is a calibrated
  probability, a rule weight or an unavailable estimate, in JSON, HTML and the console.
- **Affected modules:** `leaklens/ai/schemas.py`, `ai/semantics.py`, `ai/learned.py`,
  `ai/reasoning.py`, `report.py`, `cli.py`.
- **New modules:** none.
- **Integration:** `Hypothesis` gains `confidence_basis`
  (`calibrated_probability|heuristic_weight|unavailable`) and `sufficiency`
  (`supported|weak`); rule scores keep their values. `report.py` renders "Confidence
  (heuristic rule strength)" or "Confidence (uncalibrated ranking score)" as appropriate and
  never prints a bare percentage.
- **Compatibility:** additive fields with defaults; `SCHEMA_VERSION` moves to `1.1` and old
  payloads still validate. Values are unchanged, so no finding, plan or score moves.
- **Evaluation and acceptance:** a schema test rejects a hypothesis that claims
  `calibrated_probability` without a fitted model behind it; a report test asserts the basis
  label is rendered for both rule-based and learned hypotheses; the console summary names the
  basis once per run.
- **Tests:** schema validation matrix, report rendering matrix, CLI summary snapshot.
- **Risks:** more text in the report. Mitigation: one label per hypothesis line, no prose.
- **Dependencies:** none.
- **Performance budget:** negligible.
- **Why here:** it is the cheapest way to keep the layer's central promise — never imply a
  precision it does not have.

### M4 — AI evidence anchored to the deterministic evidence graph

- **Objective:** an AI evidence span identifies the same location as the corresponding graph
  node, without changing the graph.
- **User capability:** in one report, a hypothesis and an evidence-graph node can be told
  apart and cross-referenced by anchor, so the two intelligence layers share a vocabulary.
- **Affected modules:** `ai/schemas.py` (`EvidenceSpan`), `ai/learned.py`, `ai/semantics.py`,
  `ai/analysis.py`, and a read-only lookup against `intelligence` node ids.
- **New modules:** none — the mapping is a function over existing structures.
- **Integration:** when findings exist, each span records an optional `graph_node_id` looked
  up from the existing graph by normalized location
  (`fingerprint.normalize_location`). With `--ai` off, nothing is computed.
- **Compatibility:** graph serialization, node ids and fingerprints are unchanged; the AI
  report gains a nullable field. A span whose location matches no node keeps `graph_node_id:
  null` rather than inventing an id.
- **Evaluation and acceptance:** for a fixture scan, every span that matches a node carries
  that node's real id; no span carries an id absent from the graph; `--intelligence`-only
  output is byte-identical to today.
- **Tests:** anchor-mapping test, no-invented-id test, AI-off byte-equality test.
- **Risks:** false matches on repeated locations. Mitigation: match on normalized location
  plus evidence source kind, and leave ambiguous matches null.
- **Dependencies:** none, but it is easier to test after M1 gives every kind a category.
- **Performance budget:** one dictionary build over existing nodes; no new file IO.
- **Why here:** it turns two parallel explanations into one navigable story before the
  workspace exposes either.

### M5 — Independent real-world re-score of the shipped model

- **Objective:** publish a measured external result for the model users actually get.
- **User capability:** a user can read an honest precision/recall/calibration number for the
  shipped version instead of one from a superseded model.
- **Affected modules:** `leaklens/bench_realworld.py` (holdout selection only),
  `docs/AI_REALWORLD.md` (new H2 section or a sibling document).
- **New modules:** none.
- **Integration:** reuses the existing frozen-manifest, cluster-bootstrap and calibration
  reporting path.
- **Compatibility:** H1 stays published as the Phase 6 record; H2 is added, never substituted.
- **Evaluation and acceptance:** H2 has ≥48 units from ≥12 repositories disjoint from the
  authored corpus and H1; provenance per unit; every cited path verified against a pinned
  commit; at least two reviewers with disagreements recorded; a frozen manifest digest; one
  scoring run, reported with cluster-bootstrap intervals, calibration metrics and per-family
  false positives. If any of that cannot be met, publish the shortfall in the document rather
  than a number.
- **Tests:** digest-freeze test, disjointness test, no-near-duplicate test (same guards as the
  locked test).
- **Risks:** this is the most effort-heavy milestone and produces no code. Mitigation:
  timebox the corpus authoring and accept a documented shortfall.
- **Dependencies:** reviewer time; M2 makes the "which model was scored" question unambiguous.
- **Performance budget:** n/a.
- **Why here:** every later model change needs an honest baseline to be judged against.

### M6 — Reduce real-world false positives on the four known families

- **Objective:** raise precision on the measured holdout without giving back recall.
- **User capability:** fewer false accusations on safe CV, temporal splits, CatBoost `Pool`
  and EDA-only notebooks.
- **Affected modules:** `leaklens/ai/features.py` (deterministic hard-negative guards),
  `leaklens/ai/corpus.py` (hard-negative cases), `ai/learned.py` only if a guard needs a new
  feature.
- **New modules:** none.
- **Integration:** guards are deterministic feature-level checks, so the model class, feature
  catalogue contract (ADR 0001) and threshold selection are untouched.
- **Compatibility:** the feature catalogue stays authoritative; adding an indicator is a
  registered change with an explanation table entry, per ADR 0001.
- **Evaluation and acceptance:** tune on development CV and the Phase 7 authored holdout only,
  then score H2 once. Acceptance: each of the four families has a named guard with a
  before/after count on the development corpus; the locked test does not regress below
  accuracy 0.979 and precision 1.000; every guard has a unit test and a counter-case. The H2
  target (precision ≥ 0.80 at recall ≥ 0.93) is a hypothesis, not a promise — publish whatever
  the single run shows.
- **Tests:** one positive and one negative case per guard; locked-test regression; benchmark
  delta recorded in `AI_BENCHMARK.md`.
- **Risks:** guards can mask real leaks. Mitigation: every guard is a narrowly scoped signal
  (e.g. a `Pipeline` inside a CV call), never a suppress-all rule, and the recall floor is
  part of acceptance.
- **Dependencies:** M5 (baseline), M2 (reproducible scoring).
- **Performance budget:** ≤ 3% increase in per-unit feature extraction (≈ 0.08 ms).
- **Why here:** precision is what decides whether teams keep the layer switched on.

### M7 — Selective reporting with explicit insufficiency

- **Objective:** replace "here is a probability" with "here is a decision, and whether the
  evidence supports a decision".
- **User capability:** the layer can say "insufficient calibration data for this input" instead
  of emitting a number that will be misread.
- **Affected modules:** `ai/learned.py` (selective risk and coverage from grouped out-of-fold
  probabilities), `ai/schemas.py`, `report.py`, `cli.py`.
- **New modules:** none.
- **Integration:** a selected threshold set is computed at fit time from development data; the
  runtime either asserts "supported" with an empirical selective-risk figure or returns
  "insufficient", which renders as a ranking-only hypothesis.
- **Compatibility:** `Hypothesis` keeps `confidence` and gains nothing mandatory; the
  `sufficiency` field from M3 carries the decision. A `sufficiency: insufficient` hypothesis
  still has evidence and still cannot escalate severity.
- **Evaluation and acceptance:** on H2, report coverage and empirical selective risk with a
  cluster-bootstrap interval, plus the explicit statement that the exchangeability assumption
  is violated by repository-level shift so no distribution-free guarantee is claimed. On the
  authored corpus, selective risk at the chosen coverage must not exceed the unselective
  error rate.
- **Tests:** selective risk monotonicity on synthetic data, insufficient-path rendering,
  no-guarantee-language test (asserts the docs and report never claim a guarantee).
- **Risks:** a selection threshold that discards most hypotheses, making the layer useless.
  Mitigation: report coverage; if coverage on H2 falls below 0.5, keep the unselective output
  as the default and publish the finding.
- **Dependencies:** M6 (a stable score), M5 (measurement), M2 (reproducibility).
- **Performance budget:** one extra sort over development probabilities at fit time; runtime
  unchanged.
- **Why here:** it is the only honest answer to W7, and it needs the score to be stable first.

### M8 — Opt-in AI surface in the workspace

- **Objective:** let the interactive product show the same AI report the CLI shows — only when
  asked.
- **User capability:** run a source scan or a dataset audit in the workspace with AI
  hypotheses, root-cause groups and limitations attached, without changing the trust score or
  policy outcome.
- **Affected modules:** `leaklens/workspace.py`, `leaklens/workspace_ui.py`,
  `leaklens/public_api.py` (pass-through only).
- **New modules:** none.
- **Integration:** a request flag (default off) adds `include_ai=True` to
  `run_audit_job` for dataset audits and runs `analyze` with `collect_code_sources` under the
  existing 200-file / 512 KB caps for source scans. Rendering reuses `report._ai_section` and
  the existing serialized payload; nothing is re-implemented in the workspace, per the
  workspace rule in `AGENTS.md`.
- **Compatibility:** default behaviour, endpoints, status codes and the single activity slot
  are unchanged; AI runs inside the existing engine call, so overlap still returns 409 and an
  in-flight audit still cannot be cancelled. AI failure returns the standard
  `ai_unavailable` shape, not an error page. Policy is applied to deterministic findings
  before rendering, so AI never sees a suppressed finding as active.
- **Evaluation and acceptance:** AI off is byte-identical to today; AI on never changes
  `trust_score`, findings, policy labels or exit semantics; a forced AI exception returns a
  normal result with `ai_unavailable: true`; an AI-on run still occupies exactly one slot.
- **Tests:** default-off equality, AI-on score invariance, failure isolation, slot-overlap
  regression, HTML escaping of AI content in the workspace render.
- **Risks:** a slow AI pass making the workspace feel broken. Mitigation: report a stage
  label through the existing progress channel and keep the caps; do not add a second slot.
- **Dependencies:** M1–M4 (stable, honest content worth showing), M2 (fast cold start).
- **Performance budget:** ≤ 4 s added on a 50-file project scan (current 3.4 s feature
  extraction plus the existing fit), with a hard cap from the source reader.
- **Why last:** the surface should show the fixed semantics, not the current ones.

---

## 6. Recommended first milestone

**M1.** It is hours of work, needs no model, no data and no API change, and it removes a
visible wrongness: today a SQL-lineage or missingness finding produces an "unclassified risk"
root cause and an `other:*` plan group. It also lays the shared vocabulary the later
milestones reuse (M4 anchors, M8 rendering), so doing it first reduces work later instead of
adding work.

If time allows exactly one more, take **M2**: it is the gate that makes M5 and M6
reproducible.

---

## 7. Acceptance criteria for the whole roadmap

A milestone is done only when all of these hold:

1. `pytest -q` passes; new non-trivial logic has a test that fails without it.
2. `leaklens --scan leaklens` reports no leaks and exits 0.
3. The four benchmarks run: `python -m leaklens.bench`, `bench_data`, `bench_missing`,
   `bench_drift`. AI-touching milestones also run `bench_ai` and `bench_realworld`.
4. With AI off, output for every entry point is byte-identical to the previous release —
   checked by a test, not by inspection.
5. The trust score, exit code, fingerprints, baseline comparison and policy labels are
   unchanged by AI, asserted in tests.
6. Documentation is updated in the same change: the affected `docs/AI_*.md`, plus this file's
   status line when a milestone lands.
7. Every performance number that moves is re-measured and recorded, not estimated.
8. Nothing new is required at runtime: no network, no GPU, no new mandatory dependency.

Metrics that must not silently regress: locked-test accuracy 0.979, precision 1.000, recall
0.960, FPR 0.000. Any decrease is a blocker, not a trade-off.

---

## 8. Risks, dependencies and exclusions

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| A later milestone weakens the additive contract "just this once" | Trust in every number the project publishes | Invariant tests run per milestone; CI blocks on them |
| H2 authoring is slower than expected | Roadmap stalls at M5 | Timebox; publish the shortfall; never backfill with tuned numbers |
| Confidence relabelling is read as a downgrade | Perceived regression | Ship it with the docs change that explains the three bases and why the number did not move |
| Hard-negative guards hide real leaks | Worse than the false positives they remove | Narrow scopes, recall floor in acceptance, per-guard tests |
| Selective reporting discards too much | Layer becomes noise | Coverage is reported; unselective output stays the default if coverage < 0.5 |
| Workspace AI makes scans feel slow | Product regression | Stage label, existing caps, one slot, 4 s budget |

### Dependencies

- `scikit-learn >= 1.3` (already a core dependency; `--arena` needs it too, so it cannot be
  demoted to an AI-only extra without a breaking install change).
- Reviewer time for H2 (M5). This is the only external dependency in the roadmap.
- ADR 0001 discipline for any feature-catalogue change (M6).

### Considered and not adopted

- **Move `scikit-learn` to an `[ai]` extra.** Verified finding: it is a core dependency, but
  the deterministic scanner does not import it — `arena.py` and `ai/learned.py` do.
  `--arena` already requires the `arena` extra for mlflow, so a shared `[ml]` extra would
  work. **Not adopted now:** it is a packaging release change (pyproject, Docker images, CI,
  install docs, the arena extra) whose only benefit is install size, and it makes an
  experimental module fail at import instead of at the opt-in call. Revisit only if install
  size becomes a reported complaint.
- **ONNX Runtime, Treelite or model2vec at inference.** A 108-coefficient linear model
  already predicts in ~0.6 ms. Adding a runtime or a downloaded embedding model would trade a
  real offline guarantee for a cost that does not exist.
- **Semgrep, Joern, pyre-check, CodeQL integration.** Different problem class; Joern and
  CodeQL also need a separate runtime, and pyre is archived.
- **A hosted LLM or paid API as a default path.** Violates offline-first and would put
  dataset-adjacent content on the wire.
- **LLM-written fixes applied to user code.** The layer produces suggestions; auto-editing
  user pipelines is out of scope and unsafe.
- **Croissant or datasheet export.** New product surface with no detector benefit.
- **A neural or transformer detector.** The corpus is 259 development cases; a larger model
  would overfit it and break the CPU and memory budget.
- **Non-Python pipeline languages.** `collect_code_sources` reads `.py`/`.ipynb`. Extending it
  is a scanner project (SQL already has its own module), not an AI project.
- **AI findings feeding the trust score or the exit gate.** Non-negotiable.

### Out of scope for Phase 12

- Any change to deterministic detectors, findings, fingerprints or policy.
- Any change to the locked test, the Phase 7 holdout or H1.
- Version bumps, releases, tags or packaging changes.
- A general "AI agent" surface, chat interface or autonomous fixing.

---

## 9. Sources

Every source below was fetched successfully on 2026-11-09.

| Source | Link | Verified |
|---|---|---|
| Kapoor & Narayanan, leakage reproducibility | https://arxiv.org/abs/2207.07048 | abstract |
| Yang et al., data leakage in notebooks | https://arxiv.org/abs/2209.03345 | abstract |
| LeakageDetector 2.0 | https://arxiv.org/abs/2509.15971 | abstract; no maintained artifact found |
| Li et al., IRIS neuro-symbolic static analysis | https://arxiv.org/abs/2405.17238 | abstract |
| Guo et al., calibration of modern networks | https://arxiv.org/abs/1706.04599 | abstract |
| Angelopoulos & Bates, conformal prediction | https://arxiv.org/abs/2107.07511 | abstract |
| Geifman & El-Yaniv, selective classification | https://arxiv.org/abs/1705.08500 | abstract |
| Turpin et al., unfaithful chain-of-thought | https://arxiv.org/abs/2305.04388 | abstract |
| Zheng et al., LLM-as-a-judge | https://arxiv.org/abs/2306.05685 | abstract |
| Ovadia et al., uncertainty under dataset shift | https://arxiv.org/abs/1906.02530 | abstract |
| Mitchell et al., model cards | https://arxiv.org/abs/1810.03993 | abstract |
| Gebru et al., datasheets for datasets | https://arxiv.org/abs/1803.09010 | abstract |
| Greshake et al., indirect prompt injection | https://arxiv.org/abs/2302.12173 | abstract |
| Akhtar et al., Croissant | https://arxiv.org/abs/2403.19546 | abstract |
| cleanlab | https://github.com/cleanlab/cleanlab | API metadata: Apache-2.0, active |
| deepchecks | https://github.com/deepchecks/deepchecks | API metadata: NOASSERTION, last push 2025-12-28 |
| semgrep | https://github.com/semgrep/semgrep | API metadata: LGPL-2.1, active |
| joern | https://github.com/joernio/joern | API metadata: Apache-2.0, active |
| pyre-check | https://github.com/meta/pyre-check | API metadata: MIT, archived |
| onnxruntime | https://github.com/microsoft/onnxruntime | API metadata: MIT, active |
| treelite | https://github.com/dmlc/treelite | API metadata: Apache-2.0, active |
| model2vec | https://github.com/MinishLab/model2vec | API metadata: MIT, active |
| great_expectations | https://github.com/fivetran/great_expectations | API metadata: Apache-2.0, active |

Verification limits: abstracts and repository metadata only. Capability claims about these
tools are the claims their authors and READMEs make, not independent measurements. No
external code was executed, installed or copied.
