# Future vision: LeakLens as an ML trust auditor

**Status:** design document, not a commitment. Nothing here is implemented beyond what
[AI_ROADMAP.md](AI_ROADMAP.md) records as shipped. This document states what LeakLens should
become, what it should refuse to become, and how the two are told apart.

**Companion documents:** [FUTURE_ARCHITECTURE.md](FUTURE_ARCHITECTURE.md) (how),
[AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) (when),
[AI_ML_ECOSYSTEM_RESEARCH.md](AI_ML_ECOSYSTEM_RESEARCH.md) (what exists outside),
[WORKSPACE_DESIGN_VISION.md](WORKSPACE_DESIGN_VISION.md) (interface),
[ARTIFACT_SECURITY_STRATEGY.md](ARTIFACT_SECURITY_STRATEGY.md) (supply chain),
[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) (why).

---

## 1. Product identity

LeakLens audits whether an ML result is **trustworthy**, not whether a dataset is tidy.

The question it exists to answer is narrow and hard:

> Does this reported performance come from a model that was allowed to see the answer?

Today LeakLens answers it for tabular data and Python/notebook source: pre-split
preprocessing, target columns used as features, duplicate rows across splits, group and temporal
leakage, SQL lineage, missingness signals, drift diagnostics, cross-validation misuse, and an
optional AI layer that groups findings into explanations. Its unit of value is not a metric; it
is an **evidence-backed claim** with a stable fingerprint that survives line-number churn and can
be baselined, suppressed with a reason, exported to SARIF and diffed between runs.

The long-term identity builds on that one thing rather than diluting it:

**LeakLens is the offline-first, deterministic auditor of evaluation integrity across the ML
lifecycle — datasets, source, pipelines, models, and the claims made about them.**

"Evaluation integrity" is the product. Data quality, monitoring, lineage and governance are
supporting capabilities, admitted only when they change the answer to the question above.

### Why this identity, and not "an AI/ML platform"

The proposed direction "AI/ML Engineering Intelligence and Trust Platform" is only half right.
"Intelligence platform" invites feature parity with experiment trackers, data profilers and
observability suites — a race LeakLens cannot win and should not enter. "Trust" is the part that
holds, because trust is a property nobody's tracker owns: MLflow records what you ran, Evidently
records how distributions moved, Great Expectations records whether a column matched a rule.
None of them decides whether the *evaluation itself* was valid, and none of them will tell you
that a result is untrustworthy in a way you can attach to CI.

So the validated direction is narrower and stronger: **an integrity auditor with a growing set of
evidence-producing engines**, not a platform that ingests everything.

---

## 2. Target users

| User | What they need | What LeakLens gives them |
|---|---|---|
| ML engineer shipping a model | To know, before review, whether the 0.97 AUC is real | A CLI finding with evidence, severity and remediation, in CI |
| Data scientist reviewing a notebook or PR | A second opinion that is not another human reading 400 cells | Deterministic checks on split order, preprocessing, duplicates, target usage |
| Platform / MLOps engineer | A gate that runs in CI without a cluster, a server or a GPU | Offline, CPU-only, exit-code gate, baselines and suppressions |
| ML lead / reviewer | An audit trail for a decision | Fingerprints, evidence, graph export, HTML report, baseline comparison |
| Researcher / auditor of public work | A reproducible way to challenge a published number | Dataset and code audits with published methods and explicit limits |
| Security / governance reviewer | Evidence, not marketing | SARIF, provenance, digest-pinned audit inputs, explicit non-guarantees |

Deliberately **not** a target user: someone who wants a dashboard of every metric in production.
That is the observability market, and LeakLens only touches it where a monitoring fact becomes
evidence of an invalid evaluation (for example, train-serving skew that invalidates a claimed
lift).

---

## 3. Problems solved over the lifecycle

Ordered by how directly they answer the core question.

1. **Split and preprocessing integrity** (tabular, code, notebooks) — shipped.
2. **Entity, group, temporal and duplicate leakage** (tabular, SQL, cross-file) — shipped in part.
3. **Evaluation-methodology misuse** — CV on leaked data, test-set tuning, preprocessing fit on
   test — shipped in part; deep-learning training loops are not covered yet.
4. **Pipeline-level lineage** — which transformation produced the feature that reached the model,
   across notebooks, SQL and DAG files — partly shipped (SQL lineage, flow analysis).
5. **Text and retrieval integrity** — train/test and benchmark contamination, near-duplicate
   documents, paraphrase reuse, RAG chunk overlap and retrieval leakage — not shipped.
6. **Image and video integrity** — train/test contamination by subject, scene or capture session,
   near-duplicate frames, augmentation leakage, annotation/bbox validity — not shipped.
7. **Model and training provenance** — what data a checkpoint actually saw, whether the training
   configuration can be reproduced, whether the evaluation split was disjoint — not shipped.
8. **Artifact and supply-chain trust** — untrusted datasets, unsafe serialized models, audit
   manifest integrity — partly shipped (`upload_guard`, fingerprints) and designed in
   [ARTIFACT_SECURITY_STRATEGY.md](ARTIFACT_SECURITY_STRATEGY.md).
9. **Remediation** — a reviewable patch or split correction instead of a complaint — only
   textual recommendations today.

---

## 4. Capability map

Three layers. Only the middle layer is where new engines appear; the outer layers exist today.

```
INPUTS            ENGINES (deterministic, evidence-emitting)          OUTPUTS
dataset  ───────► tabular    code/notebook    sql                     findings + evidence
source   ───────► pipeline   text/LLM         vision   dl/artifacts   evidence graph
pipeline ───────► (each engine is optional, offline, CPU-first)       trust score (authoritative)
artifact ───────►                                                    report / SARIF / workspace
                       │ evidence graph, provenance, confidence       AI hypotheses (additive)
                       ▼                                              baselines, policy, diff
                  intelligence/  ──►  ai/  ──►  remediation proposals
```

Central (must exist once, shared): finding primitives, scoring, evidence records, provenance,
confidence basis, detector metadata, audit result assembly, policy.

Independent subsystems (each optional, gated by its own extra): tabular, code, SQL/pipeline,
text, vision, deep-learning, artifacts, integrations.

Optional by default: every learned or LLM component, every heavyweight dependency, every
integration with an external system.

---

## 5. What must stay deterministic, and where AI is genuinely useful

Deterministic forever:

- Detector execution given the same inputs, tool version and dependency versions.
- Findings, severities, fingerprints, `leakage.score()`, the exit code, baselines, suppressions.
- The evidence graph structure and its exports.
- The default output of every command with optional features off.

AI is genuinely useful for exactly four jobs, and no others:

1. **Grouping** deterministic findings into root-cause clusters that a human recognises.
2. **Ranking** hypotheses so a reviewer reads the likely causes first.
3. **Explaining** a finding in domain language, anchored to evidence ids.
4. **Planning** an investigation: which additional deterministic test would discriminate between
   two explanations.

AI is not allowed to: create findings, change severity, change the score, invent evidence,
assert a calibrated probability it has not measured, or convert a hypothesis into a conclusion.

LeakLens must remain fully useful with no LLM and no network. The AI layer is a reviewer's
assistant; the auditor is the deterministic engine. That asymmetry is the product's credibility.

---

## 6. Differentiation

| Adjacent category | What it optimises | What it does not do | LeakLens position |
|---|---|---|---|
| Data-quality suites (Great Expectations, Soda, Deequ, TFDV) | Schema, distributions, expectations | Decide if evaluation was valid; inspect source | Complement: LeakLens consumes their outputs as evidence, never replaces them |
| ML observability (Evidently, whylogs) | Production drift and reports | Pre-deployment evaluation integrity | Complement; drift is a diagnostic input |
| Model/experiment tracking (MLflow, DVC, Metaflow) | Record what ran | Audit whether what ran was valid | Complement; adapters read their metadata |
| SAST (Semgrep, CodeQL) | Security defects in code | ML-specific validity (splits, targets, CV) | Complement; LeakLens adds ML semantics SAST lacks |
| Dataset tooling (Cleanlab, FiftyOne, CleanVision) | Label and dataset quality | Cross-artifact evaluation integrity | Overlap on duplicates; difference in evidence and CI gate |
| LLM eval frameworks | Score model outputs | Detect contamination of the eval set itself | Overlap risk; LeakLens audits the eval set, which they rarely do |

The defensible combination — and the reason a copycat is hard — is:

1. **Deterministic evidence with stable identity** (fingerprints, evidence graph) rather than a
   score you must trust.
2. **Offline and CPU-first** while still offering optional learned and LLM assistance.
3. **One finding vocabulary and one scoring path** across every modality, so a vision detector, a
   text detector and a SQL detector report into the same review surface.
4. **Explicit epistemic labels** — basis, sufficiency, diagnostic status — instead of a single
   confidence number presented as probability.
5. **Auditability of the auditor**: published benchmarks, published misses, published shortfalls
   (for example the H2 corpus that does not exist and is documented as a shortfall rather than
   papered over).

---

## 7. Product boundaries (what LeakLens will not become)

Each refusal has a reason, not a slogan.

- **Not an experiment tracker.** Metadata stores duplicate MLflow and invite a schema nobody asked
  for. LeakLens reads trackers; it does not become one.
- **Not a training framework or a model server.** It inspects training and serving configuration;
  it never runs them.
- **Not a general SAST or secrets scanner.** It uses ML semantics. Security findings outside that
  scope belong in Semgrep/CodeQL, and LeakLens can surface theirs as context.
- **Not a distributed data profiler.** No Spark cluster, no warehouse connector fleet. Sampling
  and streaming summaries only.
- **Not a compliance certifier.** No "GDPR compliant" badge. It produces evidence a human or an
  auditor can use; it does not certify.
- **Not a leaderboard.** No public ranking of models or vendors.
- **Not a service you must run.** No mandatory server, database, queue or cloud dependency. The
  workspace is optional and in-process.
- **Not an LLM wrapper.** Remove the LLM and the audit still works.

---

## 8. Long-term ambitions

Ordered from near to ambitious. Each is a direction, not a promise.

1. **One result shape for every modality.** A vision audit and a tabular audit produce the same
   serialized payload, the same findings, the same evidence graph and the same report. A typed
   `AuditResult` is deliberately not introduced (ADR-U1); if a second consumer needs it, that is a
   separate decision. (Phase 13.)
2. **Pipeline truth.** A feature that reaches a model can be traced to the transformation that
   produced it, across code, SQL and DAG definitions, with unresolved lineage labelled as
   unresolved. (Phases 14–15.)
3. **Evaluation integrity for text and images**, not only tables: contamination, near-duplicates
   and retrieval leakage with the same evidence standards. (Phases 15–17.)
4. **Provenance you can verify offline.** An audit manifest containing input digests, detector
   versions, evidence and the score, verifiable later without LeakLens installed.
5. **Investigation, not just detection.** A deterministic investigator that plans tests, gathers
   evidence and states what is still missing; an LLM may narrate it but never conclude it.
   (Phase 18.)
6. **Remediation as a reviewable patch**, applied to a copy, never to the user's data or
   production code. (Phase 19.)
7. **An ecosystem of detector packs** — only if the activation triggers in
   [FUTURE_ARCHITECTURE.md](FUTURE_ARCHITECTURE.md) §3 are met. Until then, in-process only.

### The line that must not move

LeakLens may grow to inspect datasets, source, pipelines, experiments, checkpoints, images, text,
embeddings, SQL and deployments. It may add engines, adapters, agents and interfaces. It may
never grow a second scoring system, a second finding vocabulary, a hidden model call, a network
dependency in the default path, or a claim it cannot reproduce. Everything else in these
documents is negotiable.
