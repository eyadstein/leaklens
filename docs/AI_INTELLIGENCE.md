# Experimental AI intelligence

LeakLens ships an **optional, offline, experimental AI layer**. It adds hypotheses and a
root-cause write-up on top of the deterministic detectors. It is deliberately *not* part of
the trust score and it never changes a deterministic finding, severity or exit code.

- **Off by default.** Nothing runs unless you ask for it.
- **No GPU, network, paid service or credentials.** The core layer is CPU-only and local.
- **Deterministic findings stay authoritative.** AI output is labelled *experimental* and
  *unverified* in the CLI, JSON, HTML report and website.
- **Evidence-bound.** Every hypothesis points at an inspectable evidence span (a code line,
  a schema entry, a relationship, a statistic or a measured model probability).
- **No improvement claim.** Measured numbers are in
  [AI_BENCHMARK.md](AI_BENCHMARK.md); they are reported honestly, including false positives.
- **Known gaps and the ordered plan to close them** are in
  [AI_ROADMAP.md](AI_ROADMAP.md). That document is a proposal; nothing in it ships until its
  milestone lands.

## What it does

The layer has three parts:

1. **Deterministic semantic rules** (`leaklens/ai/semantics.py`) parse Python and notebook
   cells without executing them, and inspect dataset metadata (column names, dtypes, target,
   group, time and test columns). They flag patterns such as concatenating train and test
   before the split, target-derived features, backward fills, centered windows,
   negative shifts, runtime clock features, target-proxy names, future-information names,
   identifier names and train/test shared keys.
2. **A small learned CPU detector** (`leaklens/ai/learned.py`) trains an in-memory
   calibrated lightweight model (a logistic regression or shallow tree ensemble, chosen on
   validation) on 108 interpretable code indicators drawn from a 112-feature extractor.
   The extractor covers split ordering, frame identity, target-derived columns, duplicate
   handling, grouping, temporal order, functional and hand-written scaling, full-frame
   imputation, `get_dummies`, CV on the test partition, `GroupKFold` without `groups`,
   repeated test scoring, frequency encoding and fold-local fitting, plus the Phase 7
   hard-negative guards (fold-local fit detection, target-only encoder fits, sort position
   relative to a real time column, group boundaries that require an id in the modelling
   frame, and forward-label columns that are explicitly dropped). The development corpus is
   split by *scenario source and template family* into 219 training cases (15 sources) and
   40 validation cases (4 further sources); a 47-case locked test from 4 more sources is
   imported only by the benchmark, and a 12-case Phase 7 holdout from 5 further source
   families was frozen before the final fit and is scored once. Grouped cross-validation
   keeps a template family inside one fold. The shipped weights live in a versioned,
   digest-checked plain-JSON artifact (`leaklens/ai/model.json`); a request loads it when
   its digest, corpus, feature catalogue and library version match the running build, and
   otherwise fits in memory and records the reason. JSON cannot execute code on load.
   Hypotheses cite **local** signed log-odds contributions for the current unit, not global
   feature importance, and report a margin plus an uncertainty band. A runtime gate emits a
   hint only when at least one context-aware semantic risk indicator is active (or the
   deterministic scanner already corroborates the unit), so a high score driven only by
   script shape cannot raise an alarm.
3. **Evidence-bound reasoning** (`leaklens/ai/reasoning.py`) groups related deterministic
   findings and AI hypotheses into root causes, then writes a remediation plan with actions,
   a code hint and verification steps. The plan is deterministic unless you opt into LLM
   rephrasing, and every claim it makes is tied to an evidence id. Its categories cover the
   scanner and dataset kinds, including SQL lineage (`sql_*`), informative missingness
   (`missingness_*`) and the opt-in drift diagnostic (`distribution_shift`).

The orchestration (`leaklens/ai/analysis.py`) accepts code, a dataset or both. Dataset
analysis receives **metadata only** — no rows, no values — so row contents cannot leak into
AI output.

The deterministic counterpart is documented separately in
[INTELLIGENCE.md](INTELLIGENCE.md): `--intelligence` groups existing findings into an
evidence graph. The AI layer can cite that evidence but never changes it.

## Use it

### CLI

```sh
leaklens --scan notebooks/ --ai            # add AI hypotheses to a code scan
leaklens train.csv --target label --ai     # add AI hypotheses to a dataset audit
leaklens --scan src --ai-llm               # optional LLM rephrasing, with fallback
```

`--ai` prints a compact summary and includes an `ai` object in `--json` output and an
*AI intelligence (experimental)* section in `--html` reports. `--ai-llm` implies `--ai` and
asks the configured LLM provider (local Ollama by default) to rephrase the plan. If the
provider is unavailable, returns malformed output, invents a group id or claims unsupported
safety, LeakLens silently falls back to the deterministic plan and records the reason.

The exit code is unchanged: it still depends only on high-severity deterministic findings
and `--fail-under`.

### Python

```python
from leaklens import ai

report = ai.analyze_code(ai.collect_code_sources(["notebooks/"]))
payload = report.to_dict()  # schema-validated, JSON-safe

print(payload["deterministic_trust_score"])  # echo only; AI never recomputes it
print(payload["changes_deterministic_findings"])  # always False
```

Dataset audits accept a `DatasetInput(train=frame, target="label")`; only schema metadata is
used. `analyze(...)` combines findings, code and an optional dataset.

### Public live-scan API and website

`POST /audit` and `POST /scan` accept `include_ai=true` (JSON field or form field). The
response gains an `ai` object (and `ai_unavailable` when the backend could not produce it);
all existing keys keep their meaning. `GET /capabilities` advertises the layer as
`opt_in`, `network: "none"` and `changes_deterministic_findings: false`.

The live page has an **Add experimental AI hypotheses** checkbox. AI output is rendered in
its own panel with `textContent` only, never mixed into the deterministic findings list.

## Safety model

- **Untrusted input is untrusted.** Code comments, notebook markdown, column names and file
  contents are data. If they look like instructions ("ignore previous instructions…"), the
  layer raises an `info`-severity `untrusted_instruction` hypothesis and keeps analysing;
  it never follows the text.
- **No execution.** Scanned code and AI-generated snippets are never run.
- **Strict schemas.** `leaklens/ai/schemas.py` rejects unknown evidence sources, control
  characters, confidence above 0.95, severity hints above `medium`, duplicate evidence ids,
  unknown group/hypothesis/evidence references and any claim that deterministic findings
  changed.
- **Prompt injection containment.** LLM prompts contain sanitized group metadata only —
  never raw code, notebook text or dataset rows or column values. LLM replies must reference
  existing group ids and cannot introduce new evidence.
- **No silent escalation.** AI hypotheses cap at `medium`; deterministic `Finding` objects
  and `leakage.score()` remain the only source of severity and the trust score.
- **No score-only alarm.** A learned hint is emitted only when a context-aware semantic risk
  indicator is active or a deterministic finding corroborates the unit. Low-margin units are
  titled `[needs review]` and carry reduced confidence.
- **No global-importance citation.** Hypotheses cite signed local contributions for that
  unit, and each contribution names the feature value it came from.
- **No zero-risk claim.** A clean AI pass says so explicitly and still lists limitations.

## Measured behaviour

Run the benchmark yourself:

```sh
python -m leaklens.bench_ai            # JSON
python -m leaklens.bench_ai --write docs/AI_BENCHMARK.md
```

[AI_BENCHMARK.md](AI_BENCHMARK.md) lists the split protocol and provenance, candidate
architectures, grouped out-of-fold and validation metrics, calibration, the unchanged Phase 5
locked test (accuracy 0.979, precision 1.000, recall 0.960, F1 0.980, FPR 0.000 on 47 authored
cases), the frozen Phase 7 holdout (6/6 leaks caught, 1 false positive on 6 legitimate units),
the runtime hypothesis gate on development (precision 1.000, recall 0.963, FPR 0.000),
per-category and per-source results, external transfer to the existing scanner benchmark, a
diagnostic threshold sweep, the semantic regression set, a local-attribution example, global
feature importances, and every false positive and miss. The external scanner set was used for
Phase 6 error analysis, so its perfect score is a development result, not independent
evidence.

The Phase 7 holdout is authored synthetic data, not a real-world benchmark: it checks that
the hard-positive/hard-negative guards generalize to unseen templates. Its one false positive
is a fold-local transformer whose slice fit the extractor cannot tie to a split boundary; it
is reported rather than tuned away.

[AI_REALWORLD.md](AI_REALWORLD.md) is the independent real-world evaluation:
48 reviewed pipeline units from 12 public repositories that never appear in training,
validation or the locked test, plus a small smoke check on eight real public notebooks with
definitive hand labels. The measured holdout result is accuracy 0.833, precision 0.767,
recall 0.958, F1 0.852 and FPR 0.292; the report lists every false positive and miss,
repository-cluster intervals and calibration. It is not a production-accuracy claim.

The shipped threshold is chosen on validation only (0.53 for the current model) and
probabilities are calibrated from grouped out-of-fold scores, so behaviour does not drift at
import time. The eleven candidates are compared when the benchmark is built; a request loads
the class-balanced logistic regression frozen after Phase 7 validation from the checked-in
JSON artifact, so a cold opt-in API call avoids the fit and stays well inside the server's
time budget. If the artifact is absent, stale or tampered, the request fits in memory instead.
Metrics can vary slightly between scikit-learn releases because a refit is not bit-identical.
[AI_PERFORMANCE.md](AI_PERFORMANCE.md) records CPU latency and memory on the holdout and on a
representative project scan.

## Limitations

- The development corpus is small, authored and Python-only; the locked test has 47 cases,
  not real repositories.
- The real-world holdout has 48 cases from 12 repositories and is a convenience sample.
  44 of the 48 units are reviewed reconstructions of a cited idiom, not verbatim upstream
  files, and one reviewer labelled all of them. A post-freeze audit found 7 of 23 cited file
  paths had moved or did not exist at the pinned commit (13 reconstruction cases); the
  affected citations are listed in [REALWORLD_CORPUS.md](REALWORLD_CORPUS.md).
- On that holdout the model reaches recall 0.958 but precision 0.767 (7 false positives on
  24 legitimate units). Most false positives are safe CV, temporal-split, CatBoost Pool and
  EDA-only patterns that the current indicators cannot separate from leaks.
- Learned probabilities are calibrated on 40 validation cases and do **not** transfer:
  holdout Brier is 0.087 and ECE is 0.111. Treat the probability as a ranking score, not a
  risk estimate.
- The Phase 7 holdout is 12 authored synthetic units. It measures whether the hard-negative
  guards transfer to unseen templates; it is not production evidence.
- Local contributions explain the linear model's log-odds for one unit. They are not causal
  and do not prove that removing a cited line makes the pipeline safe.
- The single locked-test miss is a random split over repeated applicants where the only cue
  is an unseen identifier name (`applicant_id`), which the feature extractor does not treat
  as a leak on its own.
- Semantic rules are name and control-flow heuristics, not a proof of safety.
- The LLM path is optional and may be unavailable; the deterministic plan is always the
  fallback.
- AI output is experimental. Treat it as a prompt to inspect, not as a verdict.
