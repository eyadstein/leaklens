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

## What it does

The layer has three parts:

1. **Deterministic semantic rules** (`leaklens/ai/semantics.py`) parse Python and notebook
   cells without executing them, and inspect dataset metadata (column names, dtypes, target,
   group, time and test columns). They flag patterns such as concatenating train and test
   before the split, target-derived features, backward fills, centered windows,
   negative shifts, runtime clock features, target-proxy names, future-information names,
   identifier names and train/test shared keys.
2. **A small learned CPU detector** (`leaklens/ai/learned.py`) trains an in-memory
   gradient-boosted tree on 61 interpretable code indicators from a small labelled corpus.
   The corpus is grouped by template family so the same family cannot appear in both
   training and validation folds. The model is never serialized to disk and never loaded
   from an untrusted artifact.
3. **Evidence-bound reasoning** (`leaklens/ai/reasoning.py`) groups related deterministic
   findings and AI hypotheses into root causes, then writes a remediation plan with actions,
   a code hint and verification steps. The plan is deterministic unless you opt into LLM
   rephrasing, and every claim it makes is tied to an evidence id.

The orchestration (`leaklens/ai/analysis.py`) accepts code, a dataset or both. Dataset
analysis receives **metadata only** — no rows, no values — so row contents cannot leak into
AI output.

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
- **No zero-risk claim.** A clean AI pass says so explicitly and still lists limitations.

## Measured behaviour

Run the benchmark yourself:

```sh
python -m leaklens.bench_ai            # JSON
python -m leaklens.bench_ai --write docs/AI_BENCHMARK.md
```

The current report is [AI_BENCHMARK.md](AI_BENCHMARK.md). It lists grouped cross-validation
precision/recall/F1/FPR, external transfer to the existing scanner benchmark, a threshold
sweep, the semantic regression set, global feature importances, and the actual false
positives and misses. Metrics can vary slightly between scikit-learn releases; the shipped
threshold is fixed at `0.65` so behaviour does not drift at import time.

## Limitations

- The corpus is small, synthetic and Python-only; production code may look different.
- Learned probabilities are uncalibrated and are not a guarantee against false positives.
- Semantic rules are name and control-flow heuristics, not a proof of safety.
- The LLM path is optional and may be unavailable; the deterministic plan is always the
  fallback.
- AI output is experimental. Treat it as a prompt to inspect, not as a verdict.
