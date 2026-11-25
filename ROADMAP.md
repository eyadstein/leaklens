# LeakLens roadmap

Goal: the most trusted open-source tool for finding data leakage in ML datasets and code. Each phase has an exit criterion so we know when it is really done. Tasks live in [Issues](https://github.com/eyadstein/leaklens/issues).

## Strategy: three stages, four programs

This roadmap is a sequence, not a feature list. It widens what LeakLens can prove
without weakening how it proves it. The product identity and its refusals are in
[FUTURE_VISION.md](docs/FUTURE_VISION.md); this section is the public commitment.

### Three stages

| Stage | What LeakLens is | Exit criterion | Status |
|---|---|---|---|
| **1. Specialized leakage detector** | An offline, deterministic audit of evaluation integrity for tabular data and Python/notebook source, plus optional engines for SQL, images, text, deep learning and pipeline metadata. | A reviewer can run one command in CI, get evidence-backed findings with stable fingerprints, and read a published real-world precision estimate together with a documented false-negative list. | **Current work.** |
| **2. ML data and pipeline auditing platform** | The same evidence standard across ingestion, preprocessing, splitting, training, evaluation and inference, for teams rather than a single reviewer. | Every new engine reuses the finding vocabulary, scoring path and evidence contract; one audit can cover more than one artifact without a second result shape. | Direction only. A capability enters only with Stage 1 evidence. |
| **3. Trusted ecosystem for ML auditing** | Published methods, a curated human-reviewed corpus, adapters to the tools teams already run, and detector packs contributed by others. | Outside contributors ship detectors that pass the same benchmark and review gates as the core, and the published corpus is reused outside the project. | Ambition, not a commitment. |

A stage is entered when its exit criterion is met, not when it is announced.

### Four permanent programs

| Program | Owns | Never |
|---|---|---|
| **Product intelligence** | Which leakage manifestations are real, which detectors are missing, and which gap is worth closing. | Guess demand, or copy a competitor's feature list. |
| **Engineering excellence** | Contracts, portability, supply chain, release discipline. | Ship an engine that breaks determinism or the default offline path. |
| **Ecosystem and adoption** | First-run experience, documentation, integrations, contribution paths, measurable reach. | Fabricate users or adoption numbers, or spam communities. |
| **Research and evaluation** | Corpora, labels, benchmarks and published limits. | Promote a prediction or a hypothesis to ground truth. |

Every item below names the program it serves, so a capability cannot arrive
without an owner and a validation story.

### Near, medium and long term

| Horizon | Work | Depends on |
|---|---|---|
| **Near (next 2–3 phases)** | **Product intelligence:** close the target-derived-statistics gap — a `groupby(...)["target"].transform(...)` over the full frame after the split is silent today (evidence in #63). **Research and evaluation:** finish the first hand-reviewed expansion batch and enforce the development/holdout split (#653). **Ecosystem and adoption:** publish the adoption baseline and a first-run tutorial (#217). | Nothing outside the current engine; corpus work depends on reviewer time. |
| **Medium** | **Engineering excellence:** cross-file and cross-notebook dataflow (#51); image-code understanding for augmentation-before-split and subject-level splits (#68); one audit spanning dataset + source + pipeline. | Near-term detector work; Stage 1 precision published first. |
| **Long** | **Engineering excellence:** remediation as a reviewable patch applied to a copy. **Ecosystem and adoption:** a reusable audited corpus (#513) and third-party detector packs. | Medium-term contracts; external contribution volume. |

### Measurable success criteria

Targets, not current results. Each is falsifiable from committed artifacts.

| Program | Criterion |
|---|---|
| Product intelligence | Every new detector cites a documented real-world manifestation and ships with at least one labeled case; the known-misses list is published and updated when a miss is found. |
| Engineering excellence | CI green on Python 3.10–3.12, self-scan clean, release gate validated against an **annotated** tag, and no optional dependency in the default path. |
| Research and evaluation | Every scored label has a rationale, a code reference and a named reviewer; no score is reported as real-world precision until a reviewer other than the label's author has checked the sample. |
| Ecosystem and adoption | `pip install` → first finding is documented in a five-minute quick start and re-run in CI; download and traffic trends are reported as trend signals, never as user counts. |

### What LeakLens will not become

An experiment tracker, a training framework or model server, a general SAST or
secrets scanner, a distributed profiler, a compliance certifier, a leaderboard,
a service you must run, or an LLM wrapper. Each refusal has a reason in
[FUTURE_VISION.md §7](docs/FUTURE_VISION.md#7-product-boundaries-what-leaklens-will-not-become).

The phases below are the execution record for Stage 1. They are kept because each
one has an exit criterion, not because the numbering is the strategy.

The longer-term architecture sequence uses a **separate numbering scheme** that continues
[docs/AI_ROADMAP.md](docs/AI_ROADMAP.md) (Phase 12 onward), not the phases below. It is planned in
[docs/AI_ML_EXPANSION_ROADMAP.md](docs/AI_ML_EXPANSION_ROADMAP.md), with the design in
[docs/FUTURE_ARCHITECTURE.md](docs/FUTURE_ARCHITECTURE.md). Phases 13–16 of that sequence are
shipped, Phase 17 is partly shipped (M1–M3), and Phase 17 M4–M7 plus Phases 18–19 remain
proposals, not commitments. The Phase 18 plan is
[docs/PHASE18_IMPLEMENTATION_PLAN.md](docs/PHASE18_IMPLEMENTATION_PLAN.md).

## Phase 0: Foundation (done)
Data detectors, code scanner, model arena, API, UI, Docker, CI, PyPI, pre-commit, HTML report, benchmark, study.
Exit: `pip install leaklens-ml` works and CI is green.

## Phase 1: Trust and accuracy
1. Grow the labeled benchmark from 30 to 200+ snippets, including real-world style code.
2. Hand-label 100 random flagged notebooks from the study and publish measured precision (and a recall estimate on planted leaks).
3. Inline suppression (`# leaklens: ignore`) and a `.leaklens.toml` config.
4. SARIF output so findings show up in GitHub code scanning.
5. Every finding links to a doc page explaining the leak and the fix.
Exit: published precision at or above 85% on hand-labeled real notebooks; false-positive reports get a fix or a documented reason.

## Phase 2: Detector depth (data)
1. Fuzzy near-duplicate detection across splits.
2. Time-series suite: future-looking lags/rolling windows, shuffled splits on temporal data.
3. Cross-validation leakage: preprocessing outside the CV loop, tuning on the test fold.
4. Target-encoding and group-statistic features computed on all rows.
5. Label-derived columns found by name patterns plus single-feature probes (exists) plus pairwise probes.
Exit: each detector has benchmark cases and a fix recommendation.

## Phase 3: Code analysis depth
1. Dataflow tracking: follow derived variables (X2 = X.copy(), X3 = f(X2)) instead of only call order.
2. pandas patterns: fillna(df.mean()), get_dummies, groupby-transform before the split.
3. Deep learning: normalization statistics on all data, `adapt()` on the full dataset, random_split after preprocessing, Hugging Face dataset splits.
4. Scripts split across files (follow imports within a repo).
Exit: scanner handles the top 10 leak patterns seen in the study with no regression on the benchmark.

## Phase 4: Integrations
1. GitHub Action that comments findings on pull requests.
2. SARIF upload example in the README.
3. Jupyter extension (`%load_ext leaklens`).
4. VS Code diagnostics through a small language server.
5. Docker image on GHCR for every release.
Exit: someone can get inline warnings in their editor in under two minutes.

## Phase 5: AI layer (optional, off by default)
1. LLM explanations and suggested patches: `--llm` with local Ollama or an API provider. Never sends code or data without the explicit flag; provider interface is pluggable and tested with a fake provider.
2. ML triage model that ranks findings by likelihood of being real, trained on the hand-labeled data from Phase 1 and evaluated with cross-validation against the rules alone. Ships only if it measurably raises precision.
3. Auto-fix PRs: propose a patch (Pipeline rewrite) the user reviews; never auto-merge.
Exit: AI features improve measured results and are clearly optional.

## Phase 6: Performance
1. Profile the scanner on 10,000 notebooks and publish the numbers.
2. Cheap wins first: parallel scanning, an on-disk cache keyed by file hash, skipping unchanged files.
3. Only if profiling shows a real bottleneck: a Rust core (PyO3 plus a Python parser) behind the same Python API, with the Python implementation kept as a reference for tests.
Exit: 10,000 notebooks scanned in a time we can state in the README.

## Phase 7: Credibility and research
1. Study v2: 10,000+ notebooks, hand-checked precision sample, more sources than GitHub code search.
2. Publish the benchmark as its own dataset (LeakBench) with a clear license.
3. Write up the method and results (blog post, then a short paper or preprint).
4. Compare with other tools on the same benchmark.
Exit: numbers others can reproduce.

## Phase 8: Community and growth
1. Docs site (mkdocs-material) with tutorials, including a "find the leak" notebook.
2. Keep 10+ labeled good-first-issues open; review PRs within 48 hours.
3. Monthly releases with a changelog.
4. Launch posts (Hacker News, Reddit, Kaggle forums, PyData), a short demo recording, conference lightning talks.
Exit: regular outside contributors and issues from real users.

## Phase 9: Sustainability
1. Governance and a deprecation policy; semantic versioning.
2. Security: SECURITY.md (done), Dependabot security alerts (done), signed releases through PyPI trusted publishing (done), SBOM on release.
3. Maintainer rotation: invite repeat contributors to triage.
Exit: the project keeps running if the founder steps away for a month.