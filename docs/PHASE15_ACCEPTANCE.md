# Phase 15 acceptance record (2026-11-12)

Scope is [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) §4. This record
states what shipped, what is measured and what remains limited. Phase 13 is
accepted in [PHASE13_ACCEPTANCE.md](PHASE13_ACCEPTANCE.md); Phase 14 is
[PHASE14_ACCEPTANCE.md](PHASE14_ACCEPTANCE.md).

## Milestones

| Id | Milestone | Status | Evidence |
|---|---|---|---|
| M1 | Text ingestion + normalisation with limits; local HF snapshot reader | **Shipped** | `analysis/text/normalize.py` (NFKC, case fold, whitespace collapse, 16-hex digest, 32-char default minimum, 10k/2M/256 MiB budgets); `integrations/hf.py` reads parquet and Arrow without importing `datasets`; `tests/test_text_normalize.py`, `tests/test_hf_snapshot.py` |
| M2 | Exact duplicate overlap across declared splits | **Shipped** | `analysis/text/dedup.exact_matches` / `exact_findings`; aggregate counts and digests only; `tests/test_text_dedup.py`; benchmark cases |
| M3 | `near_duplicate.py` primitive + shingle/MinHash overlap | **Shipped** | `leaklens/near_duplicate.py` (`bucket → threshold → stable cluster key`); bottom-64 sketch, 5-shingles, 0.8 threshold; `tests/test_near_duplicate.py`, `tests/test_text_dedup.py`; published recall/FPR in [TEXT_BENCHMARK.md](TEXT_BENCHMARK.md) |
| M4 | Benchmark contamination against user-supplied eval fingerprints | **Shipped** | `analysis/text/contamination.py` (`load_fingerprints`, `write_fingerprints`, hash-only matching); no eval corpus in the package; `tests/test_text_contamination.py` |
| M5 | RAG wiring analysis (loader → splitter → index → retriever) and chunk overlap | **Shipped** | `analysis/text/rag.py` (AST only, literal containment, `..` rejected); chunk overlap via `leaklens text` on chunk files; `tests/test_text_rag.py`; shared-index benchmark case |
| M6 | Optional embedding similarity (opt-in, pinned model id, labelled non-reproducible) | **Shipped, opt-in** | `analysis/text/embeddings.py` reads user-supplied `.npy` vectors (`allow_pickle=False`), requires the model id, states non-reproducibility; LeakLens never loads a model; `tests/test_text_embeddings.py` |

## Acceptance criteria

| # | Criterion | Result | Evidence |
| ---:|---|---|---|
| 1 | Deterministic path (M1–M5) needs no model, no network and no GPU | Pass | Only stdlib, `numpy` (base dependency), `pyarrow` for optional snapshot files, `availability`, `paths`, `evidence`, `leakage`; no HTTP client, no model import. M6 consumes caller-supplied arrays. |
| 2 | Evidence never contains full document text; spans and hashes only | Pass | Tests assert the source body is absent while counts and 16-hex digests are present (`test_text_dedup.py`, `test_text_contamination.py`, `test_text_rag.py`). |
| 3 | Prompt-injection text cannot change findings or score | Pass | Text is inert data; `test_prompt_injection_text_is_inert_data_not_an_instruction` and two benchmark cases; the finding remains diagnostic and `score() == 100`. |
| 4 | Near-duplicate recall and FPR published, misses listed | Pass | [TEXT_BENCHMARK.md](TEXT_BENCHMARK.md): precision 1.00, recall 0.89, F1 0.94, FPR 0.00 on the curated synthetic set, with the hard-paraphrase miss listed. Not a real-world accuracy claim. |

## Integration contract

- **Engine id**: `leaklens.text` added to `manifest.ENGINES`; the frozen-id test was
  updated additively (no existing id changed).
- **Kinds**: five new kinds, all `requires=None`, `scored=False`, `diagnostic=True`;
  split text kinds declare `modalities=("text",)`.
- **Assembly**: the CLI builds one ordinary finding list and serializes it with
  `assemble.build`; there is no second scoring path and no new payload section
  outside the subcommand's report.
- **Exit behaviour**: the new subcommand exits `0` by default; `--fail-on-findings`
  is the only opt-in failure mode. Existing commands are unchanged.
- **New modules**: `leaklens/near_duplicate.py` (package root primitive),
  `leaklens/analysis/text/` (four roadmap modules plus `embeddings.py` for M6),
  `leaklens/integrations/hf.py`, `leaklens/bench_text.py`.
- **Dependencies**: none added. `pyarrow` remains the existing `parquet` extra;
  embeddings are a user-supplied integration, never an extra.
- **No version bump** and no release/tag in this phase.

## Measurement

`python -m leaklens.bench_text` on the generating machine (10 000 synthetic
documents, single process): 4.58 s end-to-end, 83.6 MiB peak traced memory. Both
numbers are runner-dependent and are not a pass/fail gate. The corpus is generated
by LeakLens, contains no third-party data and is not a real-world evaluation.
Parameters and misses are in [TEXT_BENCHMARK.md](TEXT_BENCHMARK.md).

## Limitations and deferred work

- **RAG analysis is heuristic.** It recognises loader/index calls and compares
  literal paths; it does not perform dataflow analysis. Documented in
  [TEXT.md](TEXT.md#rag-wiring-analysis-m5).
- **Hash-only contamination is evadable** by edits to evaluation items. That is a
  property of the roadmap's "hashes only, no copied corpora" rule, not a bug.
- **The shared primitive is not yet used by `images.image_overlap`.** ADR-022 gives
  `analysis/vision` (Phase 17) as the consumer; changing the shipped image detector
  in this phase would alter a scored engine's output without a milestone.
- **No API/workspace text surface.** The roadmap does not require one; the CLI and
  shared assembly path cover the milestone.
- **M6 model execution is out of scope by design.** "Pinned model id" is recorded
  from the caller; no model runs in LeakLens, and results are labelled
  non-reproducible across model versions.
- **Scoring graduation is not claimed.** No labelled real-world text corpus was
  evaluated, so all kinds stay `scored=False` (ADR-006).

Phase 15 does not require any deferred Phase 14 milestone. Phase 17 (computer
vision) and Phase 18 (investigation/governance) are the next roadmap dependants.
