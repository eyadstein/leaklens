# Phase 15 — Text and LLM intelligence

Source: `docs/AI_ML_EXPANSION_ROADMAP.md` §4 (lines 219–286). Dependency: Phase 13 only.
Status at planning time: `main` = `88fa23770b84d85b7f8198c650149f5799b55deb`, clean except the
protected `.phase5-bench.json` and `graphify-out/`.

## Implementation boundary

### 1. Roadmap requirements (verbatim scope)

- Exact and near-duplicate text overlap across declared splits.
- Benchmark-contamination checks against known public eval sets (hashes only, no copied corpora).
- Instruction/response leakage.
- RAG chunk overlap and retrieval-path leakage.
- Wiring-level static analysis of LangChain / LlamaIndex style pipelines.
- Intake for text corpora (files, JSONL, parquet); optional embedding integration, never an extra.
- Evidence records carrying matching spans/hashes, never raw document text.
- New: `leaklens/near_duplicate.py`, `analysis/text/{normalize,dedup,contamination,rag}.py`,
  `integrations/hf.py`, `leaklens/bench_text.py`, `docs/TEXT_BENCHMARK.md`, adversarial corpus.

Milestones M1–M6 as tabled in the roadmap. Acceptance: (1) M1–M5 deterministic without model,
network or GPU; (2) no full text in evidence; (3) prompt-injection text cannot change findings or
score; (4) near-duplicate recall and FPR published with misses listed.

### 2. Dependencies already satisfied

- Phase 13 `evidence.py` (bounded fields, aggregate-only), `availability.EngineUnavailable`,
  `manifest` (`DETECTORS`/`ENGINES`/`modalities`), `assemble.build`, `paths.resolve_within`,
  `leakage.Finding` + single `leakage.score()`, architecture test, detector-contract test.
- Phase 14 conventions: diagnostic engine, `scored=False`, benchmark regression guard, deferral
  documentation. No Phase 14 deferred work (M5/M6/M8, non-deterministic ops) is required here.

### 3. Missing gates

None blocking. M6 ships as a user-supplied-vector integration (no model, no extra, no network), so
the "pinned model id" acceptance is met by recording the user-supplied id and stating that results
are not reproducible across model versions. Model execution is explicitly out of scope.

### 4. Modules and public interfaces

| Module | Interface |
|---|---|
| `leaklens/near_duplicate.py` | `Cluster(key, members)`, `band_key(band)`, `bucket(signatures, rows=4)`, `candidate_pairs(buckets)`, `jaccard(a, b)`, `cluster(signatures, threshold=0.8, rows=4, similarity=jaccard)` |
| `analysis/text/normalize.py` | `Document(id, split, text, digest, chars)`, `normalize_text(str)`, `text_digest(str)`, `documents_from_texts(texts, split, *, min_chars=32, max_docs=10000, max_chars=2_000_000, boilerplate=frozenset())`, `load_documents(path, split, *, field=None, min_chars=32, max_docs=10000)` |
| `integrations/hf.py` | `SnapshotUnavailable(EngineUnavailable)`, `read_snapshot(root, *, field=None, split=None, max_rows=10000) -> list[str]` (parquet + Arrow, never `datasets`) |
| `analysis/text/dedup.py` | `exact_matches(train, test) -> set[str]`, `near_matches(train, test, *, shingle_size=5, num_hashes=64, threshold=0.8) -> set[str]`, `signatures(docs, shingle_size, num_hashes)`, `exact_findings(train, test, field)`, `near_findings(train, test, field, ...)` |
| `analysis/text/contamination.py` | `load_fingerprints(path) -> frozenset[str]`, `write_fingerprints(path, docs) -> int`, `contamination_findings(docs, fingerprints, field)` |
| `analysis/text/rag.py` | `RagHit(file, line, loader, index, path)`, `scan_rag(root, eval_paths, *, max_files=500) -> (list[Finding], list[RagHit])` |
| `analysis/text/embeddings.py` | `load_vectors(path) -> np.ndarray`, `embedding_findings(train, test, train_vectors, test_vectors, *, model, threshold=0.95)` |
| `leaklens/cli.py` | `text(argv)` subcommand, dispatched before the main parser like `images` |
| `leaklens/bench_text.py` | `evaluate()`, `markdown()`, `main(argv)` |

Kinds (all `diagnostic=True`, `scored=False`, added to `_DIAGNOSTIC_KINDS`):

| kind | surface | severity | modality | benchmark |
|---|---|---|---|---|
| `text_exact_duplicate` | split | high | text | `exact duplicate across splits` |
| `text_near_duplicate` | split | medium | text | `near-duplicate paraphrase` |
| `text_eval_contamination` | split | high | text | `eval fingerprint contamination` |
| `rag_index_leakage` | code | high | code | `shared RAG index` |
| `text_embedding_similarity` | split | medium | text | none + `no_bench_reason` (non-reproducible) |

One engine row: `leaklens.text` (surface `split`, `requires=None`, the five kinds).
Advice keys added to `FIXES`.

### 5. Tests and benchmark cases

New: `test_near_duplicate`, `test_text_normalize`, `test_text_dedup`, `test_text_contamination`,
`test_text_rag`, `test_text_embeddings`, `test_hf_snapshot`, `test_text_cli`, `test_bench_text`.
Updated: `test_architecture` (`CORE += near_duplicate`; `analysis` → engines layer;
`integrations` layer; core forbidden to import integrations), `_meta/test_detector_contract`
(`benchmark_case_names()` imports `bench_text`), `test_public_contract` (engine id set).

Covered contracts: availability/missing extra, determinism (same input → same findings/fingerprints),
stable identity, mandatory bounded evidence, malformed input degradation, input limits, root
containment (HF + RAG), CLI consistency, no score/exit change, no architecture cycle.

Benchmark cases (curated, synthetic): exact duplicate, normalised duplicate (case/whitespace/Unicode),
instruction and response fields, near-duplicate edits, hard paraphrase (reported as a miss if it
misses), adversarial almost-identical negative, boilerplate-only negative, prompt-injection duplicate
and prompt-injection negative, short/empty documents, contamination hit/miss, RAG shared/clean index.
Metrics TP/FP/FN/precision/recall/F1, misses listed; runtime and `tracemalloc` peak recorded but not
gated. Pass/fail = deterministic case expectations, recall floor, zero adversarial FPs.

### 6. Evidence and output contracts

- Every evidence record: `source` ∈ `{split, code}`, `kind` ∈ `{duplicate, overlap, pattern}`,
  bounded locator/detail; counts and `<16-hex>` digests only. No document text, no file contents.
- Findings aggregated per field (or per RAG file) with `stable_key` = field/file so fingerprints do
  not move when counts change.
- JSON payload uses `assemble.build(...)` sections plus text stats; no second scoring path.
- New CLI exits 0 by default because the kinds are diagnostic; `--fail-on-findings` is opt-in.
  Existing exit gates unchanged.

### 7. Performance and resources

- `MAX_CHARS = 2_000_000` per document, `DEFAULT_MAX_DOCS = 10_000`, `DEFAULT_MIN_CHARS = 32`.
- Bottom-k MinHash sketch: 64 hashes, 16 bands × 4 rows, threshold 0.8. Each shingle hashed once.
- RAG: `MAX_FILES = 500`, AST only, no imports/execution.
- Embeddings: `.npy` with `allow_pickle=False`, max 20 000 rows, max 4096 dims, float32.
- Benchmark publishes seconds per 10k documents and peak memory; runner-dependent, not gated.

### 8. Security and trust boundaries

- JSONL/JSON parsed with `json`, encoding errors replaced, malformed lines counted and skipped.
- HF snapshot and RAG roots resolved per file with `paths.resolve_within`; refusals skip, never read.
- `np.load(allow_pickle=False)`; no pickle, no model load, no network.
- RAG findings expose relative paths, line numbers, call names and literal loader paths only.
- Prompt-injection text is inert data; tests prove findings and trust score are unaffected.
- No new CodeQL suppression; any new alert is reported undismissed with a source-to-sink note.

### 9. Explicitly excluded

- No LLM judging, leaderboard scoring, corpus redistribution or model serving (roadmap non-goals).
- No API/workspace surface for text (not required by §4; existing surfaces unchanged).
- No embedding model execution or new extra; no `datasets` import.
- No refactor of `images.image_overlap`; the shared primitive is consumed by text now and by
  `analysis/vision` in Phase 17 (ADR-022 status says exactly that).
- No scoring graduation; no trust-score/exit change; no version bump.
- No RAG dataflow proof: static containment is documented as a heuristic, not proof.

### 10. Milestone sequence (TDD: RED → GREEN each)

1. **M1 intake** — `test_text_normalize` + `test_hf_snapshot`; then `normalize.py`, `hf.py`.
2. **M2 exact** — `test_text_dedup` exact cases; then `dedup.exact_*`, manifest kind, advice.
3. **M3 near** — `test_near_duplicate`; then primitive; then `near_*` + field aggregation.
4. **M4 contamination** — `test_text_contamination`; then `contamination.py`.
5. **M5 RAG** — `test_text_rag`; then `rag.py`.
6. **M6 embeddings** — `test_text_embeddings`; then `embeddings.py`.
7. **CLI + assembly** — `test_text_cli`; wire `cli.text`, engine row, all manifest entries.
8. **Benchmark** — `bench_text.py`, `test_bench_text`, generate `docs/TEXT_BENCHMARK.md`.
9. **Meta/docs** — architecture test layers, detector contract, public contract, `TEXT.md`,
   `PHASE15_ACCEPTANCE.md`, README, AGENTS.md, CI, regenerate `DETECTORS.md` + site fixtures.
10. **Verification** — full suite, ruff, diff check, self-scan, manifest check, site check, all
    benchmarks; Graphify after-snapshot; Ponytail full-diff review; commit + fast-forward push.

## Ponytail challenge (before implementation)

| # | Challenge | Disposition |
|---|---|---|
| 1 | `--write-fingerprints` is not named in the roadmap. | Accepted, kept: M4 needs user-supplied fingerprints and a 5-line generator is the only way to produce them reproducibly. Documented as the M4 companion. |
| 2 | A separate `analysis/text/audit.py` wrapper would have one caller. | Rejected: dropped. The CLI orchestrates like `images`; tests call the module functions. |
| 3 | `--eval-path` duplicates information already in the TEST positional. | Rejected: inference across cwd/root is ambiguous; explicit eval path is the honest input for static analysis. |
| 4 | RAG static analysis is a heuristic, not dataflow. | Accepted: documented as wiring recognition + literal-path containment; no claim of proof; chunk overlap handled by the text audit on chunk files. |
| 5 | `.npy` row alignment for M6 is fragile. | Accepted with guard: shapes and row counts are validated against the kept documents; error names the mismatch. Model identity is mandatory. |
| 6 | Evidence schema has no numeric count/ratio fields. | Accepted: counts/ratios live in `description`/`observed`/`expected`; no schema change. |
| 7 | Long docs make the sketch slow. | Accepted: one hash per shingle (bottom-k), not 64 permutations; documented limits. |
| 8 | Shipping all five kinds as high/medium but diagnostic can surprise users. | Accepted: every kind is `diagnostic=True`, exit stays 0, `--fail-on-findings` is opt-in; tests pin it. |
| 9 | Refactoring `images.image_overlap` onto the primitive would prove reuse. | Rejected for Phase 15: it changes a shipped, scored detector's output shape for no milestone acceptance. ADR-022 says the primitive is consumed by `analysis/vision` (Phase 17). Documented. |
| 10 | Embeddings could be an extra. | Rejected: roadmap says optional integration, never an extra. User supplies vectors; LeakLens never loads a model. |

## Ponytail full-diff review (after implementation)

| # | Finding | Disposition |
|---|---|---|
| 1 | `cli.text` recomputed `exact_matches`/`near_matches` after `exact_findings`/`near_findings` had already computed them; on the 10k-document path that doubled the most expensive step for two JSON counters. | Accepted: duplicate computation and the two counters removed. Counts remain in each finding's summary evidence, and zero matches mean no finding. |
| 2 | `dedup.near_findings` could dereference an empty `best_id`: at `--threshold 0` a document can join a cluster through another test document and share no shingle with any train document. | Accepted: guard plus `test_zero_threshold_clusters_never_dereference_a_missing_match`. |
| 3 | `tests/test_hf_snapshot.py` imported `pyarrow` at module scope, unlike the repository's `pytest.importorskip` convention. | Accepted: the module now skips when the optional extra is absent. |
| 4 | `candidate_pairs(max_bucket=64)` skips bands with more than 64 members, so a very common shingle band can hide a pair. | Rejected as a defect: it is the documented boilerplate mitigation, exact duplicates stay covered by M2, and the limit is published in `TEXT.md` and the benchmark parameters. |
| 5 | Hash-only contamination can be evaded by rewording an eval item. | Rejected as a defect: "hashes only, no copied corpora" is the roadmap rule; the limitation is documented and near-duplicate matching is the complement. |
| 6 | RAG analysis is wiring recognition, not a dataflow proof. | Rejected as a defect: it is stated in `TEXT.md`, the acceptance record, and the finding's evidence; chunk overlap is covered by running the audit on chunk files. |
| 7 | `images.image_overlap` was not refactored onto `near_duplicate`. | Rejected: ADR-022 assigns the vision consumer to Phase 17; changing a scored detector here alters shipped output for no milestone. |
| 8 | A new `analysis/` package for five modules. | Rejected: the roadmap names `analysis/text/` and `integrations/hf.py`; the architecture test now encodes the layers. |
| 9 | `--write-fingerprints` is an extra surface not named in the roadmap. | Kept: M4 requires user-supplied fingerprints, and a five-line generator is the reproducible producer for them. |
| 10 | Evidence has no numeric count/ratio schema. | Accepted: counts and ratios stay inside bounded `description`/`observed`/`expected` strings; no schema change. |

