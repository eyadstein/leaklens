# Real-world expansion corpus (Mission 1)

A **candidate** corpus that grows LeakLens's real-world evidence beyond the
frozen 60-notebook `realworld/` corpus and the frozen Phase 6 holdout.

Two rules define this directory:

1. **It is not an evaluation corpus.** Nothing here scores the scanner, and no
   detector imports it. It holds candidates, provenance and a review queue.
2. **Only hand-reviewed labels may be scored.** A scanner prediction, a
   heuristic or a repository description is never promoted to ground truth.

The frozen corpora are untouched: `realworld/` (60 notebooks, 8 scored labels)
and `leaklens/ai/corpus_realworld.py` (48 cases, digest `7fde39ce`) keep their
bytes and their published results.

## Layout

| Path | Role |
|---|---|
| `leaklens/corpus_expansion.py` | Pure logic: ids, licences, code extraction, dedup, review states, queue, validation, statistics |
| `scripts/collect_expansion.py` | Acquisition: GitHub Code Search, curated repo trees, DistilKaggle byte ranges; resumable, cached, rate limited |
| `realworld_expansion/sources.json` | Every channel investigated, including rejected ones and why |
| `realworld_expansion/manifest.jsonl` | One JSON object per candidate: provenance, hashes, findings, review state |
| `realworld_expansion/review_queue.jsonl` | The prioritised review queue, exported from the manifest |
| `realworld_expansion/reviews.jsonl` | Append-only audit log of recorded human decisions |
| `realworld_expansion/stats.json` | Machine-readable counts used by the report |
| `realworld_expansion/collection_state.json` | Resume cursors (search pages, Kaggle blocks) |
| `tests/test_corpus_expansion.py` | Synthetic-fixture tests for all of the above; no network |

Notebook bytes are **not** committed. They live in a content-addressed cache
outside Git (`%LOCALAPPDATA%\leaklens-corpus-cache`, `$XDG_CACHE_HOME`, or
`~/.cache`; override with `LEAKLENS_CORPUS_CACHE`), addressed by
`objects/<sha[:2]>/<sha256>.<ext>`. A manifest row can therefore be verified by
re-fetching the recorded URL and comparing the recorded `sha256`.

## Provenance record

Each manifest row carries:

| Field | Meaning |
|---|---|
| `id` | `rwx_` + 12 hex, from `blake2b(source_id, repo, path, sha256)` — arrival-order independent |
| `source_id`, `source_kind` | Which channel produced it (`github_search`, `github_repos`, `kaggle_distilkaggle`) |
| `repo`, `repo_url` | Repository identity (for Kaggle: `kaggle/kernel-<id>`) |
| `commit_sha` | Immutable commit for the repository tree the file was read from, when the channel can pin one |
| `path`, `file_url` | Original relative path and the URL a human can open |
| `collected_at` | UTC collection timestamp |
| `license_spdx` | Repository-level SPDX id (empty string when none is declared) |
| `license_status` (computed) | `permissive` / `copyleft` / `unknown` from `license_status()` |
| `redistributable` (computed) | `license_status == "permissive"` — gates repo inclusion, **not** scanning |
| `file_type`, `bytes`, `sha256` | Exact identity of the bytes |
| `code_sha256` | Hash of the code cells only (no markdown, outputs or execution counts) |
| `sketch` | 64-value bottom-k shingle sketch used for near-duplicate clustering |
| `workflow` | Which ML workflow markers the code shows (`split`, `fit`, `cv`, `scale`, `impute`, `encode`, `resample`) |
| `ml_relevant`, `eligible_for_review` | Whether the file is ML workflow code a reviewer should see |
| `duplicate_of`, `code_duplicate_of`, `near_of`, `near_cluster` | Duplicate relationships (see below) |
| `overlaps_frozen` | `true` if the bytes are also in `realworld/notebooks` — recorded, never merged |
| `findings` | Static scanner output, kept as *candidate signal only* |
| `review` | Current review record; starts as `{"state": "not_reviewed"}` |

## Licensing

`license_status()` maps an SPDX id to one of three states:

| State | Examples | Redistributable |
|---|---|---|
| `permissive` | MIT, Apache-2.0, BSD-2/3-Clause, ISC, MPL-2.0, CC0-1.0, CC-BY-4.0 | yes |
| `copyleft` | GPL-2.0/3.0, AGPL-3.0, LGPL-2.1/3.0, EUPL-1.2 | no |
| `unknown` | none declared, `NOASSERTION`, anything unrecognised | no |

Repository-level licence is the only signal available at scale, and it can be
wrong for vendored or third-party files. Because bytes never enter Git, a wrong
licence cannot cause a redistribution problem; it only marks a sample
`redistributable: false`, which is the conservative default.

## Deduplication

A notebook mirrored across five datasets must count once. Three layers, applied
by `deduplicate()` in a fixed order and made idempotent by choosing the smallest
`id` in each group as canonical:

1. **Exact** — identical `sha256` → `duplicate_of`.
2. **Code-level** — identical `code_sha256`, i.e. the same code after
   re-execution, output changes or notebook metadata edits → `code_duplicate_of`
   (and `duplicate_of` if not already set).
3. **Near-duplicate** — bottom-k shingle sketches at Jaccard 0.85, clustered by
   the shared primitive `leaklens.near_duplicate.cluster()` (ADR-022, the same
   bucketing used by text dedup and image clustering) → `near_of` plus a shared
   `near_cluster` key.

Only rows without `duplicate_of` count as unique samples. `stats.json` reports
candidates, unique samples, duplicates, near-duplicates, repositories and unique
repositories separately, so a mirrored file cannot inflate the headline count.

## Review workflow

Five states, from `leaklens.corpus_expansion.REVIEW_STATES`:

`not_reviewed` → `potential` / `uncertain` → `confirmed_leakage` / `legitimate`

`SCORED_STATES` is only `("confirmed_leakage", "legitimate")`. `review()`
refuses a scored verdict without both `evidence` and `explanation`, refuses an
unknown state, and refuses a confidence outside `[0, 1]`. It records reviewer,
timestamp and optional leakage mechanism, and returns a new record instead of
mutating the candidate.

`queue()` builds the review queue: unreviewed, non-duplicate, review-eligible
rows, ordered by (scanner signal, workflow richness, source, repo, path) with at
most `per_repo_cap` samples per repository, so breadth beats volume. **The queue
never assigns a review state** — that is what keeps predictions out of the
ground truth.

## Label-leakage guard

Collection and review metadata must never become detector input. Two guards:

- `tests/test_corpus_expansion.py::test_detectors_never_import_collection_metadata`
  fails if any module under `leaklens/` outside `corpus_expansion.py` mentions
  `corpus_expansion` or `realworld_expansion`.
- `review()`'s state validation keeps `potential`/`uncertain` out of scored sets.

## State of the corpus

As of 2026-11-15 (`realworld_expansion/stats.json` is authoritative):

| Metric | Value |
|---|---|
| Candidates | 1,147 |
| Unique samples | 1,052 |
| GitHub repositories | 730 |
| Kaggle kernels | 119 |
| Licences | permissive 541, copyleft 38, unknown 568 |
| Review queue | 300 (276 distinct repositories) |
| Reviewed / labelled | 0 — nothing has been human-reviewed yet |
| Overlap with the frozen `realworld/` corpus | 8 samples, flagged and excluded from the queue |

## Limitations

- **Sampling, not census.** Code Search returns at most 1000 files per query and
  ranks by relevance; DistilKaggle is sampled by byte range. Neither is a random
  sample of "real-world ML notebooks", and no precision claim should be derived
  from this corpus without acknowledging that.
- **Repository licence ≠ file licence**, and a missing licence is not evidence of
  permission.
- **Kaggle reconstruction is code-only**: `code.csv` has no markdown or outputs,
  so cell-level evidence is approximate for those samples.
- **Teaching repositories skew legitimate.** Curated public repos are cleaned-up
  material; leakage is rarer there than in competition notebooks.
- **Forks and translations.** Kaggle kernels are heavily forked and teaching
  repos ship translations; both are collapsed by dedup rather than filtered
  perfectly at collection time.
- **Candidates only.** Until a human reviews a sample, it has no label. Counts in
  `stats.json` under `reviews` are the authoritative progress signal.

## Reproducing

```sh
# resumable: re-running continues where the last run stopped
python scripts/collect_expansion.py collect --channels github_repos
python scripts/collect_expansion.py collect --channels github_search --limit 500 --per-query 12
python scripts/collect_expansion.py collect --channels kaggle

# dedupe, rebuild the queue and stats from whatever is on disk
python scripts/collect_expansion.py finalize
python scripts/collect_expansion.py validate    # provenance completeness
python scripts/collect_expansion.py stats

# record one human decision (the only way a label enters the corpus)
python scripts/collect_expansion.py review --id rwx_0123456789ab \
  --state confirmed_leakage --evidence "notebook[cell 7]:2 scaler fitted before split" \
  --mechanism preprocessing_scaling --explanation "StandardScaler.fit_transform on the full frame precedes the split" \
  --confidence 0.9 --reviewer <name>
```

Requirements: `gh` authenticated (Code Search and repo metadata), and about
50 MB of cache per 300 notebooks. No notebook is ever executed; the scanner reads
text only.
