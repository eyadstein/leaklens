# realworld_expansion — candidate corpus

Machine-readable output of the Mission 1 collection pipeline. **Not an
evaluation corpus**: nothing here is scored, and no review state exists until a
human records one with `scripts/collect_expansion.py review`.

See [`docs/REALWORLD_EXPANSION.md`](../docs/REALWORLD_EXPANSION.md) for the
method, provenance schema, licence policy, dedup rules and limitations.

| File | What it is |
|---|---|
| `sources.json` | Every channel investigated, including rejected ones and why |
| `manifest.jsonl` | Candidates: provenance, hashes, workflow flags, scanner findings, review state |
| `review_queue.jsonl` | Prioritised, repository-diversified queue for human review |
| `reviews.jsonl` | Append-only log of recorded decisions (empty until a review happens) |
| `stats.json` | Counts: candidates, unique samples, repositories, licences, review progress |
| `collection_state.json` | Resume cursors |

Notebook bytes are **not** in this directory (or in Git). They are cached
content-addressed outside the repository; `sha256` in the manifest is what makes
a row verifiable.

Frozen corpora are untouched by this directory: `../realworld/` (60 notebooks)
and `../leaklens/ai/corpus_realworld.py` (48 cases).

## Commands

```sh
python scripts/collect_expansion.py collect      # resumable, all channels
python scripts/collect_expansion.py finalize     # dedupe + queue + stats
python scripts/collect_expansion.py validate      # provenance completeness
python scripts/collect_expansion.py stats
```

Tests: `pytest tests/test_corpus_expansion.py -q` (synthetic fixtures, no
network).
