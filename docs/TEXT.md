# Text and LLM intelligence

Phase 15 adds a deterministic text engine that answers one question: **was this
evaluation set contaminated?** It compares declared train/test corpora for exact
and near-duplicate overlap, checks a training corpus against user-supplied eval
fingerprints (hashes only), and statically inspects LangChain/LlamaIndex-style
repositories for an index built over the evaluation data. Optional embedding
similarity is available from user-supplied vectors; LeakLens never loads a model.

All five kinds are diagnostics (`diagnostic: true`, `scored: false`). They never
change `leakage.score()`, an existing finding's severity or the exit code of an
existing command. The new `leaklens text` subcommand exits `0` by default and
`1` only with `--fail-on-findings`.

## Quick start

```sh
# JSONL/JSON/TXT/parquet corpora; field defaults to "text" for object rows
leaklens text train.jsonl eval.jsonl
leaklens text train.parquet eval.parquet --field instruction --json

# Write the eval set's digests once, then check training data against them
leaklens text train.jsonl eval.jsonl --write-fingerprints eval.fp
leaklens text train.jsonl eval.jsonl --eval-fingerprints eval.fp

# Static RAG wiring check (repo-relative eval path)
leaklens text train.jsonl eval.jsonl --rag . --eval-path data/eval

# Optional, never default: user-supplied vectors from an unpinned model
leaklens text train.jsonl eval.jsonl \
  --embeddings train.npy eval.npy --embedding-model all-MiniLM-L6-v2
```

| Kind | Severity | Signal |
|---|---|---|
| `text_exact_duplicate` | high | a test document's normalised text also appears in train |
| `text_near_duplicate` | medium | a test document's shingle sketch matches a train document above the threshold |
| `text_eval_contamination` | high | a train digest matches a user-supplied eval fingerprint |
| `rag_index_leakage` | high | a module builds an index and loads a literal root that contains the declared eval path |
| `text_embedding_similarity` | medium | user-supplied eval vectors are cosine-similar to train vectors |

## Intake and normalisation (M1)

Supported inputs:

| Suffix | Rows |
|---|---|
| `.jsonl`, `.ndjson` | one document per non-empty line; a string row is the document, an object row uses `--field` (default `text`) |
| `.json` | a list of strings/objects, or one object |
| `.txt`, `.md`, `.text` | the whole file is one document |
| `.parquet`, `.feather`, `.arrow` | local HuggingFace-style snapshot files, read with `pyarrow` and **without importing `datasets`** |

`pyarrow` comes from the existing `parquet` extra; when it is missing the audit
reports an actionable skip instead of a traceback. CSV is not a text format here:
convert it to JSONL first.

Normalisation is NFKC, Unicode case folding and whitespace collapse. The document
digest is the first 16 hex characters of `SHA-256(normalised text)`. This makes
case, full-width characters and whitespace irrelevant to exact matching while
keeping evidence short and comparable.

Budgets: 32 normalised characters minimum by default (`--min-chars`), 10 000
documents per split (`--max-docs`), 2 000 000 characters per document and a
256 MiB file ceiling. Malformed JSONL rows are counted as skipped; a file with no
readable rows is an error. Document ids keep the original row number
(`train:4`), so a skipped row never shifts a later identity.

## Exact duplicates (M2)

Exact overlap is a digest intersection, so it is deterministic and independent of
row order. A finding is aggregated per field with the matched count and up to
eight example digests. Instruction and response leakage is the same check run
with `--field instruction` or `--field response`.

## Near-duplicates (M3)

Near-duplicate matching uses:

- word 5-shingles (`--shingle-size`);
- a bottom-64 MinHash sketch (each shingle hashed once with BLAKE2b);
- candidate generation from shared sketch values, using the shared
  `leaklens/near_duplicate.py` primitive;
- Jaccard verification at `--threshold` (default `0.8`);
- stable cluster keys from sorted members.

The shared primitive lives at package root so text and the Phase 17 vision work
use one implementation (ADR-022). `images.image_overlap` keeps its shipped
dHash/nearest-pair behaviour until Phase 17 extends it.

Bands with more than 64 members are skipped as boilerplate-like: they produce
quadratic candidate pairs and mostly noise, while exact duplicates are already
reported by M2. A hard paraphrase can legitimately miss at `0.8`; the benchmark
publishes that miss instead of hiding it.

## Boilerplate and short documents

Two documented mitigations:

1. **Minimum length rule**: documents below `--min-chars` are skipped. This is
   the primary defence against headers and licence fragments.
2. **Line boilerplate file**: `--boilerplate FILE` removes whole lines that match
   a listed boilerplate line after normalisation, before hashing.

Naming no `--boilerplate` file is not a safety claim; it means the caller has not
declared boilerplate.

## Eval contamination (M4)

`--write-fingerprints FILE` writes sorted unique digests of the evaluation split.
`--eval-fingerprints FILE` accepts those digests (one hex digest per line, `#`
comments allowed) and matches training documents by digest only. No evaluation
corpus ships with LeakLens, no eval text is stored in evidence, and a miss is a
miss: LeakLens does not download a public benchmark.

The digest is normalised-text based, so re-punctuating an eval item can evade the
hash-only check. That limitation is deliberate: near-duplicate matching covers
some of it, and LeakLens never receives the eval corpus itself.

## RAG wiring analysis (M5)

`--rag ROOT --eval-path PATH` parses Python files under `ROOT` with `ast` (never
imports or executes them) and reports a file when both are true:

1. it calls a known document loader (`DirectoryLoader`, `TextLoader`,
   `SimpleDirectoryReader`, ...);
2. it constructs a retrieval index (`from_documents`, `add_documents`,
   `FAISS`, `VectorStoreIndex`, ...);
3. a literal loader path contains the declared eval path after normalisation.

This is **wiring recognition plus literal-path containment, not a dataflow
proof**. Computed paths, environment expansion and dynamically built loaders are
not followed; `..` components never widen a match. The finding names the file,
line, loader call, index call and literal path, and carries no file contents.

Chunk-level overlap is handled by running the deterministic text audit on the
chunk files themselves:

```sh
leaklens text chunks/train.jsonl chunks/eval.jsonl
```

## Embedding similarity (M6)

`--embeddings TRAIN.npy EVAL.npy --embedding-model ID` reads two numeric `.npy`
matrices with `allow_pickle=False`. Rows must align with the documents kept after
`--min-chars`, in input order; the command fails with the mismatch instead of
guessing. Cosine similarity is computed in bounded chunks (20 000 × 4096 limit,
`--embedding-threshold`, default `0.95`).

LeakLens never loads or runs an embedding model. The model identity in the report
is the caller's declaration, and **results are not reproducible across model
versions**; the finding says so. This milestone is opt-in and has no deterministic
benchmark case for exactly that reason.

## Evidence and privacy

Evidence records carry aggregate counts, digests, relative file paths, line
numbers and call names. They never carry full document text, file contents,
credentials or exception text. Findings are aggregated per field (or per RAG
file) and use a stable key, so a changing match count does not move a fingerprint.

Prompt-injection text is inert: documents are never executed, and a test proves
that an injection document cannot change findings or the trust score.

## Scoring status

Every Phase 15 kind is `scored=False` and `diagnostic=True` (ADR-006). No labelled
real-world text corpus exists in this repository, so graduation is not claimed.
The synthetic benchmark is a regression guard, not accuracy evidence.

Measured controls: [TEXT_BENCHMARK.md](TEXT_BENCHMARK.md). The Phase 15 acceptance
record is [PHASE15_ACCEPTANCE.md](PHASE15_ACCEPTANCE.md).

## Limits

- Hash-only contamination can be evaded by edits; use near-duplicate matching too.
- Bottom-k + shared-value candidates is a sketch, not an exact Jaccard oracle.
- RAG analysis does not resolve variables, f-strings or computed roots.
- No API or workspace surface for text in this phase; the CLI is the surface.
- No embedding model execution, no network access and no GPU requirement in the
  deterministic path.
