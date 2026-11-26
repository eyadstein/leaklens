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
| `realworld_expansion/reviews.jsonl` | Append-only audit log of recorded human decisions (created by the first recorded review; absent until then) |
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

### Record contract

A candidate is only usable as evidence when each of these can be answered from
committed files. This is the contract every future case must satisfy; "gap" means
the field is not stored yet and the case cannot be promoted until it is.

| Contract field | Where it lives | Status |
|---|---|---|
| Stable case id | `id` (`rwx_` + 12 hex, derived from provenance and content, never from list order) | present |
| Source repository or publication | `repo`, `repo_url`, `source_id` | present |
| Original URL and retrieval date | `file_url`, `collected_at` | present |
| Commit or revision | `commit_sha` (required for GitHub rows) | present |
| Licence and permitted usage | `license_spdx`, `license_status`, `redistributable` | present |
| Leakage taxonomy and manifestation | decision `mechanism` + dossier `mechanism`; taxonomy is the detector-kind vocabulary in `docs/DETECTORS.md` | partial: no free-text "manifestation" field |
| Human review status and rationale | `reviews.jsonl` decision: `state`, `rationale`, `evidence_refs`, `reviewer` | present |
| Confidence and ambiguity | decision `confidence` (0–1, optional) plus `potential` / `uncertain` for ambiguity | present |
| Expected detector behaviour | not stored; asserted in the regression test that cites the case id | gap |
| Regression test linkage | the test docstring names the candidate id and expected kind | convention, not a field |
| Privacy and redistribution restrictions | `license_status`, excerpt guard (permissive only), notebook bytes never committed | present |

When a case is promoted to a regression test, the test asserts what the detector
must do (fire kind `X`, or stay silent) and cites the candidate id, so the
expected behaviour is versioned with the code rather than restated in prose.

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

### Source intake and licensing findings

External sources are cited for taxonomy and evidence, not bulk-copied. Status was
verified on 2026-11-25; `sources.json` remains the machine-readable registry and
records rejected channels with the reason.

| Source | Verified status | Reuse decision |
|---|---|---|
| Sasse et al., "Overview of leakage scenarios in supervised machine learning", *Journal of Big Data* 12:135 (2025-05-29), DOI `10.1186/s40537-025-01193-8` | Peer-reviewed; CC BY 4.0 | Cite the taxonomy. Do not copy its examples into the corpus. |
| Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* (2023), DOI `10.1016/j.patter.2023.100804` | Peer-reviewed; CC BY 4.0; companion collection reports 41 review papers across 30 fields affecting 648 papers and publishes an 8-type taxonomy and model info sheets | Use the taxonomy and info-sheet fields as a cross-check. Do not re-host the collection. |
| "Understanding How Data Leakage Manifests in Real-World Machine Learning Notebooks", SSRN preprint `10.2139/ssrn.6388954` (2026) | **Preprint**, not peer-reviewed; abstract reports "over 600" notebook/model pairs — the exact count and any dataset availability were not verified | Cite only. Ingest nothing; do not repeat a specific corpus size as fact. |
| DistilKaggle notebook corpus (Zenodo 10317389) | Record metadata CC BY 4.0; upstream Kaggle notebook terms may still apply | Code-only candidates with references; no notebook redistribution. |
| Hugging Face `jupyter-agent/kaggle-notebooks-edu-v0`, `bigcomputer/meta-kaggle-notebook`, `lvwerra/jupyter-notebooks-edu-*` | No licence tag on the dataset card | **Rejected.** No permission to reuse. |
| Kaggle public API and public notebooks | Terms restrict scraping and redistribution; per-kernel licence often unspecified | **Rejected** as an acquisition channel. |
| UCI and documented research datasets | Licence is per dataset and often requires attribution | Not a notebook source; a dataset may be used only with its own recorded licence. |

The rule that follows: a source enters the pipeline only when reuse permission is
recorded, and a corpus row may carry a reference even when it may not carry bytes.

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

`SCORED_STATES` is only `("confirmed_leakage", "legitimate")`. `validate_decision()`
refuses a scored verdict without both a rationale and at least one code
reference, refuses an unknown state, a placeholder reviewer, a `source` other
than `human`, and a confidence outside `[0, 1]`.

A decision is **appended** to `realworld_expansion/reviews.jsonl` and nowhere
else. `apply_decisions()` merges the log into the records at read time, so
`queue()` and `scored_records()` see decisions without `manifest.jsonl` being
rewritten. The manifest is history; the decision log is the audit trail, and
`review()` (a pure function) still returns a new record rather than mutating a
candidate.

Independence is computed from the log, not asserted: `review_status()` reports
`single_reviewed`, `double_reviewed` (two distinct reviewer ids that agree and a
blind pass), `double_reviewed_unblinded` (two ids, no blind pass),
`disputed`, `adjudicated` or `uncertain`. New decisions record their `role`
(`first_pass`/`adjudication`), `materials` (`full`/`blind`) and the
`corpus_revision` (first 16 hex of the manifest digest) they were decided
against. The 26 decisions recorded before those fields existed carry none of
them, so they cannot be tied to a precise corpus revision; the append-only log
keeps them as historical evidence instead of rewriting them. A disagreement
stays visible after an adjudication. As of this writing every recorded decision
has one reviewer, so no candidate is independently validated and the corpus
cannot support a precision claim.
`show --blind` withholds the scanner signal, the workflow summary, the AI
mechanism, `why_it_may_apply`, the counter-evidence list, the reviewer questions,
the per-reference notes, review history, the recorded decision and the derived
audit status. The provenance, licence, split side and cited line numbers stay,
because a reviewer needs them to read the code; the split side is label-blind.
`blind-pack` rebuilds the whole batch with those same fields removed
(`review_batch_01.blind.jsonl` and `REVIEW_BATCH_01.blind.md`), so a second
reviewer can start from material that never contained an expected answer. It is
still not a review: no case becomes `double_reviewed` without a distinct human
`--blind` decision.

The first batch is `realworld_expansion/review_batch_01.jsonl` plus
`review_batch_01.analysis.json` and the human-readable
`REVIEW_BATCH_01.md`. Building it (`batch`) is deterministic: dossiers are
regenerated from the authored analysis, never hand-edited, and every dossier
carries `decision: null`. See [`REVIEW_GUIDE.md`](REVIEW_GUIDE.md) for what a
reviewer does with one.

The pack is a snapshot **as built**, not a live status. Twenty-six of its thirty
candidates now have decisions in `reviews.jsonl`; the dossiers keep
`decision: null` on purpose, so `show <id>` (which now applies the decision log)
and `stats.json` are the current state. A new pack is built only after an authored
analysis exists for the next selection.

`queue()` builds the review queue: unreviewed, non-duplicate, review-eligible
rows, ordered by (scanner signal, workflow richness, source, repo, path) with at
most `per_repo_cap` samples per repository, so breadth beats volume. **The queue
never assigns a review state** — that is what keeps predictions out of the
ground truth.

### Four-verdict protocol

Every candidate ends in exactly one of four human verdicts. The names below are
the review states above; "positive"/"negative" describe the *finding*, not the
code's quality.

| Verdict | State | Scored | Meaning | Minimum evidence |
|---|---|---|---|---|
| Confirmed positive | `confirmed_leakage` | yes | Information crosses the intended evaluation boundary and changes a reported result. | A rationale naming the mechanism and at least one code reference. |
| Confirmed negative | `legitimate` | yes | The apparent signal is a false positive, or the workflow is sound despite matching a heuristic. | A rationale explaining why the boundary holds (for example, `fit` on the train partition only) and a code reference. |
| Potential | `potential` | no | The pattern is present, but the reviewer cannot establish whether information crosses the boundary. | Rationale naming what is missing. |
| Uncertain | `uncertain` | no | Insufficient or ambiguous evidence; the sample needs a different reviewer or more context. | Rationale naming the blocker. |

Review steps: open the dossier, fetch the source at the recorded commit, trace
the workflow (ingest → preprocess → split → fit → evaluate), cite `cell N:L`
references, then record the verdict. A **published precision or recall claim
requires a second reviewer** who did not author the verdict; until then the label
is evidence, not a benchmark number.

### Development and holdout splits

Real-world labels are only useful if the detector was not tuned on them. The
separation rule is enforced by `corpus_expansion.split_roles()`:

- **Unit of separation is the repository**, not the file, because near-duplicates
  mostly share a repository. All samples from one repository go to the same side.
- **Assignment is deterministic and label-blind**: a stable hash of `repo` names
  the side. No reviewer chooses a side after seeing a verdict.
- **Holdout samples are never inspected while writing a detector.** They are run
  at release or benchmark time and their results are reported once.
- **No cluster may straddle the boundary.** A copy or near-duplicate group
  (`duplicate_of`, `code_duplicate_of`, `near_of`) is wholly on one side; if any
  member hashes to development, the whole group moves to development and loses
  its holdout side. A newly collected copy can therefore only move a group toward
  development, never promote a development sample into holdout.
- **Near-duplicate banding must be shift-invariant.** `deduplicate` feeds sorted
  bottom-k sketches to `near_duplicate.cluster` with `rows=1`. Positional bands
  assume a minhash signature where position is the hash function; with a sorted
  sketch one extra low value shifts every later position and two 0.97-similar
  notebooks share no band. With `rows=4` six development/holdout near-copy pairs
  went undetected, all copies of one `CS.ipynb`; `rows=1` links them and the
  straddling record moves to development. The committed manifest is re-checked by
  an independent candidate generator in `tests/test_review_batch.py`, so a future
  banding change cannot silently reopen the gap.

`finalize` writes the side into every `review_queue.jsonl` record and counts both
sides in `stats.json` (`development + holdout == unique_samples`). `show` prints
the side and `--side development` refuses to open a holdout dossier, so a
reviewer can work one side without seeing the other. Any number computed from the
expansion corpus today is still a *development* measurement: the holdout side has
not been run and its first results are a release-time event, not a claim.

The split was introduced on 2026-11-26, after the first 26 decisions were
recorded. Sides are label-blind, but history is not retroactively clean: 18 of
those decisions fall on development and 8 on holdout, and the reviewers did not
know a side existed. Treat the first holdout run as contaminated for those 8
cases; a pristine holdout needs repositories collected and reviewed after this
rule existed. No Phase 20L detector was tuned on a reviewed expansion sample;
workflow verification opened one holdout dossier after the detector was written.
Phase 20N tightened the cluster rule and re-ran `finalize`: one unreviewed record
(`simpleniick/NIDS`) moved from holdout to development because its `CS.ipynb` is
a 0.95-similar copy of six development notebooks. No reviewed holdout case moved,
so the contaminated-history count stays at 8.

### Locked evaluation protocol (not yet run)

No pristine holdout exists today, so the protocol below is the requirement for
the first locked evaluation, not a description of a completed one.

1. **Freeze the case set.** Commit a manifest of candidate ids, repository,
   source revision and content digest. After the freeze, a new sample cannot
   enter the set without a recorded inclusion decision and a new revision.
2. **Freeze the split.** The development/holdout assignment is the repository
   hash plus the cluster rule above; record the resulting side per id in the
   manifest. A later collection may move a cluster to development, so the frozen
   manifest must be the one evaluated.
3. **Freeze the detector.** Record the LeakLens commit and version. A detector
   change after the freeze starts a new evaluation revision; it cannot be scored
   against the old one.
4. **Label independently.** Every scored case needs two distinct reviewers, at
   least one blind pass (`show --blind`), preserved rationales and an
   adjudication that does not overwrite either verdict. A single-reviewer label
   is evidence, not a metric.
5. **Apply exclusions explicitly.** Duplicates, byte-identical frozen overlaps
   and repository-level overlaps are reported and excluded, never dropped
   silently. Unknown or non-redistributable licences stay in the corpus as
   metadata but out of any redistributed case pack.
6. **Report counts, not just ratios.** Publish the number of scored positives,
   negatives, disputed and excluded cases with every precision/recall figure, and
   state that the sample is not a random census. No confidence interval is
   claimed from a handful of cases.

Reproduce the inputs with `python scripts/collect_expansion.py finalize`, then
`stats`; open a holdout dossier only with
`python scripts/collect_expansion.py show --id <id> --blind --side holdout`.
The first holdout result is a release-time event, not a number to tune against.


## Label-leakage guard

Collection and review metadata must never become detector input. Two guards:

- `tests/test_corpus_expansion.py::test_detectors_never_import_collection_metadata`
  fails if any module under `leaklens/` outside `corpus_expansion.py` mentions
  `corpus_expansion` or `realworld_expansion`.
- `review()`'s state validation keeps `potential`/`uncertain` out of scored sets.

## State of the corpus

As of 2026-11-26 (`realworld_expansion/stats.json` is authoritative):

| Metric | Value |
|---|---|
| Candidates | 1,147 |
| Unique samples | 1,051 |
| Duplicates / near-duplicates | 96 / 64 |
| GitHub repositories | 730 |
| Kaggle kernels | 119 |
| Licences | permissive 541, copyleft 38, unknown 568 |
| Review queue | 300 (275 distinct repositories) |
| Development / holdout | 571 / 480 unique samples (repository-hash, cluster-safe) |
| Human decisions recorded | 26 — 15 `confirmed_leakage`, 7 `legitimate`, 1 `potential`, 3 `uncertain` |
| Scored labels | 22 (only the 15 positives and 7 negatives) |
| Overlap with the frozen `realworld/` corpus | 8 samples, flagged and excluded from the queue |

The 26 decisions come from one reviewer and are evidence, not a published
benchmark. The frozen `realworld/` corpus remains at 60 cases with 8 definitive
labels; the expansion corpus is a separate, still-growing pool.

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

# rebuild the review dossiers from the authored analysis, then read one
python scripts/collect_expansion.py batch
python scripts/collect_expansion.py show --id rwx_0123456789ab

# record one human decision (the only way a label enters the corpus)
python scripts/collect_expansion.py review --id rwx_0123456789ab \
  --state confirmed_leakage \
  --rationale "StandardScaler.fit_transform on the full frame precedes the split" \
  --ref "cell 7:2" --mechanism preprocess_before_split --reviewer <name>
```

Requirements: `gh` authenticated (Code Search and repo metadata), and about
50 MB of cache per 300 notebooks. No notebook is ever executed; the scanner reads
text only.
