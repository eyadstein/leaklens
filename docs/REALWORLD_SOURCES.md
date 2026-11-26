# Real-world sources examined (Phase 20N)

**Date:** 2026-11-26. **Scope:** a genuine attempt to expand LeakLens' real-world
evidence base beyond the 1,147 candidates already in `realworld_expansion/`.
Every claim below was checked against the source on that date; nothing was
downloaded, copied or added to the corpus unless its licence allows it.

**Result: no new corpus entry was added.** The reasons are recorded per source.
This is not a claim that no suitable case exists — it is a record of what was
checked and why each candidate failed the licence, schema or reproducibility bar.

## 1. What was checked

| Source | Licence (verified) | What it contains | Decision |
|---|---|---|---|
| `malusamayo/leakage-analysis` | MIT | ASE'22 static analyser for notebook leakage; code only | **Reference only** — a tool, not a case. Compare against it; do not vendor it. |
| `malusamayo/GitHubAPI-Crawler` | **none declared** | `evaluation_materials/` with `ground-truth.csv`, `manual-labels.csv`, `analysis-result.csv`, `sample-notebooks.zip` (42.8 MB) | **Excluded — no licence.** The closest external labelled corpus found. Reusing it needs the authors' permission; contacting them is not authorised in this phase. |
| `SE4AIResearch/DataLeakage_JupyterNotebook_Fall2024` | MIT | VS Code extension; leakage taxonomy (multi-test, overlap, preprocessing) | **Reference only** — tool/fixtures, not a licensed case corpus. |
| `SE4AIResearch/DataLeakage_JupyterNotebook_Website_Fall2024` | **none declared** | Documentation site | **Excluded — no licence.** |
| `tahh1/LeakageRepairPackage` | already in the corpus | Repair-package notebooks | Already represented; no new case. |
| arXiv 2611.18598, 2608.24665, 2609.28725 | papers | Leakage measurements, not case data | **Reference only** — see `docs/LEAKAGE_LANDSCAPE.md` §5. |
| GitHub repository search for leakage notebooks (sorted by stars) | mixed | Mostly unlicensed student projects or the same ASE'22 lineage | **Rejected** — no licence, or the notebook is a tutorial duplicate of material already in the corpus. |

## 2. Why the obvious corpus did not work

The ASE'22 evaluation materials would be the single most valuable addition:
external labels, produced by a research group, on real notebooks. They are
published in a repository with **no licence file**, which under default copyright
means no right to copy, redistribute or derive. Two consequences:

1. LeakLens cannot use them as a benchmark, even locally, without permission.
2. A cross-tool comparison (LeakLens vs the ASE'22 analyser) is therefore also
   blocked on permission, not on engineering.

Recording this is more useful than adding unlicensed notebooks to a corpus whose
licence policy exists precisely to prevent that.

## 3. Coverage of the existing reviewed set

The 26 recorded decisions (one reviewer) cover ten mechanisms:

| Mechanism | Decisions |
|---|---|
| `legitimate_lookalike` | 5 |
| `none_observed` | 5 |
| `cv_contamination` | 4 |
| `preprocess_before_split` | 4 |
| `test_set_reuse` | 3 |
| `group_entity_leakage` | 1 |
| `feature_selection_on_holdout` | 1 |
| `duplicate_contamination` | 1 |
| `target_leakage` | 1 |
| `temporal_leakage` | 1 |

Missing from the reviewed set: `target_encoding_leakage`,
`augmentation_leakage`. A new cohort should deliberately include them, but a
deliberate inclusion is a selection decision and must be recorded as such — the
current set is a queue sample, not a census.

## 4. What a new cohort needs

The protocol is in `docs/REALWORLD_EXPANSION.md` ("Locked evaluation protocol").
For a cohort collected *after* this phase, the practical requirements are:

1. **A licence that permits at least reference and local analysis.** Permissive
   (MIT/Apache/BSD) repositories, or an explicit written permission. Public
   visibility alone is not enough.
2. **A recorded revision.** Commit SHA plus content SHA-256, both already in the
   manifest schema.
3. **Cluster-safe isolation from everything already collected** — repository
   hash, content hash, code hash and near-duplicate sketch. Phase 20N fixed the
   banding bug that let six near-copies straddle the split; a new cohort must be
   checked against the same invariant.
4. **Independent labels.** Two distinct humans, at least one blind pass. A new
   cohort with one reviewer is more data, not better evidence.
5. **A freeze.** The ids, revisions, digests and sides recorded in one committed
   file before the first evaluation run.

Until (4) is satisfied for a cohort, adding more single-reviewed cases raises the
count without raising the confidence.

## 5. Reproducing this pass

```sh
# what the committed corpus already contains
python scripts/collect_expansion.py stats
python scripts/collect_expansion.py validate

# a licensed collection run (writes to realworld_expansion/; review the diff)
python scripts/collect_expansion.py collect --channels github_search --limit 50
python scripts/collect_expansion.py finalize
```

The collector records licence status per candidate; unknown licences stay in the
corpus as metadata and out of any redistributed pack. Nothing in this pass was
added to the corpus.
