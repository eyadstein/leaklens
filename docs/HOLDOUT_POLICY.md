# Holdout policy for the real-world corpus (#655)

**Status:** owner decision required. This document is a decision memo, not an
applied policy. Nothing here changes the manifest, the queue, the freeze or any
label.

**Date:** 2026-11-26. **Baseline:** `main` at Phase 20P.

## 1. The measured problem

The expansion corpus (`realworld_expansion/`) and the frozen `realworld/` corpus
are separate collections. The expansion corpus already flags byte-identical
copies of a frozen notebook (`overlaps_frozen`) and keeps them out of the review
queue, but file-level identity is not the documented separation unit. The unit is
the **repository**: code, vocabulary and label decisions from one repository can
inform another notebook from the same repository even when the bytes differ.

Measured on the current manifest (1,147 expansion records, 60 frozen records from
57 repositories):

| Measure | Value |
|---|---|
| Expansion records whose repository appears in the frozen corpus | 9 |
| Distinct such repositories | 6 |
| Byte-identical to a frozen notebook (`overlaps_frozen`) | 7 |
| Same repository, not byte-identical | 2 (`HOXOMInc/feature-engineering-book` `3.ipynb`, `9.ipynb`) |
| Records with a recorded review decision | 0 |
| Records in the committed 300-item queue | 0 |
| Development / holdout side of the 9 | 6 / 3 |
| Side reason for the 9 | `repo-hash` (none is a straddling cluster) |

The three holdout-side records are `Annie00000/Project` `5_5.ipynb` and
`5_6.ipynb`, and `CarlaAstudillo/pandas-nicar-2016` `Part 1.ipynb`.

Two further facts make the current holdout non-pristine independently of #655:

1. **No labels are frozen.** The split is deterministic, but the decisions live
   in an append-only log and can change; a freeze records the case set, not a
   verdict.
2. **Eight holdout decisions predate the split rule.** They were recorded on
   2026-11-15/18, before the repository-hash side existed, by one reviewer who
   saw the scanner signal and the authored analysis (the historical records have
   no `materials` field). They are evidence of work done, not independent
   holdout labels.

So the question is not "should 9 unlabelled records be relabelled". It is "what
may a future holdout number be computed from".

## 2. What each option protects

### A. Retain historical assignments; permanently exclude contaminated cases

Keep the current sides and ids. Record every same-repository and every pre-split
decision as excluded from any pristine holdout claim; compute a pristine number
only from the remainder.

- **Data integrity:** high for the remaining set; the exclusion is explicit and
  auditable.
- **Historical comparability:** highest — ids and sides do not move, so a later
  run can be compared with an earlier one case by case.
- **Leakage and near-duplicate risk:** the 9 repository-overlap records and the
  8 pre-split decisions are removed from the pristine claim; cluster
  co-location already prevents near-duplicate straddling.
- **Reviewer independence:** unchanged; every remaining case still needs two
  independent reviews.
- **Operational complexity:** low — one exclusion field plus a queue filter.
- **Reproducibility:** high if the exclusion set is committed and frozen.
- **Risk of silent drift:** medium — `permanently exclude` is easy to forget in
  a later query; the exclusion list must itself be frozen.

### B. Rebuild a new holdout with strict provenance and cluster separation

Discard the current sides, re-collect or re-partition into a new holdout with
strict repository/cluster separation and fresh labels.

- **Data integrity:** highest at the design level.
- **Historical comparability:** broken — no prior result is comparable, and
  there is no valid prior holdout result to preserve.
- **Leakage and near-duplicate risk:** lowest, if the new partition is checked
  against both corpora.
- **Reviewer independence:** still required; rebuilding the split does not
  create a second reviewer.
- **Operational complexity:** high — a new collection/review cycle.
- **Reproducibility:** high with a new freeze, but the new set's provenance is
  unproven until reviewed.
- **Risk of silent drift:** low for the new set, high for the old result if the
  old number is still quoted.

### C. Separate historical, development and prospective evaluation sets

Name three cohorts explicitly: `historical` (the 26 existing decisions and their
sides), `development` (the rest of the current corpus) and `prospective` (a new
holdout, repository-disjoint from both corpora, frozen before labelling). Each
cohort has its own purpose and is never pooled into one precision number.

- **Data integrity:** highest overall — each claim names the cohort it comes
  from, and no historical label is laundered into a pristine set.
- **Historical comparability:** preserved inside `historical`; a prospective
  number is deliberately not comparable to it.
- **Leakage and near-duplicate risk:** the prospective cohort is the only one
  allowed to carry a pristine claim; it is checked for repository and cluster
  separation before any label is recorded.
- **Reviewer independence:** required for the prospective cohort; historical
  labels stay single-reviewer and are labelled as such.
- **Operational complexity:** medium — one `cohort` field and one new freeze
  label, plus the same review work any option needs.
- **Reproducibility:** high if each cohort is frozen separately.
- **Risk of silent drift:** low if the cohort field is required and the freeze
  verifies it.

### D. The narrow rule the issue itself proposes

Treat repository-level overlap as disqualifying for a **holdout** claim and a
warning for development; keep the current split otherwise. Add the pre-split
decisions to the same exclusion.

- **Data integrity:** good for the holdout claim; development keeps useful
  material.
- **Historical comparability:** high.
- **Leakage risk:** addressed at the documented unit (repository), without
  claiming that every notebook in a shared repository is contaminated.
- **Reviewer independence:** unchanged.
- **Operational complexity:** lowest — a flag, a stats count and a filter.
- **Reproducibility:** high.
- **Risk of silent drift:** medium — "warning in development" can be ignored
  and later quoted by mistake.

## 3. Recommendation

**Recommend option C, implemented with option D's narrow rule and A's
exclusion records.**

Why:

1. The current set has **zero valid independent holdout labels**. Eight holdout
   decisions predate the split and none is blind or double-reviewed, so there is
   no historical holdout number worth protecting from a rebuild (which removes
   the main argument for B).
2. Naming cohorts is the only option that prevents the historical decisions from
   being quoted as pristine later. Option A excludes cases but leaves one
   undifferentiated set; option D warns in development but still allows a
   mistaken pooled query.
3. The expensive work — two independent reviews of a clean cohort — is required
   by every option. C adds only a `cohort` field and a separate freeze, not a
   second review cycle.
4. The narrow repository rule is justified by the documented separation unit:
   same-repository material is a holdout disqualifier, not necessarily a
   development exclusion. That keeps the development corpus useful without
   pretending the holdout is clean.

**Concrete transition, if the owner accepts C:**

1. Add a `cohort` value (`historical` for the 26 decided cases; `development`
   for the rest) without rewriting any side or decision.
2. Mark the 9 same-repository records and the 8 pre-split decisions as
   `exclude_from_pristine` with a reason string; keep them in the corpus.
3. Freeze the exclusion set and the cohort assignment in a **new** snapshot
   (never overwrite `freezes/2026-11-26.json`).
4. Only after that, collect or select a `prospective` cohort of repositories
   disjoint from both corpora, freeze it, and give it to two reviewers who have
   not seen the historical decisions.
5. Quote a holdout precision number only when the cohort is `prospective`, the
   freeze verifies clean and the cases are `double_reviewed`.

**What this memo does not do:** it does not choose for the owner, does not change
the manifest or split, and does not imply that the existing 26 decisions are
wrong. They are historical single-reviewer evidence, which is what they will
remain until a second reviewer records independent decisions.
