# H2 — independent real-world re-score of the shipped model

> **Status: H2 is not available. No H2 number is published.**
>
> `docs/AI_ROADMAP.md` (M5) requires a second real-world holdout (H2) with ≥48 units from
> ≥12 repositories, disjoint from the authored corpus and H1, each unit carrying provenance,
> every cited path verified at a pinned commit, at least two reviewers with disagreements
> recorded, and a frozen manifest digest — then **one** scoring run reported with
> cluster-bootstrap intervals, calibration metrics and per-family false positives.
>
> Those conditions are not met, and the roadmap says: *"If any of that cannot be met, publish
> the shortfall in the document rather than a number."* This document is that shortfall.
>
> **H1** (`docs/AI_REALWORLD.md`) remains the published Phase 6 record. It is not replaced,
> re-scored as H2, or edited here.

## 1. Why H2 does not exist

No H2 corpus was authored for M5. The two real-world corpora in the repository are both
ineligible for H2:

| Corpus | Units | Repositories | Definitive labels | Why it is not H2 |
|---|---:|---:|---:|---|
| H1 — `leaklens/ai/corpus_realworld.py` (`phase6-realworld-holdout-v1`) | 48 | 12 | 48 | It **is** H1. Re-scoring it cannot be a second, independent holdout, and H1 must stay the Phase 6 record. |
| Notebook corpus — `realworld/` (60 `.ipynb`) | 60 | 57 | 8 | Same authoring programme as the scanner corpus, not a disjoint holdout; only 8 of 60 labels are definitive; 46 of the 50 `uncertain` labels were auto-seeded, not hand-reviewed. |

Consequently, for M5:

- H2 units: **0** of ≥48.
- H2 repositories disjoint from the authored corpus and H1: **0** of ≥12.
- H2 reviewers: **0** of ≥2.
- H2 freeze digest: **none** (no corpus to freeze).
- Cited paths verified at a pinned commit for H2 units: **n/a** (no H2 units).
- Scoring runs against H2: **0**.

## 2. Acceptance checklist (roadmap M5)

| Requirement | Target | Actual | Met |
|---|---|---|---|
| Units | ≥48 | 0 | ✗ |
| Repositories | ≥12 | 0 | ✗ |
| Disjoint from authored corpus and H1 | required | n/a | ✗ |
| Per-unit provenance | required | n/a | ✗ |
| Every cited path verified at a pinned commit | required | n/a | ✗ |
| ≥2 reviewers, disagreements recorded | required | 0 reviewers | ✗ |
| Frozen manifest digest | required | none | ✗ |
| One scoring run with cluster-bootstrap intervals, calibration, per-family FP | required | none on H2 | ✗ |

Output per roadmap: **shortfall documented; no metric reported.**

## 3. Corpus audit performed for this milestone

The following was verified read-only on the current tree (see §7 for the reproduction script).
Nothing was edited, relabelled or removed.

### 3.1 H1 (frozen Phase 6 record) — unchanged

| Property | Value |
|---|---|
| Holdout id | `phase6-realworld-holdout-v1` |
| Units | 48 (24 leaky / 24 legitimate) |
| Repositories | 12 |
| Pipeline families | 39 |
| Categories | 14 |
| Unique case names / unique code bodies | 48 / 48 |
| Review stamp | `manual-review-2026-11-05` (single review pass) |
| Manifest digest | `7fde39ce94d34ae7f05243163e2761d5410f4b3a01f9770eabc2a24ea45d5742` (matches `FROZEN_DIGEST`) |
| Verbatim / reconstructed excerpts | 4 / 44 |

H1 is a **single-review** corpus. M5's reviewer requirement (≥2 reviewers, disagreements
recorded) is not satisfied by H1 either, so re-scoring H1 would still not meet H2 acceptance.

### 3.2 Notebook corpus (`realworld/`) — labels and exclusions

| Verdict | Count | Scored | Provenance |
|---|---:|---|---|
| `confirmed_leakage` | 5 | Yes (positive) | Hand written reason per notebook |
| `legitimate` | 3 | Yes (negative) | Hand written reason per notebook |
| `potential_leakage` | 2 | **No** — excluded | Hand written reason |
| `uncertain` | 50 | **No** — excluded | 46 auto-seeded from scanner findings, 4 other |
| **Total** | **60** | **8 scored, 52 excluded** | |

- 60 fixtures, 60 manifest records, 57 distinct source repositories, 60 unique 16-character
  SHA-256 prefixes, no duplicate fixtures, no manifest/label id mismatch.
- All 60 fixture hashes match their manifest prefixes, so no fixture changed after freezing.
- The 52 `potential`/`uncertain` records are **not** treated as negatives and are not folded
  into any denominator. Reporting the 8-case smoke check as if it described 60 notebooks would
  be exactly the silent-relabelling error M5 forbids.
- The 8 definitive notebooks come from 8 distinct repositories, but they are part of the
  same real-world collection the project already curated; they are a **smoke check**, not a
  disjoint 48-unit holdout. Even with perfect labels they could not satisfy H2's unit and
  reviewer requirements.

### 3.3 Overlap and near-duplicate checks

| Comparison | Result |
|---|---|
| H1 repositories ∩ development sources | 0 |
| H1 code bodies identical to development | 0 |
| H1 max token Jaccard vs development | 0.7895 (< 0.8 guard) |
| Notebook repositories ∩ H1 repositories | 0 |
| Notebook repositories ∩ development sources | 0 |
| Notebook definitive cases: max Jaccard vs development | 0.2979 (< 0.8) |
| Notebook definitive cases: max Jaccard vs H1 | 0.3103 (< 0.8) |

Disjointness is clean for both corpora. That is necessary for H2 but not sufficient: the
blocking shortfall is units, reviewers and a freeze, not contamination.

## 4. Model under evaluation (pinned)

Any future H2 run must score exactly this artifact; the regression test in
`tests/test_ai_h2.py` fails if it drifts.

| Property | Value |
|---|---|
| Model | `leaklens-ai-p7` |
| Version | `leaklens-ai-p7-2026.11` |
| Artifact file | `leaklens/ai/model.json` |
| Artifact file SHA-256 | `65508af13318069efefe9a6c3276dd47b837d6a9e3869bf9774f6bdae1795e01` |
| Card `artifact_digest` | `sha256:ff15fdff173522db62e32bc46c04d534e5a09d824989395b27651fab25950475` |
| Model features | 108 (`MODEL_FEATURE_NAMES`) |
| Shipped threshold | **0.53** (selected on validation only) |
| Library version | scikit-learn 1.9.1 |

## 5. Diagnostic only: shipped-model H1 re-score (not H2)

For transparency the shipped P7 model was applied once to the **existing H1 holdout**. This is
**not** an independent result and **not** an H2 number:

- H1 was not tuned against P7 (P7 changed the authored corpus and extraction guards, and H1
  was deliberately not re-scored during Phase 7), so it is legitimate diagnostic evidence.
- But H1 is the *first* holdout, already published as the Phase 6 record, and single-review.
  Using it again would be a re-score of H1, which the roadmap forbids as an H2 substitute.

Measured at the shipped threshold 0.53 on all 48 H1 units:

| Metric | Value | Denominator |
|---|---:|---|
| TP / FP / FN / TN | 22 / 6 / 2 / 18 | 48 |
| Accuracy | 0.8333 | 48 |
| Precision | 0.7857 | 28 predicted leaky |
| Recall | 0.9167 | 24 actual leaky |
| F1 | 0.8462 | — |
| Specificity | 0.7500 | 24 actual legitimate |
| False-positive rate | 0.2500 | 24 actual legitimate |
| Brier score | 0.1496 | 48 |
| Expected calibration error | 0.1460 | 48 |
| Mean predicted probability | 0.6004 | 48 |
| Observed positive rate | 0.5000 | 48 |

At the Phase 6 threshold 0.50, for comparison: TP/FP/FN/TN = 22/7/2/17, accuracy 0.8125,
precision 0.7586, recall 0.9167, F1 0.8302, FPR 0.2917.

Per-family mistakes at 0.53:

- False positives (6, one each): `catboost-timeseries`, `lightgbm-simple`, `optuna-cv`,
  `sklearn-scaling`, `xgboost-cv`, `xgboost-timeseries`.
- False negatives (2, one each): `autosklearn-duplicates`, `catboost-duplicates`.

Score distribution: leaky units have median 0.9977 (mean 0.9178); legitimate units have
median 0.0803 (mean 0.2830). Leaky minimum is 0.0024 and legitimate maximum is 0.999, so the
two misses and six false alarms are confident mistakes, not threshold-edge cases. With n=48
these counts move precision by ~0.04 per case; no interval tighter than the H1
repository-cluster bootstrap should be quoted.

### 5.1 Provenance of the historical claims

The historical real-world figures are the **Phase 6** record at threshold **0.50** on H1:

| Historical claim | Value | Source and denominator |
|---|---:|---|
| Precision | 0.767 | `docs/AI_REALWORLD.md`, P6 on H1, 23 TP / 30 predicted leaky |
| Recall | 0.958 | 23 TP / 24 actual leaky |
| FPR | 0.292 | 7 FP / 24 actual legitimate |
| Accuracy | 0.833 | 40 correct / 48 |
| F1 | 0.852 | — |
| Brier | 0.1149 | 48 units |
| ECE | 0.1264 | 48 units |
| Mean predicted probability | 0.6232 | 48 units |
| Observed positive rate | 0.5000 | 48 units |

The P7 diagnostic above is on the **same 48 H1 units** but a different model and threshold, so
the differences (precision 0.767 → 0.786, Brier 0.1149 → 0.1496, mean probability 0.6232 →
0.6004) are model/threshold deltas on one corpus, not evidence about H2. Brier and ECE both
worsened, so the shipped model is less calibrated on H1 than the Phase 6 record; this is not
calibrated probability evidence for users, because H1 is a small single-review holdout and
P7's card reports development calibration separately. M3's `confidence_basis` semantics are
untouched and must not be read as calibrated probabilities.

## 6. H2 verdict

**Inconclusive — not evaluated.** There is no H2 corpus, so there is no H2 number. H2 is
neither supported nor rejected; the experiment the roadmap specifies has not been run.

This is deliberately not resolved by:

- re-scoring H1 and calling it H2 (same corpus, single review, already published);
- scoring the 60-notebook corpus (only 8 definitive labels, 46 auto-seeded `uncertain`);
- treating `potential`/`uncertain` records as negatives;
- using model predictions as labels; or
- changing the corpus, labels or protocol after seeing these diagnostic numbers.

## 7. Reproduction

The audit above is reproducible read-only from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'

# Corpus counts, hashes, disjointness and near-duplicate guards
.\.venv\Scripts\python.exe -m pytest tests/test_ai_h2.py tests/test_ai_realworld.py -q

# H1 freeze digest and shipped-model identity
.\.venv\Scripts\python.exe -c "from leaklens.ai.corpus_realworld import manifest_digest; print(manifest_digest())"

# Historical Phase 6 record + the diagnostic P6/P7 numbers quoted above
.\.venv\Scripts\python.exe -m leaklens.bench_realworld --no-profile
```

`python -m leaklens.bench_realworld` prints the shipped-model H1 metrics that §5 quotes and
the notebook smoke check. It does not produce an H2 result, and it must not be renamed as one.

## 8. What would unblock H2

1. Author a new holdout (`phase7-realworld-holdout-h2` or later) of **≥48 pipeline units**
   from **≥12 repositories** that appear in neither the authored development corpus nor H1
   (and are not the 57 notebook-corpus repositories if those are considered part of the same
   programme).
2. Record per unit: repository, licence, commit-pinned URL, path, retrieval date, provenance
   class (verbatim vs reconstruction) and a label rationale.
3. Verify **every** cited path at its pinned commit, not a sample (H1 resolved 16 of 23 cited
   paths and documented 7 unresolved).
4. Have **two independent reviewers** label each unit; record disagreements and how they were
   resolved; keep machine-generated labels out of ground truth.
5. Freeze the corpus and publish a manifest digest, with a test that fails on any post-freeze
   edit.
6. Run the existing `bench_realworld` path once against the pinned artifact and threshold,
   with repository-cluster bootstrap intervals, calibration metrics and per-family false
   positives, into a sibling H2 report.

Until then, treat H1 as the only measured real-world evidence, read it as a Phase 6 record,
and do not cite any H2 number — because none exists.

## 9. Limitations

- The diagnostic re-score is small (n=48) and single-review; no interval tighter than the
  repository-cluster bootstrap is meaningful.
- 44 of 48 H1 units are reviewed reconstructions, not verbatim upstream code.
- The notebook corpus cannot supply independent labels at the sample size H2 needs.
- M5 produces no product code; the only guarantees added are regression guards in
  `tests/test_ai_h2.py`.
