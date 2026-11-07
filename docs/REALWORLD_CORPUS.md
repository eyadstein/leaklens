# Real-world holdout corpus (Phase 6)

This document records the provenance, labelling method and freeze state of the
**Phase 6 real-world holdout** used by `python -m leaklens.bench_realworld`.

The corpus lives in `leaklens/ai/corpus_realworld.py`. The machine-readable
freeze record is `docs/realworld_holdout.json`.

- Holdout id: `phase6-realworld-holdout-v1`
- Role: `single_pass_evaluation`
- Retrieval date: 2026-11-05
- Cases: **48** (24 leaky, 24 legitimate)
- Repositories: **12** (permissively licensed only)
- Pipeline families: **39** (each family lives in exactly one repository)
- Categories: **14**
- Provenance: **4 verbatim excerpts**, **44 reviewed reconstructions**
- Frozen manifest digest:
  `7fde39ce94d34ae7f05243163e2761d5410f4b3a01f9770eabc2a24ea45d5742`

The digest is the SHA-256 of every labelled field (code, label, category,
rationale, provenance, path, URL). It is checked by the test suite and printed
by the benchmark, so editing the corpus after it has been scored is visible.

## Source repositories

All repositories are public and permissively licensed. The commit is the tip of
the default branch when the corpus was assembled; URLs are commit-pinned.

| Repository | Licence | Commit | Retrieved |
|---|---|---|---|
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause | `3c5e5ef` | 2026-11-05 |
| [mlflow/mlflow](https://github.com/mlflow/mlflow) | Apache-2.0 | `5c6af89` | 2026-11-05 |
| [optuna/optuna-examples](https://github.com/optuna/optuna-examples) | MIT | `9450de9` | 2026-11-05 |
| [ageron/handson-ml3](https://github.com/ageron/handson-ml3) | Apache-2.0 | `ef1c9f5` | 2026-11-05 |
| [feature-engine/feature_engine](https://github.com/feature-engine/feature_engine) | BSD-3-Clause | `d866312` | 2026-11-05 |
| [microsoft/FLAML](https://github.com/microsoft/FLAML) | MIT | `798cacb` | 2026-11-05 |
| [aladdinpersson/Machine-Learning-Collection](https://github.com/aladdinpersson/Machine-Learning-Collection) | MIT | `558557c` | 2026-11-05 |
| [scikit-learn-contrib/imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | MIT | `8504e95` | 2026-11-05 |
| [automl/auto-sklearn](https://github.com/automl/auto-sklearn) | BSD-3-Clause | `6732112` | 2026-11-05 |
| [dmlc/xgboost](https://github.com/dmlc/xgboost) | Apache-2.0 | `f109cc0` | 2026-11-05 |
| [microsoft/LightGBM](https://github.com/microsoft/LightGBM) | MIT | `3ee03c2` | 2026-11-05 |
| [catboost/catboost](https://github.com/catboost/catboost) | Apache-2.0 | `ecf5e12` | 2026-11-05 |

Verbatim excerpts are redistributed under those licences with attribution to the
upstream project. Each case stores the repository, commit and file path in
`source_url`. For the four verbatim excerpts that is the exact upstream file. For
the 44 reconstructions the path identifies the upstream idiom the unit was
modelled on, not a byte-exact source line; the 2026-11-06 audit below records
where that citation is only approximate.

## Provenance audit (2026-11-06)

After the corpus was frozen, every distinct cited path was requested at its
pinned commit. **16 of 23 paths resolve; 7 do not, affecting 13 of 48 cases.**
The failures are recorded here rather than by editing the frozen corpus, so the
manifest digest and the single-pass result stay intact.

| Cited path (frozen) | Cases | Actual state at the pinned commit |
|---|---:|---|
| `aladdinpersson/.../ML/Scikit_learn/KNN/knn.ipynb` | 3 | Directory is `ML/algorithms/knn/`, and the file is `knn.py`, a from-scratch k-NN classifier — there is no notebook, imputer or scaler in it. |
| `aladdinpersson/.../ML/Scikit_learn/SVM/svm.ipynb` | 1 | Directory is `ML/algorithms/svm/`, and the file is `svm.py`, a from-scratch SVM. The `sklearn.preprocessing.scale` idiom is not present. |
| `automl/auto-sklearn/.../example_methods_to_load_and_export_models.py` | 4 | No file of that name at the pinned commit. The nearest examples are `examples/20_basic/example_classification.py` (split before `fit`) and `examples/40_advanced/example_pandas_train_test.py` (test data passed to `fit`). |
| `catboost/.../tutorials/regression/regression_tutorial.ipynb` | 1 | No generic regression tutorial at that commit; the directory holds `poisson.ipynb`, `survival.ipynb` and `tweedie.ipynb`. |
| `imbalanced-learn/.../examples/over_sampling/plot_random_over_sampling.py` | 1 | Directory is `examples/over-sampling/`; the file is `plot_comparison_over_sampling.py`. |
| `imbalanced-learn/.../examples/over_sampling/plot_smote.py` | 2 | Directory is `examples/over-sampling/`; SMOTE is demonstrated in `plot_comparison_over_sampling.py`. |
| `imbalanced-learn/.../examples/under_sampling/plot_tomek_links.py` | 1 | Directory is `examples/under-sampling/`; the file is `plot_illustration_tomek_links.py`. |

What this does and does not mean:

- All 13 affected cases are `reconstructed_excerpt`, not verbatim excerpts, so
  no upstream code is misattributed as copied text.
- The code, labels, rationales and the scored probabilities are unchanged. The
  manifest digest is unchanged, and the holdout was not rescored for this audit.
- The aladdinpersson entries are the weakest citations: that repository contains
  from-scratch implementations rather than scikit-learn notebook pipelines, so
  the two `notebook_order` units are generic notebook idioms attached to the
  wrong repository. They remain valid leakage examples, but their provenance
  claim should be read as "synthetic" rather than "modelled on this file".
- The remaining four repositories with broken citations still contain the
  families involved (auto-sklearn holdout/AutoML, CatBoost ordered splits,
  imbalanced-learn resampling); the specific file names drifted.

## What is in the corpus

| Category | Cases | Leaky |
|---|---:|---:|
| `preprocessing_scaling` | 6 | 5 |
| `preprocessing_order` | 3 | 0 |
| `target_encoding` | 6 | 4 |
| `tuning` | 6 | 3 |
| `cv_misuse` | 5 | 1 |
| `grouping` | 4 | 2 |
| `temporal` | 3 | 1 |
| `duplicates` | 3 | 3 |
| `resampling` | 3 | 1 |
| `imputation` | 2 | 1 |
| `feature_selection` | 2 | 1 |
| `notebook_order` | 2 | 1 |
| `complex_transform` | 2 | 1 |
| `eda_only` | 1 | 0 |

The corpus deliberately includes near-misses as well as headline leaks:

- full-frame work that is *not* a leak because it never feeds a score
  (`sklearn_scaling_for_plot_only`);
- target-independent binning used only for stratification
  (`homl_stratified_holdout`);
- cross-validation on the full dataset with no holdout, labelled legitimate but
  flagged in the rationale as selection bias (`optuna_cross_val_objective`);
- early stopping on the test set in an upstream example, labelled leaky on
  review (`lightgbm_early_stopping_on_test`);
- exactly balanced leaky/legitimate labels overall.

## Labelling method

1. Read the cited file (or reconstruct the shortest faithful variant of its
   idiom) and write the pipeline unit.
2. Decide the label from the **data flow**, not from identifier names: does any
   statistic, fitted state, duplicated row or selected iteration number cross
   the train/evaluation boundary?
3. Record a one-sentence rationale naming what crosses the boundary, or why
   nothing does.
4. Assign exactly one repository and one pipeline family. A family never spans
   repositories.
5. Re-read every case once, with the label hidden, before freezing.

Labels are the author's reviewed judgement. They are not adjudicated by an
independent third party, and the reconstructed cases are not verbatim upstream
code. Both caveats are repeated in the generated report and in the model card.

## Freeze and evaluation protocol

- The corpus was written and labelled **before** Phase 6 changed feature
  extraction, model candidates or thresholds.
- No holdout case is imported by `learned.py`, `corpus.py`,
  `corpus_validation.py` or `corpus_test.py`.
- Model family, calibration and threshold are selected on grouped development
  CV, validation and the existing external scanner cases. Only then is the
  frozen model applied to this holdout, once.
- The holdout result is reported as measured, including every false positive and
  false negative. It is not used to choose features, labels or thresholds.
- The Phase 5 locked authored test in `leaklens/ai/corpus_test.py` is unchanged.

## Measured result

`python -m leaklens.bench_realworld` applies the frozen model to this holdout
once and writes [AI_REALWORLD.md](AI_REALWORLD.md). On the current frozen
model the measured result is **accuracy 0.833, precision 0.767, recall 0.958,
F1 0.852, FPR 0.292** (TP/FP/FN/TN = 23/7/1/17). The report lists every mistake,
per-category and per-repository metrics, repository-cluster bootstrap intervals,
calibration, the candidate comparison and the separate public-notebook smoke
check. The score is not a production-accuracy claim.

## Known limits

- 48 cases from 12 repositories is a **convenience sample** of public code, not
  a random sample of production machine-learning pipelines.
- One author labelled everything; inter-rater agreement is not measured.
- 44 of 48 cases are reconstructions. They preserve the control flow of a real
  idiom but are not verbatim upstream files.
- Repository-level and category-level metrics rest on 2–6 cases each, so their
  confidence intervals are wide by construction.
- Public examples are written to be instructive and often avoid leaks. The
  corpus is therefore not representative of leak *frequency* in the wild.
