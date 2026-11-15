# Review batch 01 - 30 candidates

Human adjudication queue. **No decision has been recorded**: every dossier
below ends with an empty decision, and the analysis is an AI hypothesis
marked `unverified`, not a label.

States: `confirmed_leakage`, `legitimate`, `potential`, `uncertain`,
`not_reviewed`. Only the first two are scored, and only after a human
records them. See [`docs/REVIEW_GUIDE.md`](../docs/REVIEW_GUIDE.md).

Evidence is cited as `cell N:L` (zero-based notebook cell, one-based line).
Run `python scripts/collect_expansion.py show <id>` to print the cached
source those references point at. Code text appears below only for
permissive licences; copyleft and unknown-licence candidates carry
references instead of quoted code.

## Batch composition

| Dimension | Distribution |
|---|---|
| Scanner signal | high 11, low 8, medium 7, none 4 |
| Licence | copyleft 8, permissive 15, unknown 7 |
| Channel | github_repos 8, github_search 15, kaggle_distilkaggle 7 |
| Hypothesis mechanism | cv_contamination 5, group_entity_leakage 1, legitimate_lookalike 4, none_observed 7, preprocess_before_split 9, temporal_leakage 1, test_set_reuse 3 |

Independent repositories: 30 of 30 candidates.

## Index

| # | Candidate | Repo | Licence | Signal | Hypothesis | Confidence |
|---|---|---|---|---|---|---|
| 1 | `rwx_cf8cb6a83808` | ageron/tf2_course | Apache-2.0 | high | legitimate_lookalike | low |
| 2 | `rwx_cf4b451b8c8a` | rasbt/python-machine-learning-book-3rd-edition | MIT | high | legitimate_lookalike | low |
| 3 | `rwx_88f14155caaa` | alirezadir/Machine-Learning-Interviews | MIT | high | legitimate_lookalike | low |
| 4 | `rwx_032348a621ed` | mrdbourke/zero-to-mastery-ml | unknown | medium | preprocess_before_split | medium |
| 5 | `rwx_e9e72e2f1f81` | codebasics/py | unknown | medium | cv_contamination | medium |
| 6 | `rwx_b1fa16d2c10a` | khanhnamle1994/cracking-the-data-science-interview | unknown | low | preprocess_before_split | low |
| 7 | `rwx_0e6934e5ce4f` | wandb/examples | unknown | low | test_set_reuse | medium |
| 8 | `rwx_5fa76ac23c47` | rasbt/machine-learning-book | MIT | none | none_observed | low |
| 9 | `rwx_d5c845943c82` | DDDavides/MLProj | GPL-3.0 | high | preprocess_before_split | high |
| 10 | `rwx_13ee93bec8f7` | ashcode028/Music-Genre-Classification | GPL-3.0 | high | group_entity_leakage | high |
| 11 | `rwx_102dfc6c7c9e` | vermasrijan/srijan-gsoc-2020 | MIT | high | cv_contamination | high |
| 12 | `rwx_11f261a63388` | pratush-mt/Project | Unlicense | high | preprocess_before_split | high |
| 13 | `rwx_9c204d7611a6` | MichalKorzycki/PythonDataScience | Unlicense | high | cv_contamination | medium |
| 14 | `rwx_b65af046ffc5` | brainiak/brainiak-tutorials | Apache-2.0 | high | cv_contamination | high |
| 15 | `rwx_2d4cbad5fb67` | flatiron-school/ds-k-nearest_neighbors-kbo33 | GPL-3.0 | medium | legitimate_lookalike | low |
| 16 | `rwx_73a6de31b37b` | PrasoonPratham/Titanic-Challenge | GPL-3.0 | medium | preprocess_before_split | medium |
| 17 | `rwx_73ed47531b70` | Chaan0210/study-ai | unknown | medium | preprocess_before_split | medium |
| 18 | `rwx_c2ae52031c08` | stdereka/CourseraML | GPL-3.0 | low | none_observed | low |
| 19 | `rwx_2ecc75f87a32` | beritzellerplumhoff/DataScience_MSF_CDMA | GPL-3.0 | low | none_observed | low |
| 20 | `rwx_934050e90066` | tuhinsharma121/deeplearning | GPL-3.0 | low | none_observed | low |
| 21 | `rwx_fe6685882b36` | samanyougarg/Group-Emotion-Recognition | GPL-3.0 | low | test_set_reuse | medium |
| 22 | `rwx_19b2c543d74f` | martin-he543/novozymes-enzyme-stability-prediction-kaggle-2022 | unknown | none | none_observed | low |
| 23 | `rwx_46e12ee43c76` | learn-co-curriculum/dsc-cross-validation-lab | NOASSERTION | none | none_observed | low |
| 24 | `rwx_6bfa5ed1946c` | kaggle/kernel-41941594 | CC-BY-4.0 | high | cv_contamination | high |
| 25 | `rwx_26cf9b2d1280` | kaggle/kernel-36442984 | CC-BY-4.0 | high | preprocess_before_split | high |
| 26 | `rwx_2a33fe770c62` | kaggle/kernel-25227697 | CC-BY-4.0 | medium | preprocess_before_split | high |
| 27 | `rwx_1c064ba9af63` | kaggle/kernel-36301788 | CC-BY-4.0 | medium | temporal_leakage | high |
| 28 | `rwx_d44c61fa976b` | kaggle/kernel-40709602 | CC-BY-4.0 | low | none_observed | low |
| 29 | `rwx_7aaf49ab3e5d` | kaggle/kernel-27437467 | CC-BY-4.0 | low | test_set_reuse | medium |
| 30 | `rwx_69b642bc801b` | kaggle/kernel-21994782 | CC-BY-4.0 | none | preprocess_before_split | medium |

---

## 1. `rwx_cf8cb6a83808`

- **Source**: github_repos
- **Repository**: ageron/tf2_course at `cb9a7263c76bbad302cd177835895992a556c4b9`
- **Path**: `01_neural_nets_with_keras.ipynb`
- **URL**: https://github.com/ageron/tf2_course/blob/cb9a7263c76bbad302cd177835895992a556c4b9/01_neural_nets_with_keras.ipynb
- **Licence**: Apache-2.0 (permissive; redistribution allowed)
- **SHA-256**: `c807380f33f5b646bf45b4058e42c7045f8697ac78727a605c48248bcad5af1b`
- **Scanner signal**: high (high preprocess_before_split, low no_stratify, low no_stratify)

**Workflow.** Keras/Fashion-MNIST tutorial: loads the official train/test split, carves a 5,000-row validation block out of the training half, standardises with statistics fitted on the training rows only, then trains and evaluates a small feed-forward network.

**Possible mechanism: `legitimate_lookalike`** (hypothesis confidence: low)

The high finding names StandardScaler.fit_transform at cell 128 line 4. Cell 15 has already taken X_test straight from load_data() and sliced X_valid/X_train out of X_train_full, so the scaler is fitted on training rows only and cell 128 lines 5-6 transform the validation and test blocks with those training statistics. The scanner can see the operation but not which array it received, which is why this is a useful calibration case: the shape of the code looks like leakage, the data flow is clean.

**Evidence against, or alternative explanations.**

- cell 15:2 loads (X_train_full, y_train_full), (X_test, y_test) and keeps X_test untouched for the whole notebook
- cell 128:5-6 call scaler.transform (not fit_transform) on X_valid and X_test
- the housing regression block repeats the correct order: split at cell 186:3-4, scaler fitted at 189:4, held-out block transformed at 189:6
- the second finding at cell 186 is only no_stratify, and Fashion-MNIST is a balanced 10-class target

**Questions the reviewer must resolve.**

1. Do you accept that fit_transform on a variable already named X_train after an explicit split is not leakage, even though the scanner cannot prove the provenance of the argument?
2. If so, should this be recorded as legitimate with a rationale pointing at cell 15, so the batch has a documented false positive?
3. Is the missing stratify on a balanced 10-class benchmark worth any concern, or noise?

**Code references.**

- `cell 15:2-5` - official split loaded and validation block carved from the training half
- `cell 128:4` - the flagged fit_transform, applied to X_train
- `cell 128:5-6` - transform-only calls for validation and test
- `cell 189:3-6` - same correct order in the housing regression block

```python
# cell 15:2-5
(X_train_full, y_train_full), (X_test, y_test) = (
    fashion_mnist.load_data())
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# cell 128:4
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28)

# cell 128:5-6
X_valid_scaled = scaler.transform(X_valid.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28)
X_test_scaled = scaler.transform(X_test.astype(np.float32).reshape(-1, 28 * 28)).reshape(-1, 28, 28)

# cell 189:3-6
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

```

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The high finding names StandardScaler.fit_transform at cell 128 line 4. Cell 15 has already taken X_test straight from load_data() and sliced X_valid/X_train out of X_train_full, so the scaler is fitted on training rows only and cell 128 lines 5-6 transform the validation and test blocks with those training statistics. The scanner can see the operation but not which array it received, which is why this is a useful calibration case: the shape of the code looks like leakage, the data flow is clean.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_cf8cb6a83808 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 15:2-5' --reviewer <your name>
```

---

## 2. `rwx_cf4b451b8c8a`

- **Source**: github_repos
- **Repository**: rasbt/python-machine-learning-book-3rd-edition at `bf62496f44a92a21d23250d5ed13d2d2e617e0e2`
- **Path**: `ch10/ch10.ipynb`
- **URL**: https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/bf62496f44a92a21d23250d5ed13d2d2e617e0e2/ch10/ch10.ipynb
- **Licence**: MIT (permissive; redistribution allowed)
- **SHA-256**: `2b308b5901db768014e51f94f878efa4442f62b1fc1296639888158a1d7066f3`
- **Scanner signal**: high (high preprocess_before_split, high preprocess_before_split, low no_stratify, low no_stratify)

**Workflow.** Chapter 10 of a machine-learning textbook: simple and multiple linear regression on a one-column toy dataset, with a train/test split for the evaluation section and several curve-fitting illustrations that make no generalisation claim.

**Possible mechanism: `legitimate_lookalike`** (hypothesis confidence: low)

Both high findings point at cells 35:6-7, where StandardScaler is fitted on the entire toy dataset (X and y) to demonstrate gradient-descent scaling. The model that is actually evaluated is fitted in cell 56 on the raw, unscaled column produced by the cell 55 split, and scored in cell 62. So the flagged scaling never feeds a reported score, but a reviewer still has to decide whether an unused teaching block should be labelled.

**Evidence against, or alternative explanations.**

- cell 55:6 splits the raw data; cell 56:3 fits sklearn's LinearRegression on X_train, not on the scaled arrays
- cells 76-84 fit polynomial and log transforms on all rows, but present no train/test metric at all - they are curve-shape illustrations
- the dataset is a two-column teaching example (rooms vs price), so the practical effect of any scaling choice is nil

**Questions the reviewer must resolve.**

1. Is a scaling block that is never used for evaluation a false positive, or does intent not matter for the label?
2. If recorded as legitimate, does the rationale need to name cell 55-62 as the evaluation path so the judgement is not mistaken for approval of cell 35?
3. Cell 92 splits again for the random-forest demo - is one clean evaluation path enough for the whole notebook?

**Code references.**

- `cell 35:4-7` - the flagged whole-dataset scaling for the gradient-descent demo
- `cell 55:1-6` - the actual train/test split
- `cell 56:3-5` - fit and predict on the split, unscaled
- `cell 62:4-9` - train/test MSE and R2 reported from the split model

```python
# cell 35:4-7
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

# cell 55:1-6
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# cell 56:3-5
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# cell 62:4-9
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))

```

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. Both high findings point at cells 35:6-7, where StandardScaler is fitted on the entire toy dataset (X and y) to demonstrate gradient-descent scaling. The model that is actually evaluated is fitted in cell 56 on the raw, unscaled column produced by the cell 55 split, and scored in cell 62. So the flagged scaling never feeds a reported score, but a reviewer still has to decide whether an unused teaching block should be labelled.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_cf4b451b8c8a \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 35:4-7' --reviewer <your name>
```

---

## 3. `rwx_88f14155caaa`

- **Source**: github_repos
- **Repository**: alirezadir/Machine-Learning-Interviews at `9ee330857b953e6e95376a901f8b12cd89bd2234`
- **Path**: `cn/src/MLC/notebooks/feedforward.ipynb`
- **URL**: https://github.com/alirezadir/Machine-Learning-Interviews/blob/9ee330857b953e6e95376a901f8b12cd89bd2234/cn/src/MLC/notebooks/feedforward.ipynb
- **Licence**: MIT (permissive; redistribution allowed)
- **SHA-256**: `31fd2ba721a4a15cd37e3d61b9fe0a161d193a42b1717f4c52920414751fa35a`
- **Scanner signal**: high (low no_stratify, high manual_scale_before_split)

**Workflow.** A from-scratch NumPy neural network on a small tabular dataset: a hand-written forward/backward pass, an explicit train/test split, and per-column standardisation using training statistics.

**Possible mechanism: `legitimate_lookalike`** (hypothesis confidence: low)

The high finding at cell 22:29 flags x - np.max(x, axis=1, keepdims=True) as scaling from all rows. That line is the max-subtraction stability trick inside a softmax, applied per row along axis=1, so it cannot carry information between rows. The real scaling is at cell 24:12-15, which computes mean and std from X_train after the cell 24:9 split and applies them to both blocks.

**Evidence against, or alternative explanations.**

- axis=1 reductions act within a single row, so no dataset-level statistic is involved
- cell 24:9 split precedes cell 24:12-15 fit/apply, which is the correct order
- only 26 cells: there is no later block that re-splits or re-uses the test rows

**Questions the reviewer must resolve.**

1. Should manual_scale_before_split be understood as row-wise when the reduction axis is 1? Record the observation for the detector's backlog - do not change the detector in this mission.
2. Is the whole notebook legitimate as a teaching implementation of a network, given it has no cross-validation at all?

**Code references.**

- `cell 22:29` - flagged line: per-row softmax max-subtraction
- `cell 24:9` - the train/test split
- `cell 24:12-15` - training-only mean and std applied to both blocks

```python
# cell 22:29
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))

# cell 24:9
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cell 24:12-15
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

```

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The high finding at cell 22:29 flags x - np.max(x, axis=1, keepdims=True) as scaling from all rows. That line is the max-subtraction stability trick inside a softmax, applied per row along axis=1, so it cannot carry information between rows. The real scaling is at cell 24:12-15, which computes mean and std from X_train after the cell 24:9 split and applies them to both blocks.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_88f14155caaa \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 22:29' --reviewer <your name>
```

---

## 4. `rwx_032348a621ed`

- **Source**: github_repos
- **Repository**: mrdbourke/zero-to-mastery-ml at `81492352d12d7a52caef57bba7744cbdc34af33f`
- **Path**: `section-2-data-science-and-ml-tools/introduction-to-scikit-learn-OLD.ipynb`
- **URL**: https://github.com/mrdbourke/zero-to-mastery-ml/blob/81492352d12d7a52caef57bba7744cbdc34af33f/section-2-data-science-and-ml-tools/introduction-to-scikit-learn-OLD.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `941e3c3e77c9f31dd84c2acb6ac27917e11f874c6ea1e4504c121d45a594f773`
- **Scanner signal**: medium (low no_random_state, low no_stratify, low no_random_state, low no_stratify)

**Workflow.** A long introductory scikit-learn walkthrough: heart-disease classification, car-price regression, imputation of missing car data, and finally a Pipeline that wraps preprocessing and model together.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: medium)

Cell 49 fits a ColumnTransformer containing OneHotEncoder on the whole car frame and the transformed matrix is what cell 56 splits. Cells 61 and 75 do the same with an imputer on the missing-data frame, so the imputation statistics are computed over rows that later become test rows. These are the classic preprocessor-before-split shapes, but the notebook then demonstrates the correct pattern at cells 90-94 and again with a full Pipeline at cell 288.

**Evidence against, or alternative explanations.**

- the encoder in cell 49 is one-hot over Make/Colour/Doors: it carries no target statistic, so its practical contribution is close to zero
- cells 90:7-8 show fit_transform(X_train) followed by transform(X_test), i.e. the notebook teaches the correct order a few cells later
- cell 288 builds a single Pipeline for imputer + encoder + model, which is the safest form in the notebook
- no result in the notebook is presented as a scientific claim; it is numbered coursework

**Questions the reviewer must resolve.**

1. Does a deliberately wrong-then-right teaching sequence count as legitimate pedagogy, or as potential leakage because the earlier cells still compute a test-informed transform?
2. Is the line between cell 49 (nominal one-hot, no target statistic) and cell 61 (mean imputation of numeric columns) worth drawing in the label?
3. Several splits in this notebook omit random_state - is irreproducibility part of this review's scope or a separate concern?

**Code references.**

- `cell 49:6-11` - ColumnTransformer fitted on the full car frame before the split
- `cell 56:3` - the split that consumes the pre-transformed matrix
- `cell 61:7-12` - full-frame imputation of the missing-data frame
- `cell 90:6-8` - the correct pattern demonstrated later in the same notebook

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_032348a621ed`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cell 49 fits a ColumnTransformer containing OneHotEncoder on the whole car frame and the transformed matrix is what cell 56 splits. Cells 61 and 75 do the same with an imputer on the missing-data frame, so the imputation statistics are computed over rows that later become test rows. These are the classic preprocessor-before-split shapes, but the notebook then demonstrates the correct pattern at cells 90-94 and again with a full Pipeline at cell 288.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_032348a621ed \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 49:6-11' --reviewer <your name>
```

---

## 5. `rwx_e9e72e2f1f81`

- **Source**: github_repos
- **Repository**: codebasics/py at `3ee4bde332ae7a10499c2679900094fa22ae191f`
- **Path**: `DataScience/BangloreHomePrices/model/banglore_home_prices_final.ipynb`
- **URL**: https://github.com/codebasics/py/blob/3ee4bde332ae7a10499c2679900094fa22ae191f/DataScience/BangloreHomePrices/model/banglore_home_prices_final.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `135bc20efe7d62180e226086c55eab14ae769db8240dd1d955a8b521661c2c21`
- **Scanner signal**: medium (low get_dummies_before_split, low no_stratify, medium tuning_on_full_data, low notebook_out_of_order_execution)

**Workflow.** Bengaluru house-price regression: outlier trimming, one-hot location encoding, a single train/test split with a linear baseline, then a helper that runs GridSearchCV over several models on the full feature matrix.

**Possible mechanism: `cv_contamination`** (hypothesis confidence: medium)

Cell 90 splits off 20% and cell 91 fits and scores a linear model. Cell 96 then runs GridSearchCV over the entire X, y - the same rows used as the test set in cell 91 - and returns a best estimator. From the notebook alone a reviewer cannot tell whether the reported test score was produced before or after that tuning, and the flagged finding is exactly that ambiguity. Cell 81 also one-hot encodes location over the full frame before the split.

**Evidence against, or alternative explanations.**

- cell 90-91 does hold out 20% before the first fit and score, which is the correct starting order
- the grid search sits in a separate helper whose output is a model comparison table, not a re-scored test metric
- location dummies carry no target statistic, so the encoding concern is minor on its own
- the notebook was executed out of order, so cell order is weak evidence for what fed what

**Questions the reviewer must resolve.**

1. Was cell 96 the final model-selection step behind the numbers the author reports, or an exploratory side branch?
2. Given the out-of-order execution count, is cell order admissible evidence here, or should the decision stay potential/uncertain?
3. Is tuning on all rows a confirmed leak when the notebook also contains a clean holdout earlier?

**Code references.**

- `cell 90:1-2` - the 20% holdout with a fixed random_state
- `cell 91:3-4` - linear model fitted and scored on the holdout
- `cell 96:32-33` - GridSearchCV fitted on the full X, y
- `cell 81:1` - full-frame location dummies before the split

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_e9e72e2f1f81`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cell 90 splits off 20% and cell 91 fits and scores a linear model. Cell 96 then runs GridSearchCV over the entire X, y - the same rows used as the test set in cell 91 - and returns a best estimator. From the notebook alone a reviewer cannot tell whether the reported test score was produced before or after that tuning, and the flagged finding is exactly that ambiguity. Cell 81 also one-hot encodes location over the full frame before the split.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_e9e72e2f1f81 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 90:1-2' --reviewer <your name>
```

---

## 6. `rwx_b1fa16d2c10a`

- **Source**: github_repos
- **Repository**: khanhnamle1994/cracking-the-data-science-interview at `2553974e7109224b05b8861681d2f52d7e338b29`
- **Path**: `EBooks/Machine-Learning-For-Finance/2 structured data.ipynb`
- **URL**: https://github.com/khanhnamle1994/cracking-the-data-science-interview/blob/2553974e7109224b05b8861681d2f52d7e338b29/EBooks/Machine-Learning-For-Finance/2 structured data.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `6a28b85ea6ac95d8b6b6fb300f57849a9678318f517391c8f1243cadb5d244e7`
- **Scanner signal**: low (low get_dummies_before_split, low no_stratify, low no_stratify, low notebook_out_of_order_execution)

**Workflow.** Fraud-detection notebook on a large transaction log: a hand-written rule, class imbalance handling with SMOTE on the training block, then tree, forest and boosted models scored on a held-out test block.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: low)

Cell 39 one-hot encodes the transaction type column over the whole frame before the cell 53 split. The notebook also evaluates a hand-built Fraud_Heuristic column against the isFraud labels on the same rows, which reads like a model result but is an in-sample rule check. Neither is a fitted preprocessing statistic leaking from test rows, but both are the kind of thing a reviewer should clear explicitly.

**Evidence against, or alternative explanations.**

- cell 56 calls SMOTE fit_sample on X_train only; the validation and test blocks are never resampled
- cell 54 splits validation out of the training block before any resampling
- the type column has four levels and encodes no target information
- the heuristic in cell 13 is an analyst rule, not a model that could have seen the test rows

**Questions the reviewer must resolve.**

1. Is the rule-versus-label comparison in cell 13 being presented as model evidence? If so, does that make the notebook misleading rather than leaky?
2. Does a nominal one-hot on four levels deserve any label at all, or is this legitimate with a note?

**Code references.**

- `cell 39:2` - full-frame one-hot of the type column
- `cell 53:1` - train/test split
- `cell 56:2` - SMOTE applied to the training block only
- `cell 13:1` - heuristic column scored against the labels in-sample

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_b1fa16d2c10a`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. Cell 39 one-hot encodes the transaction type column over the whole frame before the cell 53 split. The notebook also evaluates a hand-built Fraud_Heuristic column against the isFraud labels on the same rows, which reads like a model result but is an in-sample rule check. Neither is a fitted preprocessing statistic leaking from test rows, but both are the kind of thing a reviewer should clear explicitly.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_b1fa16d2c10a \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 39:2' --reviewer <your name>
```

---

## 7. `rwx_0e6934e5ce4f`

- **Source**: github_repos
- **Repository**: wandb/examples at `a03988fc6b6a41576956822bf241f5ae84422391`
- **Path**: `colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb`
- **URL**: https://github.com/wandb/examples/blob/a03988fc6b6a41576956822bf241f5ae84422391/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `9e25e00783150d899f17e52f0afce6c03a8f16d9c02eadac34829b1f4bf917ac`
- **Scanner signal**: low (low no_stratify)

**Workflow.** A Weights & Biases sweep example: one train() function that splits, fits an XGBoost model, evaluates on the test split and logs accuracy, invoked by wandb.agent for 25 trials.

**Possible mechanism: `test_set_reuse`** (hypothesis confidence: medium)

The sweep objective is the accuracy computed on X_test inside train(), and the agent runs it 25 times with different hyperparameters, so hyperparameter selection is performed on the same split that the notebook reports as its accuracy. Each individual fit only sees X_train, which is why the scanner's only finding is a missing stratify - the reuse is one level up, in the sweep driver.

**Evidence against, or alternative explanations.**

- each trial trains on X_train only and never resamples the test block
- train_test_split is called inside train() with no random_state, so the 25 trials do not share one fixed test set
- the notebook is product documentation for sweep mechanics, not a performance claim

**Questions the reviewer must resolve.**

1. Does model selection on the evaluation split meet the confirmed-leakage bar for a documentation example, or is potential the honest state?
2. Does the re-drawn split in every trial change that judgement?

**Code references.**

- `cell 16:26` - the split inside train()
- `cell 16:33-40` - fit on train, score on test, log that score as the sweep objective
- `cell 19:1` - wandb.agent running the objective 25 times

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_0e6934e5ce4f`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. The sweep objective is the accuracy computed on X_test inside train(), and the agent runs it 25 times with different hyperparameters, so hyperparameter selection is performed on the same split that the notebook reports as its accuracy. Each individual fit only sees X_train, which is why the scanner's only finding is a missing stratify - the reuse is one level up, in the sweep driver.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_0e6934e5ce4f \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 16:26' --reviewer <your name>
```

---

## 8. `rwx_5fa76ac23c47`

- **Source**: github_repos
- **Repository**: rasbt/machine-learning-book at `63bee951ca45420ac33ccf803ceb337afc3c0a91`
- **Path**: `ch08/logistic-regression-bag-of-words/log-reg.ipynb`
- **URL**: https://github.com/rasbt/machine-learning-book/blob/63bee951ca45420ac33ccf803ceb337afc3c0a91/ch08/logistic-regression-bag-of-words/log-reg.ipynb
- **Licence**: MIT (permissive; redistribution allowed)
- **SHA-256**: `165412cc15c6aecef252e2f92f516908c88bfbb8d2fbeee655ef95edfb1a0c08`
- **Scanner signal**: none (none)

**Workflow.** Bag-of-words sentiment classification on IMDb: the merged review frames are shuffled, rows 0-25000 become train and 25000-50000 test, and a Pipeline of TF-IDF plus logistic regression is tuned with GridSearchCV on the training rows.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

The scanner reports no findings and the inspected code is the recommended pattern: the vectoriser lives inside the Pipeline, so GridSearchCV refits vocabulary and IDF per fold on training data only. The one question that remains is the split itself - rows are taken from a shuffled merge of the dataset's own train and test frames rather than from the official folders, which makes near-duplicate reviews across the boundary more likely.

**Evidence against, or alternative explanations.**

- the TF-IDF vectoriser is a Pipeline step, so it is fitted inside each CV fold and on X_train only
- the test block is taken once by row index and never re-scored during tuning
- this is published textbook code whose evaluation order is a deliberate teaching point

**Questions the reviewer must resolve.**

1. Does the shuffle-then-slice split (cell 9) instead of the official train/test folders raise a duplicate-contamination question, or is it out of scope for a leakage label?
2. With no scanner finding at all, should this be recorded as legitimate so the batch documents a clean negative?

**Code references.**

- `cell 9:5` - shuffle of the merged review frames
- `cell 17:1-4` - row-index train/test split
- `cell 21:24-28` - TF-IDF plus classifier inside a Pipeline, tuned by GridSearchCV
- `cell 23:1` - grid fitted on X_train only

```python
# cell 9:5
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

# cell 17:1-4
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# cell 21:24-28
lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

# cell 23:1
gs_lr_tfidf.fit(X_train, y_train)

```

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The scanner reports no findings and the inspected code is the recommended pattern: the vectoriser lives inside the Pipeline, so GridSearchCV refits vocabulary and IDF per fold on training data only. The one question that remains is the split itself - rows are taken from a shuffled merge of the dataset's own train and test frames rather than from the official folders, which makes near-duplicate reviews across the boundary more likely.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_5fa76ac23c47 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 9:5' --reviewer <your name>
```

---

## 9. `rwx_d5c845943c82`

- **Source**: github_search
- **Repository**: DDDavides/MLProj at `8304dbe4e9e652e7138fa12c7ecc675495fb652f`
- **Path**: `NN.ipynb`
- **URL**: https://github.com/DDDavides/MLProj/blob/8304dbe4e9e652e7138fa12c7ecc675495fb652f/NN.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `ffcd9c1cfe3172937dc627931092e64f9352c30590ca19c2cc2452406ad5ba07`
- **Scanner signal**: high (low encoder_before_split, high preprocess_before_split, low no_random_state, low no_stratify)

**Workflow.** A short coursework notebook: load a tabular dataset, one-hot the target, mean-impute the features, split, then compare a random forest with a grid search and a small Keras network.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: high)

Cell 1 line 6 fits a SimpleImputer on the complete feature matrix and cell 2 splits the imputed result, so every training row's missing values are filled from column statistics that include the test rows. That is a genuine transductive step and the clearest example of the mechanism in this batch after the explicit test-file rescaling case.

**Evidence against, or alternative explanations.**

- the imputer uses feature means/medians, not the target, so the effect on a random forest is small
- the GridSearchCV in cell 5 is fitted on x_train only, so model selection never sees the test rows
- the file is 12 cells of coursework with no reported benchmark

**Questions the reviewer must resolve.**

1. Does a feature-only imputer fitted across the boundary clear the confirmed-leakage bar here, or is the small effect better recorded as potential?
2. Should the target one-hot at cell 1:5 be judged separately from the feature imputer at cell 1:6?

**Code references.**

- `cell 1:5-6` - target one-hot and full-frame feature imputation
- `cell 2:1` - the split that consumes the imputed matrix
- `cell 5:6-7` - grid search correctly fitted on the training block

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_d5c845943c82`._

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 1 line 6 fits a SimpleImputer on the complete feature matrix and cell 2 splits the imputed result, so every training row's missing values are filled from column statistics that include the test rows. That is a genuine transductive step and the clearest example of the mechanism in this batch after the explicit test-file rescaling case.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_d5c845943c82 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 1:5-6' --reviewer <your name>
```

---

## 10. `rwx_13ee93bec8f7`

- **Source**: github_search
- **Repository**: ashcode028/Music-Genre-Classification at `e9ab98e2f7269e0dc64b32dba12546e256ee5b45`
- **Path**: `NB_SGD_KNN.ipynb`
- **URL**: https://github.com/ashcode028/Music-Genre-Classification/blob/e9ab98e2f7269e0dc64b32dba12546e256ee5b45/NB_SGD_KNN.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `88e1727881e9299702aa0e618bc555ffa8150516233fbe6fb8073dbce462c0b1`
- **Scanner signal**: high (low encoder_before_split, high preprocess_before_split, low no_random_state, low no_stratify)

**Workflow.** Music-genre classification on the 3-second segment variant of GTZAN: per-segment spectral features, label encoding, scaling, chi-square feature selection, then KNN, naive Bayes and SGD compared with grid search.

**Possible mechanism: `group_entity_leakage`** (hypothesis confidence: high)

The file is features_3_sec.csv, the segment-level cut of GTZAN in which one recording contributes many near-identical rows; train_test_split at cell 5 splits rows, not recordings, so segments of the same track appear on both sides of the split and a nearest-neighbour model can recognise the recording rather than the genre. Cell 4 separately fits StandardScaler on the full feature frame before that split.

**Evidence against, or alternative explanations.**

- the chi-square selection at cell 5:5 is fitted on X_train only, which is the correct order
- the scaler fit at cell 4:3 is not obviously used to transform a held-out block differently from the training block
- GTZAN's segment-level benchmark is a common public setup with a known track-level protocol, so the reviewer may judge the notebook against community practice

**Questions the reviewer must resolve.**

1. Does the review treat segment-level contamination within the same recording as leakage for a public benchmark, and if so which state?
2. Can the reviewer confirm from the dataset documentation that each recording contributes multiple rows, so the group structure is not merely assumed?
3. Is the scaler fit in cell 4 independently enough for a confirmed verdict?

**Code references.**

- `cell 1:1` - reads the 3-second segment feature file
- `cell 4:3` - scaler fitted on the full feature frame
- `cell 5:2` - row-level split with no recording grouping
- `cell 5:5` - feature selection correctly fitted on X_train

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_13ee93bec8f7`._

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. The file is features_3_sec.csv, the segment-level cut of GTZAN in which one recording contributes many near-identical rows; train_test_split at cell 5 splits rows, not recordings, so segments of the same track appear on both sides of the split and a nearest-neighbour model can recognise the recording rather than the genre. Cell 4 separately fits StandardScaler on the full feature frame before that split.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_13ee93bec8f7 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 1:1' --reviewer <your name>
```

---

## 11. `rwx_102dfc6c7c9e`

- **Source**: github_search
- **Repository**: vermasrijan/srijan-gsoc-2020 at `97a3c8fb5494dbb79ad24d4f4bc3daa53c1f26d5`
- **Path**: `notebooks/GTEx_V8_NN-regression.ipynb`
- **URL**: https://github.com/vermasrijan/srijan-gsoc-2020/blob/97a3c8fb5494dbb79ad24d4f4bc3daa53c1f26d5/notebooks/GTEx_V8_NN-regression.ipynb
- **Licence**: MIT (permissive; redistribution allowed)
- **SHA-256**: `d816749848f400f3be094ece8ad1f3c0f7590dab8f1a2c9471a4b245d5e7c3cc`
- **Scanner signal**: high (high preprocess_before_split)

**Workflow.** A GTEx gene-expression experiment: standardise then min-max scale the whole expression matrix, attach tissue/sex dummies, and run 5-fold cross-validated Keras regression and classification models with no separate holdout.

**Possible mechanism: `cv_contamination`** (hypothesis confidence: high)

Cell 1 line 109 fits a two-step scaling pipeline on every row of the expression matrix, and cells 8 and 10 then build KFold(n_splits=5, shuffle=True) splits over those same rows. Each fold's validation rows therefore contributed to the mean, std, min and max used to transform that fold's training data, and because there is no holdout the cross-validated score is the only reported estimate.

**Evidence against, or alternative explanations.**

- the leaked quantity is scaling statistics, not target information
- tissue and sex encodings at cell 1 are nominal dummies with no target statistic
- the notebook is an exploration notebook from a summer project rather than a published result

**Questions the reviewer must resolve.**

1. With no holdout anywhere in the notebook, is CV contamination here confirmed or merely potential?
2. Would a reviewer require a re-run with the scaler inside the fold before confirming, or is the static evidence sufficient?

**Code references.**

- `cell 1:90-92` - scaler pipeline fitted on the full expression matrix
- `cell 1:109` - the flagged MinMaxScaler fit_transform
- `cell 8:4-13` - KFold over all rows with a shuffled random_state

```python
# cell 1:90-92
        steps = [('standardization', StandardScaler()), ('normalization', MinMaxScaler())]
        pre_processing_pipeline = Pipeline(steps)
        transformed_data = pre_processing_pipeline.fit_transform(x)

# cell 1:109
        x_scaled = min_max_scaler.fit_transform(x)

# cell 8:4-13
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

rmse = []
mae = []
r2 = []
huber_loss = []

for train, test in kfold.split(X, Y):
    model = optimized_age_model_regression(X[train], X[test], Y[train], Y[test])
    model.fit(X[train], Y[train], validation_data=[X[test], Y[test]], epochs=150, batch_size=256, verbose=1)

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 1 line 109 fits a two-step scaling pipeline on every row of the expression matrix, and cells 8 and 10 then build KFold(n_splits=5, shuffle=True) splits over those same rows. Each fold's validation rows therefore contributed to the mean, std, min and max used to transform that fold's training data, and because there is no holdout the cross-validated score is the only reported estimate.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_102dfc6c7c9e \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 1:90-92' --reviewer <your name>
```

---

## 12. `rwx_11f261a63388`

- **Source**: github_search
- **Repository**: pratush-mt/Project at `15b2ce7536016792bce863a43cb2e5d3912345f5`
- **Path**: `CS.ipynb`
- **URL**: https://github.com/pratush-mt/Project/blob/15b2ce7536016792bce863a43cb2e5d3912345f5/CS.ipynb
- **Licence**: Unlicense (permissive; redistribution allowed)
- **SHA-256**: `ad9d0bd005d1c176ea31780d574590ad44e6dd84d729e3c6ada1cf64b84e1a5c`
- **Scanner signal**: high (low encoder_before_split, high preprocess_before_split, high fit_on_eval_data, low no_stratify)

**Workflow.** Network-intrusion classification with separate competition train and test files: label encoding, RFE feature selection, feature scaling, a 70/30 internal split, then logistic regression, KNN, decision tree and a 10-fold CV comparison.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: high)

Cell 21 scales twice: line 2 fits a StandardScaler on the training features and line 3 fits a second, fresh scaler on the test file, so the held-out rows define the transformation applied to themselves and the two partitions end up on different scales. The same holdout is then scored five times across cells 33, 43, 50 and the CV loop, which is the reuse the scanner counts.

**Evidence against, or alternative explanations.**

- RFE selection at cell 19 is fitted on X_train before the internal split, which is the correct order
- the external test file carries no labels in the notebook, so the reported scores come from the internal 30% holdout
- refitting a scaler on consistent feature columns is a mild transform when the same rows both fit and transform

**Questions the reviewer must resolve.**

1. Is the double fit_transform at cell 21 the decisive evidence, or the five scores on one holdout?
2. Is the judgeable partition the internal 30% split at cell 22, or the external test file?
3. Does the parse error at cell 36 matter for a notebook used as evidence?

**Code references.**

- `cell 21:1-3` - scaler fitted on train, then refitted on test
- `cell 22:1` - internal 70/30 split of the training file
- `cell 19:4` - RFE fitted on the training block only
- `cell 33:1` - the held-out block scored again

```python
# cell 21:1-3
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
test = scale.fit_transform(test)

# cell 22:1
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.70, random_state=2)

# cell 19:4
rfe = rfe.fit(X_train, Y_train)

# cell 33:1
lg_train, lg_test = lg_model.score(x_train , y_train), lg_model.score(x_test , y_test)

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 21 scales twice: line 2 fits a StandardScaler on the training features and line 3 fits a second, fresh scaler on the test file, so the held-out rows define the transformation applied to themselves and the two partitions end up on different scales. The same holdout is then scored five times across cells 33, 43, 50 and the CV loop, which is the reuse the scanner counts.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_11f261a63388 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 21:1-3' --reviewer <your name>
```

---

## 13. `rwx_9c204d7611a6`

- **Source**: github_search
- **Repository**: MichalKorzycki/PythonDataScience at `86d47b266ee389da2ca7ee4737bcacd4fe76b939`
- **Path**: `14.MachineLearning_2z5_Regresja.ipynb`
- **URL**: https://github.com/MichalKorzycki/PythonDataScience/blob/86d47b266ee389da2ca7ee4737bcacd4fe76b939/14.MachineLearning_2z5_Regresja.ipynb
- **Licence**: Unlicense (permissive; redistribution allowed)
- **SHA-256**: `c4f409c77140fe0b3904640d0bed47350f9cde090c442acaa8571217fbe20a4b`
- **Scanner signal**: high (low no_stratify, medium fit_on_full_data, low no_stratify, medium fit_on_full_data)

**Workflow.** Polish-language course notebook: SMS spam classification with TF-IDF and naive Bayes, then a property-price regression section with label/one-hot encoding and a cross-validated linear model.

**Possible mechanism: `cv_contamination`** (hypothesis confidence: medium)

The scanner reports TfidfVectorizer.fit_transform(data) as fitting on the full dataset, but data is x_train.tolist() from cell 18 line 3, so that particular claim does not hold - the vectoriser is fitted on training messages and applied to the test block. What does hold is cell 32: a 5-fold CV over a matrix whose vocabulary and IDF weights were fitted on all of that matrix's rows, which is a real if small contamination of the CV estimate.

**Evidence against, or alternative explanations.**

- cell 18:10-11 fits the vectoriser on the training messages and only transforms the test block
- the quantity fitted outside the fold is corpus-level IDF and vocabulary, not target information
- cells 51-52 report an in-sample regression score without presenting it as a held-out estimate
- the advert section's label and one-hot encoders transform nominal categories only

**Questions the reviewer must resolve.**

1. Should the fit_on_full_data finding on cell 18 be recorded as a scanner false positive in the rationale?
2. Does IDF fitted outside the fold qualify as confirmed leakage for a text classifier, or potential?
3. Is the in-sample regression score at cells 51-52 a separate concern worth a second label?

**Code references.**

- `cell 18:3` - data is the training messages only
- `cell 18:10-11` - vectoriser fitted on train, applied to test
- `cell 32:1` - cross_val_score over a matrix transformed outside the folds
- `cell 51:6` - regression fitted and scored on all advert rows

```python
# cell 18:3
data = x_train.tolist()

# cell 18:10-11
features_train_transformed = vectorizer.fit_transform(data)  
features_test_transformed  = vectorizer.transform(x_test) 

# cell 32:1
scores = cross_val_score(MultinomialNB(), features_train_transformed, y_train, cv=5)

# cell 51:6
reg = LinearRegression().fit(X, y)

```

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. The scanner reports TfidfVectorizer.fit_transform(data) as fitting on the full dataset, but data is x_train.tolist() from cell 18 line 3, so that particular claim does not hold - the vectoriser is fitted on training messages and applied to the test block. What does hold is cell 32: a 5-fold CV over a matrix whose vocabulary and IDF weights were fitted on all of that matrix's rows, which is a real if small contamination of the CV estimate.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_9c204d7611a6 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 18:3' --reviewer <your name>
```

---

## 14. `rwx_b65af046ffc5`

- **Source**: github_search
- **Repository**: brainiak/brainiak-tutorials at `fb62ede943d9694fe703aee0df5f43ecf5558415`
- **Path**: `tutorials/04-dimensionality-reduction.ipynb`
- **URL**: https://github.com/brainiak/brainiak-tutorials/blob/fb62ede943d9694fe703aee0df5f43ecf5558415/tutorials/04-dimensionality-reduction.ipynb
- **Licence**: Apache-2.0 (permissive; redistribution allowed)
- **SHA-256**: `db6ce57919d2a71934d553e81f4ebb95ee45adbdadc00d7e76d0f456918d60d7`
- **Scanner signal**: high (high preprocess_before_split, high preprocess_before_split)

**Workflow.** fMRI decoding tutorial: load per-run BOLD data and labels, demonstrate a whole-brain baseline, then a PCA baseline, and finish with a Pipeline combined with PredefinedSplit over run ids.

**Possible mechanism: `cv_contamination`** (hypothesis confidence: high)

Cells 17 and 32 fit PCA on every run and then cross-validate on the reduced data, so the component basis is learned from the same runs that the folds hold out. Because fMRI runs are the natural group here, this is both a preprocessing-before-CV problem and a group-structure problem, and the tutorial's own closing cells show the correct version.

**Evidence against, or alternative explanations.**

- cells 39-41 use a Pipeline and PredefinedSplit(run_ids), so the notebook ends by teaching run-level, pipeline-contained evaluation
- PCA statistics are unsupervised and carry no target information
- the flagged cells are explicitly baseline comparisons in a tutorial about why the baseline is wrong

**Questions the reviewer must resolve.**

1. Should the reviewer judge the whole notebook (which teaches the correct pattern) or only the scored baseline cells?
2. Does the presence of PredefinedSplit show the authors understood group structure, which may argue for legitimate?
3. Is a documentary statement of intent admissible, or should the label describe only what the cells compute?

**Code references.**

- `cell 17:4` - PCA fitted on all runs
- `cell 31:3-11` - cross-validation on the pre-reduced data
- `cell 32:3` - second PCA fit on all runs
- `cell 39:3-10` - the correct Pipeline plus PredefinedSplit version

```python
# cell 17:4
bold_pca = pca.fit_transform(bold_data)

# cell 31:3-11
# Get baseline, whole-brain decoding accuracy without PCA
print('Baseline classification')
print('Original size: ', bold_normalized.shape)
svc = SVC(kernel="linear", C=1)

start = time()
models, scores = decode(bold_normalized, labels, run_ids, svc)
end = time()
print('Accuracy: ', scores)

# cell 32:3
bold_pca_normalized = pca.fit_transform(bold_normalized)

# cell 39:3-10
pipe = Pipeline([
    ('reduce_dim', PCA(n_components=20)),
    ('classify', SVC(kernel="linear", C=1)),
])

# Run the pipeline with cross-validation
ps = PredefinedSplit(run_ids) # Provides train/test indices to split data in train/test sets
clf_pipe = cross_validate(
    pipe,bold_normalized,labels,cv=ps,
    return_train_score=True

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cells 17 and 32 fit PCA on every run and then cross-validate on the reduced data, so the component basis is learned from the same runs that the folds hold out. Because fMRI runs are the natural group here, this is both a preprocessing-before-CV problem and a group-structure problem, and the tutorial's own closing cells show the correct version.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_b65af046ffc5 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 17:4' --reviewer <your name>
```

---

## 15. `rwx_2d4cbad5fb67`

- **Source**: github_search
- **Repository**: flatiron-school/ds-k-nearest_neighbors-kbo33 at `f616154a6738126bbf39e8347d81cc86981af42b`
- **Path**: `knn.ipynb`
- **URL**: https://github.com/flatiron-school/ds-k-nearest_neighbors-kbo33/blob/f616154a6738126bbf39e8347d81cc86981af42b/knn.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `c7790b439e09ea10e6a3ef91be47d373cec2fc9c2c93d635de670eaa4faac59e`
- **Scanner signal**: medium (low encoder_before_split, low no_stratify, low no_stratify, low no_stratify)

**Workflow.** A KNN lesson: iris as a toy demonstration, then Titanic with train/test and train/validation splits, scaling demonstrations, a k-fold loop for choosing k, and a final scaled fit and test score.

**Possible mechanism: `legitimate_lookalike`** (hypothesis confidence: low)

Every fit_on_full_data finding names X_t or X_train, which are the training partitions created at cells 43-45, not the full dataset; the finding text cannot tell a training partition from the whole frame. Cell 102 then demonstrates the canonical pair: fit_transform on X_train, transform on X_test.

**Evidence against, or alternative explanations.**

- cell 43:1 and cell 45:1 define X_test and X_val before X_t is used
- cell 94 fits a fresh scaler inside each CV fold, on the training fold only
- cell 102:4-8 fits on X_train and calls transform for X_test
- the notebook is lesson material whose whole arc is showing the correct scaler pattern

**Questions the reviewer must resolve.**

1. Do the teaching cells that fit separate scalers for train and validation (cells 85, 89) matter once the final score uses the correct pattern?
2. Should a fit_on_full_data finding be treated as a false positive whenever its argument is a training partition?

**Code references.**

- `cell 43:1` - train/test split
- `cell 45:1` - train/validation split: X_t is a training partition
- `cell 94:16-22` - scaler refitted inside each CV fold
- `cell 102:4-8` - fit on train, transform on test

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_2d4cbad5fb67`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. Every fit_on_full_data finding names X_t or X_train, which are the training partitions created at cells 43-45, not the full dataset; the finding text cannot tell a training partition from the whole frame. Cell 102 then demonstrates the canonical pair: fit_transform on X_train, transform on X_test.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2d4cbad5fb67 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 43:1' --reviewer <your name>
```

---

## 16. `rwx_73a6de31b37b`

- **Source**: github_search
- **Repository**: PrasoonPratham/Titanic-Challenge at `20a7d72b305fe064ebfc93be41c7e84e05ffb027`
- **Path**: `main.ipynb`
- **URL**: https://github.com/PrasoonPratham/Titanic-Challenge/blob/20a7d72b305fe064ebfc93be41c7e84e05ffb027/main.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `fdc6bb0aea0f452e689e96eb8d65d78b065c66ae4a94a4af72920b040356628a`
- **Scanner signal**: medium (low no_random_state, low no_stratify, medium fillna_before_split, medium fillna_before_split)

**Workflow.** A first Titanic submission: median-fill Age and Embarked, split the training file into fit and validation blocks, fit a decision tree, score it, then predict the Kaggle test file and write a submission.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: medium)

Cells 10 and 12 fill Age and Embarked with medians computed over every training row, and only then does cell 14 split that frame into trainData and valData, so validation-row values contribute to the imputation. The reported accuracy at cell 17 is therefore optimistic relative to a clean split.

**Evidence against, or alternative explanations.**

- the imputed columns are a small part of the frame and the model is an untuned decision tree
- the external test file at cell 25 is never scored locally, so no submission information re-enters the fit
- the notebook has no hyperparameter search, so the only reuse is the single validation score

**Questions the reviewer must resolve.**

1. Does median-filling roughly 177 missing ages before the split rise above potential?
2. Is a first-submission script worth a scored label at all, or should the batch record it as legitimate-with-caveat?

**Code references.**

- `cell 10:1` - Age median fill over all training rows
- `cell 12:1` - Embarked median fill over all training rows
- `cell 14:2` - the split that should have come first
- `cell 17:1` - validation accuracy reported after the leakage-prone fill

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_73a6de31b37b`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cells 10 and 12 fill Age and Embarked with medians computed over every training row, and only then does cell 14 split that frame into trainData and valData, so validation-row values contribute to the imputation. The reported accuracy at cell 17 is therefore optimistic relative to a clean split.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_73a6de31b37b \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 10:1' --reviewer <your name>
```

---

## 17. `rwx_73ed47531b70`

- **Source**: github_search
- **Repository**: Chaan0210/study-ai at `61de17c65e65661edfb3071c891b0d2819a329ab`
- **Path**: `datamining/datamining_project.ipynb`
- **URL**: https://github.com/Chaan0210/study-ai/blob/61de17c65e65661edfb3071c891b0d2819a329ab/datamining/datamining_project.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `ab69c74676e8d3ae25ad794698bada4f55f73d540a45b96a516b85316dcaff13`
- **Scanner signal**: medium (low get_dummies_before_split, low encoder_before_split, low no_stratify, low no_stratify)

**Workflow.** Startup-valuation regression: log transforms, categorical encoding, group-relative features, outlier clipping, then a grid-searched linear/ridge/lasso comparison and a random-forest sweep over four preprocessing variants.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: medium)

The group medians flagged at cell 2 lines 32-33 are medians of Growth Rate and Investment Amount, not of the target Valuation (USD), so the target-encoding shape the finding suggests does not apply. What does apply is that remove_outliers and the encoders are applied to the whole frame before the split inside the cell 5 loop, so quantile thresholds for clipping are computed with the test rows included.

**Evidence against, or alternative explanations.**

- cell 5:20-21 defines the target as Valuation (USD), which is not one of the grouped columns
- the group-relative features are scale-free ratios, so leakage through them would require them to encode the target
- the model search in cell 5:30 is fitted on X_tr only, after the split
- clipping thresholds are a robust statistic; their influence on a tree ensemble is small

**Questions the reviewer must resolve.**

1. Is the outlier clipping at cell 2 and cell 5 (via remove_outliers on the full frame) enough for a confirmed verdict, or does the split-first model search make the notebook legitimate apart from that step?
2. Should the group-relative feature construction be recorded as a false positive of group_stat_before_split, since it groups a feature rather than the target?

**Code references.**

- `cell 2:32-33` - group medians of feature columns, not of the target
- `cell 5:19-21` - split performed after full-frame preprocessing, target is Valuation (USD)
- `cell 5:30` - grid search fitted on the training block only
- `cell 7:35-42` - random-forest variant repeats the same split-then-fit order

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_73ed47531b70`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. The group medians flagged at cell 2 lines 32-33 are medians of Growth Rate and Investment Amount, not of the target Valuation (USD), so the target-encoding shape the finding suggests does not apply. What does apply is that remove_outliers and the encoders are applied to the whole frame before the split inside the cell 5 loop, so quantile thresholds for clipping are computed with the test rows included.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_73ed47531b70 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 2:32-33' --reviewer <your name>
```

---

## 18. `rwx_c2ae52031c08`

- **Source**: github_search
- **Repository**: stdereka/CourseraML at `49ebf7ac31bf88c11b983b69f1672d469b8250e8`
- **Path**: `NB.ipynb`
- **URL**: https://github.com/stdereka/CourseraML/blob/49ebf7ac31bf88c11b983b69f1672d469b8250e8/NB.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `639a945f616e6051b9627d6c369e7b7e9470c19437038def919069f84f920de9`
- **Scanner signal**: low (low parse_error)

**Workflow.** A five-cell scikit-learn exercise: load the breast-cancer and digits datasets built into scikit-learn, then compare three naive Bayes variants with cross-validation.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

There is nothing to leak: the datasets are bundled with scikit-learn and every statement is a cross_val_score call with default folds. The only finding is a parse error, because the notebook uses Python 2 print statements and the removed sklearn.cross_validation module, so it cannot run on any current environment.

**Evidence against, or alternative explanations.**

- no train/test split, no fitted preprocessing step, and no reused holdout exists in the notebook
- cross_val_score over a bundled dataset is the textbook minimal example
- the parse errors mean the code as stored is not even executable today

**Questions the reviewer must resolve.**

1. Should a non-executable Python 2 notebook be recorded as legitimate, or excluded from the corpus with a reason?
2. Does a five-cell snippet belong in a review batch at all, or is it only here to keep the low-signal quota honest?

**Code references.**

- `cell 1:2` - import from the removed sklearn.cross_validation module
- `cell 2:1-3` - Python 2 print statements that fail to parse
- `cell 3:1-3` - cross-validation on the bundled datasets

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_c2ae52031c08`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. There is nothing to leak: the datasets are bundled with scikit-learn and every statement is a cross_val_score call with default folds. The only finding is a parse error, because the notebook uses Python 2 print statements and the removed sklearn.cross_validation module, so it cannot run on any current environment.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_c2ae52031c08 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 1:2' --reviewer <your name>
```

---

## 19. `rwx_2ecc75f87a32`

- **Source**: github_search
- **Repository**: beritzellerplumhoff/DataScience_MSF_CDMA at `45795c3c0edb9e2850123a39f1fbb2eb03fd8028`
- **Path**: `PCA_kmeans/NN_BP_PCA_kmeans_empty.ipynb`
- **URL**: https://github.com/beritzellerplumhoff/DataScience_MSF_CDMA/blob/45795c3c0edb9e2850123a39f1fbb2eb03fd8028/PCA_kmeans/NN_BP_PCA_kmeans_empty.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `9b14f512adbba35447e64ba65cd56f132720459d2400159f5234962c8ece9884`
- **Scanner signal**: low (low no_random_state, low no_stratify)

**Workflow.** Iris exercise on the built-in dataset: split, fit a naive Bayes classifier, plot a partial_fit learning curve, and separately run PCA with k-means for visualisation.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

The split at cell 7 precedes every fit, and the k-means block at cell 33 is an unsupervised visualisation over the iris matrix rather than an evaluation of the classifier. The only findings are the missing random_state and stratify, both reproducibility concerns rather than leakage.

**Evidence against, or alternative explanations.**

- cell 7:6 splits before cell 11:3 fits the classifier
- the partial_fit loop at cell 13 evaluates on X_test but never trains on it
- the k-means fit at cell 33 is unsupervised and feeds a scatter plot, not a score

**Questions the reviewer must resolve.**

1. Do the missing random_state and stratify belong in this review at all, given the review is about leakage?
2. Is the Clusterer visualisation block worth a separate look, or clearly out of scope?

**Code references.**

- `cell 7:6` - split on the bundled iris data
- `cell 11:3` - classifier fitted on the training block
- `cell 33:15` - k-means used for a visualisation only

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_2ecc75f87a32`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The split at cell 7 precedes every fit, and the k-means block at cell 33 is an unsupervised visualisation over the iris matrix rather than an evaluation of the classifier. The only findings are the missing random_state and stratify, both reproducibility concerns rather than leakage.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2ecc75f87a32 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 7:6' --reviewer <your name>
```

---

## 20. `rwx_934050e90066`

- **Source**: github_search
- **Repository**: tuhinsharma121/deeplearning at `9f50a4a3cd5e721ae0e2a3b18b34bf332da177bc`
- **Path**: `CNN.ipynb`
- **URL**: https://github.com/tuhinsharma121/deeplearning/blob/9f50a4a3cd5e721ae0e2a3b18b34bf332da177bc/CNN.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `267b6820db81cdc3cafb120922a15ee8739363a0b2e83909cb92a1b2a8d16149`
- **Scanner signal**: low (low notebook_out_of_order_execution)

**Workflow.** A Keras CNN on a pre-split image dataset stored as separate train and test directories, with a label encoder fitted on the training labels and three successive training runs.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

The train/test boundary is defined by the dataset's directory layout, not by a scikit-learn call, and the label encoder at cell 8 is fitted on train_label only. The single finding is an out-of-order execution count, which is a provenance smell rather than a leakage mechanism.

**Evidence against, or alternative explanations.**

- cell 8:3 fits the encoder on the training labels alone
- the test arrays are only used in compile/evaluate blocks after each fit
- the out-of-order execution count affects reproducibility of the stored outputs, not the data flow of the code

**Questions the reviewer must resolve.**

1. Is an out-of-order execution count enough to make a dossier uncertain, given the code itself is clean?
2. Does the notebook train on validation_split=0.1 of the training set - is that split visible in the code or hidden in Keras?

**Code references.**

- `cell 8:3` - label encoder fitted on training labels
- `cell 18:1` - fit on train_X with an internal validation split
- `cell 21:1-3` - evaluation on the held-out directory

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_934050e90066`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The train/test boundary is defined by the dataset's directory layout, not by a scikit-learn call, and the label encoder at cell 8 is fitted on train_label only. The single finding is an out-of-order execution count, which is a provenance smell rather than a leakage mechanism.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_934050e90066 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 8:3' --reviewer <your name>
```

---

## 21. `rwx_fe6685882b36`

- **Source**: github_search
- **Repository**: samanyougarg/Group-Emotion-Recognition at `19118ffa91bed3c0522d6eaced03c23182c4cbc0`
- **Path**: `CNN.ipynb`
- **URL**: https://github.com/samanyougarg/Group-Emotion-Recognition/blob/19118ffa91bed3c0522d6eaced03c23182c4cbc0/CNN.ipynb
- **Licence**: GPL-3.0 (copyleft; references only, no code text)
- **SHA-256**: `809047529ed64495e33c377560badec06bae4158d5bb4c9dd5d5a6390e89ef2d`
- **Scanner signal**: low (low encoder_before_split, low no_stratify, low notebook_out_of_order_execution)

**Workflow.** Group-emotion recognition with Keras: label encoding, an 80/20 split, an ImageDataGenerator augmented only on the training block, a transfer-learning CNN, then 400 epochs with evaluation on the test block.

**Possible mechanism: `test_set_reuse`** (hypothesis confidence: medium)

Cell 59:11-13 passes validation_data=(X_test, y_test) to fit_generator for 400 epochs alongside callbacks, so whatever the callbacks and the author select (best epoch, best weights loaded at cell 49) is chosen by looking at the test block. Each fit still trains on X_train only, so the reuse is in model selection rather than in fitting.

**Evidence against, or alternative explanations.**

- cell 53:12 fits the augmentation generator on X_train only, which keeps augmentation out of the test path
- cell 30:6 label-encodes the label column before the split but with no target statistic
- the notebook is a class project with no reported benchmark

**Questions the reviewer must resolve.**

1. Do the callbacks in cell 59 select a checkpoint by test loss? If so, is that confirmed test-set reuse or potential?
2. Does loading pretrained weights at cell 49 change anything about the split?

**Code references.**

- `cell 30:6` - label encoding before the split
- `cell 32:1` - the 80/20 split
- `cell 53:2-12` - augmentation generator fitted on the training block
- `cell 59:11-13` - training with the test block as validation data

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_fe6685882b36`._

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cell 59:11-13 passes validation_data=(X_test, y_test) to fit_generator for 400 epochs alongside callbacks, so whatever the callbacks and the author select (best epoch, best weights loaded at cell 49) is chosen by looking at the test block. Each fit still trains on X_train only, so the reuse is in model selection rather than in fitting.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_fe6685882b36 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 30:6' --reviewer <your name>
```

---

## 22. `rwx_19b2c543d74f`

- **Source**: github_search
- **Repository**: martin-he543/novozymes-enzyme-stability-prediction-kaggle-2022 at `b878472b8a061c2b1277852d7cfb55274e22cde6`
- **Path**: `demo.ipynb`
- **URL**: https://github.com/martin-he543/novozymes-enzyme-stability-prediction-kaggle-2022/blob/b878472b8a061c2b1277852d7cfb55274e22cde6/demo.ipynb
- **Licence**: unknown (unknown; references only, no code text)
- **SHA-256**: `6f2d2715401f5ff0d831584cabc33c6565922773526edadb160895c2f81dac8a`
- **Scanner signal**: none (none)

**Workflow.** A single-cell Kaggle demo notebook for an enzyme-stability competition: imports plotting and metric helpers and defines nothing else.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

There is no modelling code to inspect. The file is one cell of imports, so no split, no fit and no evaluation exists; the scanner reports nothing and there is nothing a reviewer could confirm.

**Evidence against, or alternative explanations.**

- the notebook has exactly one cell, containing imports only
- no dataset is read and no model is fitted

**Questions the reviewer must resolve.**

1. Should a one-cell demo be recorded as legitimate, or excluded from the corpus as non-substantive?
2. Is an unexplained absence of any workflow a reason for uncertain rather than legitimate?

**Code references.**

- `cell 0:9` - plotly import helper
- `cell 0:24` - metric imports, no modelling code

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_19b2c543d74f`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. There is no modelling code to inspect. The file is one cell of imports, so no split, no fit and no evaluation exists; the scanner reports nothing and there is nothing a reviewer could confirm.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_19b2c543d74f \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 0:9' --reviewer <your name>
```

---

## 23. `rwx_46e12ee43c76`

- **Source**: github_search
- **Repository**: learn-co-curriculum/dsc-cross-validation-lab at `9ac04e45629233e2551fd9100c909567d0de53b0`
- **Path**: `index.ipynb`
- **URL**: https://github.com/learn-co-curriculum/dsc-cross-validation-lab/blob/9ac04e45629233e2551fd9100c909567d0de53b0/index.ipynb
- **Licence**: NOASSERTION (unknown; references only, no code text)
- **SHA-256**: `e74e0aa8c0b816a857761c325776843da9b7bd7f3692608d733aec2889e4e8f3`
- **Scanner signal**: none (none)

**Workflow.** A cross-validation lab exercise: Ames housing data, one-hot encoded categoricals, and a kfolds() template whose bodies are left as None for the student to fill in.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

The workbook ships with the answers removed, so the dataset-splitting calls appear only in comments and the fold loop assigns X_train = None, X_test = None. With no executed split and no fitted model, there is no mechanism to assess - the interesting review question is whether an empty template belongs in a leakage corpus at all.

**Evidence against, or alternative explanations.**

- cells 26-27 assign None to the train and test partitions rather than computing them
- the only real transformation, one-hot encoding the categoricals, carries no target statistic
- no model is fitted anywhere in the notebook

**Questions the reviewer must resolve.**

1. Should answer-stripped lab templates be excluded from the queue with a stated reason rather than labelled?
2. If labelled, is legitimate the honest state for a notebook with no computation?

**Code references.**

- `cell 2:27` - one-hot encoding of categorical columns
- `cell 26:6-10` - fold split placeholders left as None
- `cell 27:15` - evaluation placeholder never executed

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show rwx_46e12ee43c76`._

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The workbook ships with the answers removed, so the dataset-splitting calls appear only in comments and the fold loop assigns X_train = None, X_test = None. With no executed split and no fitted model, there is no mechanism to assess - the interesting review question is whether an empty template belongs in a leakage corpus at all.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_46e12ee43c76 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 2:27' --reviewer <your name>
```

---

## 24. `rwx_6bfa5ed1946c`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-41941594 at `HEAD`
- **Path**: `code.csv#kernel=41941594`
- **URL**: https://www.kaggle.com/kernels/41941594
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `df6d8c31cb61a9e6e2b12dc89933a43fd4eee47393364b98d3b9f962f31bcc5e`
- **Scanner signal**: high (high preprocess_before_split, high preprocess_before_split, low encoder_before_split, low encoder_before_split)

**Workflow.** A Kaggle datathon notebook on road-segment average speed: OSM tag extraction, datetime and holiday features, haversine distance, sine/cosine time encodings, label encoding, then cross-validated models refitted on all rows before predicting the competition test set.

**Possible mechanism: `cv_contamination`** (hypothesis confidence: high)

Cell 44 runs cross_val_score(model, X, y, cv=5) over the full training frame and cell 44 also refits model on all of X before predicting; there is no held-out split at that point, so the CV estimate is the only evidence and it is produced by a model later trained on the same rows. The function-transformers flagged at cell 26 are fitted on train and applied to test, which is the correct order, but a series-level split only appears much later at cells 52 and 65.

**Evidence against, or alternative explanations.**

- cell 26:5-6 fits the sine/cosine transformers on the training frame and cell 26:10-11 only transforms the test frame
- the notebook does contain a series-level split (cells 52 and 65) grouping by road segment ids, which is the group-aware pattern
- no single hyperparameter value is reported as final; the notebook ends with a submission prediction

**Questions the reviewer must resolve.**

1. Given the series-level split exists later, is the earlier full-frame cross_val_score exploratory or does it produce the reported estimate?
2. Is refitting on all rows after CV a legitimate final-model step for a competition submission, or test_set_reuse?
3. Are the parse errors at cells 37 and 55 material to the evidence?

**Code references.**

- `cell 26:5-11` - transformers fitted on train, applied to test
- `cell 44:31-37` - cross_val_score on all rows, then refit on all rows
- `cell 52:13` - series-level split of training data
- `cell 65:37` - series-level split repeated in the final function

```python
# cell 26:5-11
    df[f'{column_name}_sin'] = sin_transformer.fit_transform(df[[column_name]])
    df[f'{column_name}_cos'] = cos_transformer.fit_transform(df[[column_name]])
    
#     test_df[f'{column_name}_sin'] = sin_transformer.transform(test_df[[column_name]])
#     test_df[f'{column_name}_cos'] = cos_transformer.transform(test_df[[column_name]])
    test[f'{column_name}_sin'] = sin_transformer.transform(test[[column_name]])
    test[f'{column_name}_cos'] = cos_transformer.transform(test[[column_name]])

# cell 44:31-37
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    avg_cv_score = np.mean(-cv_scores)
    std_cv_score = np.std(-cv_scores)
    model.fit(X,y)
    elapsed_time = time.time() - start_time
    
    y_pred = model.predict(test2)

# cell 52:13
    train_series_data, test_series_data = train_test_split(series_data, test_size=test_percentage, random_state=42)

# cell 65:37
        train_series_data, test_series_data = train_test_split(series_data, test_size=test_percentage, random_state=random_state)

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 44 runs cross_val_score(model, X, y, cv=5) over the full training frame and cell 44 also refits model on all of X before predicting; there is no held-out split at that point, so the CV estimate is the only evidence and it is produced by a model later trained on the same rows. The function-transformers flagged at cell 26 are fitted on train and applied to test, which is the correct order, but a series-level split only appears much later at cells 52 and 65.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_6bfa5ed1946c \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 26:5-11' --reviewer <your name>
```

---

## 25. `rwx_26cf9b2d1280`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-36442984 at `HEAD`
- **Path**: `code.csv#kernel=36442984`
- **URL**: https://www.kaggle.com/kernels/36442984
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `bbeb0403ead23d0d88c96ff486ef4494237b82cda0069ccf8c007290a682371a`
- **Scanner signal**: high (low get_dummies_before_split, low get_dummies_before_split, high preprocess_before_split, high preprocess_before_split)

**Workflow.** Hitters salary regression in the style of a Turkish data-science course: dummies, min-max scaling, KNN imputation of the salary column, outlier clipping, then tree and ensemble models with grid search.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: high)

Cell 13 min-max scales the whole frame - including the Salary column that becomes the target - and cell 14 fits a KNNImputer over all rows to fill those salaries, so the target values of the training rows are produced from a neighbourhood that includes the rows later held out. The split only happens at cell 23, after the imputed target has been written back at cell 15 and the original Salary column dropped at cell 16.

**Evidence against, or alternative explanations.**

- cell 16 drops the original Salary column and cell 23 keeps only the imputed copy as the target, so the six-cell target leak of a repeated column does not occur
- the target is an aggregate salary statistic, and KNN imputation uses feature-space neighbourhoods rather than labels
- every model search at cells 39, 40 and 42 is fitted on X_train only after the split

**Questions the reviewer must resolve.**

1. This is a 710-row dataset; do you treat transductive target imputation across the boundary as confirmed leakage, or as potential given the small fraction of missing salaries?
2. Does the RobustScaler at cell 35 fit on the full frame change the answer, or is it the same mechanism repeated?
3. Is the target being left in 0-1 scaled units a separate data-quality problem worth noting in the rationale?

**Code references.**

- `cell 13:3-6` - min-max scaling of the frame including Salary
- `cell 14:2-3` - KNN imputer fitted over all rows
- `cell 15:1` - imputed values written back as the future target
- `cell 23:2-4` - split performed after imputation
- `cell 35:5` - full-frame robust scaling repeated later

```python
# cell 13:3-6
dff=pd.get_dummies(df[cat_cols+num_cols],drop_first=True)
#Standardization
scaler=MinMaxScaler()
dff=pd.DataFrame(scaler.fit_transform(dff),columns=dff.columns)

# cell 14:2-3
imputer=KNNImputer(n_neighbors=5)
dff=pd.DataFrame(imputer.fit_transform(dff),columns=dff.columns)

# cell 15:1
df["Salary_Imputed_Knn"]=dff[["Salary"]] #To see the change of missing values in the data.

# cell 23:2-4
X=df.drop("Salary_Imputed_Knn",axis=1)
y=df[["Salary_Imputed_Knn"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=17)

# cell 35:5
    dff[col]=RobustScaler().fit_transform(dff[[col]])

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 13 min-max scales the whole frame - including the Salary column that becomes the target - and cell 14 fits a KNNImputer over all rows to fill those salaries, so the target values of the training rows are produced from a neighbourhood that includes the rows later held out. The split only happens at cell 23, after the imputed target has been written back at cell 15 and the original Salary column dropped at cell 16.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_26cf9b2d1280 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 13:3-6' --reviewer <your name>
```

---

## 26. `rwx_2a33fe770c62`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-25227697 at `HEAD`
- **Path**: `code.csv#kernel=25227697`
- **URL**: https://www.kaggle.com/kernels/25227697
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `4b6ef533fe90bfecb6b8c0927c671b49978ffe1d27f2632fc2d6742f97c65041`
- **Scanner signal**: medium (low get_dummies_before_split, low get_dummies_before_split, low no_stratify, medium fillna_before_split)

**Workflow.** Titanic classification in a Kaggle-style script: train and test files appended, Sex and Embarked dummied, Age and Fare mean-filled, then a 67/33 split, standardisation, a grid search and repeated scoring.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: high)

Cell 12 appends the test frame to the training frame, cells 19 and 22 mean-fill Age and Fare over the combined frame, and only at cell 37 is the combined frame split. The imputation statistics therefore include the rows that become the test block, and the appended frame also means the pipeline was designed around the two files together rather than around a clean holdout.

**Evidence against, or alternative explanations.**

- cell 40:3-4 standardises with a scaler fitted on X_train and applied to X_test, the correct order despite the earlier frame-level work
- the imputed columns are Age and Fare, and the mean of either is a weak statistic on a 1,300-row frame
- the dummies at cells 25 and 28 are nominal encodings of Sex and Embarked with no target statistic

**Questions the reviewer must resolve.**

1. Does appending the official test file before imputation (cell 12) make this a stronger case than a plain fillna-before-split?
2. Is the correct scaler at cell 40 enough to record legitimate, or does the frame-level imputation stand?

**Code references.**

- `cell 12:1` - test frame appended to the training frame
- `cell 19:1` - Age mean fill over the combined frame
- `cell 22:1` - Fare mean fill over the combined frame
- `cell 37:2` - the split, after imputation
- `cell 40:3-4` - scaling correctly fitted on train only

```python
# cell 12:1
df_final = df_train.append(df_test)

# cell 19:1
df_final['Age'] = df_final['Age'].fillna(df_final['Age'].mean())

# cell 22:1
df_final['Fare'] = df_final['Fare'].fillna(df_final['Fare'].mean())

# cell 37:2
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

# cell 40:3-4
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Cell 12 appends the test frame to the training frame, cells 19 and 22 mean-fill Age and Fare over the combined frame, and only at cell 37 is the combined frame split. The imputation statistics therefore include the rows that become the test block, and the appended frame also means the pipeline was designed around the two files together rather than around a clean holdout.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2a33fe770c62 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 12:1' --reviewer <your name>
```

---

## 27. `rwx_1c064ba9af63`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-36301788 at `HEAD`
- **Path**: `code.csv#kernel=36301788`
- **URL**: https://www.kaggle.com/kernels/36301788
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `f5b5d512806a14cf291a920297c3c51b35a337a5608ef497bc8cca0c2b79658f`
- **Scanner signal**: medium (low get_dummies_before_split, low no_stratify, low no_stratify, medium tuning_on_full_data)

**Workflow.** Istanbul reservoir-occupancy regression: two weather CSVs sorted by date, missing solar and UV values imputed with linear regressions, seasonal dummies, then a random 80/20 split and several ensemble regressors compared by MSE.

**Possible mechanism: `temporal_leakage`** (hypothesis confidence: high)

Both frames are sorted by date and the time column is parsed, but cell 49 splits with train_test_split and no shuffle=False or ordering, so observations from the same period appear on both sides of the boundary. The solar-energy and UV-index imputations at cells 30 and 32 also fit a linear regression over the full series, and cell 60 grid-searches on the training frame. Time-ordered environmental data with rows randomly mixed is the temporal mechanism the scanner did not flag.

**Evidence against, or alternative explanations.**

- the target is reservoir occupancy, an aggregated daily measurement, so a leaked neighbouring day carries less information than a duplicated row
- cells 2 and 4 do sort by date, showing the author thought about time ordering
- the notebook is exploratory and compares models rather than publishing a held-out score

**Questions the reviewer must resolve.**

1. Do you accept a temporal read that the scanner did not flag, based on cells 2, 4 and 49?
2. Is the register-level target stationary enough that random splitting is acceptable practice here?
3. Should the two linear-regression imputations of solar energy and UV index be considered part of the same label or a separate one?

**Code references.**

- `cell 2:9` - occupancy frame sorted by date
- `cell 30:8-15` - solar energy imputed by a regression over the full series
- `cell 49:7` - random split of time-ordered observations
- `cell 60:9-13` - grid search and scoring after the random split

```python
# cell 2:9
baraj_doluluk.sort_values("Tarih", axis = 0, ascending = True,
                 inplace = True, na_position ='first')

# cell 30:8-15
X_train = baraj_doluluk[['solarradiation']][~baraj_doluluk['solarenergy'].isnull()]
y_train = baraj_doluluk['solarenergy'][~baraj_doluluk['solarenergy'].isnull()]

# Modeli eğitin
reg = LinearRegression().fit(X_train, y_train)

# Eksik değerleri tahmin edin
missing_df['solarenergy'] = reg.predict(missing_df[['solarradiation']])

# cell 49:7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cell 60:9-13
voting_reg = GridSearchCV(voting_reg, param_grid=params,  scoring='neg_mean_squared_error', n_jobs=-1)

voting_model = voting_reg.fit(X_train, y_train)

y_pred_Voting = voting_model.predict(X_test)

```

**AI hypothesis (unverified - not ground truth).** high confidence, author `ai_unverified`. Both frames are sorted by date and the time column is parsed, but cell 49 splits with train_test_split and no shuffle=False or ordering, so observations from the same period appear on both sides of the boundary. The solar-energy and UV-index imputations at cells 30 and 32 also fit a linear regression over the full series, and cell 60 grid-searches on the training frame. Time-ordered environmental data with rows randomly mixed is the temporal mechanism the scanner did not flag.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_1c064ba9af63 \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 2:9' --reviewer <your name>
```

---

## 28. `rwx_d44c61fa976b`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-40709602 at `HEAD`
- **Path**: `code.csv#kernel=40709602`
- **URL**: https://www.kaggle.com/kernels/40709602
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `a166ecd3b2ffb11c3719bca939e134d9ed483316f32039d3ce1e5fbfdda7a06d`
- **Scanner signal**: low (low no_stratify, low preprocess_outside_cv)

**Workflow.** ICR age-related-conditions competition notebook: separate train/test files, median fill, random oversampling of the training block, KNN imputation fitted on train, standardisation, a grid search and an ensemble.

**Possible mechanism: `none_observed`** (hypothesis confidence: low)

The pipeline is close to the recommended order: cell 10 splits before any resampling, cell 11 oversamples the training block only, and cell 12 fits the KNN imputer on train and transforms validation and test. The only findings are a missing stratify and a scaler fitted outside the grid search but on training rows only. The notable defect is structural: cells 16 and 21 score X_val_norm, which is never defined.

**Evidence against, or alternative explanations.**

- cell 11:6 resamples X_train only, leaving the validation block untouched
- cell 12:7-9 fit the imputer on train and transform the other blocks
- cell 14:9-11 fit the scaler on X_train and transform X_test
- the imputed medians at cell 10 come from X_train, not the combined frame

**Questions the reviewer must resolve.**

1. Cells 16 and 21 reference X_val_norm, which the notebook never defines - does that make the reported validation numbers unreproducible and therefore uncertain?
2. Is the scaler fitted outside GridSearchCV (cell 14) worth a finding when it uses training rows only?

**Code references.**

- `cell 10:7-15` - split first, then median fill from training statistics
- `cell 11:5-6` - oversampling of the training block only
- `cell 12:6-9` - imputer fitted on train, applied to validation and test
- `cell 16:4` - references X_val_norm, which is never defined

```python
# cell 10:7-15
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.1, random_state=42)

# X_train['EJ'].fillna('B', inplace=True) # fillna with mode 
# X_val['EJ'].fillna('B', inplace=True)
# X_test['EJ'].fillna('B', inplace=True)

X_train.fillna(X_train.median(), inplace=True)
X_val.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)

# cell 11:5-6
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

# cell 12:6-9
imputer = KNNImputer(n_neighbors=5)
X_train[null_cols] = imputer.fit_transform(X_train[null_cols])
X_val[null_cols] = imputer.transform(X_val[null_cols])
X_test[null_cols] = imputer.transform(X_test[null_cols])

# cell 16:4
y_pred_val_proba = best_model.predict_proba(X_val_norm)[:, 1]

```

**AI hypothesis (unverified - not ground truth).** low confidence, author `ai_unverified`. The pipeline is close to the recommended order: cell 10 splits before any resampling, cell 11 oversamples the training block only, and cell 12 fits the KNN imputer on train and transforms validation and test. The only findings are a missing stratify and a scaler fitted outside the grid search but on training rows only. The notable defect is structural: cells 16 and 21 score X_val_norm, which is never defined.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_d44c61fa976b \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 10:7-15' --reviewer <your name>
```

---

## 29. `rwx_7aaf49ab3e5d`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-27437467 at `HEAD`
- **Path**: `code.csv#kernel=27437467`
- **URL**: https://www.kaggle.com/kernels/27437467
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `23ee94747ab943a6c801021991a0eb5a7c694e712cca8ee1ab6d346380fd3a9a`
- **Scanner signal**: low (low get_dummies_before_split, low no_stratify, low preprocess_outside_cv, low parse_error)

**Workflow.** Titanic on the titanic3 variant: Excel load, drop-first dummies, a 70/30 split, iterative imputation fitted on the training block, standardisation, then a score, k-fold CV on the concatenated train and test partitions, and a stacked model.

**Possible mechanism: `test_set_reuse`** (hypothesis confidence: medium)

Cell 17 splits and cells 21 and 27 do the imputation and scaling in the correct order, fitting on the training block. But cell 31 concatenates X_train with X_test and y_train with y_test and then cross-validates over the union at cell 32, so the block that had already been scored at cell 29 re-enters the evaluation used to compare models.

**Evidence against, or alternative explanations.**

- cell 21:9 fits the iterative imputer on X_train1 and only transforms X_test1
- cell 27:5-7 fits the scaler on X_train1 and transforms X_test1
- the concatenated CV is presented as a robustness check after the single score, not as the headline result
- the notebook stores commented-out expected accuracy values, suggesting it was written as an exercise

**Questions the reviewer must resolve.**

1. Is re-using the test partition inside a later CV estimate confirmed test_set_reuse, or an acceptable robustness check when the headline score is separate?
2. Do the parse errors in cells 0 and 1 affect the evidence?

**Code references.**

- `cell 17:1` - 70/30 split
- `cell 21:9-14` - imputation fitted on train only
- `cell 29:6-7` - first fit and score on the holdout
- `cell 31:2-3` - train and test partitions concatenated
- `cell 32:6-7` - cross-validation over the union

```python
# cell 17:1
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2210)

# cell 21:9-14
imputed = imputer.fit_transform(X_train1[cols])

X_train1.loc[:, cols] = imputed

imputed = imputer.transform(X_test1[cols])
X_test1.loc[:, cols] = imputed

# cell 29:6-7
bm.fit(X_train, y_train)
bm.score(X_test, y_test) 

# cell 31:2-3
X_kcv = pd.concat([X_train, X_test])
y_kcv = pd.concat([y_train, y_test])

# cell 32:6-7
    s = model_selection.cross_val_score(
        cls, X_kcv, y_kcv, scoring = "roc_auc", cv = kfold)

```

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cell 17 splits and cells 21 and 27 do the imputation and scaling in the correct order, fitting on the training block. But cell 31 concatenates X_train with X_test and y_train with y_test and then cross-validates over the union at cell 32, so the block that had already been scored at cell 29 re-enters the evaluation used to compare models.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_7aaf49ab3e5d \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 17:1' --reviewer <your name>
```

---

## 30. `rwx_69b642bc801b`

- **Source**: kaggle_distilkaggle
- **Repository**: kaggle/kernel-21994782 at `HEAD`
- **Path**: `code.csv#kernel=21994782`
- **URL**: https://www.kaggle.com/kernels/21994782
- **Licence**: CC-BY-4.0 (permissive; redistribution allowed)
- **SHA-256**: `03ecf959f74f94b68c97272d2d4e2569e9b97a9b9356dd83e95b2b71292eea09`
- **Scanner signal**: none (none)

**Workflow.** Titanic with engineered imputation: train and test frames appended, submission labels attached to the test rows, dummies, cross_val_predict over the combined frame, then a linear model that predicts missing ages before a final genuine train/test split.

**Possible mechanism: `preprocess_before_split`** (hypothesis confidence: medium)

Cell 59 builds the age model from the appended frame: X_train is every row with a known age and X_test every row with a missing one, so training and imputation both span the official boundary. The notebook also attaches the submission file's Survived column to the test rows at cell 8 and then cross-validates over the union at cells 54-56, which mixes the future test rows into model comparison.

**Evidence against, or alternative explanations.**

- cell 81 does finish with a genuine split by index and a clean classification report
- the imputed quantity is Age, a feature, not the target
- the attached Survived labels are only used for the in-notebook CV comparison, never as training rows for the final model at cell 81

**Questions the reviewer must resolve.**

1. Does the age-imputation model trained across the official boundary (cell 59) constitute leakage if the age values are only features?
2. Is the concatenated cross-validation at cells 54-56 a separate label from the imputation, or the same underlying boundary problem?
3. Is attaching the submission labels at cell 8 acceptable as a convenience, or does it invalidate the CV comparison?

**Code references.**

- `cell 8:1-2` - submission labels attached to the test rows
- `cell 54:12` - cross_val_predict over the combined frame
- `cell 59:1-4` - age model trained on known ages from both partitions
- `cell 81:3-10` - final split and clean train/test report

```python
# cell 8:1-2
df_test['Survived'] = df_sub['Survived']
df = df_train.append(df_test,ignore_index=False)

# cell 54:12
pred = cross_val_predict(logmodel, X, y, cv=30)

# cell 59:1-4
X_test = dfml2[dfml2['Age'].isna()].drop(['Survived','Age'],axis=1)
X_train = dfml2[(dfml2['Age'].isna())==False].drop('Survived',axis=1)
y_train = X_train['Age']
X_train = X_train.drop('Age',axis=1)

# cell 81:3-10
X_train = dfml_train.drop('Survived',axis=1)
y_train = dfml_train['Survived']

X_test = dfml_test.drop('Survived',axis=1)
y_test = dfml_test['Survived']

logmodel.fit(X_train, y_train)
pred = logmodel.predict(X_test)

```

**AI hypothesis (unverified - not ground truth).** medium confidence, author `ai_unverified`. Cell 59 builds the age model from the appended frame: X_train is every row with a known age and X_test every row with a missing one, so training and imputation both span the official boundary. The notebook also attaches the submission file's Survived column to the test rows at cell 8 and then cross-validates over the union at cells 54-56, which mixes the future test rows into model comparison.

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_69b642bc801b \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 8:1-2' --reviewer <your name>
```

---

## What happens to a decision

`review` appends one validated JSON line to `reviews.jsonl`. It never
rewrites `manifest.jsonl`, `review_queue.jsonl` or this document. The
decision log is the only place a label exists, and only
`confirmed_leakage` / `legitimate` rows are eligible for a scored set.
