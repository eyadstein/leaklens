# Review batch 01 - blind pack

Second-reviewer material. Every scanner signal, AI hypothesis, workflow
summary, list of counter-evidence and reviewer question has been removed,
so nothing here states the expected answer. No verdict from the first
review is included either.

Decide each case from the cited source lines, then record it with
`--blind` and your own reviewer id:

```sh
python scripts/collect_expansion.py show --id <id> --blind
python scripts/collect_expansion.py review --id <id> --blind \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' --ref '<cell N:L>' \
  --reviewer <your name>
```

States: `confirmed_leakage` (a real leak), `legitimate` (correct
practice), `potential` (suspicious but unresolved), `uncertain` (cannot
tell from this source). Only the first two are scored. See
[`docs/REVIEW_GUIDE.md`](../docs/REVIEW_GUIDE.md).

Rationale template: what the code does at the cited lines, whether the
information crosses the split boundary, and why that is or is not a leak.

Code text is included only for permissive licences. For the rest, read it
with `show --blind`, which prints the cached source without commentary.

Cases: 30.

## Index

| # | Candidate | Repo | Licence |
|---|---|---|---|
| 1 | `rwx_cf8cb6a83808` | ageron/tf2_course | Apache-2.0 |
| 2 | `rwx_cf4b451b8c8a` | rasbt/python-machine-learning-book-3rd-edition | MIT |
| 3 | `rwx_88f14155caaa` | alirezadir/Machine-Learning-Interviews | MIT |
| 4 | `rwx_032348a621ed` | mrdbourke/zero-to-mastery-ml | unknown |
| 5 | `rwx_e9e72e2f1f81` | codebasics/py | unknown |
| 6 | `rwx_b1fa16d2c10a` | khanhnamle1994/cracking-the-data-science-interview | unknown |
| 7 | `rwx_0e6934e5ce4f` | wandb/examples | unknown |
| 8 | `rwx_5fa76ac23c47` | rasbt/machine-learning-book | MIT |
| 9 | `rwx_d5c845943c82` | DDDavides/MLProj | GPL-3.0 |
| 10 | `rwx_13ee93bec8f7` | ashcode028/Music-Genre-Classification | GPL-3.0 |
| 11 | `rwx_102dfc6c7c9e` | vermasrijan/srijan-gsoc-2020 | MIT |
| 12 | `rwx_11f261a63388` | pratush-mt/Project | Unlicense |
| 13 | `rwx_9c204d7611a6` | MichalKorzycki/PythonDataScience | Unlicense |
| 14 | `rwx_b65af046ffc5` | brainiak/brainiak-tutorials | Apache-2.0 |
| 15 | `rwx_2d4cbad5fb67` | flatiron-school/ds-k-nearest_neighbors-kbo33 | GPL-3.0 |
| 16 | `rwx_73a6de31b37b` | PrasoonPratham/Titanic-Challenge | GPL-3.0 |
| 17 | `rwx_73ed47531b70` | Chaan0210/study-ai | unknown |
| 18 | `rwx_c2ae52031c08` | stdereka/CourseraML | GPL-3.0 |
| 19 | `rwx_2ecc75f87a32` | beritzellerplumhoff/DataScience_MSF_CDMA | GPL-3.0 |
| 20 | `rwx_934050e90066` | tuhinsharma121/deeplearning | GPL-3.0 |
| 21 | `rwx_fe6685882b36` | samanyougarg/Group-Emotion-Recognition | GPL-3.0 |
| 22 | `rwx_19b2c543d74f` | martin-he543/novozymes-enzyme-stability-prediction-kaggle-2022 | unknown |
| 23 | `rwx_46e12ee43c76` | learn-co-curriculum/dsc-cross-validation-lab | NOASSERTION |
| 24 | `rwx_6bfa5ed1946c` | kaggle/kernel-41941594 | CC-BY-4.0 |
| 25 | `rwx_26cf9b2d1280` | kaggle/kernel-36442984 | CC-BY-4.0 |
| 26 | `rwx_2a33fe770c62` | kaggle/kernel-25227697 | CC-BY-4.0 |
| 27 | `rwx_1c064ba9af63` | kaggle/kernel-36301788 | CC-BY-4.0 |
| 28 | `rwx_d44c61fa976b` | kaggle/kernel-40709602 | CC-BY-4.0 |
| 29 | `rwx_7aaf49ab3e5d` | kaggle/kernel-27437467 | CC-BY-4.0 |
| 30 | `rwx_69b642bc801b` | kaggle/kernel-21994782 | CC-BY-4.0 |

---

## 1. `rwx_cf8cb6a83808`

- **Source**: github_repos
- **Repository**: ageron/tf2_course at `cb9a7263c76bbad302cd177835895992a556c4b9`
- **Path**: `01_neural_nets_with_keras.ipynb`
- **URL**: https://github.com/ageron/tf2_course/blob/cb9a7263c76bbad302cd177835895992a556c4b9/01_neural_nets_with_keras.ipynb
- **Licence**: Apache-2.0 (permissive; redistribution allowed)
- **SHA-256**: `c807380f33f5b646bf45b4058e42c7045f8697ac78727a605c48248bcad5af1b`

**Code references.**

- `cell 15:2-5`
- `cell 128:4`
- `cell 128:5-6`
- `cell 189:3-6`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_cf8cb6a83808 --blind \
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

**Code references.**

- `cell 35:4-7`
- `cell 55:1-6`
- `cell 56:3-5`
- `cell 62:4-9`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_cf4b451b8c8a --blind \
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

**Code references.**

- `cell 22:29`
- `cell 24:9`
- `cell 24:12-15`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_88f14155caaa --blind \
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

**Code references.**

- `cell 49:6-11`
- `cell 56:3`
- `cell 61:7-12`
- `cell 90:6-8`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_032348a621ed --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_032348a621ed --blind \
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

**Code references.**

- `cell 90:1-2`
- `cell 91:3-4`
- `cell 96:32-33`
- `cell 81:1`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_e9e72e2f1f81 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_e9e72e2f1f81 --blind \
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

**Code references.**

- `cell 39:2`
- `cell 53:1`
- `cell 56:2`
- `cell 13:1`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_b1fa16d2c10a --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_b1fa16d2c10a --blind \
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

**Code references.**

- `cell 16:26`
- `cell 16:33-40`
- `cell 19:1`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_0e6934e5ce4f --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_0e6934e5ce4f --blind \
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

**Code references.**

- `cell 9:5`
- `cell 17:1-4`
- `cell 21:24-28`
- `cell 23:1`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_5fa76ac23c47 --blind \
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

**Code references.**

- `cell 1:5-6`
- `cell 2:1`
- `cell 5:6-7`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_d5c845943c82 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_d5c845943c82 --blind \
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

**Code references.**

- `cell 1:1`
- `cell 4:3`
- `cell 5:2`
- `cell 5:5`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_13ee93bec8f7 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_13ee93bec8f7 --blind \
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

**Code references.**

- `cell 1:90-92`
- `cell 1:109`
- `cell 8:4-13`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_102dfc6c7c9e --blind \
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

**Code references.**

- `cell 21:1-3`
- `cell 22:1`
- `cell 19:4`
- `cell 33:1`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_11f261a63388 --blind \
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

**Code references.**

- `cell 18:3`
- `cell 18:10-11`
- `cell 32:1`
- `cell 51:6`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_9c204d7611a6 --blind \
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

**Code references.**

- `cell 17:4`
- `cell 31:3-11`
- `cell 32:3`
- `cell 39:3-10`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_b65af046ffc5 --blind \
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

**Code references.**

- `cell 43:1`
- `cell 45:1`
- `cell 94:16-22`
- `cell 102:4-8`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_2d4cbad5fb67 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2d4cbad5fb67 --blind \
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

**Code references.**

- `cell 10:1`
- `cell 12:1`
- `cell 14:2`
- `cell 17:1`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_73a6de31b37b --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_73a6de31b37b --blind \
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

**Code references.**

- `cell 2:32-33`
- `cell 5:19-21`
- `cell 5:30`
- `cell 7:35-42`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_73ed47531b70 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_73ed47531b70 --blind \
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

**Code references.**

- `cell 1:2`
- `cell 2:1-3`
- `cell 3:1-3`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_c2ae52031c08 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_c2ae52031c08 --blind \
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

**Code references.**

- `cell 7:6`
- `cell 11:3`
- `cell 33:15`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_2ecc75f87a32 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2ecc75f87a32 --blind \
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

**Code references.**

- `cell 8:3`
- `cell 18:1`
- `cell 21:1-3`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_934050e90066 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_934050e90066 --blind \
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

**Code references.**

- `cell 30:6`
- `cell 32:1`
- `cell 53:2-12`
- `cell 59:11-13`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_fe6685882b36 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_fe6685882b36 --blind \
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

**Code references.**

- `cell 0:9`
- `cell 0:24`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_19b2c543d74f --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_19b2c543d74f --blind \
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

**Code references.**

- `cell 2:27`
- `cell 26:6-10`
- `cell 27:15`

_Code text omitted: licence is not permissive. Run `python scripts/collect_expansion.py show --id rwx_46e12ee43c76 --blind`._

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_46e12ee43c76 --blind \
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

**Code references.**

- `cell 26:5-11`
- `cell 44:31-37`
- `cell 52:13`
- `cell 65:37`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_6bfa5ed1946c --blind \
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

**Code references.**

- `cell 13:3-6`
- `cell 14:2-3`
- `cell 15:1`
- `cell 23:2-4`
- `cell 35:5`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_26cf9b2d1280 --blind \
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

**Code references.**

- `cell 12:1`
- `cell 19:1`
- `cell 22:1`
- `cell 37:2`
- `cell 40:3-4`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_2a33fe770c62 --blind \
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

**Code references.**

- `cell 2:9`
- `cell 30:8-15`
- `cell 49:7`
- `cell 60:9-13`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_1c064ba9af63 --blind \
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

**Code references.**

- `cell 10:7-15`
- `cell 11:5-6`
- `cell 12:6-9`
- `cell 16:4`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_d44c61fa976b --blind \
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

**Code references.**

- `cell 17:1`
- `cell 21:9-14`
- `cell 29:6-7`
- `cell 31:2-3`
- `cell 32:6-7`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_7aaf49ab3e5d --blind \
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

**Code references.**

- `cell 8:1-2`
- `cell 54:12`
- `cell 59:1-4`
- `cell 81:3-10`

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

**Decision.** none recorded

```sh
python scripts/collect_expansion.py review --id rwx_69b642bc801b --blind \
  --state <confirmed_leakage|legitimate|potential|uncertain> \
  --rationale '<what you concluded and why>' \
  --ref 'cell 8:1-2' --reviewer <your name>
```

---

## What this pack does not establish

Producing it is not a review. Nothing here is a label, and no case becomes
`double_reviewed` until a distinct human records a `--blind` decision.
