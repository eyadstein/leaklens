"""Phase 7 development corpus: hard positives and hard negatives.

Motivation
----------
The Phase 5/6 corpora pair obvious leaks with obvious fixes.  A reliability
audit of the learned detector found the remaining errors concentrated in
*lookalikes*: scripts that contain the same surface constructs as a leak
(``scaler``, ``sort_values``, ``groupby``, ``shift(-1)``, ``KFold``) but wire
them correctly, and subtle leaks that never call ``fit`` on the full frame.

This module is **independently authored for Phase 7**.  No example is copied
from the Phase 5 locked test, the Phase 6 real-world holdout, the scanner
benchmarks or any upstream repository.  Every snippet is synthetic Python
written to expose one decisive construct.

Provenance and licence
----------------------
* Author: LeakLens maintainers, 2026.
* Licence: same as the project (MIT); the snippets are original text.
* Label source: manual review against the leakage definition in
  ``docs/AI_INTELLIGENCE.md``.  No model output was used to assign labels.
* Grouping: a ``group`` is one template family; every leaky/safe pair shares a
  group so grouped cross-validation never separates the pair.  A ``source`` is
  a domain/idiom family and lives entirely in one split.

Splits
------
``PHASE7_CASES`` (``split="train"``) is development data and is used for
fitting.  ``PHASE7_HOLDOUT_CASES`` (``split="holdout"``) is frozen before the
final model is fitted and is only scored by :mod:`leaklens.bench_ai`; it is
never imported by the training path.  Keeping both in one module makes the
family-level separation auditable.

Limitations
-----------
The corpus is small, synthetic and English-identifier biased.  It measures
whether a specific construct is recognised, not production accuracy.  A safe
label means "no leakage path is visible in this snippet under the documented
definition"; it is not a proof of correctness.
"""

from __future__ import annotations

from .corpus_types import HOLDOUT, TRAIN, CorpusCase

_SPLIT = "from sklearn.model_selection import train_test_split\n"

PHASE7_CASES: tuple[CorpusCase, ...] = (
    # ------------------------------------------------------------------
    # Ordering: a transformer can be present in both variants; only the
    # position relative to the fold boundary decides the label.
    # ------------------------------------------------------------------
    CorpusCase(
        "p7_scaler_fold_loop",
        "scaler fitted before a manual fold loop",
        "from sklearn.model_selection import KFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "scaled = StandardScaler().fit_transform(X)\n"
        "for train_idx, valid_idx in KFold(5, shuffle=True).split(scaled):\n"
        "    model.fit(scaled[train_idx], y[train_idx])\n",
        True,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_scaler_fold_loop",
        "scaler fitted inside a manual fold loop",
        "from sklearn.model_selection import KFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "for train_idx, valid_idx in KFold(5, shuffle=True).split(X):\n"
        "    scaler = StandardScaler().fit(X[train_idx])\n"
        "    model.fit(scaler.transform(X[train_idx]), y[train_idx])\n",
        False,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_pipeline_search",
        "full frame scaled before a hyperparameter search",
        "from sklearn.model_selection import GridSearchCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "scaled = StandardScaler().fit_transform(X)\n"
        "search = GridSearchCV(model, {'C': [0.1, 1]}, cv=4)\n"
        "search.fit(scaled, y)\n",
        True,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_pipeline_search",
        "hyperparameter search over a pipeline",
        "from sklearn.model_selection import GridSearchCV\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "pipe = make_pipeline(StandardScaler(), model)\n"
        "search = GridSearchCV(pipe, {'model__C': [0.1, 1]}, cv=4)\n"
        "search.fit(X, y)\n",
        False,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_onehot",
        "one hot encoding before the split",
        "import pandas as pd\n" + _SPLIT + "encoded = pd.get_dummies(frame)\n"
        "X_train, X_test, y_train, y_test = train_test_split(encoded, y)\n",
        True,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_onehot",
        "one hot encoder fitted on the training split",
        "from sklearn.preprocessing import OneHotEncoder\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)\n"
        "X_train = encoder.transform(X_train)\n"
        "X_test = encoder.transform(X_test)\n",
        False,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_imputer",
        "imputer fitted on the full frame before the split",
        "from sklearn.impute import SimpleImputer\n"
        + _SPLIT
        + "filled = SimpleImputer(strategy='median').fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(filled, y)\n",
        True,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_imputer",
        "imputer fitted on the training split only",
        "from sklearn.impute import SimpleImputer\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "imputer = SimpleImputer(strategy='median').fit(X_train)\n"
        "X_train = imputer.transform(X_train)\n"
        "X_test = imputer.transform(X_test)\n",
        False,
        "preprocessing",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_feature_select",
        "selector fitted with the labels before the split",
        "from sklearn.feature_selection import SelectKBest\n"
        + _SPLIT
        + "selected = SelectKBest(k=8).fit_transform(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(selected, y)\n",
        True,
        "selection",
        TRAIN,
        "train:p7-order",
    ),
    CorpusCase(
        "p7_feature_select",
        "selector kept inside a cross-validated pipeline",
        "from sklearn.feature_selection import SelectKBest\n"
        "from sklearn.model_selection import cross_val_score\n"
        "from sklearn.pipeline import make_pipeline\n"
        "pipe = make_pipeline(SelectKBest(k=8), model)\n"
        "scores = cross_val_score(pipe, X, y, cv=5)\n",
        False,
        "selection",
        TRAIN,
        "train:p7-order",
    ),
    # ------------------------------------------------------------------
    # Temporal lookalikes.  A ``sort_values`` call alone is not a leak.
    # ------------------------------------------------------------------
    CorpusCase(
        "p7_sort_report",
        "events sorted by timestamp then split with the default shuffle",
        "from sklearn.model_selection import train_test_split\n"
        "events = log.sort_values('timestamp')\n"
        "train, test = train_test_split(events, test_size=0.25, random_state=3)\n",
        True,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_sort_report",
        "rows sorted by amount for a report, then split normally",
        "from sklearn.model_selection import train_test_split\n"
        "report = log.sort_values('amount')\n"
        "train, test = train_test_split(log, test_size=0.25, random_state=3)\n",
        False,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_sort_after",
        "chronological sort before a random split",
        "from sklearn.model_selection import train_test_split\n"
        "ordered = visits.sort_values('visit_date')\n"
        "X_train, X_test, y_train, y_test = train_test_split(ordered, ordered['y'])\n",
        True,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_sort_after",
        "split first, then sorted only for the report",
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    visits, visits['y'], shuffle=False\n"
        ")\n"
        "report = X_test.sort_values('visit_date')\n",
        False,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_centered_window",
        "centered rolling window over the full series",
        "from sklearn.model_selection import train_test_split\n"
        "frame['smooth'] = frame['sales'].rolling(7, center=True).mean()\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['smooth']], frame['sales']\n"
        ")\n",
        True,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_centered_window",
        "trailing rolling window on past values only",
        "from sklearn.model_selection import train_test_split\n"
        "frame['smooth'] = frame['sales'].shift(1).rolling(7).mean()\n"
        "frame = frame.dropna()\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['smooth']], frame['sales'], shuffle=False\n"
        ")\n",
        False,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_kfold_time",
        "shuffled k-fold over calendar features",
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "X = frame[['month', 'orders']]\n"
        "scores = cross_val_score(model, X, frame['sales'], cv=KFold(5, shuffle=True))\n",
        True,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    CorpusCase(
        "p7_kfold_time",
        "time series split over the same calendar features",
        "from sklearn.model_selection import TimeSeriesSplit, cross_val_score\n"
        "X = frame[['month', 'orders']]\n"
        "scores = cross_val_score(model, X, frame['sales'], cv=TimeSeriesSplit(5))\n",
        False,
        "temporal",
        TRAIN,
        "train:p7-temporal",
    ),
    # ------------------------------------------------------------------
    # Entity groups: the identifier may be present, dropped, or used for
    # grouping; only repeated entities split across folds are a leak.
    # ------------------------------------------------------------------
    CorpusCase(
        "p7_group_feature",
        "repeated store rows split randomly with the id as a feature",
        "from sklearn.model_selection import train_test_split\n"
        "X = sales[['store_id', 'revenue', 'visits']]\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, sales['target'], random_state=2\n"
        ")\n",
        True,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_group_feature",
        "repeated store rows split by store group",
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "X = sales[['store_id', 'revenue', 'visits']]\n"
        "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)\n"
        "train_idx, test_idx = next(\n"
        "    splitter.split(X, sales['target'], groups=sales['store_id'])\n"
        ")\n",
        False,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_group_dedupe",
        "one row per customer kept before a random split",
        "from sklearn.model_selection import train_test_split\n"
        "unique = visits.drop_duplicates(subset=['customer_id', 'month'])\n"
        "features = unique.drop(columns=['customer_id'])\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    features, unique['churn'], random_state=4\n"
        ")\n",
        False,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_group_dedupe",
        "repeated customer rows split randomly after a cosmetic dedupe",
        "from sklearn.model_selection import train_test_split\n"
        "visits = visits.drop_duplicates()\n"
        "features = visits[['customer_id', 'spend', 'sessions']]\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    features, visits['churn'], random_state=4\n"
        ")\n",
        True,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_group_stats",
        "store level mean computed on all rows before the split",
        "from sklearn.model_selection import train_test_split\n"
        "sales['store_mean'] = sales.groupby('store_id')['revenue'].transform('mean')\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    sales[['store_mean', 'visits']], sales['target']\n"
        ")\n",
        True,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_group_stats",
        "store level mean computed from the training split only",
        "from sklearn.model_selection import GroupShuffleSplit\n"
        "splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=5)\n"
        "train_idx, test_idx = next(\n"
        "    splitter.split(sales, sales['target'], groups=sales['store_id'])\n"
        ")\n"
        "X_train = sales.iloc[train_idx].copy()\n"
        "means = X_train.groupby('store_id')['revenue'].transform('mean')\n"
        "X_train['store_mean'] = means\n",
        False,
        "grouping",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_test_concat",
        "test rows appended before the split",
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "pooled = pd.concat([train_rows, test_rows], ignore_index=True)\n"
        "X_train, X_test, y_train, y_test = train_test_split(pooled, pooled['y'])\n",
        True,
        "overlap",
        TRAIN,
        "train:p7-groups",
    ),
    CorpusCase(
        "p7_test_concat",
        "split first, predictions concatenated afterwards",
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(rows, rows['y'])\n"
        "predictions = pd.concat([\n"
        "    pd.Series(model.predict(X_train), index=X_train.index),\n"
        "    pd.Series(model.predict(X_test), index=X_test.index),\n"
        "])\n",
        False,
        "overlap",
        TRAIN,
        "train:p7-groups",
    ),
    # ------------------------------------------------------------------
    # Target handling: encoding or lagging the label is not the same as
    # turning the label into a feature.
    # ------------------------------------------------------------------
    CorpusCase(
        "p7_label_encode_target",
        "label encoder fitted on a categorical feature before the split",
        "from sklearn.preprocessing import LabelEncoder\n"
        + _SPLIT
        + "frame['city_code'] = LabelEncoder().fit_transform(frame['city'])\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['city_code', 'spend']], frame['churn']\n"
        ")\n",
        True,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_label_encode_target",
        "label encoder fitted on the target vector only",
        "from sklearn.preprocessing import LabelEncoder\n"
        + _SPLIT
        + "encoder = LabelEncoder()\n"
        "y = encoder.fit_transform(y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)\n",
        False,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_forward_lag",
        "next period demand used as a feature",
        "from sklearn.model_selection import train_test_split\n"
        "frame['next_demand'] = frame['demand'].shift(-1)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['next_demand', 'price']], frame['sales'], shuffle=False\n"
        ")\n",
        True,
        "temporal",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_forward_lag",
        "next period target used as the label and dropped from the features",
        "from sklearn.model_selection import train_test_split\n"
        "frame['next_sales'] = frame['sales'].shift(-1)\n"
        "frame = frame.dropna()\n"
        "features = frame.drop(columns=['next_sales', 'sales'])\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    features, frame['next_sales'], shuffle=False\n"
        ")\n",
        False,
        "temporal",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_target_bucket",
        "target quantiles turned into a feature before the split",
        "import pandas as pd\n"
        + _SPLIT
        + "frame['target_bucket'] = pd.qcut(frame['target'], 4, labels=False)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['target_bucket', 'spend']], frame['target']\n"
        ")\n",
        True,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_target_bucket",
        "target quantiles computed on the training split only",
        "import pandas as pd\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame, frame['target'], random_state=6\n"
        ")\n"
        "train_bucket = pd.qcut(X_train['target'], 4, labels=False)\n",
        False,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_frequency_encode",
        "frequency encoding from every row before the split",
        "from sklearn.model_selection import train_test_split\n"
        "counts = frame['city'].value_counts()\n"
        "frame['city_freq'] = frame['city'].map(counts)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['city_freq', 'spend']], frame['churn']\n"
        ")\n",
        True,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    CorpusCase(
        "p7_frequency_encode",
        "frequency encoding fitted on the training split",
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame, frame['churn'], random_state=7\n"
        ")\n"
        "counts = X_train['city'].value_counts()\n"
        "X_train['city_freq'] = X_train['city'].map(counts)\n"
        "X_test['city_freq'] = X_test['city'].map(counts).fillna(0)\n",
        False,
        "target",
        TRAIN,
        "train:p7-target",
    ),
    # ------------------------------------------------------------------
    # Summary statistics and cross-validation boundaries.
    # ------------------------------------------------------------------
    CorpusCase(
        "p7_stats_normalise",
        "feature standardised with full frame statistics",
        "from sklearn.model_selection import train_test_split\n"
        "frame['x'] = (frame['x'] - frame['x'].mean()) / frame['x'].std()\n"
        "X_train, X_test, y_train, y_test = train_test_split(frame, frame['target'])\n",
        True,
        "preprocessing",
        TRAIN,
        "train:p7-cv",
    ),
    CorpusCase(
        "p7_stats_normalise",
        "mean printed for inspection, scaler fitted after the split",
        "from sklearn.model_selection import train_test_split\n"
        "print(frame['x'].mean())\n"
        "X_train, X_test, y_train, y_test = train_test_split(frame, frame['target'])\n"
        "mean = X_train['x'].mean()\n"
        "X_train['x'] = X_train['x'] - mean\n"
        "X_test['x'] = X_test['x'] - mean\n",
        False,
        "preprocessing",
        TRAIN,
        "train:p7-cv",
    ),
    CorpusCase(
        "p7_cv_test",
        "cross validation repeated on the test partition",
        "from sklearn.model_selection import cross_val_score\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "scores = cross_val_score(model, X_test, y_test, cv=5)\n",
        True,
        "resampling",
        TRAIN,
        "train:p7-cv",
    ),
    CorpusCase(
        "p7_cv_test",
        "cross validation on the training partition only",
        "from sklearn.model_selection import cross_val_score\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
        "print(model.fit(X_train, y_train).score(X_test, y_test))\n",
        False,
        "resampling",
        TRAIN,
        "train:p7-cv",
    ),
    CorpusCase(
        "p7_tune_split",
        "hyperparameters tuned on the full frame before the split",
        "from sklearn.model_selection import GridSearchCV\n"
        + _SPLIT
        + "search = GridSearchCV(model, {'C': [0.1, 1]}, cv=4)\n"
        "search.fit(X, y)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        True,
        "tuning",
        TRAIN,
        "train:p7-cv",
    ),
    CorpusCase(
        "p7_tune_split",
        "hyperparameters tuned on the training split",
        "from sklearn.model_selection import GridSearchCV\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "search = GridSearchCV(model, {'C': [0.1, 1]}, cv=4)\n"
        "search.fit(X_train, y_train)\n",
        False,
        "tuning",
        TRAIN,
        "train:p7-cv",
    ),
)

PHASE7_HOLDOUT_CASES: tuple[CorpusCase, ...] = (
    CorpusCase(
        "p7h_scaler_folds",
        "scaler fitted on all rows before manual folds",
        "from sklearn.preprocessing import StandardScaler\n"
        "scaled = StandardScaler().fit_transform(readings)\n"
        "for fold in range(4):\n"
        "    start, stop = fold * 25, (fold + 1) * 25\n"
        "    model.fit(scaled[start:stop], labels[start:stop])\n",
        True,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-order",
    ),
    CorpusCase(
        "p7h_scaler_folds",
        "scaler fitted on each training fold",
        "from sklearn.preprocessing import StandardScaler\n"
        "for fold in range(4):\n"
        "    start, stop = fold * 25, (fold + 1) * 25\n"
        "    scaler = StandardScaler().fit(readings[start:stop])\n"
        "    model.fit(scaler.transform(readings[start:stop]), labels[start:stop])\n",
        False,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-order",
    ),
    CorpusCase(
        "p7h_sort_report",
        "sensor rows sorted by event time then randomly split",
        "from sklearn.model_selection import train_test_split\n"
        "ordered = sensors.sort_values('event_time')\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    ordered, ordered['failure'], random_state=8\n"
        ")\n",
        True,
        "temporal",
        HOLDOUT,
        "holdout:p7-temporal",
    ),
    CorpusCase(
        "p7h_sort_report",
        "sensor rows sorted by temperature, split normally",
        "from sklearn.model_selection import train_test_split\n"
        "report = sensors.sort_values('temperature')\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    sensors, sensors['failure'], random_state=8\n"
        ")\n",
        False,
        "temporal",
        HOLDOUT,
        "holdout:p7-temporal",
    ),
    CorpusCase(
        "p7h_group_kfold",
        "plain k-fold over repeated store rows with the id as a feature",
        "from sklearn.model_selection import KFold, cross_val_score\n"
        "features = visits[['store_id', 'basket', 'hour']]\n"
        "scores = cross_val_score(model, features, visits['spend'], cv=KFold(4))\n",
        True,
        "grouping",
        HOLDOUT,
        "holdout:p7-groups",
    ),
    CorpusCase(
        "p7h_group_kfold",
        "group k-fold that keeps every store in one fold",
        "from sklearn.model_selection import GroupKFold, cross_val_score\n"
        "features = visits[['store_id', 'basket', 'hour']]\n"
        "splitter = GroupKFold(4)\n"
        "scores = cross_val_score(\n"
        "    model, features, visits['spend'], cv=splitter, groups=visits['store_id']\n"
        ")\n",
        False,
        "grouping",
        HOLDOUT,
        "holdout:p7-groups",
    ),
    CorpusCase(
        "p7h_forward_lag",
        "next week demand used as a feature",
        "from sklearn.model_selection import train_test_split\n"
        "frame['next_week'] = frame['demand'].shift(-7)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    frame[['next_week', 'promo']], frame['orders'], shuffle=False\n"
        ")\n",
        True,
        "temporal",
        HOLDOUT,
        "holdout:p7-target",
    ),
    CorpusCase(
        "p7h_forward_lag",
        "next week demand used as the label and excluded from features",
        "from sklearn.model_selection import train_test_split\n"
        "frame['next_week'] = frame['demand'].shift(-7)\n"
        "frame = frame.dropna()\n"
        "features = frame.drop(columns=['next_week', 'demand'])\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    features, frame['next_week'], shuffle=False\n"
        ")\n",
        False,
        "temporal",
        HOLDOUT,
        "holdout:p7-target",
    ),
    CorpusCase(
        "p7h_eda_mean",
        "amount centred with the mean of every row",
        "from sklearn.model_selection import train_test_split\n"
        "frame['amount'] = frame['amount'] - frame['amount'].mean()\n"
        "X_train, X_test, y_train, y_test = train_test_split(frame, frame['fraud'])\n",
        True,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-stats",
    ),
    CorpusCase(
        "p7h_eda_mean",
        "mean displayed, centring applied to the training split",
        "from sklearn.model_selection import train_test_split\n"
        "print(frame['amount'].mean())\n"
        "X_train, X_test, y_train, y_test = train_test_split(frame, frame['fraud'])\n"
        "mean = X_train['amount'].mean()\n"
        "X_train['amount'] = X_train['amount'] - mean\n"
        "X_test['amount'] = X_test['amount'] - mean\n",
        False,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-stats",
    ),
    CorpusCase(
        "p7h_onehot",
        "dummy columns built from every row",
        "import pandas as pd\n"
        + _SPLIT
        + "table = pd.get_dummies(orders[['channel', 'region']])\n"
        "X_train, X_test, y_train, y_test = train_test_split(table, orders['y'])\n",
        True,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-order",
    ),
    CorpusCase(
        "p7h_onehot",
        "dummy encoder fitted after the split",
        "from sklearn.preprocessing import OneHotEncoder\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    orders[['channel', 'region']], orders['y']\n"
        ")\n"
        "encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)\n"
        "train_matrix = encoder.transform(X_train)\n"
        "test_matrix = encoder.transform(X_test)\n",
        False,
        "preprocessing",
        HOLDOUT,
        "holdout:p7-order",
    ),
)

__all__ = ["PHASE7_CASES", "PHASE7_HOLDOUT_CASES"]
