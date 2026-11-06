"""Phase 6 development cases for real-world idioms.

The Phase 5 corpus covers scikit-learn objects and a few hand-written
aggregations.  Phase 6 adds paired leaky/safe cases for idioms that appear in
public notebooks and in pre-pipeline code:

* functional scalers (``sklearn.preprocessing.scale``) and hand-written
  normalisation such as ``X / X.max()``;
* full-frame ``fillna`` with ``mode``/``mean`` and ``get_dummies`` before a
  split;
* cross-validation and repeated scoring on the test partition;
* ``GroupKFold`` without a ``groups`` argument;
* frequency encoding from the whole frame;
* sorting by time and then calling ``train_test_split`` without
  ``shuffle=False``;
* transforms fitted per fold instead of once on the full frame;
* ``GridSearchCV`` over a pipeline and one-hot-only preprocessing;
* train-only transformers and positive target lags.

Each pair shares a template group so grouped cross-validation always scores the
leaky and safe variants together.  These cases are **development** data: they
are not in the Phase 5 locked test, and they are not the Phase 6 real-world
holdout.  They exist because the external error analysis showed the Phase 5
feature set could not name these patterns at all.
"""

from __future__ import annotations

from .corpus_types import TRAIN, CorpusCase

SOURCE = "train:phase6-idioms"
_SPLIT = "from sklearn.model_selection import train_test_split\n"


def _case(
    group: str,
    name: str,
    code: str,
    leaky: bool,
    category: str,
) -> CorpusCase:
    return CorpusCase(
        group=group,
        name=name,
        code=code,
        leaky=leaky,
        category=category,
        split=TRAIN,
        source=SOURCE,
    )


PHASE6_CASES: tuple[CorpusCase, ...] = (
    # Functional scaling -----------------------------------------------------
    _case(
        "p6_functional_scale",
        "functional scale before split",
        "from sklearn.preprocessing import scale\n" + _SPLIT + "X = scale(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_functional_scale",
        "functional scale on train only",
        "from sklearn.preprocessing import scale\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "X_train = scale(X_train)\n"
        "X_test = scale(X_test)\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_power_transform",
        "power transform before split",
        "from sklearn.preprocessing import power_transform\n"
        + _SPLIT
        + "X = power_transform(X)\n"
        "tr, te, ytr, yte = train_test_split(X, y, random_state=1)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_power_transform",
        "power transform fitted on train",
        "from sklearn.preprocessing import PowerTransformer\n"
        + _SPLIT
        + "tr, te, ytr, yte = train_test_split(X, y, random_state=1)\n"
        "transformer = PowerTransformer().fit(tr)\n"
        "tr = transformer.transform(tr)\n"
        "te = transformer.transform(te)\n",
        False,
        "preprocessing",
    ),
    # Hand-written normalisation --------------------------------------------
    _case(
        "p6_manual_scale",
        "divide by full frame maximum",
        "import numpy as np\n" + _SPLIT + "X = X / X.max()\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_manual_scale",
        "divide by training maximum",
        _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
        "X_train = X_train / X_train.max()\n"
        "X_test = X_test / X_train.max()\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_column_minmax",
        "column min max scaling before split",
        _SPLIT + "df['amount'] = (df['amount'] - df['amount'].min()) / "
        "(df['amount'].max() - df['amount'].min())\n"
        "X_train, X_test, y_train, y_test = train_test_split(df[['amount']], df['y'])\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_column_minmax",
        "column min max scaling on train only",
        _SPLIT + "train, test = train_test_split(df, random_state=2)\n"
        "low, high = train['amount'].min(), train['amount'].max()\n"
        "train['amount'] = (train['amount'] - low) / (high - low)\n"
        "test['amount'] = (test['amount'] - low) / (high - low)\n",
        False,
        "preprocessing",
    ),
    # Imputation -------------------------------------------------------------
    _case(
        "p6_mode_fill",
        "mode fill on the full frame",
        _SPLIT + "frame = frame.fillna(frame.mode().iloc[0])\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "frame[['amount', 'city']], frame['target'], random_state=9)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_mode_fill",
        "mode fill learned on train",
        _SPLIT + "train, test = train_test_split(df, random_state=5)\n"
        "modes = train.mode().iloc[0]\n"
        "train = train.fillna(modes)\n"
        "test = test.fillna(modes)\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_mean_fill",
        "median fill from every row",
        _SPLIT + "X = X.fillna(X.median())\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_mean_fill",
        "median fill from training rows",
        _SPLIT + "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "medians = X_train.median()\n"
        "X_train = X_train.fillna(medians)\n"
        "X_test = X_test.fillna(medians)\n",
        False,
        "preprocessing",
    ),
    # Categorical encoding ---------------------------------------------------
    _case(
        "p6_get_dummies",
        "get dummies on the full frame",
        "import pandas as pd\n" + _SPLIT + "X = pd.get_dummies(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n",
        True,
        "preprocessing",
    ),
    _case(
        "p6_get_dummies",
        "get dummies after the split",
        "import pandas as pd\n"
        + _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
        "X_train = pd.get_dummies(X_train)\n"
        "X_test = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_frequency_encoding",
        "frequency encoding from all rows",
        "import pandas as pd\n"
        + _SPLIT
        + "cardinality = frame['city'].value_counts()\n"
        "frame['city_frequency'] = frame['city'].map(cardinality)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "frame[['city_frequency', 'amount']], frame['target'], random_state=5)\n",
        True,
        "target_encoding",
    ),
    _case(
        "p6_frequency_encoding",
        "frequency encoding from training rows",
        "import pandas as pd\n"
        + _SPLIT
        + "train, test = train_test_split(df, random_state=0)\n"
        "freq = train['city'].value_counts()\n"
        "train['city_freq'] = train['city'].map(freq)\n"
        "test['city_freq'] = test['city'].map(freq).fillna(0)\n",
        False,
        "target_encoding",
    ),
    # Cross-validation discipline -------------------------------------------
    _case(
        "p6_cv_on_test",
        "cross validation on the test partition",
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_fit, X_test, y_fit, y_test = train_test_split("
        "features, labels, test_size=0.2, random_state=13)\n"
        "scores = cross_val_score(estimator, X_test, y_test, cv=4)\n",
        True,
        "tuning",
    ),
    _case(
        "p6_cv_on_test",
        "cross validation on the training partition",
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
        "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
        "print(model.fit(X_train, y_train).score(X_test, y_test))\n",
        False,
        "tuning",
    ),
    _case(
        "p6_repeated_test_scoring",
        "three models scored on the test partition",
        _SPLIT + "X_fit, X_test, y_fit, y_test = train_test_split("
        "features, labels, test_size=0.2, random_state=17)\n"
        "baseline = LinearRegression().fit(X_fit, y_fit)\n"
        "print('linear', baseline.score(X_test, y_test))\n"
        "forest = RandomForestRegressor().fit(X_fit, y_fit)\n"
        "print('forest', forest.score(X_test, y_test))\n",
        True,
        "tuning",
    ),
    _case(
        "p6_repeated_test_scoring",
        "one final score on the test partition",
        _SPLIT + "X_fit, X_test, y_fit, y_test = train_test_split("
        "features, labels, test_size=0.2, random_state=21)\n"
        "estimator = RandomForestClassifier().fit(X_fit, y_fit)\n"
        "print(estimator.score(X_test, y_test))\n",
        False,
        "tuning",
    ),
    # Grouping ---------------------------------------------------------------
    _case(
        "p6_group_kfold",
        "group kfold without a groups argument",
        "from sklearn.model_selection import GroupKFold, cross_val_score\n"
        "splitter = GroupKFold(n_splits=4)\n"
        "scores = cross_val_score(estimator, features, labels, cv=splitter)\n",
        True,
        "grouping",
    ),
    _case(
        "p6_group_kfold",
        "group kfold with groups",
        "from sklearn.model_selection import GroupKFold, cross_val_score\n"
        "kf = GroupKFold(n_splits=5)\n"
        "scores = cross_val_score(model, X, y, cv=kf, groups=df['customer_id'])\n",
        False,
        "grouping",
    ),
    # Temporal ordering ------------------------------------------------------
    _case(
        "p6_default_shuffle",
        "sorted frame split with default shuffle",
        "import pandas as pd\n" + _SPLIT + "events = log.sort_values('timestamp')\n"
        "train, test = train_test_split(events, test_size=0.25, random_state=11)\n",
        True,
        "temporal",
    ),
    _case(
        "p6_default_shuffle",
        "sorted frame split by position",
        "import pandas as pd\n"
        "ordered = df.sort_values('date').reset_index(drop=True)\n"
        "cutoff = int(len(ordered) * 0.8)\n"
        "train, test = ordered.iloc[:cutoff], ordered.iloc[cutoff:]\n",
        False,
        "temporal",
    ),
    # Fold-local fitting -----------------------------------------------------
    _case(
        "p6_fold_local",
        "transformer fitted inside the fold loop",
        "from sklearn.model_selection import KFold\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "for train_idx, valid_idx in kf.split(X):\n"
        "    scaler = StandardScaler().fit(X[train_idx])\n"
        "    model.fit(scaler.transform(X[train_idx]), y[train_idx])\n"
        "    print(model.score(scaler.transform(X[valid_idx]), y[valid_idx]))\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_fold_local",
        "transformer fitted once before cross validation",
        "from sklearn.model_selection import cross_val_score\n"
        "scaler = StandardScaler().fit_transform(X)\n"
        "scores = cross_val_score(model, scaler, y, cv=5)\n",
        True,
        "preprocessing",
    ),
    # Tuning -----------------------------------------------------------------
    _case(
        "p6_search_pipeline",
        "grid search over a pipeline",
        "from sklearn.model_selection import GridSearchCV\n"
        "pipe = make_pipeline(StandardScaler(), LogisticRegression())\n"
        "search = GridSearchCV(pipe, {'logisticregression__C': [0.1, 1, 10]}, cv=5)\n"
        "search.fit(X, y)\n",
        False,
        "tuning",
    ),
    _case(
        "p6_search_pipeline",
        "grid search fitted on the test partition",
        _SPLIT
        + "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
        "search = GridSearchCV(model, {'max_depth': [2, 4, 8]}, cv=5)\n"
        "search.fit(X_test, y_test)\n",
        True,
        "tuning",
    ),
    # Category-only preprocessing -------------------------------------------
    _case(
        "p6_onehot_only",
        "one hot encoder fitted on the full frame",
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        + _SPLIT
        + "preprocess = ColumnTransformer([('cat', OneHotEncoder(), ['city'])])\n"
        "encoded = preprocess.fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(encoded, y)\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_onehot_only",
        "target encoder fitted on the full frame",
        "from sklearn.preprocessing import TargetEncoder\n"
        + _SPLIT
        + "encoder = TargetEncoder().fit(X[['city']], y)\n"
        "X['city'] = encoder.transform(X[['city']])\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
        True,
        "target_encoding",
    ),
    # Train-only transformers ------------------------------------------------
    _case(
        "p6_train_only_imputer",
        "iterative imputer fitted on train only",
        "from sklearn.experimental import enable_iterative_imputer\n"
        "from sklearn.impute import IterativeImputer\n"
        + _SPLIT
        + "X_fit, X_test, y_fit, y_test = train_test_split("
        "features, labels, test_size=0.2, random_state=3)\n"
        "imputer = IterativeImputer(random_state=3).fit(X_fit)\n"
        "X_fit = imputer.transform(X_fit)\n"
        "X_test = imputer.transform(X_test)\n",
        False,
        "preprocessing",
    ),
    _case(
        "p6_train_only_imputer",
        "iterative imputer fitted before the split",
        "from sklearn.experimental import enable_iterative_imputer\n"
        "from sklearn.impute import IterativeImputer\n"
        + _SPLIT
        + "features = IterativeImputer(random_state=3).fit_transform(features)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "features, labels, random_state=3)\n",
        True,
        "preprocessing",
    ),
    # Target lags ------------------------------------------------------------
    _case(
        "p6_target_lag",
        "past target lag before split",
        _SPLIT + "df['target_lag'] = df['target'].shift(1)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df[['target_lag', 'x']], df['target'], shuffle=False)\n",
        False,
        "temporal",
    ),
    _case(
        "p6_target_lag",
        "future target lag before split",
        _SPLIT + "df['next_target'] = df['target'].shift(-1)\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "df[['next_target', 'x']], df['target'], shuffle=False)\n",
        True,
        "temporal",
    ),
)


__all__ = ["PHASE6_CASES", "SOURCE"]
