"""Advanced detection cases added in the 0.14.0 detection-engine phase.

Kept separate from ``bench_cases.EXTRA_CASES`` so the historical benchmark
counts stay stable and regressions in either set are visible on their own.
"""

from __future__ import annotations

from .bench_cases import BenchmarkCase

ADVANCED_CASES: list[BenchmarkCase] = []

_T = "from sklearn.model_selection import train_test_split\n"

# ---------------------------------------------------------------------------
# Hyperparameter tuning and test-set reuse.
# ---------------------------------------------------------------------------
_TUNING_HEAD = (
    "from sklearn.model_selection import GridSearchCV, train_test_split\n"
    "from sklearn.linear_model import LogisticRegression\n"
)

ADVANCED_CASES.extend(
    [
        BenchmarkCase(
            "search fitted on named holdout split",
            _TUNING_HEAD + "X_train, X_holdout, y_train, y_holdout = train_test_split("
            "X, y, random_state=0)\n"
            + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
            + "gs.fit(X_holdout, y_holdout)\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "search fitted on positional second split value",
            _TUNING_HEAD
            + "a, b, c, d = train_test_split(X, y, random_state=0)\n"
            + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
            + "gs.fit(b, d)\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "search fitted on train split",
            _TUNING_HEAD + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
            + "gs.fit(X_train, y_train)\n"
            + "print(gs.score(X_test, y_test))\n",
            False,
            "tuning",
        ),
        BenchmarkCase(
            "nested cross-validation around a search",
            "from sklearn.model_selection import GridSearchCV, cross_val_score, "
            "train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            "gs = GridSearchCV(model, grid, cv=5)\n"
            "scores = cross_val_score(gs, X_train, y_train, cv=5)\n"
            "print(gs.fit(X_train, y_train).score(X_test, y_test))\n",
            False,
            "tuning",
        ),
        BenchmarkCase(
            "search fitted on full data before the split",
            _TUNING_HEAD
            + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
            + "gs.fit(X, y)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "cross-validation on the test partition",
            "from sklearn.model_selection import cross_val_score, train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            "scores = cross_val_score(model, X_test, y_test, cv=5)\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "cross-validation on the training partition",
            "from sklearn.model_selection import cross_val_score, train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            "scores = cross_val_score(model, X_train, y_train, cv=5)\n",
            False,
            "tuning",
        ),
        BenchmarkCase(
            "hand-rolled tuning loop on test data",
            _T + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "for C in [0.1, 1.0, 10.0]:\n"
            + "    model = LogisticRegression(C=C)\n"
            + "    model.fit(X_test, y_test)\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "hand-rolled tuning loop on train data",
            _T + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "for C in [0.1, 1.0, 10.0]:\n"
            + "    model = LogisticRegression(C=C)\n"
            + "    model.fit(X_train, y_train)\n",
            False,
            "tuning",
        ),
        BenchmarkCase(
            "tuning on a validation split is allowed",
            _T + "X_train, X_val, y_train, y_val = train_test_split("
            "X_train, y_train, random_state=0)\n"
            + "gs = GridSearchCV(model, grid, cv=5)\n"
            + "gs.fit(X_train, y_train)\n"
            + "print(gs.score(X_val, y_val))\n",
            False,
            "tuning",
        ),
        BenchmarkCase(
            "group CV without groups",
            "from sklearn.model_selection import GroupKFold, cross_val_score\n"
            "kf = GroupKFold(n_splits=5)\n"
            "scores = cross_val_score(model, X, y, cv=kf)\n",
            True,
            "group_cv",
        ),
        BenchmarkCase(
            "group CV with groups",
            "from sklearn.model_selection import GroupKFold, cross_val_score\n"
            "kf = GroupKFold(n_splits=5)\n"
            "scores = cross_val_score(model, X, y, cv=kf, groups=patient_id)\n",
            False,
            "group_cv",
        ),
        BenchmarkCase(
            "repeated scoring on the test partition",
            _T + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "print(model_a.fit(X_train, y_train).score(X_test, y_test))\n"
            + "print(model_b.fit(X_train, y_train).score(X_test, y_test))\n"
            + "print(model_c.fit(X_train, y_train).score(X_test, y_test))\n",
            True,
            "tuning",
        ),
        BenchmarkCase(
            "single score on the test partition",
            _T + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "model.fit(X_train, y_train)\n"
            + "print(model.score(X_test, y_test))\n",
            False,
            "tuning",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Encodings built from the label or from the full column distribution.
# ---------------------------------------------------------------------------
_ENCODING_SPLIT = (
    "from sklearn.model_selection import train_test_split\n"
    "X_train, X_test, y_train, y_test = train_test_split("
    "df.drop(columns=['target']), df['target'], random_state=0)\n"
)

ADVANCED_CASES.extend(
    [
        BenchmarkCase(
            "target mean mapped into a column before split",
            "import pandas as pd\n"
            + _T
            + "stats = df.groupby('city')['target'].mean()\n"
            + "df['city_te'] = df['city'].map(stats)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "df.drop(columns=['target']), df['target'], random_state=0)\n",
            True,
            "encoding",
        ),
        BenchmarkCase(
            "inline groupby target mean mapped before split",
            "import pandas as pd\n"
            + _T
            + "df['city_te'] = df['city'].map(df.groupby('city')['target'].mean())\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "df.drop(columns=['target']), df['target'], random_state=0)\n",
            True,
            "encoding",
        ),
        BenchmarkCase(
            "TargetEncoder fitted before split",
            "from category_encoders import TargetEncoder\n"
            + _T
            + "enc = TargetEncoder()\n"
            + "X = enc.fit_transform(X, y)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n",
            True,
            "encoding",
        ),
        BenchmarkCase(
            "TargetEncoder inside a pipeline",
            "from category_encoders import TargetEncoder\n"
            "from sklearn.pipeline import make_pipeline\n"
            + _T
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "pipe = make_pipeline(TargetEncoder(), model)\n"
            + "pipe.fit(X_train, y_train)\n"
            + "print(pipe.score(X_test, y_test))\n",
            False,
            "encoding",
        ),
        BenchmarkCase(
            "frequency encoding from all rows before split",
            "import pandas as pd\n"
            + _T
            + "freq = df['city'].value_counts()\n"
            + "df['city_freq'] = df['city'].map(freq)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "df.drop(columns=['target']), df['target'], random_state=0)\n",
            True,
            "encoding",
        ),
        BenchmarkCase(
            "frequency encoding computed on train only",
            "import pandas as pd\n"
            + _ENCODING_SPLIT
            + "freq = X_train['city'].value_counts()\n"
            + "X_train['city_freq'] = X_train['city'].map(freq)\n"
            + "X_test['city_freq'] = X_test['city'].map(freq).fillna(0)\n",
            False,
            "encoding",
        ),
        BenchmarkCase(
            "target encoding computed on train only",
            "import pandas as pd\n"
            + _ENCODING_SPLIT
            + "train = X_train.assign(target=y_train)\n"
            + "stats = train.groupby('city')['target'].mean()\n"
            + "X_train['city_te'] = X_train['city'].map(stats)\n"
            + "X_test['city_te'] = X_test['city'].map(stats)\n",
            False,
            "encoding",
        ),
        BenchmarkCase(
            "target relabelled before split is harmless",
            "import pandas as pd\n"
            + _T
            + "df['target'] = df['target'].replace({'yes': 1, 'no': 0})\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "df.drop(columns=['target']), df['target'], random_state=0)\n",
            False,
            "encoding",
        ),
        BenchmarkCase(
            "frequency summary not used as a feature",
            "import pandas as pd\n"
            + _T
            + "summary = df['city'].value_counts()\n"
            + "print(summary)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "df.drop(columns=['target']), df['target'], random_state=0)\n",
            False,
            "encoding",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Feature selection learns from the label.
# ---------------------------------------------------------------------------
ADVANCED_CASES.extend(
    [
        BenchmarkCase(
            "SelectKBest fitted before split",
            "from sklearn.feature_selection import SelectKBest\n"
            + _T
            + "sel = SelectKBest(k=10)\n"
            + "X = sel.fit_transform(X, y)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n",
            True,
            "feature_selection",
        ),
        BenchmarkCase(
            "RFE fitted before split",
            "from sklearn.feature_selection import RFE\n"
            + _T
            + "rfe = RFE(model, n_features_to_select=5)\n"
            + "X = rfe.fit_transform(X, y)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n",
            True,
            "feature_selection",
        ),
        BenchmarkCase(
            "SelectFromModel fitted before split",
            "from sklearn.feature_selection import SelectFromModel\n"
            + _T
            + "sel = SelectFromModel(model)\n"
            + "X = sel.fit_transform(X, y)\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n",
            True,
            "feature_selection",
        ),
        BenchmarkCase(
            "feature selection inside a pipeline",
            "from sklearn.feature_selection import SelectKBest\n"
            "from sklearn.pipeline import make_pipeline\n"
            + _T
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "pipe = make_pipeline(SelectKBest(k=10), model)\n"
            + "pipe.fit(X_train, y_train)\n",
            False,
            "feature_selection",
        ),
        BenchmarkCase(
            "feature selector fitted on train only",
            "from sklearn.feature_selection import SelectKBest\n"
            + _T
            + "X_train, X_test, y_train, y_test = train_test_split("
            "X, y, random_state=0)\n"
            + "sel = SelectKBest(k=10)\n"
            + "X_train = sel.fit_transform(X_train, y_train)\n"
            + "X_test = sel.transform(X_test)\n",
            False,
            "feature_selection",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Chronological order.
# ---------------------------------------------------------------------------
ADVANCED_CASES.extend(
    [
        BenchmarkCase(
            "time-sorted frame split with default shuffle",
            "import pandas as pd\n"
            + _T
            + "ordered = df.sort_values('date')\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "ordered[['x']], ordered['y'], random_state=0)\n",
            True,
            "temporal",
        ),
        BenchmarkCase(
            "time-sorted frame split with explicit shuffle",
            "import pandas as pd\n"
            + _T
            + "ordered = df.sort_values('timestamp')\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "ordered[['x']], ordered['y'], shuffle=True, random_state=0)\n",
            True,
            "temporal",
        ),
        BenchmarkCase(
            "time-sorted frame split chronologically",
            "import pandas as pd\n"
            + _T
            + "ordered = df.sort_values('date')\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "ordered[['x']], ordered['y'], shuffle=False)\n",
            False,
            "temporal",
        ),
        BenchmarkCase(
            "sorting an unrelated column before split",
            "import pandas as pd\n"
            + _T
            + "ordered = df.sort_values('sales')\n"
            + "X_train, X_test, y_train, y_test = train_test_split("
            "ordered[['x']], ordered['y'], random_state=0)\n",
            False,
            "temporal",
        ),
    ]
)
