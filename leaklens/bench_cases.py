"""Additional categorized cases for the expanded scanner benchmark."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    code: str
    leaky: bool
    category: str


EXTRA_CASES: list[BenchmarkCase] = []

# Scaling and normalization.
_SCALERS = [
    ("standard scaler", "StandardScaler"),
    ("minmax scaler", "MinMaxScaler"),
    ("robust scaler", "RobustScaler"),
    ("maxabs scaler", "MaxAbsScaler"),
    ("normalizer", "Normalizer"),
    ("quantile transformer", "QuantileTransformer"),
    ("power transformer", "PowerTransformer"),
    ("pca", "PCA"),
]

for _label, _cls in _SCALERS:
    EXTRA_CASES.append(
        BenchmarkCase(
            f"{_label} before split",
            f"from sklearn.preprocessing import {_cls}\n"
            "from sklearn.model_selection import train_test_split\n"
            f"obj = {_cls}()\nX = obj.fit_transform(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n",
            True,
            "scaling",
        )
    )

for _label, _cls in _SCALERS:
    EXTRA_CASES.append(
        BenchmarkCase(
            f"{_label} train only",
            f"from sklearn.preprocessing import {_cls}\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            f"obj = {_cls}()\nX_train = obj.fit_transform(X_train)\n"
            "X_test = obj.transform(X_test)\n",
            False,
            "scaling",
        )
    )

# Imputation leakage: distinct APIs and operations.
_IMPUTATION_BAD = [
    (
        "simple imputer fit transform",
        (
            "from sklearn.impute import SimpleImputer\n"
            "imp = SimpleImputer(strategy='median')\n"
            "X = imp.fit_transform(X)\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        ),
    ),
    (
        "knn imputer fit transform",
        (
            "from sklearn.impute import KNNImputer\n"
            "imputer = KNNImputer(n_neighbors=3)\n"
            "features = imputer.fit_transform(features)\n"
            "train_test_split(features, labels)\n"
        ),
    ),
    (
        "iterative imputer fit transform",
        (
            "from sklearn.experimental import enable_iterative_imputer\n"
            "from sklearn.impute import IterativeImputer\n"
            "filled = IterativeImputer(max_iter=5).fit_transform(data)\n"
            "train_test_split(filled, target)\n"
        ),
    ),
    (
        "imputer fit then transform full data",
        (
            "from sklearn.impute import SimpleImputer\n"
            "imp = SimpleImputer(strategy='mean')\n"
            "imp.fit(matrix)\n"
            "matrix = imp.transform(matrix)\n"
            "train_test_split(matrix, y)\n"
        ),
    ),
    (
        "dataframe mean fill before split",
        (
            "import pandas as pd\n"
            "df = df.fillna(df.mean(numeric_only=True))\n"
            "train_test_split(df.drop(columns=['target']), df['target'])\n"
        ),
    ),
    (
        "column median fill before split",
        (
            "df['age'] = df['age'].fillna(df['age'].median())\n"
            "train_test_split(df.drop(columns=['target']), df['target'])\n"
        ),
    ),
    (
        "dataframe mode fill before split",
        (
            "df = df.fillna(df.mode().iloc[0])\n"
            "train_test_split(df.drop(columns=['target']), df['target'])\n"
        ),
    ),
]

for _name, _code in _IMPUTATION_BAD:
    EXTRA_CASES.append(
        BenchmarkCase(
            _name,
            "from sklearn.model_selection import train_test_split\n" + _code,
            True,
            "imputation",
        )
    )

# Clean controls: learn imputation statistics from training data only.
_IMPUTATION_GOOD = [
    (
        "simple imputer median train only",
        (
            "from sklearn.impute import SimpleImputer\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "imp = SimpleImputer(strategy='median')\n"
            "X_train = imp.fit_transform(X_train)\n"
            "X_test = imp.transform(X_test)\n"
        ),
    ),
    (
        "knn imputer train only",
        (
            "from sklearn.impute import KNNImputer\n"
            "a, b, c, d = train_test_split(features, labels)\n"
            "imp = KNNImputer(n_neighbors=5)\n"
            "a = imp.fit_transform(a)\n"
            "b = imp.transform(b)\n"
        ),
    ),
    (
        "iterative imputer train only",
        (
            "from sklearn.experimental import enable_iterative_imputer\n"
            "from sklearn.impute import IterativeImputer\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "imp = IterativeImputer(max_iter=10)\n"
            "train_filled = imp.fit_transform(X_train)\n"
            "test_filled = imp.transform(X_test)\n"
        ),
    ),
    (
        "most frequent imputer train only",
        (
            "from sklearn.impute import SimpleImputer\n"
            "train, test, y_train, y_test = train_test_split(X, y)\n"
            "imp = SimpleImputer(strategy='most_frequent')\n"
            "train = imp.fit_transform(train)\n"
            "test = imp.transform(test)\n"
        ),
    ),
    (
        "constant imputer train only",
        (
            "from sklearn.impute import SimpleImputer\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "imp = SimpleImputer(strategy='constant', fill_value=0)\n"
            "X_train = imp.fit_transform(X_train)\n"
            "X_test = imp.transform(X_test)\n"
        ),
    ),
    (
        "column mean from training data",
        (
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "age_mean = X_train['age'].mean()\n"
            "X_train['age'] = X_train['age'].fillna(age_mean)\n"
            "X_test['age'] = X_test['age'].fillna(age_mean)\n"
        ),
    ),
    (
        "column median from training data",
        (
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "income_median = X_train['income'].median()\n"
            "X_train['income'] = X_train['income'].fillna(income_median)\n"
            "X_test['income'] = X_test['income'].fillna(income_median)\n"
        ),
    ),
    (
        "separate dataframe imputation",
        (
            "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
            "medians = X_train.median(numeric_only=True)\n"
            "X_train = X_train.fillna(medians)\n"
            "X_test = X_test.fillna(medians)\n"
        ),
    ),
]

for _name, _code in _IMPUTATION_GOOD:
    EXTRA_CASES.append(
        BenchmarkCase(
            _name,
            "from sklearn.model_selection import train_test_split\n" + _code,
            False,
            "imputation",
        )
    )

# Issue #4: future-looking target-derived features.
_FUTURE = "from sklearn.model_selection import train_test_split\n"

EXTRA_CASES.extend(
    [
        BenchmarkCase(
            "negative target shift before split",
            _FUTURE
            + "future = y.shift(-1)\na = train_test_split(X, y, random_state=42)\n",
            True,
            "future_target",
        ),
        BenchmarkCase(
            "positive target lag before split",
            _FUTURE + "lag = y.shift(1)\na = train_test_split(X, y, random_state=42)\n",
            False,
            "future_target",
        ),
        BenchmarkCase(
            "centered target rolling before split",
            _FUTURE
            + "smooth = y.rolling(5, center=True).mean()\na = train_test_split(X, y, random_state=42)\n",
            True,
            "future_target",
        ),
        BenchmarkCase(
            "trailing target rolling before split",
            _FUTURE
            + "smooth = y.rolling(5).mean()\na = train_test_split(X, y, random_state=42)\n",
            False,
            "future_target",
        ),
        BenchmarkCase(
            "get_dummies on full data before split",
            "import pandas as pd\n"
            + _FUTURE
            + "X = pd.get_dummies(X)\na = train_test_split(X, y, random_state=42)\n",
            True,
            "encoding",
        ),
    ]
)
