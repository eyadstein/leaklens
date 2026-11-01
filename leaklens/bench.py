"""Labeled regression benchmark for the code scanner.

Usage: python -m leaklens.bench [--write docs/BENCHMARK.md]
"""
from __future__ import annotations

import sys
from pathlib import Path

from .scanner import scan_source

SPLIT = "from sklearn.model_selection import train_test_split\n"
CASES: list[tuple[str, str, bool]] = []

for _cls, _m in [("StandardScaler", "fit_transform"), ("MinMaxScaler", "fit_transform"),
                 ("RobustScaler", "fit_transform"), ("SimpleImputer", "fit_transform"),
                 ("PCA", "fit_transform"), ("TfidfVectorizer", "fit_transform"),
                 ("SelectKBest", "fit_transform"), ("SMOTE", "fit_resample"), ("ADASYN", "fit_resample")]:
    head = f"from x import {_cls}\n" + SPLIT
    CASES.append((f"{_cls} fitted before split", head + f"obj = {_cls}()\nX = obj.{_m}(X)\na = train_test_split(X, y)\n", True))
    CASES.append((f"{_cls} fitted after split on train", head + f"a = train_test_split(X, y)\nobj = {_cls}()\nX_train = obj.{_m}(X_train)\n", False))

CASES += [
    ("scale() before split",
     "from sklearn.preprocessing import scale\n" + SPLIT + "X = scale(X)\na = train_test_split(X, y)\n", True),
    ("scaler fitted on test data",
     "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler()\nZ = sc.fit_transform(X_test)\n", True),
    ("scaler before KFold",
     "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold\n"
     "sc = StandardScaler()\nX = sc.fit_transform(X)\nkf = KFold(5)\nfor tr, te in kf.split(X):\n    pass\n", True),
    ("imputer .fit before split",
     "from sklearn.impute import SimpleImputer\n" + SPLIT + "imp = SimpleImputer()\nimp.fit(X)\na = train_test_split(X, y)\n", True),
    ("fit on train, transform test",
     "from sklearn.preprocessing import StandardScaler\n" + SPLIT +
     "a = train_test_split(X, y)\nsc = StandardScaler()\nX_train = sc.fit_transform(X_train)\nX_test = sc.transform(X_test)\n", False),
    ("pipeline inside cross_val_score",
     "from sklearn.pipeline import make_pipeline\nfrom sklearn.model_selection import cross_val_score\n"
     "pipe = make_pipeline(StandardScaler(), LogisticRegression())\nscores = cross_val_score(pipe, X, y)\n", False),
    ("label encoder before split (benign)",
     "from sklearn.preprocessing import LabelEncoder\n" + SPLIT + "le = LabelEncoder()\ny = le.fit_transform(y)\na = train_test_split(X, y)\n", False),
    ("one-hot before split (benign)",
     "from sklearn.preprocessing import OneHotEncoder\n" + SPLIT + "ohe = OneHotEncoder()\nX = ohe.fit_transform(X)\na = train_test_split(X, y)\n", False),
    ("transform only",
     "from sklearn.preprocessing import StandardScaler\nsc = StandardScaler()\nX_test = sc.transform(X_test)\n", False),
    ("fit inside KFold loop",
     "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold\n"
     "kf = KFold(5)\nfor tr, te in kf.split(X):\n    sc = StandardScaler()\n    Xtr = sc.fit_transform(X[tr])\n", False),
    ("no preprocessing",
     SPLIT + "from sklearn.linear_model import LogisticRegression\na = train_test_split(X, y)\nm = LogisticRegression()\nm.fit(X_train, y_train)\n", False),
    ("scale() on train after split",
     "from sklearn.preprocessing import scale\n" + SPLIT + "a = train_test_split(X, y)\nXs = scale(X_train)\n", False),
]


CASES += [
    ("fillna with full-data mean before split",
     SPLIT + "df = df.fillna(df.mean())\na = train_test_split(df, y, random_state=42)\n", True),
    ("fillna with full-data median before split",
     SPLIT + "df['c'] = df['c'].fillna(df['c'].median())\na = train_test_split(df, y, random_state=42)\n", True),
    ("fillna with constant before split",
     SPLIT + "df = df.fillna(0)\na = train_test_split(df, y, random_state=42)\n", False),
    ("fillna with train mean after split",
     SPLIT + "a = train_test_split(df, y, random_state=42)\ndf['c'] = df['c'].fillna(df_train['c'].mean())\n", False),
]

_H = "from sklearn.preprocessing import StandardScaler\n" + SPLIT
CASES += [
    ("chained StandardScaler before split", _H + "X = StandardScaler().fit_transform(X)\na = train_test_split(X, y)\n", True),
    ("chained scaler after split on train", _H + "a = train_test_split(X, y)\nX_train = StandardScaler().fit_transform(X_train)\n", False),
    ("fit then transform before cross-validation",
     _H + "from sklearn.model_selection import cross_val_score\nsc = StandardScaler()\nsc.fit(X)\nXs = sc.transform(X)\ns = cross_val_score(m, Xs, y)\n", True),
    ("scaler fitted on full X after split", _H + "sc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nZ = sc.fit_transform(X)\n", True),
    ("derived copy fitted after split", _H + "X2 = X.copy()\nsc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nsc.fit(X2)\n", True),
    ("fit on train partition only", _H + "sc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nsc.fit(X_train)\nZ = sc.transform(X_test)\n", False),
]

# issue #20: features built from the target or from group statistics on all rows
_P = "import pandas as pd\n" + SPLIT
CASES += [
    ("groupby target transform before split",
     _P + "df['enc'] = df.groupby('city')[y].transform('mean')\na = train_test_split(X, y)\n", True),
    ("groupby target-name column transform before split",
     _P + "df['enc'] = df.groupby('city')['target'].transform('mean')\na = train_test_split(df.drop('target'), df['target'])\n", True),
    ("groupby feature transform before split",
     _P + "df['m'] = df.groupby('city')['sales'].transform('mean')\na = train_test_split(X, y)\n", True),
    ("groupby transform after the split",
     _P + "a = train_test_split(X, y)\nX_train['m'] = X_train.groupby('city')['sales'].transform('mean')\n", False),
    ("groupby agg into a column before split",
     _P + "df['m'] = df.groupby('city')['sales'].agg('mean')\na = train_test_split(X, y)\n", True),
    ("groupby agg summary before split (EDA, not a feature)",
     _P + "summary = df.groupby('city')['sales'].agg('mean')\na = train_test_split(X, y)\n", False),
]


# issue #77: hand-rolled normalization with full-data statistics
_N = "import numpy as np\n" + SPLIT
CASES += [
    ("z-score by hand before split",
     _N + "X = (X - X.mean()) / X.std()\na = train_test_split(X, y, random_state=0)\n", True),
    ("divide by max before split",
     _N + "X = X / X.max()\na = train_test_split(X, y, random_state=0)\n", True),
    ("np.mean subtraction before split",
     _N + "X = X - np.mean(X)\na = train_test_split(X, y, random_state=0)\n", True),
    ("column max scaling before split",
     _N + "df['a'] = df['a'] / df['a'].max()\na = train_test_split(X, y, random_state=0)\n", True),
    ("manual z-score with train statistics after split",
     _N + "a = train_test_split(X, y, random_state=0)\nX_test = (X_test - X_train.mean()) / X_train.std()\n", False),
    ("statistics from a different frame",
     _N + "Z = (A - B.mean()) / B.std()\na = train_test_split(X, y, random_state=0)\n", False),
]

# Low-severity cases (reproducibility and pre-split category detection):
LOW_CASES: list[tuple[str, str, bool]] = [
    ("train_test_split without random_state",
     SPLIT + "a = train_test_split(X, y)\n", True),
    ("train_test_split with random_state",
     SPLIT + "a = train_test_split(X, y, random_state=42)\n", False),
    ("KFold shuffle without random_state",
     "from sklearn.model_selection import KFold\nkf = KFold(5, shuffle=True)\n", True),
    ("KFold shuffle with random_state",
     "from sklearn.model_selection import KFold\nkf = KFold(5, shuffle=True, random_state=42)\n", False),
    ("KFold without shuffle stays deterministic",
     "from sklearn.model_selection import KFold\nkf = KFold(5)\n", False),
]


LOW_CASES += [
    ("train_test_split shuffle=False is deterministic", SPLIT + "a = train_test_split(X, y, shuffle=False)\n", False),
    ("ShuffleSplit without random_state", "from sklearn.model_selection import ShuffleSplit\nss = ShuffleSplit(5)\n", True),
    ("ShuffleSplit with random_state", "from sklearn.model_selection import ShuffleSplit\nss = ShuffleSplit(5, random_state=1)\n", False),
    ("get_dummies before split","import pandas as pd\n" + SPLIT +
     "X = pd.get_dummies(df)\na = train_test_split(X, y, random_state=0)\n", True),
    ("get_dummies after split",
     "import pandas as pd\n" + SPLIT +
     "a = train_test_split(X, y, random_state=0)\nX_train = pd.get_dummies(X_train)\n", False),
    ("unrelated get_dummies",
     "import pandas as pd\n" + SPLIT +
     "summary = pd.get_dummies(df)\na = train_test_split(X, y, random_state=0)\n", False),

]


def evaluate() -> dict:
    tp = fp = tn = fn = 0
    errors = []
    for name, code, leaky in CASES:
        flagged = any(f.severity in ("high", "medium") for f in scan_source(code, name))
        if leaky and flagged:
            tp += 1
        elif leaky:
            fn += 1
            errors.append(f"missed: {name}")
        elif flagged:
            fp += 1
            errors.append(f"false alarm: {name}")
        else:
            tn += 1
    # low-severity reproducibility cases, scored the same way on their own list
    low_tp = low_fp = low_tn = low_fn = 0
    low_kinds = {
        "get_dummies before split": "get_dummies_before_split",
        "get_dummies after split": "get_dummies_before_split",
        "unrelated get_dummies": "get_dummies_before_split",
    }
    for name, code, leaky in LOW_CASES:
        expected_kind = low_kinds.get(name, "no_random_state")
        flagged = any(f.kind == expected_kind for f in scan_source(code, name))
        if leaky and flagged:
            low_tp += 1
        elif leaky:
            low_fn += 1
            errors.append(f"missed (low): {name}")
        elif flagged:
            low_fp += 1
            errors.append(f"false alarm (low): {name}")
        else:
            low_tn += 1
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(CASES), "leaky": tp + fn, "clean": tn + fp, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "low_n": len(LOW_CASES), "low_tp": low_tp, "low_fp": low_fp, "low_tn": low_tn, "low_fn": low_fn,
            "precision": prec, "recall": rec, "f1": f1, "errors": errors}


_CV = "from sklearn.model_selection import cross_val_score, GridSearchCV\nfrom sklearn.preprocessing import StandardScaler\n"
CASES += [
    ("scaler fitted on all data then cross_val_score", _CV + "X = StandardScaler().fit_transform(X)\nscores = cross_val_score(model, X, y, cv=5)\n", True),
    ("SMOTE on all data then cross_val_score", "from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import cross_val_score\nX, y = SMOTE().fit_resample(X, y)\nscores = cross_val_score(model, X, y, cv=5)\n", True),
    ("scaler on all data then GridSearchCV.fit", _CV + "sc = StandardScaler()\nXs = sc.fit_transform(X)\ngs = GridSearchCV(model, grid)\ngs.fit(Xs, y)\n", True),
    ("pipeline in cross_val_score with raw X", _CV + "pipe = make_pipeline(StandardScaler(), model)\nscores = cross_val_score(pipe, X, y, cv=5)\n", False),
    ("GridSearchCV over a pipeline with raw X", _CV + "gs = GridSearchCV(make_pipeline(StandardScaler(), model), grid)\ngs.fit(X, y)\n", False),
    ("label-encoded target then cross_val_score", "from sklearn.preprocessing import LabelEncoder\nfrom sklearn.model_selection import cross_val_score\ny = LabelEncoder().fit_transform(y)\nscores = cross_val_score(model, X, y, cv=5)\n", False),
]


_F = "from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.model_selection import StratifiedKFold, train_test_split, permutation_test_score\n"
CASES += [
    ("scaler on all data then manual KFold loop", _F + "X = MinMaxScaler().fit_transform(x)\ncv = StratifiedKFold(5)\nfor tr, te in cv.split(X, y):\n    pass\n", True),
    ("full-data scaler then a second split", _F + "a = train_test_split(X, y, random_state=0)\nX = MinMaxScaler().fit_transform(x)\nb = train_test_split(X, y, random_state=0)\n", True),
    ("ColumnTransformer with a scaler before split", _F + "ct = ColumnTransformer([('s', StandardScaler(), [0])])\nX = ct.fit_transform(X)\na = train_test_split(X, y, random_state=0)\n", True),
    ("ColumnTransformer with only one-hot before split", _F + "ct = ColumnTransformer([('o', OneHotEncoder(), [0])])\nX = ct.fit_transform(X)\na = train_test_split(X, y, random_state=0)\n", False),
    ("scaler on all data then permutation_test_score", _F + "Xs = MinMaxScaler().fit_transform(x)\nr = permutation_test_score(model, Xs, y, cv=5)\n", True),
    ("scaler fitted per fold inside the loop", _F + "cv = StratifiedKFold(5)\nfor tr, te in cv.split(X, y):\n    sc = MinMaxScaler()\n    Xtr = sc.fit_transform(X[tr])\n    Xte = sc.transform(X[te])\n", False),
]


# Contributed snippets (issue #14), adapted from common notebook patterns.
CASES += [
    ("QuantileTransformer fit on all data before split",
     "from sklearn.preprocessing import QuantileTransformer\n" + SPLIT +
     "qt = QuantileTransformer()\nX = qt.fit_transform(X)\na = train_test_split(X, y)\n", True),
    ("power_transform() on all data before split",
     "from sklearn.preprocessing import power_transform\n" + SPLIT +
     "X = power_transform(X)\na = train_test_split(X, y)\n", True),
    ("RandomOverSampler on all data before split",
     "from imblearn.over_sampling import RandomOverSampler\n" + SPLIT +
     "X_res, y_res = RandomOverSampler(random_state=0).fit_resample(X, y)\n"
     "X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state=0)\n", True),
    ("named-step Pipeline fitted on train only",
     "from sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\n"
     "from sklearn.preprocessing import StandardScaler\n" + SPLIT +
     "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
     "pipe = Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler()), ('clf', model)])\n"
     "pipe.fit(X_train, y_train)\nscore = pipe.score(X_test, y_test)\n", False),
    ("imputer fitted on train only, applied to test",
     "from sklearn.impute import SimpleImputer\n" + SPLIT +
     "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
     "imp = SimpleImputer()\nX_train = imp.fit_transform(X_train)\nX_test = imp.transform(X_test)\n", False),
]


def markdown(m: dict) -> str:
    miss = "\n".join(f"- {e}" for e in m["errors"]) or "- none"
    return (
        "# Scanner benchmark\n\n"
        f"{m['n']} hand-written labeled snippets ({m['leaky']} leaky, {m['clean']} clean). "
        "A snippet counts as flagged when the scanner reports a high or medium finding.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Precision | {m['precision']:.2f} |\n| Recall | {m['recall']:.2f} |\n| F1 | {m['f1']:.2f} |\n"
        f"| True / false positives | {m['tp']} / {m['fp']} |\n| True negatives / false negatives | {m['tn']} / {m['fn']} |\n\n"
        f"Low-severity cases (reproducibility and pre-split category detection): {m['low_tp'] + m['low_tn']}/{m['low_n']} correct, "
        f"{m['low_fp'] + m['low_fn']} misclassified.\n\n"
        f"Misclassified:\n{miss}\n\n"
        "## Caveats\n"
        "The set is small, hand-written and partly shaped by the scanner's own rules, so it guards against "
        "regressions; it does not measure real-world accuracy. For prevalence on real notebooks, "
        "see [STUDY.md](STUDY.md) (also a heuristic).\n"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    m = evaluate()
    text = markdown(m)
    print(text)
    if len(argv) >= 2 and argv[0] == "--write":
        out = Path(argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 1 if m["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())