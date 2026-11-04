from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class Finding:
    severity: str
    kind: str
    column: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _encode(s: pd.Series) -> np.ndarray:
    if s.dtype.kind in "biuf":
        fill = s.median() if s.notna().any() else 0.0
        return s.astype(float).fillna(fill).to_numpy().reshape(-1, 1)
    return s.astype("category").cat.codes.to_numpy().astype(float).reshape(-1, 1)


class _SkippedType:
    """Sentinel marking a column that could not be cross-validated."""

    __slots__ = ()


_SKIPPED = _SkippedType()


def single_feature_leaks(df, target, threshold=0.95, max_rows=20000, seed=0):
    d = df[df[target].notna()]
    if len(d) > max_rows:
        d = d.sample(max_rows, random_state=seed)
    y_raw = d[target]
    clf = y_raw.dtype.kind not in "iuf" or y_raw.nunique() <= 20
    if clf:
        y = y_raw.astype("category").cat.codes.to_numpy()
        scoring = "roc_auc" if len(np.unique(y)) == 2 else "balanced_accuracy"
        make = lambda: DecisionTreeClassifier(max_depth=4, random_state=seed)
    else:
        y = y_raw.astype(float).to_numpy()
        scoring = "r2"
        make = lambda: DecisionTreeRegressor(max_depth=4, random_state=seed)
    out = []
    for c in d.columns:
        if c == target:
            continue
        try:
            sc = float(
                cross_val_score(make(), _encode(d[c]), y, cv=5, scoring=scoring).mean()
            )
        except Exception:  # noqa: BLE001 - a column that cannot be cross-validated is skipped below
            sc = _SKIPPED
        if sc is _SKIPPED:
            continue
        if sc >= threshold:
            sev = "high"
        elif sc >= threshold - 0.1:
            sev = "medium"
        else:
            continue
        out.append(
            Finding(
                sev, "target_leakage", c, f"alone predicts target ({scoring}={sc:.3f})"
            )
        )
    return out


def id_like(df, target, ratio=0.98):
    out = []
    if len(df) < 50:
        return out
    for c in df.columns:
        if c == target or df[c].dtype.kind not in "OSUiu":
            continue
        r = df[c].nunique(dropna=True) / len(df)
        if r >= ratio:
            out.append(
                Finding(
                    "medium",
                    "id_like",
                    c,
                    f"{r:.1%} unique values; likely an identifier",
                )
            )
    return out


def split_overlap(train, test):
    cols = [c for c in train.columns if c in test.columns]
    a = set(pd.util.hash_pandas_object(train[cols], index=False))
    n = int(pd.util.hash_pandas_object(test[cols], index=False).isin(a).sum())
    if n == 0:
        return []
    return [
        Finding(
            "high",
            "split_overlap",
            "*",
            f"{n} test rows ({n / len(test):.1%}) also appear in train",
        )
    ]


def temporal_overlap(train, test, time_col):
    a = pd.to_datetime(train[time_col], errors="coerce")
    b = pd.to_datetime(test[time_col], errors="coerce")
    if a.max() > b.min():
        return [
            Finding(
                "high",
                "temporal_leakage",
                time_col,
                f"train extends to {a.max()} but test starts {b.min()}",
            )
        ]
    return []


def categorical_target_leaks(df, target, threshold=0.95, min_rows=50):
    """High-cardinality categorical columns that encode the target.

    Distinct from id_like (which checks uniqueness ratio >=0.98): this checks
    whether knowing the category alone determines (or strongly predicts) the
    target via conditional-entropy reduction. A column is flagged when each
    category is near-pure w.r.t. the target while having many categories.
    Requires >=50 rows to avoid small-sample false positives.
    """
    out = []
    if len(df) < min_rows or target not in df.columns:
        return out
    y = df[target].astype("category").cat.codes
    y_counts = y.value_counts(normalize=True)
    if len(y_counts) < 2:
        return out
    base_entropy = float(-(y_counts * np.log(y_counts)).sum())
    if base_entropy == 0:
        return out
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        # Only categorical/object/string columns with many categories qualify.
        if s.dtype.kind not in "OSU" and str(s.dtype) not in ("category", "object"):
            continue
        n_unique = s.nunique(dropna=True)
        # High cardinality: at least 10 categories, but not so many it's an ID
        # (id_like handles the >=98% uniqueness case; require <= 50% unique ratio).
        if n_unique < 10 or n_unique / max(len(df), 1) > 0.5:
            continue
        codes = s.astype("category").cat.codes
        if codes.min() < 0:
            continue
        # Conditional entropy H(Y|X)
        cond = 0.0
        sizes = codes.value_counts()
        for code, count in sizes.items():
            mask = codes == code
            yy = y[mask]
            yc = yy.value_counts(normalize=True)
            cond += (count / len(df)) * float(-(yc * np.log(yc)).sum())
        gain = 1.0 - (cond / base_entropy)
        if gain >= 0.9:
            sev = "high"
        elif gain >= 0.8:
            sev = "medium"
        else:
            continue
        out.append(
            Finding(
                sev,
                "categorical_target_leakage",
                c,
                f"category almost determines target (entropy reduction={gain:.2f}, "
                f"{n_unique} categories in {len(df)} rows); the column likely encodes the target",
            )
        )
    return out


def scan(df, target, threshold=0.95):
    return (
        single_feature_leaks(df, target, threshold)
        + id_like(df, target)
        + categorical_target_leaks(df, target)
    )


def score(findings) -> int:
    pen = {"high": 35, "medium": 15, "low": 5}
    return max(0, 100 - sum(pen.get(f.severity, 0) for f in findings))


def group_risk(df, group_col):
    if group_col not in df.columns:
        return [Finding("low", "group_missing", group_col, "group column not found")]
    sizes = df.groupby(group_col).size()
    share = float(sizes[sizes > 1].sum() / max(len(df), 1))
    if share < 0.5:
        return []
    return [
        Finding(
            "medium",
            "group_leakage_risk",
            group_col,
            f"{share:.0%} of rows share a {group_col} with other rows; a random split leaks entities, split by group",
        )
    ]


def group_overlap(train, test, group_col):
    if group_col not in train.columns or group_col not in test.columns:
        return []
    shared = set(train[group_col].dropna()) & set(test[group_col].dropna())
    if not shared:
        return []
    n = int(test[group_col].isin(shared).sum())
    return [
        Finding(
            "high",
            "group_overlap",
            group_col,
            f"{len(shared)} {group_col} values in both splits ({n / len(test):.1%} of test rows)",
        )
    ]
