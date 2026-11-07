from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .evidence import Evidence, coerce_evidence
from .fingerprint import fingerprint_for


@dataclass
class Finding:
    severity: str
    kind: str
    column: str
    detail: str
    evidence: tuple[Evidence, ...] = ()
    stable_key: str | None = None

    def __post_init__(self) -> None:
        self.evidence = coerce_evidence(self.evidence)
        if self.stable_key is not None:
            if not isinstance(self.stable_key, str):
                raise TypeError("stable_key must be a string or None")
            if len(self.stable_key) > 200 or any(
                ord(ch) < 32 or ord(ch) == 127 for ch in self.stable_key
            ):
                raise ValueError("stable_key is not a valid stable identifier")

    @property
    def fingerprint(self) -> str:
        """Stable identity for the logical finding (versioned SHA-256)."""
        return fingerprint_for(self)

    def to_dict(self, root: str | Path | None = None) -> dict:
        """Backward-compatible JSON payload; evidence and fingerprint are additive."""
        return {
            "severity": self.severity,
            "kind": self.kind,
            "column": self.column,
            "detail": self.detail,
            "evidence": [item.to_dict() for item in self.evidence],
            "fingerprint": fingerprint_for(self, root=root),
        }


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
                sev,
                "target_leakage",
                c,
                f"alone predicts target ({scoring}={sc:.3f})",
                evidence=(
                    Evidence(
                        source="data",
                        kind="statistic",
                        description=(
                            "one feature predicts the target at or above the "
                            "leakage threshold"
                        ),
                        location=c,
                        observed=f"{scoring}={sc:.3f}",
                        expected=f"{scoring} < {threshold:.3f}",
                    ),
                ),
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
            evidence=(
                Evidence(
                    source="split",
                    kind="overlap",
                    description="test rows are exact duplicates of train rows",
                    observed=f"{n} of {len(test)} test rows shared",
                    expected="0 shared rows",
                ),
            ),
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
                evidence=(
                    Evidence(
                        source="split",
                        kind="ordering",
                        description=(
                            "the training window ends after the test window starts"
                        ),
                        location=time_col,
                        observed="max(train time) > min(test time)",
                        expected="max(train time) <= min(test time)",
                    ),
                ),
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
            evidence=(
                Evidence(
                    source="split",
                    kind="overlap",
                    description=(
                        "group identifiers occur in both splits, so a model can "
                        "memorise the same entity"
                    ),
                    location=group_col,
                    observed=f"{len(shared)} shared groups; {n} affected test rows",
                    expected="0 shared groups",
                ),
            ),
        )
    ]


def _bounded_pair(train, test, max_rows):
    """Deterministic, bounded copies so duplicate checks stay cheap on large files."""
    if max_rows <= 0:
        return None, None
    a = train if len(train) <= max_rows else train.sample(max_rows, random_state=0)
    b = test if len(test) <= max_rows else test.sample(max_rows, random_state=0)
    return a.reset_index(drop=True), b.reset_index(drop=True)


def feature_overlap(train, test, target=None, max_rows=50_000):
    """Test rows whose feature vector also appears in train.

    ``split_overlap`` hashes every shared column, including the label.  Rows that
    share features but carry a different label are invisible there, yet they are
    exactly the duplicates that inflate held-out scores (the model has seen the
    features and memorised one of the conflicting labels).
    """
    a, b = _bounded_pair(train, test, max_rows)
    if a is None or b is None or a.empty or b.empty:
        return []
    cols = [c for c in a.columns if c in b.columns and c != target]
    if not cols:
        return []
    try:
        left = pd.util.hash_pandas_object(a[cols], index=False)
        right = pd.util.hash_pandas_object(b[cols], index=False)
    except (TypeError, ValueError):
        return []
    overlap = right.isin(set(left))
    n = int(overlap.sum())
    if n == 0:
        return []
    conflicts = 0
    if target is not None and target in a.columns and target in b.columns:
        train_labels = (
            pd.DataFrame(
                {
                    "hash": left.to_numpy(),
                    "label": a[target].astype(str).to_numpy(),
                }
            )
            .groupby("hash")["label"]
            .agg(set)
            .to_dict()
        )
        test_labels = (
            pd.DataFrame(
                {
                    "hash": right[overlap].to_numpy(),
                    "label": b.loc[overlap, target].astype(str).to_numpy(),
                }
            )
            .drop_duplicates()
            .itertuples(index=False)
        )
        conflicts = sum(
            1 for h, label in test_labels if label not in train_labels.get(h, set())
        )
    rows = "row" if n == 1 else "rows"
    if conflicts:
        cro = "row" if conflicts == 1 else "rows"
        return [
            Finding(
                "high",
                "feature_overlap",
                "*",
                f"{conflicts} test {cro} share a feature vector with train rows but "
                f"carry a different label; duplicate samples with conflicting labels "
                f"leak across the split",
            )
        ]
    return [
        Finding(
            "medium",
            "feature_overlap",
            "*",
            f"{n} test {rows} ({n / len(b):.1%}) share an identical feature vector with "
            f"train rows (the label column is ignored); deduplicate before splitting",
        )
    ]


def near_duplicate_overlap(train, test, target=None, max_rows=20_000, max_columns=64):
    """Rows that match after case/whitespace normalisation and numeric rounding.

    This is deliberately conservative: it does not claim fuzzy matching, only
    that two rows are identical once text is lower-cased/stripped and numbers
    are rounded to four decimals.  The label column is ignored when ``target``
    is given, because near-duplicates with different labels are the most
    damaging case.  A cheap deterministic sample keeps runtime bounded.
    """
    a, b = _bounded_pair(train, test, max_rows)
    if a is None or b is None or a.empty or b.empty:
        return []
    cols = [c for c in a.columns if c in b.columns and c != target]
    if not cols or len(cols) > max_columns:
        return []
    try:
        norm_a = a[cols].copy()
        norm_b = b[cols].copy()
        for c in cols:
            if pd.api.types.is_numeric_dtype(
                norm_a[c]
            ) and pd.api.types.is_numeric_dtype(norm_b[c]):
                norm_a[c] = norm_a[c].round(4)
                norm_b[c] = norm_b[c].round(4)
            else:
                norm_a[c] = norm_a[c].astype(str).str.strip().str.lower()
                norm_b[c] = norm_b[c].astype(str).str.strip().str.lower()
        left = pd.util.hash_pandas_object(norm_a, index=False)
        right = pd.util.hash_pandas_object(norm_b, index=False)
    except (TypeError, ValueError):
        return []
    n = int(right.isin(set(left)).sum())
    if n == 0:
        return []
    return [
        Finding(
            "medium",
            "near_duplicate_overlap",
            "*",
            f"{n} test rows ({n / len(b):.1%}) match a train row after "
            f"case/whitespace normalisation and rounding numbers to 4 decimals; "
            f"check for near-duplicate samples before splitting",
            evidence=(
                Evidence(
                    source="split",
                    kind="duplicate",
                    description=(
                        "test rows match train rows after case/whitespace and "
                        "numeric normalisation"
                    ),
                    observed=f"{n} of {len(b)} test rows matched",
                    expected="0 matches after normalisation",
                ),
            ),
        )
    ]
