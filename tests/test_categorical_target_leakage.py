"""Detector 2.5: high-cardinality categorical target leakage."""

import numpy as np
import pandas as pd

from leaklens.leakage import categorical_target_leaks, scan


def _df_with_cat_target_encoding(n=200, seed=0):
    r = np.random.default_rng(seed)
    # 20 categories each strongly associated with the target: cat i -> target i%2
    y = r.integers(0, 2, size=n)
    # make high-cardinality by refining: cat_i_sub_j
    base = [f"c{int(v)}" for v in y]
    # add 20 distinct categories overall (10 per class)
    sub = r.integers(0, 10, size=n)
    cat_col = [f"{b}_{s}" for b, s in zip(base, sub)]
    return pd.DataFrame({"cat": cat_col, "noise": r.normal(size=n), "y": y}), y


def test_high_cardinality_cat_encoding_target_flagged():
    d, _ = _df_with_cat_target_encoding(n=400)
    findings = categorical_target_leaks(d, "y")
    assert any(f.column == "cat" and f.severity == "high" for f in findings), findings
    assert all(f.kind == "categorical_target_leakage" for f in findings)


def test_integrated_into_scan():
    d, _ = _df_with_cat_target_encoding(n=400)
    assert any(f.kind == "categorical_target_leakage" for f in scan(d, "y"))


def test_random_categorical_not_flagged():
    r = np.random.default_rng(1)
    n = 400
    y = r.integers(0, 2, size=n)
    cat = [f"c{r.integers(0, 20)}" for _ in range(n)]  # independent of y
    d = pd.DataFrame({"cat": cat, "noise": r.normal(size=n), "y": y})
    findings = categorical_target_leaks(d, "y")
    assert not [f for f in findings if f.column == "cat"], findings


def test_low_cardinality_not_flagged():
    r = np.random.default_rng(2)
    n = 400
    y = r.integers(0, 2, size=n)
    # only 2 categories (low cardinality) — even if predictive, not our detector
    cat = [f"c{int(v)}" for v in y]
    d = pd.DataFrame({"cat": cat, "y": y})
    assert not [f for f in categorical_target_leaks(d, "y") if f.column == "cat"]


def test_small_sample_not_flagged():
    r = np.random.default_rng(3)
    n = 30  # below min_rows
    y = r.integers(0, 2, size=n)
    cat = [f"c{i % 20}" for i in range(n)]
    d = pd.DataFrame({"cat": cat, "y": y})
    assert categorical_target_leaks(d, "y") == []


def test_numeric_column_not_flagged():
    r = np.random.default_rng(4)
    n = 200
    y = (r.normal(size=n) > 0).astype(int)
    d = pd.DataFrame({"num": r.normal(size=n), "y": y})
    assert not [f for f in categorical_target_leaks(d, "y") if f.column == "num"]


def test_missing_target_graceful():
    d = pd.DataFrame({"a": list(range(100)), "b": [f"x{i % 10}" for i in range(100)]})
    assert categorical_target_leaks(d, "nonexistent") == []


def test_constant_target_graceful():
    n = 100
    d = pd.DataFrame({"cat": [f"c{i % 15}" for i in range(n)], "y": [1] * n})
    assert categorical_target_leaks(d, "y") == []


def test_id_like_still_separate():
    # a near-unique id column is id_like (ratio>=0.98), NOT categorical_target_leakage
    r = np.random.default_rng(5)
    n = 100
    y = r.integers(0, 2, size=n)
    uid = [f"u{i}" for i in range(n)]
    d = pd.DataFrame({"uid": uid, "y": y})
    findings = categorical_target_leaks(d, "y")
    assert not [f for f in findings if f.column == "uid"], findings
