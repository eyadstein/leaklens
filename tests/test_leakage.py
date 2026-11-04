import numpy as np
import pandas as pd

from leaklens.cli import main
from leaklens.leakage import scan, score, split_overlap, temporal_overlap


def _df(n=600, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (x + r.normal(scale=1.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "noise": r.normal(size=n), "y": y})


def test_clean_has_no_high():
    assert not [f for f in scan(_df(), "y") if f.severity == "high"]


def test_target_leak_detected():
    d = _df()
    d["leak"] = d["y"] * 2.0
    assert any(f.column == "leak" and f.severity == "high" for f in scan(d, "y"))


def test_id_like_detected():
    d = _df()
    d["uid"] = [f"u{i}" for i in range(len(d))]
    assert any(f.kind == "id_like" for f in scan(d, "y"))


def test_split_overlap():
    d = _df()
    assert split_overlap(d.iloc[:400], d.iloc[300:])


def test_temporal_overlap():
    tr = pd.DataFrame({"t": pd.date_range("2024-02-01", periods=10)})
    te = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=10)})
    assert temporal_overlap(tr, te, "t")
    assert not temporal_overlap(te, tr, "t")


def test_score_and_cli(tmp_path):
    d = _df()
    d["leak"] = d["y"] * 2.0
    f = tmp_path / "d.csv"
    d.to_csv(f, index=False)
    assert score(scan(d, "y")) < 100
    assert main([str(f), "--target", "y"]) == 1


def test_single_feature_leaks_skips_uncrossvalidatable_columns(monkeypatch):
    """A column whose cross-validation raises must be skipped, not crash the scan."""
    from leaklens import leakage

    d = _df(n=600)
    real = leakage.cross_val_score
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # first candidate column raises
            raise ValueError("cannot fit")
        return real(*args, **kwargs)

    monkeypatch.setattr(leakage, "cross_val_score", flaky)
    findings = leakage.single_feature_leaks(d, "y", threshold=0.99)
    # Must not raise, and must still be able to run the remaining columns.
    assert isinstance(findings, list)
