import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlflow")
from leaklens.arena import run


def _uri(tmp_path):
    return f"sqlite:///{(tmp_path / 'm.db').as_posix()}"


def _df(n=600, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (x + r.normal(scale=1.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "noise": r.normal(size=n), "y": y})


def test_classification_drops_leaks(tmp_path):
    d = _df()
    d["leak"] = d["y"] * 2.0
    d["uid"] = [f"u{i}" for i in range(len(d))]
    res = run(d, "y", out_dir=str(tmp_path / "models"), tracking_uri=_uri(tmp_path))
    assert len(res) == 3
    assert {"leak", "uid"} <= set(res.attrs["dropped"])
    assert (tmp_path / "models" / "best.joblib").exists()
    assert res["pr_auc"].max() > 0.6


def test_temporal_split(tmp_path):
    d = _df()
    d["t"] = pd.date_range("2024-01-01", periods=len(d), freq="h")
    res = run(
        d,
        "y",
        time_col="t",
        out_dir=str(tmp_path / "models"),
        tracking_uri=_uri(tmp_path),
    )
    assert len(res) == 3


def test_regression(tmp_path):
    r = np.random.default_rng(1)
    x = r.normal(size=500)
    d = pd.DataFrame(
        {"x": x, "noise": r.normal(size=500), "y": 3 * x + r.normal(scale=3, size=500)}
    )
    res = run(d, "y", out_dir=str(tmp_path / "models"), tracking_uri=_uri(tmp_path))
    assert len(res) == 3 and "rmse" in res.columns
