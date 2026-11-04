import pandas as pd

from leaklens.cli import main
from leaklens.leakage import group_overlap, group_risk


def _df():
    return pd.DataFrame(
        {"pid": [1, 1, 2, 2, 3, 3], "x": range(6), "y": [0, 1, 0, 1, 0, 1]}
    )


def test_group_risk():
    assert group_risk(_df(), "pid")[0].kind == "group_leakage_risk"
    assert not group_risk(pd.DataFrame({"pid": [1, 2, 3, 4]}), "pid")
    assert group_risk(_df(), "nope")[0].kind == "group_missing"


def test_group_overlap():
    d = _df()
    f = group_overlap(d.iloc[:4], d.iloc[3:], "pid")
    assert f and f[0].severity == "high"
    assert not group_overlap(d.iloc[:2], d.iloc[4:], "pid")


def test_cli_group(tmp_path):
    d = _df()
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    d.iloc[:4].to_csv(a, index=False)
    d.iloc[3:].to_csv(b, index=False)
    assert main([str(a), "--target", "y", "--test", str(b), "--group", "pid"]) == 1
