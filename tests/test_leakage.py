import numpy as np
import pandas as pd
import pytest

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


def test_unique_numeric_measurement_is_not_id_like():
    d = _df()
    d["score"] = np.arange(len(d), dtype=np.int64) * 3
    assert not [f for f in scan(d, "y") if f.kind == "id_like"]


def test_unique_float_measurement_is_not_id_like():
    d = _df()
    d["price"] = np.arange(len(d)) * 1.5
    assert not [f for f in scan(d, "y") if f.kind == "id_like"]


def test_timestamp_counter_is_not_id_like():
    d = _df()
    d["epoch_seconds"] = np.arange(len(d), dtype=np.int64) * 60 + 1_700_000_000
    assert not [f for f in scan(d, "y") if f.kind == "id_like"]


@pytest.mark.parametrize(
    "name",
    ["customer_id", "uuid", "guid", "key", "index", "idx", "row_id", "row"],
)
def test_numeric_identifier_names_stay_id_like(name):
    d = _df()
    d[name] = np.arange(len(d), dtype=np.int64)
    fs = [f for f in scan(d, "y") if f.kind == "id_like"]
    assert fs and fs[0].column == name and fs[0].severity == "medium"


def test_string_id_detail_states_evidence_only():
    d = _df()
    d["uid"] = [f"u{i}" for i in range(len(d))]
    detail = next(f for f in scan(d, "y") if f.kind == "id_like").detail
    assert "100.0% unique values" in detail and "one value per row" in detail
    assert "likely an identifier" not in detail


def test_numeric_id_detail_names_the_name_evidence():
    d = _df()
    d["customer_id"] = np.arange(len(d), dtype=np.int64)
    detail = next(f for f in scan(d, "y") if f.kind == "id_like").detail
    assert "identifier-like name" in detail


def test_split_overlap():
    d = _df()
    assert split_overlap(d.iloc[:400], d.iloc[300:])


def test_temporal_overlap():
    tr = pd.DataFrame({"t": pd.date_range("2024-02-01", periods=10)})
    te = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=10)})
    assert temporal_overlap(tr, te, "t")
    assert not temporal_overlap(te, tr, "t")


def test_temporal_overlap_mixed_awareness():
    """A tz-aware export and a naive export must be comparable, not raise."""
    tr = pd.DataFrame({"t": ["2024-01-01T00:00:00+00:00", "2024-01-05T00:00:00+00:00"]})
    te = pd.DataFrame({"t": ["2024-01-02T00:00:00", "2024-01-06T00:00:00"]})
    fs = temporal_overlap(tr, te, "t")
    assert fs and fs[0].kind == "temporal_leakage"
    assert fs[0].severity == "high"


def test_temporal_overlap_compares_instants_not_clock_text():
    """02:00+02:00 is 00:00Z, before the test window; naive text would say overlap."""
    tr = pd.DataFrame({"t": ["2024-01-01T02:00:00+02:00"]})
    te = pd.DataFrame({"t": ["2024-01-01T01:00:00+00:00"]})
    assert temporal_overlap(tr, te, "t") == []


def test_temporal_overlap_coerces_unparseable_values():
    tr = pd.DataFrame({"t": ["2025-01-01", "not-a-date"]})
    te = pd.DataFrame({"t": ["2025-01-02", "also-bad"]})
    assert temporal_overlap(tr, te, "t") == []


def test_temporal_overlap_all_nat_is_empty():
    tr = pd.DataFrame({"t": ["not-a-date", "junk"]})
    te = pd.DataFrame({"t": ["junk", "more junk"]})
    assert temporal_overlap(tr, te, "t") == []


def test_temporal_overlap_keeps_naive_rendering():
    """Both columns naive: the message must stay naive (no synthetic UTC offset)."""
    tr = pd.DataFrame({"t": pd.date_range("2024-02-01", periods=3)})
    te = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=3)})
    detail = temporal_overlap(tr, te, "t")[0].detail
    assert "2024-02-03" in detail and "2024-01-01" in detail
    assert "+00:00" not in detail


def test_cli_mixed_timezone_reports_finding_without_traceback(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "t": ["2024-01-01T00:00:00+00:00", "2024-01-05T00:00:00+00:00"],
            "v": [1, 3],
            "y": [0, 1],
        }
    ).to_csv(train, index=False)
    pd.DataFrame(
        {"t": ["2024-01-02T00:00:00", "2024-01-06T00:00:00"], "v": [4, 6], "y": [0, 1]}
    ).to_csv(test, index=False)
    # exit 1 because the high temporal finding is real, not because of a traceback
    assert main([str(train), "--test", str(test), "--target", "y", "--time", "t"]) == 1


def test_cli_offset_aware_pair_without_overlap_exits_clean(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "t": ["2024-01-01T02:00:00+02:00", "2024-01-02T02:00:00+02:00"],
            "v": [1, 3],
            "y": [0, 1],
        }
    ).to_csv(train, index=False)
    pd.DataFrame(
        {
            "t": ["2024-01-03T00:00:00+00:00", "2024-01-04T00:00:00+00:00"],
            "v": [4, 6],
            "y": [0, 1],
        }
    ).to_csv(test, index=False)
    assert main([str(train), "--test", str(test), "--target", "y", "--time", "t"]) == 0


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
