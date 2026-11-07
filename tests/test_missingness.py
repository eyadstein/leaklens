"""Split-aware missingness: signals, safeguards, determinism, privacy."""

import numpy as np
import pandas as pd

from leaklens.missingness import (
    DEFAULT_SETTINGS,
    MissingnessSettings,
    audit_missingness,
)


def _binary_target_frame(n=800, seed=1, high=0.70, low=0.03):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    missing = rng.random(n) < np.where(y == 1, high, low)
    x = rng.normal(size=n)
    x[missing] = np.nan
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})


def _kinds(report):
    return {finding.kind for finding in report.findings}


def test_binary_target_association_is_medium_and_explained():
    report = audit_missingness(_binary_target_frame(), "y")
    finding = next(f for f in report.findings if f.kind == "missingness_target_signal")
    assert finding.severity == "medium"
    assert "missing" in finding.detail.lower()
    assert finding.evidence and finding.evidence[0].kind == "statistic"
    assert report.target_checked and report.status == "ok"


def test_string_target_is_handled():
    frame = _binary_target_frame().assign(
        y=lambda d: np.where(d["y"] == 1, "yes", "no")
    )
    assert "missingness_target_signal" in _kinds(audit_missingness(frame, "y"))


def test_numeric_target_correlation_is_detected():
    rng = np.random.default_rng(3)
    n = 600
    missing = rng.random(n) < 0.35
    y = 2.0 * missing + rng.normal(scale=1.0, size=n)
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    report = audit_missingness(frame, "y")
    assert "missingness_target_signal" in _kinds(report)


def _split(train_rate, test_rate, seed=4, column="x"):
    rng = np.random.default_rng(seed)
    n = 400
    train = pd.DataFrame(
        {
            column: np.where(rng.random(n) < train_rate, np.nan, rng.normal(size=n)),
            "noise": rng.normal(size=n),
            "y": rng.integers(0, 2, n),
        }
    )
    test = pd.DataFrame(
        {
            column: np.where(rng.random(n) < test_rate, np.nan, rng.normal(size=n)),
            "noise": rng.normal(size=n),
            "y": rng.integers(0, 2, n),
        }
    )
    return train, test


def test_train_test_missingness_difference_is_reported():
    train, test = _split(0.05, 0.60)
    report = audit_missingness(train, "y", test=test)
    finding = next(f for f in report.findings if f.kind == "missingness_split_shift")
    assert finding.severity == "medium"
    assert "train" in finding.evidence[0].observed
    assert report.split_checked


def test_column_missing_in_one_split_only_is_reported():
    train, test = _split(1.0, 0.0, seed=6)
    assert "missingness_split_shift" in _kinds(audit_missingness(train, "y", test=test))


def test_split_check_is_skipped_without_test_metadata():
    train, _ = _split(0.60, 0.60)
    report = audit_missingness(train, "y")
    assert report.split_checked is False
    assert "missingness_split_shift" not in _kinds(report)


def test_group_association_is_low_and_hides_group_labels():
    rng = np.random.default_rng(9)
    n = 600
    groups = np.where(np.arange(n) % 2 == 0, "site-A", "site-B")
    missing = rng.random(n) < np.where(groups == "site-A", 0.03, 0.60)
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=n),
            "site": groups,
            "y": rng.integers(0, 2, n),
        }
    )
    report = audit_missingness(frame, "y", group="site")
    finding = next(f for f in report.findings if f.kind == "missingness_group_signal")
    assert finding.severity == "low"
    text = str(finding.to_dict())
    assert "site-A" not in text and "site-B" not in text
    assert report.group_checked


def test_ordinary_random_missingness_is_not_reported():
    rng = np.random.default_rng(11)
    n = 400
    frame = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "z": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "y": rng.integers(0, 2, n),
        }
    )
    assert audit_missingness(frame, "y").findings == ()


def test_small_sample_is_explicit_and_never_flagged():
    frame = pd.DataFrame(
        {"x": np.where(np.arange(80) % 2, np.nan, 1.0), "y": [0, 1] * 40}
    )
    report = audit_missingness(frame, "y")
    assert report.status == "insufficient_data"
    assert report.findings == ()
    assert report.skipped["dataset_too_small"] == 1


def test_constant_and_all_missing_columns_are_skipped():
    rng = np.random.default_rng(15)
    n = 400
    frame = pd.DataFrame(
        {
            "constant": np.full(n, 7.0),
            "empty": np.full(n, np.nan),
            "noise": rng.normal(size=n),
            "y": rng.integers(0, 2, n),
        }
    )
    report = audit_missingness(frame, "y")
    assert report.findings == ()
    assert report.skipped["all_missing"] == 1
    assert report.skipped["no_missing"] >= 2


def test_results_are_deterministic():
    frame = _binary_target_frame()
    first = audit_missingness(frame, "y").to_dict()
    second = audit_missingness(frame, "y").to_dict()
    assert first == second
    assert [f.fingerprint for f in audit_missingness(frame, "y").findings] == [
        f.fingerprint for f in audit_missingness(frame, "y").findings
    ]


def test_audit_never_mutates_the_input():
    frame = _binary_target_frame()
    before = frame.copy(deep=True)
    audit_missingness(frame, "y")
    pd.testing.assert_frame_equal(frame, before)


def test_settings_can_disable_the_audit():
    report = audit_missingness(
        _binary_target_frame(), "y", settings=MissingnessSettings(enabled=False)
    )
    assert report.status == "disabled"
    assert report.findings == ()


def test_missing_target_column_is_explicit():
    report = audit_missingness(_binary_target_frame(), "nope")
    assert report.status == "no_target"
    assert report.findings == ()


def test_thresholds_are_configurable_and_bounded():
    assert DEFAULT_SETTINGS.min_rows >= 100
    strict = MissingnessSettings(min_rate_delta=0.99)
    assert (
        audit_missingness(_binary_target_frame(), "y", settings=strict).findings == ()
    )


def test_severity_never_exceeds_medium():
    frame = _binary_target_frame()
    train, test = _split(0.05, 0.60)
    reports = [audit_missingness(frame, "y"), audit_missingness(train, "y", test=test)]
    severities = {f.severity for report in reports for f in report.findings}
    assert severities <= {"low", "medium"}
