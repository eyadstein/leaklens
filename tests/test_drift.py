"""Opt-in drift diagnostics: signals, safeguards, privacy, score isolation."""

import numpy as np
import pandas as pd

from leaklens.drift import DEFAULT_SETTINGS, DriftSettings, audit_drift
from leaklens.leakage import score

ENABLED = DriftSettings(enabled=True)


def _scan(train, test, **kwargs):
    kwargs.setdefault("settings", ENABLED)
    return audit_drift(train, test, **kwargs)


def test_numeric_shift_is_low_diagnostic_and_explained():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(0.0, 1.0, 500)})
    test = pd.DataFrame({"x": rng.normal(2.0, 1.0, 500)})
    report = _scan(train, test)
    finding = next(f for f in report.findings if f.kind == "distribution_shift")
    assert finding.severity == "low"
    assert finding.diagnostic is True
    assert finding.evidence[0].kind == "statistic"
    assert "ks=" in finding.evidence[0].observed
    assert report.status == "ok"
    assert score(report.findings) == 100


def test_categorical_and_missingness_shifts_are_detected():
    categorical = _scan(
        pd.DataFrame({"segment": ["a"] * 320 + ["b"] * 80}),
        pd.DataFrame({"segment": ["a"] * 80 + ["b"] * 320}),
    )
    assert "distribution_shift" in {f.kind for f in categorical.findings}
    rng = np.random.default_rng(2)
    n = 500
    train = pd.DataFrame(
        {"x": np.where(rng.random(n) < 0.05, np.nan, rng.normal(size=n))}
    )
    test = pd.DataFrame(
        {"x": np.where(rng.random(n) < 0.60, np.nan, rng.normal(size=n))}
    )
    missing = _scan(train, test)
    finding = next(f for f in missing.findings if f.kind == "distribution_shift")
    assert "missingness=" in finding.evidence[0].observed


def test_identical_distributions_are_quiet():
    rng = np.random.default_rng(3)
    values = rng.normal(size=500)
    report = _scan(pd.DataFrame({"x": values}), pd.DataFrame({"x": values.copy()}))
    assert report.findings == ()
    assert report.checked_columns == 1


def test_small_samples_are_suppressed():
    rng = np.random.default_rng(4)
    report = _scan(
        pd.DataFrame({"x": rng.normal(size=50)}),
        pd.DataFrame({"x": rng.normal(3.0, 1.0, 50)}),
    )
    assert report.status == "insufficient_data"
    assert report.findings == ()
    assert report.skipped["small_sample"] == 1


def test_constant_and_all_missing_columns_are_skipped():
    constant = _scan(
        pd.DataFrame({"x": np.ones(500)}), pd.DataFrame({"x": np.ones(500)})
    )
    assert constant.findings == ()
    assert constant.skipped["constant"] == 1
    missing = _scan(
        pd.DataFrame({"x": [np.nan] * 500}), pd.DataFrame({"x": [np.nan] * 500})
    )
    assert missing.findings == ()
    assert missing.skipped["constant"] == 1


def test_high_cardinality_is_capped_and_hides_labels():
    train = pd.DataFrame({"code": [f"c{i}" for i in range(50)] * 10})
    test = pd.DataFrame({"code": [f"d{i}" for i in range(50)] * 10})
    report = _scan(train, test)
    finding = next(f for f in report.findings if f.kind == "distribution_shift")
    evidence = [item.to_dict() for item in finding.evidence]
    text = " ".join([finding.detail, *map(str, evidence)])
    assert "c0" not in text and "d0" not in text
    assert "tvd=" in finding.evidence[0].observed


def test_target_column_is_never_compared():
    rng = np.random.default_rng(5)
    values = rng.normal(size=500)
    train = pd.DataFrame({"x": values, "y": [0] * 500})
    test = pd.DataFrame({"x": values.copy(), "y": [1] * 500})
    report = _scan(train, test, target="y")
    assert report.findings == ()
    assert report.checked_columns == 1


def test_without_test_split_or_without_opt_in_nothing_runs():
    rng = np.random.default_rng(6)
    train = pd.DataFrame({"x": rng.normal(size=500)})
    assert audit_drift(train, None, settings=ENABLED).status == "skipped"
    assert audit_drift(train, train, settings=DEFAULT_SETTINGS).status == "disabled"
    assert audit_drift(train, train).findings == ()


def test_output_is_deterministic():
    rng = np.random.default_rng(7)
    train = pd.DataFrame({"x": rng.normal(size=400)})
    test = pd.DataFrame({"x": rng.normal(1.0, 1.0, 400)})
    first = _scan(train, test)
    second = _scan(train, test)
    assert [f.fingerprint for f in first.findings] == [
        f.fingerprint for f in second.findings
    ]
    assert first.to_dict() == second.to_dict()


def test_report_payload_is_sorted_and_aggregate_only():
    rng = np.random.default_rng(8)
    train = pd.DataFrame({"x": rng.normal(size=400)})
    test = pd.DataFrame({"x": rng.normal(2.0, 1.0, 400)})
    payload = _scan(train, test).to_dict()
    assert list(payload["skipped"]) == sorted(payload["skipped"])
    assert set(payload) == {
        "status",
        "rows",
        "test_rows",
        "checked_columns",
        "skipped",
        "findings",
    }
