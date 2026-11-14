"""Phase 18 M5: read-only monitoring intake is diagnostic and never raises."""

from __future__ import annotations

import json

from leaklens.cli import main
from leaklens.leakage import score
from leaklens.monitoring import MAX_REPORT_BYTES, audit_monitoring


def _write(path, payload) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_breached_drift_metric_is_low_diagnostic(tmp_path):
    report = _write(
        tmp_path / "drift.json",
        {
            "kind": "drift",
            "metrics": [
                {"name": "amount", "value": 0.42, "threshold": 0.2},
                {"name": "region", "value": 0.05, "threshold": 0.2},
            ],
        },
    )
    result = audit_monitoring([report], root=tmp_path)
    assert result.status == "ok"
    assert result.reports == 1
    assert result.metrics == 2
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.kind == "monitoring_drift_report"
    assert finding.severity == "low"
    assert finding.diagnostic is True
    assert "amount" in finding.column
    evidence = finding.evidence[0]
    assert evidence.location.endswith(":amount")
    assert evidence.observed == "0.42"
    assert evidence.expected == "<= 0.2"


def test_skew_report_and_explicit_breach_flag(tmp_path):
    report = _write(
        tmp_path / "skew.json",
        {
            "kind": "skew",
            "metrics": [{"name": "age", "value": 3, "breached": True}],
        },
    )
    result = audit_monitoring([report], root=tmp_path)
    assert [f.kind for f in result.findings] == ["monitoring_skew_report"]
    assert result.findings[0].evidence[0].expected == "not breached"


def test_metric_without_a_breach_signal_is_skipped(tmp_path):
    report = _write(
        tmp_path / "drift.json",
        {"kind": "drift", "metrics": [{"name": "amount", "value": 0.4}]},
    )
    result = audit_monitoring([report], root=tmp_path)
    assert result.findings == ()
    assert result.skipped == {"no_breach_signal": 1}


def test_missing_malformed_oversized_and_traversal_are_skips(tmp_path):
    (tmp_path / "malformed.json").write_text("{", encoding="utf-8")
    (tmp_path / "unsupported.json").write_text(
        json.dumps({"kind": "bias", "metrics": []}), encoding="utf-8"
    )
    (tmp_path / "oversized.json").write_bytes(b"{" + b" " * MAX_REPORT_BYTES)

    result = audit_monitoring(
        [
            "missing.json",
            "malformed.json",
            "unsupported.json",
            "oversized.json",
            "../outside.json",
        ],
        root=tmp_path,
    )
    assert result.findings == ()
    assert result.reports == 0
    assert result.skipped == {
        "malformed": 1,
        "missing": 1,
        "outside_root": 1,
        "oversized": 1,
        "unsupported_kind": 1,
    }


def test_invalid_metric_names_and_values_are_skips(tmp_path):
    report = _write(
        tmp_path / "drift.json",
        {
            "kind": "drift",
            "metrics": [
                {"name": "bad\x1bname", "value": 1, "breached": True},
                {"name": "ok", "value": True, "breached": True},
                {"name": "ok2", "value": 1, "breached": "yes"},
            ],
        },
    )
    result = audit_monitoring([report], root=tmp_path)
    assert result.findings == ()
    assert result.skipped == {"invalid_metric": 3}


def test_score_ignores_monitoring_findings(tmp_path):
    report = _write(
        tmp_path / "drift.json",
        {"kind": "drift", "metrics": [{"name": "x", "value": 1, "breached": True}]},
    )
    result = audit_monitoring([report], root=tmp_path)
    assert score(result.findings) == 100


def test_cli_surfaces_the_payload_without_failing_the_gate(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    report = _write(
        tmp_path / "drift.json",
        {"kind": "drift", "metrics": [{"name": "x", "value": 1, "breached": True}]},
    )
    monkeypatch.chdir(tmp_path)
    assert main(["--scan", "clean.py", "--monitoring-report", report, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["monitoring"]["findings"] == 1
    assert payload["monitoring"]["skipped"] == {}
    assert payload["trust_score"] == 100
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "monitoring_drift_report" in kinds
