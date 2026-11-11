"""``assemble.build`` is the shared, single-policy payload path (Phase 13 M2)."""

from __future__ import annotations

from leaklens import assemble
from leaklens.baseline import Baseline
from leaklens.leakage import Finding, score
from leaklens.suppression import parse_rules


def _findings() -> list[Finding]:
    return [
        Finding("high", "target_leakage", "leak", "copied"),
        Finding("low", "parse_error", "a.py", "bad"),
    ]


def test_build_without_policy_matches_finding_payloads():
    findings = _findings()
    sections = assemble.build(findings)
    assert sections.trust_score == score(findings)
    assert sections.findings == [f.to_dict() for f in findings]
    assert sections.counts == {"high": 1, "medium": 0, "low": 1, "total": 2}
    assert sections.policy == {}
    assert sections.resolved == []


def test_build_labels_suppressed_findings_without_changing_them():
    findings = _findings()
    rules = parse_rules(
        [{"fingerprint": findings[0].fingerprint, "reason": "documented"}]
    )
    sections = assemble.build(findings, rules=rules)
    assert sections.trust_score == score([findings[1]])
    assert sections.findings[0]["status"] == "suppressed"
    assert sections.findings[0]["severity"] == findings[0].severity
    assert sections.policy["suppressions"]["suppressed"] == 1


def test_build_serializes_the_baseline_policy_and_resolved_entries():
    findings = _findings()
    baseline = Baseline.from_findings(findings)
    sections = assemble.build(findings, baseline=baseline, baseline_path="base.json")
    assert sections.policy["baseline"]["path"] == "base.json"
    assert sections.policy["baseline"]["unchanged"] == 2


def test_public_api_uses_the_shared_sections(monkeypatch):
    from leaklens import public_api
    from leaklens.scanner import scan_source

    source = "train_test_split(X, y)"
    expected = assemble.build(scan_source(source))
    payload = public_api.run_code_job(source, "snippet.py")
    assert payload["trust_score"] == expected.trust_score
    assert payload["counts"] == expected.counts
    assert [f["kind"] for f in payload["findings"]] == [
        f["kind"] for f in expected.findings
    ]
