"""AI evidence anchored to the deterministic evidence graph (M4)."""

from __future__ import annotations

import json

import pytest

from leaklens import cli
from leaklens.ai.analysis import analyze
from leaklens.ai.schemas import SCHEMA_VERSION, EvidenceSpan
from leaklens.ai.semantics import CodeSource
from leaklens.intelligence.provenance import build_evidence_graph
from leaklens.leakage import Finding

LEAKY = (
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "X = StandardScaler().fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
)


def finding(kind: str, column: str, severity: str = "high") -> Finding:
    return Finding(severity, kind, column, "deterministic detail")


def graph_ids(findings) -> set[str]:
    return {item.evidence_id for item in build_evidence_graph(findings).evidence}


def code_spans(report):
    return [
        span
        for hypothesis in report.to_dict()["hypotheses"]
        for span in hypothesis["evidence"]
        if span["source"] == "code"
    ]


def test_code_evidence_links_to_the_matching_graph_node():
    findings = (finding("preprocess_before_split", "model.py:3"),)
    report = analyze(findings, code=(CodeSource(label="model.py", source=LEAKY),))

    linked = [span for span in code_spans(report) if span["graph_node_id"]]

    assert linked
    assert all(span["graph_node_id"] in graph_ids(findings) for span in linked)
    assert linked[0]["graph_node_id"] == "scanner-000000"


def test_multiple_files_link_to_their_own_graph_nodes():
    findings = (
        finding("preprocess_before_split", "a.py:3"),
        finding("preprocess_before_split", "b.py:3"),
    )
    report = analyze(
        findings,
        code=(
            CodeSource(label="a.py", source=LEAKY),
            CodeSource(label="b.py", source=LEAKY),
        ),
    )
    by_location = {
        span["location"].split(":")[0]: span["graph_node_id"]
        for span in code_spans(report)
    }

    assert by_location["a.py"] == "scanner-000000"
    assert by_location["b.py"] == "scanner-000001"


def test_missing_findings_leave_every_span_unlinked():
    report = analyze(code=(CodeSource(label="model.py", source=LEAKY),))

    assert report.hypotheses
    assert all(span["graph_node_id"] is None for span in code_spans(report))


def test_ambiguous_matches_are_left_null_instead_of_guessed():
    findings = (
        finding("preprocess_before_split", "model.py:3"),
        finding("no_random_state", "model.py:4", "low"),
    )
    report = analyze(findings, code=(CodeSource(label="model.py", source=LEAKY),))

    assert any(span["graph_node_id"] for span in code_spans(report)) is False


def test_model_evidence_never_claims_a_deterministic_node():
    findings = (finding("preprocess_before_split", "model.py:3"),)
    report = analyze(findings, code=(CodeSource(label="model.py", source=LEAKY),))
    model_spans = [
        span
        for hypothesis in report.to_dict()["hypotheses"]
        for span in hypothesis["evidence"]
        if span["source"] == "model"
    ]

    assert model_spans
    assert all(span["graph_node_id"] is None for span in model_spans)


def test_links_are_deterministic_and_bounded():
    findings = (finding("preprocess_before_split", "model.py:3"),)
    source = (CodeSource(label="model.py", source=LEAKY),)

    first = analyze(findings, code=source).to_dict()
    second = analyze(findings, code=source).to_dict()
    spans = [span for item in first["hypotheses"] for span in item["evidence"]]
    ids = graph_ids(findings)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len([span for span in spans if span["graph_node_id"]]) <= len(spans)
    assert {span["graph_node_id"] for span in spans if span["graph_node_id"]} <= ids


def test_linking_does_not_change_the_graph_or_the_trust_score():
    findings = (finding("preprocess_before_split", "model.py:3"),)
    before = build_evidence_graph(findings).to_dict()
    report = analyze(findings, code=(CodeSource(label="model.py", source=LEAKY),))
    after = build_evidence_graph(findings).to_dict()

    assert before == after
    assert report.deterministic_trust_score == 65
    assert report.deterministic_findings == 1
    assert report.changes_deterministic_findings is False


def test_legacy_spans_and_payloads_stay_valid():
    span = EvidenceSpan(
        evidence_id="ev-abc",
        source="code",
        kind="data_flow",
        location="model.py:3",
        excerpt="x = 1",
    )

    assert span.graph_node_id is None
    assert span.to_dict()["graph_node_id"] is None
    assert SCHEMA_VERSION == "1.1"
    assert EvidenceSpan(**span.to_dict()) == span

    with pytest.raises(ValueError, match="graph_node_id"):
        EvidenceSpan(
            evidence_id="ev-abc",
            source="code",
            kind="data_flow",
            location="model.py:3",
            excerpt="x = 1",
            graph_node_id="not a node id!",
        )


def test_serialized_links_round_trip():
    findings = (finding("preprocess_before_split", "model.py:3"),)
    report = analyze(findings, code=(CodeSource(label="model.py", source=LEAKY),))
    encoded = json.loads(json.dumps(report.to_dict()))

    for hypothesis in encoded["hypotheses"]:
        for span in hypothesis["evidence"]:
            assert span["graph_node_id"] in graph_ids(findings) | {None}
            assert EvidenceSpan(**span).to_dict() == span


def test_cli_scan_links_evidence_to_the_graph_node(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.py").write_text(LEAKY, encoding="utf-8")

    assert cli.main(["--scan", "model.py", "--ai", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    linked = [
        span
        for hypothesis in payload["ai"]["hypotheses"]
        for span in hypothesis["evidence"]
        if span["graph_node_id"]
    ]

    assert linked
    assert all(span["graph_node_id"].startswith("scanner-") for span in linked)


def test_intelligence_only_output_has_no_ai_linkage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.py").write_text(LEAKY, encoding="utf-8")

    assert cli.main(["--scan", "model.py", "--intelligence", "--json"]) == 1
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert "ai" not in payload
    assert "graph_node_id" not in out
    assert payload["intelligence"]["evidence"]
