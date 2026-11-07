"""Wave 4 integration: the evidence graph stays additive and exit-neutral."""

import json

import pandas as pd

from leaklens.cli import main
from leaklens.intelligence import analyze_findings
from leaklens.report import render


def _frames(tmp_path):
    rows = {
        "row_id": list(range(8)),
        "label": [0, 1, 0, 1, 0, 1, 0, 1],
        "target_copy": [0, 1, 0, 1, 0, 1, 0, 1],
        "group": ["a", "a", "b", "b", "a", "a", "b", "b"],
    }
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(train, index=False)
    pd.DataFrame(rows).to_csv(test, index=False)
    return train, test


def _args(train, test):
    return [
        str(train),
        "--target",
        "label",
        "--test",
        str(test),
        "--group",
        "group",
        "--json",
    ]


def _run(args, capsys):
    exit_code = main(args)
    return exit_code, json.loads(capsys.readouterr().out)


def test_intelligence_keeps_score_findings_and_exit_code(tmp_path, capsys):
    train, test = _frames(tmp_path)
    base = _args(train, test)

    plain_exit, plain = _run(base, capsys)
    enriched_exit, enriched = _run([*base, "--intelligence"], capsys)

    assert plain_exit == enriched_exit == 1
    assert plain["trust_score"] == enriched["trust_score"]
    assert plain["findings"] == enriched["findings"]
    assert "intelligence" not in plain
    assert enriched["intelligence"]["finding_count"] == len(plain["findings"])


def test_intelligence_json_is_deterministic(tmp_path, capsys):
    train, test = _frames(tmp_path)
    args = [*_args(train, test), "--intelligence"]

    _, first = _run(args, capsys)
    _, second = _run(args, capsys)

    assert first["intelligence"] == second["intelligence"]
    clusters = first["intelligence"]["clusters"]
    assert clusters
    evidence_ids = {item["evidence_id"] for item in first["intelligence"]["evidence"]}
    for cluster in clusters:
        assert set(cluster["evidence_ids"]) <= evidence_ids
    for link in first["intelligence"]["links"]:
        assert link["source_id"] in evidence_ids
        assert link["target_id"] in evidence_ids


def test_measured_impact_is_routed_to_the_split_cluster(tmp_path, capsys):
    train, test = _frames(tmp_path)
    _, result = _run([*_args(train, test), "--intelligence"], capsys)

    by_anchor = {
        cluster["anchor"]: cluster for cluster in result["intelligence"]["clusters"]
    }
    assert any(
        "duplicate training rows" in note
        for note in by_anchor["split"]["measured_impact"]
    )


def test_html_renders_the_evidence_cluster_table(tmp_path, capsys):
    train, test = _frames(tmp_path)
    _, result = _run([*_args(train, test), "--intelligence"], capsys)

    html = render(result)

    assert "Evidence clusters" in html
    for cluster in result["intelligence"]["clusters"]:
        assert cluster["anchor"] in html


def test_report_defaults_keep_the_previous_shape():
    report = analyze_findings([])

    assert report.clusters == ()
    assert report.links == ()
    assert report.to_dict()["clusters"] == ()
    assert report.to_dict()["links"] == ()
