"""Phase 18 M3: skips are enumerated with their own reason and count."""

from __future__ import annotations

import json

import pytest

from leaklens.availability import EngineUnavailable
from leaklens.cli import main
from leaklens.missing_evidence import (
    MAX_ENTRIES,
    MissingRecord,
    build,
    effect_for,
    engine_unavailable,
    from_payload,
    from_skips,
)


def test_skips_are_one_record_per_reason_with_counts():
    records = from_skips(
        "pipeline",
        "pipeline.skipped",
        {"yaml_not_installed": 1, "malformed_yaml": 2, "zero": 0},
    )
    assert [(item.reason, item.count) for item in records] == [
        ("malformed_yaml", 2),
        ("yaml_not_installed", 1),
    ]
    assert all(item.area == "pipeline" for item in records)


def test_absent_extra_malformed_and_unsupported_each_keep_a_reason():
    records = from_payload(
        {
            "pipeline": {"skipped": {"yaml_not_installed": 1}},
            "monitoring": {
                "skipped": {"malformed": 1, "unsupported_kind": 1, "oversized": 1}
            },
        }
    )
    reasons = {item.reason for item in records}
    assert {
        "yaml_not_installed",
        "malformed",
        "unsupported_kind",
        "oversized",
    } <= reasons


def test_engine_unavailable_names_the_missing_extra():
    error = EngineUnavailable(
        'SQL lineage needs sqlglot: pip install "leaklens-ml[sql]"'
    )
    record = engine_unavailable(error, "sql")
    assert record.reason == "engine_unavailable"
    assert record.source == "sql.availability"
    assert "sqlglot" in record.effect


def test_text_document_counts_keep_the_split_as_source():
    records = from_payload({"text": {"skipped": {"train": 2, "test": 0}}})
    assert len(records) == 1
    assert records[0].reason == "document_skipped"
    assert records[0].source == "text.train"
    assert records[0].count == 2


def test_split_limitations_keep_their_message():
    message = "declaration cap (100) reached; remaining entries were not read"
    records = from_payload({"splits": {"limitations": [message]}})
    assert len(records) == 1
    assert records[0].reason == "limitation"
    assert records[0].effect == message


def test_report_orders_entries_and_counts_every_reason():
    records = from_skips("drift", "drift.skipped", {"constant": 2, "small_sample": 1})
    records += from_skips("pipeline", "pipeline.skipped", {"yaml_not_installed": 3})
    report = build(records).to_dict()
    assert [entry["reason"] for entry in report["entries"]] == [
        "constant",
        "small_sample",
        "yaml_not_installed",
    ]
    assert report["counts"] == {
        "constant": 2,
        "small_sample": 1,
        "yaml_not_installed": 3,
    }
    assert report["total"] == 6


def test_entry_cap_is_reported_not_silent():
    records = [
        MissingRecord(
            area="drift",
            reason=f"reason_{index:03d}",
            effect=effect_for("drift"),
            source="drift.skipped",
        )
        for index in range(MAX_ENTRIES + 10)
    ]
    report = build(records)
    assert len(report.entries) == MAX_ENTRIES
    assert report.entries[-1].reason == "entry_cap"
    assert report.entries[-1].count == 11


def test_record_rejects_empty_reasons_and_zero_counts():
    with pytest.raises(ValueError, match="reason"):
        MissingRecord(area="drift", reason=" ", effect="e", source="s")
    with pytest.raises(ValueError, match="count"):
        MissingRecord(area="drift", reason="r", effect="e", source="s", count=0)


def _dataset(path):
    path.write_text("a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")
    return path


def test_cli_enumerates_a_malformed_report_without_touching_the_score(
    tmp_path, monkeypatch, capsys
):
    _dataset(tmp_path / "train.csv")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["train.csv", "--target", "target", "--json"]) == 0
    clean = json.loads(capsys.readouterr().out)

    assert (
        main(
            [
                "train.csv",
                "--target",
                "target",
                "--monitoring-report",
                "bad.json",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["trust_score"] == clean["trust_score"]
    assert payload["monitoring"]["skipped"] == {"malformed": 1}
    counts = payload["missing_evidence"]["counts"]
    assert counts["malformed"] == 1
    assert counts["dataset_too_small"] == 1
    entry = next(
        item
        for item in payload["missing_evidence"]["entries"]
        if item["area"] == "monitoring"
    )
    assert entry["source"] == "monitoring.skipped"
    assert entry["count"] == 1
