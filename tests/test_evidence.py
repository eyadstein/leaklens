"""Structured evidence: validation, privacy, detector migration and serialization."""

import json

import numpy as np
import pandas as pd
import pytest

from leaklens.evidence import MAX_ITEMS, MAX_TEXT, Evidence, coerce_evidence
from leaklens.leakage import (
    Finding,
    group_overlap,
    near_duplicate_overlap,
    scan,
    split_overlap,
    temporal_overlap,
)
from leaklens.sarif import to_sarif

SENTINEL = "sk-live-SENTINEL-9f3a8c"


def _frame(n=120, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    y = (x + r.normal(scale=1.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "noise": r.normal(size=n), "y": y})


def test_existing_positional_constructor_still_works():
    finding = Finding("high", "target_leakage", "leak", "predicts target")
    assert finding.severity == "high"
    assert finding.kind == "target_leakage"
    assert finding.column == "leak"
    assert finding.detail == "predicts target"
    assert finding.evidence == ()


def test_empty_evidence_serializes_as_empty_list():
    payload = Finding("low", "parse_error", "a.py", "bad syntax").to_dict()
    assert payload["evidence"] == []
    assert isinstance(payload["fingerprint"], str)


def test_evidence_serialization_is_compact_and_ordered():
    evidence = Evidence(
        source="data",
        kind="statistic",
        description="one feature predicts the target",
        observed="score=0.99",
        expected="score < 0.95",
    )
    payload = evidence.to_dict()
    assert list(payload) == [
        "source",
        "kind",
        "description",
        "observed",
        "expected",
    ]
    assert evidence.to_dict() == payload
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "unknown", "kind": "statistic", "description": "x"},
        {"source": "data", "kind": "mystery", "description": "x"},
        {"source": "data", "kind": "statistic", "description": ""},
        {"source": "data", "kind": "statistic", "description": "line\nbreak"},
        {"source": "data", "kind": "statistic", "description": "x", "observed": 3},
        {"source": "data", "kind": "statistic", "description": "x" * (MAX_TEXT + 1)},
        {"source": "data", "kind": "statistic", "description": "\x00"},
    ],
)
def test_invalid_evidence_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Evidence(**kwargs)


def test_finding_rejects_unsupported_evidence_values():
    with pytest.raises(TypeError):
        Finding("high", "split_overlap", "*", "x", evidence=[{"source": "split"}])
    with pytest.raises((TypeError, ValueError)):
        Finding("high", "split_overlap", "*", "x", evidence=("not evidence",))


def test_evidence_collection_is_bounded():
    item = Evidence(source="split", kind="overlap", description="rows overlap")
    assert coerce_evidence([item] * MAX_ITEMS) == (item,) * MAX_ITEMS
    with pytest.raises(ValueError):
        coerce_evidence([item] * (MAX_ITEMS + 1))


def _by_kind(findings, kind):
    return next(f for f in findings if f.kind == kind)


def test_target_leakage_has_evidence_without_raw_values():
    frame = _frame()
    frame["leak"] = frame["y"] * 2.0
    frame["secret"] = SENTINEL
    finding = _by_kind(scan(frame, "y"), "target_leakage")
    assert finding.evidence
    first = finding.evidence[0]
    assert (first.source, first.kind) == ("data", "statistic")
    assert "predict" in first.description.lower()
    assert SENTINEL not in json.dumps(finding.to_dict())


def test_split_overlap_evidence_reports_counts_not_rows():
    frame = _frame(n=400)
    frame["secret"] = SENTINEL
    finding = _by_kind(
        split_overlap(frame.iloc[:300], frame.iloc[200:]), "split_overlap"
    )
    assert finding.evidence
    text = json.dumps(finding.to_dict())
    assert SENTINEL not in text
    assert "test rows" in finding.evidence[0].description.lower()


def test_temporal_leakage_evidence_does_not_copy_timestamps():
    train = pd.DataFrame({"t": pd.date_range("2024-02-01", periods=10)})
    test = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=10)})
    finding = _by_kind(temporal_overlap(train, test, "t"), "temporal_leakage")
    evidence_text = json.dumps([item.to_dict() for item in finding.evidence])
    assert "2024-02-01" not in evidence_text and "2024-01-01" not in evidence_text
    assert finding.evidence[0].source == "split"
    assert finding.evidence[0].kind == "ordering"


def test_group_overlap_evidence_never_contains_group_values():
    train = pd.DataFrame({"customer": [SENTINEL] * 3 + ["a", "b"], "v": range(5)})
    test = pd.DataFrame({"customer": [SENTINEL] * 2 + ["c"], "v": range(3)})
    finding = _by_kind(group_overlap(train, test, "customer"), "group_overlap")
    text = json.dumps(finding.to_dict())
    assert SENTINEL not in text
    assert finding.evidence[0].observed
    assert "group" in finding.evidence[0].description.lower()


def test_near_duplicate_overlap_evidence_reports_matches_not_values():
    train = pd.DataFrame({"name": ["A ", SENTINEL], "v": [1.0, 2.0]})
    test = pd.DataFrame({"name": ["a", SENTINEL], "v": [1.0, 2.0]})
    finding = _by_kind(
        near_duplicate_overlap(train, test, target=None),
        "near_duplicate_overlap",
    )
    text = json.dumps(finding.to_dict())
    assert SENTINEL not in text
    assert finding.evidence[0].source == "split"


def test_image_overlap_evidence_reports_pairs_not_bytes(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    rng = np.random.default_rng(0)
    pixels = Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))
    pixels.save(train / "a.png")
    pixels.save(test / "copy.png")
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        test / "other.png"
    )
    from leaklens.images import image_overlap

    findings, _pairs, _stats = image_overlap(train, test)
    finding = _by_kind(findings, "image_overlap")
    assert finding.evidence[0].source == "image"
    assert "pair" in finding.evidence[0].description.lower()
    assert "PNG" not in json.dumps(finding.to_dict())


def test_json_and_sarif_serialization_are_deterministic():
    finding = Finding(
        "high",
        "split_overlap",
        "*",
        "3 test rows also appear in train",
        evidence=(
            Evidence(
                source="split",
                kind="overlap",
                description="test rows also appear in train",
                observed="3 of 10 test rows",
                expected="0 shared rows",
            ),
        ),
    )
    assert finding.to_dict() == finding.to_dict()
    sarif_a = to_sarif([finding], "dataset.csv")
    sarif_b = to_sarif([finding], "dataset.csv")
    assert sarif_a == sarif_b
    result = sarif_a["runs"][0]["results"][0]
    assert result["properties"]["leaklens"]["evidence"] == [
        finding.evidence[0].to_dict()
    ]


def test_api_payload_keeps_evidence_and_adds_no_raw_values():
    pytest.importorskip("fastapi")
    from leaklens.api import _finding_payloads
    from leaklens.policy import apply_policy

    finding = Finding(
        "high",
        "group_overlap",
        "customer",
        "2 customer values in both splits",
        evidence=(
            Evidence(
                source="split",
                kind="overlap",
                description="entity identifiers occur in both splits",
                observed="2 shared groups",
                expected="0 shared groups",
            ),
        ),
    )
    payload = _finding_payloads(apply_policy([finding]))[0]
    assert payload["evidence"] == [finding.evidence[0].to_dict()]
    assert payload["fingerprint"] == finding.fingerprint
