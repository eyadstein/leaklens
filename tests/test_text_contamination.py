"""Benchmark contamination against user-supplied eval fingerprints (hash-only)."""

from __future__ import annotations

import pytest

from leaklens.analysis.text.contamination import (
    contamination_findings,
    load_fingerprints,
    write_fingerprints,
)
from leaklens.analysis.text.normalize import documents_from_texts, text_digest
from leaklens.leakage import score

BASE = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
)
OTHER = "one two three four five six seven eight nine ten eleven twelve thirteen"


def _docs(texts, split="train"):
    return documents_from_texts(texts, split, min_chars=10)[0]


def test_fingerprints_round_trip_and_ignore_comments(tmp_path):
    docs = _docs([BASE, OTHER])
    path = tmp_path / "eval.fp"
    assert write_fingerprints(path, docs) == 2
    path.write_text(
        "# generated\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert load_fingerprints(path) == frozenset({docs[0].digest, docs[1].digest})


def test_invalid_fingerprint_line_is_rejected_without_echoing_it(tmp_path):
    path = tmp_path / "eval.fp"
    path.write_text("not-a-fingerprint\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_fingerprints(path)


def test_matching_is_hash_only_and_reports_counts(tmp_path):
    eval_docs = _docs([BASE], split="test")
    path = tmp_path / "eval.fp"
    write_fingerprints(path, eval_docs)
    train = _docs([BASE, OTHER])
    findings = contamination_findings(train, load_fingerprints(path), "text")

    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == "text_eval_contamination"
    assert finding.severity == "high"
    assert finding.diagnostic is True
    assert finding.column == "text"
    blob = " ".join(
        str(value) for item in finding.evidence for value in item.to_dict().values()
    )
    assert train[0].digest in blob
    assert "matched=1" in blob
    assert BASE not in blob
    assert score(findings) == 100


def test_no_match_produces_no_finding(tmp_path):
    path = tmp_path / "eval.fp"
    write_fingerprints(path, _docs([OTHER], split="test"))
    assert contamination_findings(_docs([BASE]), load_fingerprints(path), "text") == []


def test_fingerprint_is_stable_when_the_match_count_changes(tmp_path):
    eval_docs = _docs([BASE, OTHER], split="test")
    path = tmp_path / "eval.fp"
    write_fingerprints(path, eval_docs)
    fingerprints = load_fingerprints(path)
    one = contamination_findings(_docs([BASE]), fingerprints, "text")[0]
    two = contamination_findings(_docs([BASE, OTHER]), fingerprints, "text")[0]
    assert one.fingerprint == two.fingerprint


def test_an_empty_fingerprint_file_is_valid_and_matches_nothing(tmp_path):
    path = tmp_path / "eval.fp"
    path.write_text("", encoding="utf-8")
    assert load_fingerprints(path) == frozenset()
    assert contamination_findings(_docs([BASE]), frozenset(), "text") == []
    assert text_digest(BASE) != "0000000000000000"
