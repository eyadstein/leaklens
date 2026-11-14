"""Versioned baselines: validation, deterministic output, comparison, writes."""

import json
import os
import socket
import threading

import pytest

from leaklens.baseline import (
    BASELINE_SCHEMA,
    BASELINE_VERSION,
    Baseline,
    BaselineEntry,
    BaselineError,
    compare,
)
from leaklens.fingerprint import SCHEMA_VERSION
from leaklens.leakage import Finding


def _findings():
    return [
        Finding("high", "target_leakage", "leak", "predicts target"),
        Finding("medium", "id_like", "uid", "looks like an identifier"),
        Finding("low", "missingness_group_signal", "x", "group pattern"),
    ]


def _fp():
    return "a" * 64


def test_serialization_is_deterministic_and_sorted():
    first = Baseline.from_findings(_findings())
    second = Baseline.from_findings(list(reversed(_findings())))
    assert first.to_json() == second.to_json()
    payload = json.loads(first.to_json())
    fingerprints = [item["fingerprint"] for item in payload["findings"]]
    assert fingerprints == sorted(fingerprints)
    assert first.to_json().endswith("\n")


def test_round_trip_preserves_entries(tmp_path):
    baseline = Baseline.from_findings(_findings())
    path = baseline.write(tmp_path / "base.json")
    assert Baseline.load(path) == baseline


def test_baseline_never_contains_detail_or_rows(tmp_path):
    baseline = Baseline.from_findings(_findings())
    text = baseline.write(tmp_path / "base.json").read_text(encoding="utf-8")
    assert "predicts target" not in text
    assert set(json.loads(text)) == {
        "schema",
        "version",
        "fingerprint_schema",
        "tool_version",
        "findings",
    }


def test_write_refuses_overwrite_without_update(tmp_path):
    baseline = Baseline.from_findings(_findings())
    path = baseline.write(tmp_path / "base.json")
    with pytest.raises(BaselineError, match="already exists"):
        baseline.write(path)
    assert baseline.write(path, update=True) == path


def test_write_requires_an_existing_parent(tmp_path):
    baseline = Baseline.from_findings(_findings())
    with pytest.raises(BaselineError, match="does not exist"):
        baseline.write(tmp_path / "missing" / "base.json")


def test_write_refuses_a_directory(tmp_path):
    with pytest.raises(BaselineError, match="directory"):
        Baseline.from_findings(_findings()).write(tmp_path)


@pytest.mark.skipif(
    os.name == "nt", reason="creating symlinks needs elevated Windows privileges"
)
def test_write_refuses_a_symlink(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(BaselineError, match="symlink"):
        Baseline.from_findings(_findings()).write(link, update=True)


def test_load_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(BaselineError, match="--write-baseline"):
        Baseline.load(tmp_path / "nope.json")


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "base.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineError, match="invalid JSON"):
        Baseline.load(path)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda doc: doc.update(schema="other"), "schema"),
        (lambda doc: doc.update(version=2), "version"),
        (lambda doc: doc.update(fingerprint_schema=99), "fingerprint schema"),
        (lambda doc: doc.update(extra=1), "unsupported baseline keys"),
        (lambda doc: doc["findings"][0].update(extra=1), "unsupported keys"),
        (lambda doc: doc["findings"][0].pop("kind"), "missing: kind"),
        (lambda doc: doc["findings"][0].update(fingerprint="ABC"), "fingerprint"),
        (lambda doc: doc["findings"].append(dict(doc["findings"][0])), "duplicate"),
        (lambda doc: doc.update(findings={}), "findings must be an array"),
    ],
)
def test_load_rejects_malformed_baselines(tmp_path, mutate, message):
    doc = Baseline.from_findings(_findings()).to_dict()
    mutate(doc)
    path = tmp_path / "base.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(BaselineError, match=message):
        Baseline.load(path)


def test_incompatible_fingerprint_schema_is_refused(tmp_path):
    doc = Baseline.from_findings(_findings()).to_dict()
    doc["fingerprint_schema"] = SCHEMA_VERSION + 1
    path = tmp_path / "base.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(BaselineError, match="regenerate the baseline"):
        Baseline.load(path)


def test_from_dict_accepts_the_documented_schema():
    doc = {
        "schema": BASELINE_SCHEMA,
        "version": BASELINE_VERSION,
        "fingerprint_schema": SCHEMA_VERSION,
        "tool_version": "0.13.0",
        "findings": [
            {"fingerprint": _fp(), "kind": "k", "severity": "low", "column": ""}
        ],
    }
    baseline = Baseline.from_dict(doc)
    assert baseline.entries[0].fingerprint == _fp()


def test_compare_classifies_new_unchanged_changed_and_resolved():
    kept = Finding("medium", "id_like", "uid", "looks like an identifier")
    baseline = Baseline.from_findings(
        [kept, Finding("high", "target_leakage", "removed", "gone")]
    )
    promoted = Finding("high", "target_leakage", "leak2", "new")
    same_identity_higher = Finding("high", "id_like", "uid", "same fingerprint")
    assert same_identity_higher.fingerprint == kept.fingerprint
    diff = compare(baseline, [same_identity_higher, promoted, _findings()[0]])
    assert [f.column for f in diff.new] == ["leak", "leak2"]
    assert [f.column for f in diff.changed] == ["uid"]
    assert diff.unchanged == ()
    assert [entry.column for entry in diff.resolved] == ["removed"]
    assert diff.counts == {"new": 2, "unchanged": 0, "changed": 1, "resolved": 1}


def test_entry_validation_rejects_bad_fingerprint():
    with pytest.raises(BaselineError):
        BaselineEntry("short", "kind", "low", "")


def test_load_refuses_a_directory(tmp_path):
    with pytest.raises(BaselineError, match="not a regular file"):
        Baseline.load(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX named pipes")
def test_load_refuses_a_fifo_without_blocking(tmp_path):
    fifo = tmp_path / "baseline.json"
    os.mkfifo(fifo)
    outcome = {}

    def load():
        try:
            Baseline.load(fifo)
        except BaselineError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=load, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "Baseline.load blocked on a FIFO"
    assert "error" in outcome


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(socket, "AF_UNIX"), reason="POSIX sockets"
)
def test_load_refuses_a_socket(tmp_path):
    path = tmp_path / "baseline.json"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        with pytest.raises(BaselineError, match="not a regular file"):
            Baseline.load(path)
    finally:
        server.close()
