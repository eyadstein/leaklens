import json

import pytest

from leaklens.intelligence import (
    GraphBaselineError,
    GraphSnapshot,
    compare_graph_snapshots,
    graph_baseline,
)
from leaklens.intelligence.graph_baseline import SNAPSHOT_SCHEMA
from leaklens.intelligence.schemas import (
    ConfidenceAssessment,
    Evidence,
    EvidenceCluster,
    EvidenceLink,
)
from leaklens.leakage import Finding


def _finding(
    severity="high",
    kind="target_leakage",
    column="amount",
    detail="detail",
    evidence=(),
):
    return Finding(severity, kind, column, detail, evidence=evidence)


def _snapshot(findings, *, complete=True, context=None):
    return GraphSnapshot.from_findings(
        findings, complete=complete, context=context or {"mode": "dataset"}
    )


def _confidence(score=0.7, level="high"):
    return ConfidenceAssessment(score, level, ("reason",), 1, 0)


def _manual(evidence, *, clusters=(), links=(), fingerprints=()):
    return GraphSnapshot(
        evidence=tuple(evidence),
        clusters=tuple(clusters),
        links=tuple(links),
        finding_fingerprints=tuple(fingerprints),
    )


def test_snapshot_roundtrip_is_deterministic():
    finding = _finding()
    snapshot = _snapshot([finding])

    assert snapshot.to_json() == snapshot.to_json()
    loaded = GraphSnapshot.from_dict(json.loads(snapshot.to_json()))

    assert loaded == snapshot
    assert loaded.to_json() == snapshot.to_json()
    assert loaded.graph.to_dict() == snapshot.graph.to_dict()


def test_snapshot_write_refuses_overwrite_without_update(tmp_path):
    path = tmp_path / "graph.json"
    snapshot = _snapshot([_finding()])

    assert snapshot.write(path) == path
    assert GraphSnapshot.load(path) == snapshot

    with pytest.raises(GraphBaselineError, match="already exists"):
        snapshot.write(path)

    assert snapshot.write(path, update=True) == path
    assert GraphSnapshot.load(path) == snapshot


def test_snapshot_write_rejects_missing_directory(tmp_path):
    with pytest.raises(GraphBaselineError, match="directory does not exist"):
        _snapshot([]).write(tmp_path / "missing" / "graph.json")


def test_compare_reports_new_unchanged_changed_resolved():
    first = _finding()
    second = _finding("medium", "id_like", "uid", "id")
    changed = _finding("medium", "target_leakage", "amount", "detail")

    same = compare_graph_snapshots(_snapshot([first]), _snapshot([first]))
    assert same.counts == {
        "new": 0,
        "unchanged": 1,
        "changed": 0,
        "resolved": 0,
        "uncertain": 0,
    }

    added = compare_graph_snapshots(_snapshot([]), _snapshot([first]))
    assert added.counts["new"] == 1

    altered = compare_graph_snapshots(_snapshot([first]), _snapshot([changed]))
    assert altered.counts["changed"] == 1
    assert altered.evidence[0].changes == ("metadata",)

    removed = compare_graph_snapshots(_snapshot([first]), _snapshot([second]))
    assert removed.counts["new"] == 1
    assert removed.counts["resolved"] == 1


def test_matching_ignores_order_and_evidence_ids():
    first = _finding()
    second = _finding("medium", "id_like", "uid", "id")

    forward = _snapshot([first, second])
    reverse = _snapshot([second, first])
    assert compare_graph_snapshots(forward, reverse).counts["unchanged"] == 2

    baseline_evidence = Evidence(
        "old-id",
        "scanner",
        "target_leakage",
        first.detail,
        finding_index=0,
        location=first.column,
        metadata=(("severity", first.severity),),
    )
    current_evidence = Evidence(
        "new-id",
        "scanner",
        "target_leakage",
        first.detail,
        finding_index=0,
        location=first.column,
        metadata=(("severity", first.severity),),
    )
    baseline = _manual([baseline_evidence], fingerprints=((0, first.fingerprint),))
    current = _manual([current_evidence], fingerprints=((0, first.fingerprint),))

    diff = compare_graph_snapshots(baseline, current)
    assert diff.counts["unchanged"] == 1
    assert diff.evidence[0].baseline_id == "old-id"
    assert diff.evidence[0].current_id == "new-id"
    assert diff.relationships == ()


def test_relationship_only_change_marks_incident_evidence_changed():
    first = _finding()
    second = _finding("medium", "id_like", "amount", "id")
    baseline_evidence = [
        Evidence(
            "a",
            "scanner",
            "target_leakage",
            first.detail,
            finding_index=0,
            location="amount",
            metadata=(("severity", "high"),),
        ),
        Evidence(
            "b",
            "scanner",
            "id_like",
            second.detail,
            finding_index=1,
            location="amount",
            metadata=(("severity", "medium"),),
        ),
    ]
    fingerprints = ((0, first.fingerprint), (1, second.fingerprint))
    baseline = _manual(
        baseline_evidence,
        links=(EvidenceLink("a", "b", "corroborates", "same anchor"),),
        fingerprints=fingerprints,
    )
    current = _manual(
        baseline_evidence,
        links=(EvidenceLink("b", "a", "corroborates", "same anchor"),),
        fingerprints=fingerprints,
    )

    diff = compare_graph_snapshots(baseline, current)

    assert diff.counts["changed"] == 2
    assert {item.state for item in diff.relationships} == {"added", "removed"}
    assert all("relationships" in item.changes for item in diff.evidence)


def test_cluster_membership_change_is_reported_separately():
    finding = _finding()
    evidence = Evidence(
        "a",
        "scanner",
        "target_leakage",
        finding.detail,
        finding_index=0,
        location="amount",
        metadata=(("severity", "high"),),
    )
    fingerprints = ((0, finding.fingerprint),)
    baseline_cluster = EvidenceCluster(
        "cluster-amount",
        "dataset:amount",
        "feature",
        ("target_leakage",),
        (0,),
        ("a",),
        ("scanner",),
        "high",
        "high",
        _confidence(),
    )
    current_cluster = EvidenceCluster(
        "cluster-uid",
        "dataset:uid",
        "feature",
        ("target_leakage",),
        (0,),
        ("a",),
        ("scanner",),
        "high",
        "high",
        _confidence(),
    )

    diff = compare_graph_snapshots(
        _manual([evidence], clusters=(baseline_cluster,), fingerprints=fingerprints),
        _manual([evidence], clusters=(current_cluster,), fingerprints=fingerprints),
    )

    assert diff.counts["unchanged"] == 1
    assert diff.cluster_counts["new"] == 1
    assert diff.cluster_counts["resolved"] == 1


def test_incomplete_or_different_context_is_uncertain_not_resolved():
    finding = _finding()

    incomplete = _snapshot([finding], complete=False)
    diff = compare_graph_snapshots(incomplete, _snapshot([]))
    assert diff.counts["resolved"] == 0
    assert diff.counts["uncertain"] == 1
    assert diff.resolution_complete is False
    assert any("incomplete" in warning for warning in diff.warnings)

    baseline = _snapshot([finding], context={"target": "a"})
    current = _snapshot([], context={"target": "b"})
    diff = compare_graph_snapshots(baseline, current)
    assert diff.counts["uncertain"] == 1
    assert any("context differs" in warning for warning in diff.warnings)


def test_empty_graphs_compare_cleanly():
    diff = compare_graph_snapshots(_snapshot([]), _snapshot([]))

    assert diff.counts == {
        "new": 0,
        "unchanged": 0,
        "changed": 0,
        "resolved": 0,
        "uncertain": 0,
    }
    assert diff.clusters == ()
    assert diff.relationships == ()


def test_same_fingerprint_with_different_content_is_changed():
    first = _finding()
    altered = _finding(detail="same identity, different message")

    assert altered.fingerprint == first.fingerprint
    diff = compare_graph_snapshots(_snapshot([first]), _snapshot([altered]))

    # One record on each side: identity matches, content moved, so `changed`.
    assert diff.counts["changed"] == 1
    assert diff.counts["new"] == 0
    assert diff.evidence[0].changes == ("description",)


def test_duplicate_fingerprints_keep_ordinals_and_report_duplicates():
    first = _finding()
    duplicate = _finding(detail="same identity, different message")
    assert duplicate.fingerprint == first.fingerprint

    diff = compare_graph_snapshots(
        _snapshot([first, duplicate]), _snapshot([first, duplicate])
    )

    assert diff.counts["unchanged"] == 2
    assert diff.counts["new"] == 0
    assert diff.duplicate_evidence == 2
    identities = [item.identity for item in diff.evidence]
    assert identities == sorted(identities)
    assert identities[0].endswith("#0") and identities[1].endswith("#1")


def test_cluster_membership_ignores_evidence_id_changes():
    finding = _finding()
    fingerprints = ((0, finding.fingerprint),)
    payload = {
        "cluster_id": "cluster-amount",
        "anchor": "dataset:amount",
        "anchor_kind": "feature",
        "finding_kinds": ("target_leakage",),
        "finding_indices": (0,),
        "sources": ("scanner",),
        "severity": "high",
        "priority": "high",
        "confidence": _confidence(),
    }

    def evidence(identifier):
        return Evidence(
            identifier,
            "scanner",
            "target_leakage",
            finding.detail,
            finding_index=0,
            location="amount",
            metadata=(("severity", "high"),),
        )

    def cluster(identifier):
        return EvidenceCluster(evidence_ids=(identifier,), **payload)

    baseline = _manual(
        [evidence("old-id")],
        clusters=(cluster("old-id"),),
        fingerprints=fingerprints,
    )
    current = _manual(
        [evidence("new-id")],
        clusters=(cluster("new-id"),),
        fingerprints=fingerprints,
    )

    diff = compare_graph_snapshots(baseline, current)

    assert diff.counts["unchanged"] == 1
    assert diff.cluster_counts["unchanged"] == 1
    assert diff.clusters[0].changes == ()


def test_snapshot_rejects_non_utf8_files(tmp_path):
    path = tmp_path / "graph.json"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(GraphBaselineError, match="not valid UTF-8"):
        GraphSnapshot.load(path)


def test_compare_rejects_different_fingerprint_schemas():
    finding = _finding()
    baseline = _snapshot([finding])
    current = GraphSnapshot(
        evidence=baseline.evidence,
        clusters=baseline.clusters,
        links=baseline.links,
        finding_fingerprints=baseline.finding_fingerprints,
        fingerprint_schema=2,
    )

    with pytest.raises(GraphBaselineError, match="different fingerprint schemas"):
        compare_graph_snapshots(baseline, current)


def test_malformed_snapshot_payloads_are_rejected():
    payload = _snapshot([_finding()]).to_dict()

    def reject(key, value):
        broken = json.loads(json.dumps(payload))
        broken[key] = value
        with pytest.raises(GraphBaselineError):
            GraphSnapshot.from_dict(broken)

    reject("schema", "other")
    reject("version", 99)
    reject("fingerprint_schema", 99)
    reject("complete", "yes")
    reject("unknown", True)
    reject("summary", {"evidence": 99, "clusters": 0, "links": 0})

    broken = json.loads(json.dumps(payload))
    broken["links"] = [
        {
            "source_id": "missing",
            "target_id": "also-missing",
            "relation": "supports",
            "rationale": "why",
        }
    ]
    with pytest.raises(GraphBaselineError, match="unknown evidence"):
        GraphSnapshot.from_dict(broken)

    broken = json.loads(json.dumps(payload))
    broken["finding_fingerprints"] = [[0, "not-a-fingerprint"]]
    with pytest.raises(GraphBaselineError, match="64 lowercase"):
        GraphSnapshot.from_dict(broken)


def test_load_rejects_bad_json_and_oversized_files(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphBaselineError, match="invalid JSON"):
        GraphSnapshot.load(bad)

    huge = tmp_path / "huge.json"
    huge.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(graph_baseline, "MAX_SNAPSHOT_BYTES", 1)
    with pytest.raises(GraphBaselineError, match="exceeds"):
        GraphSnapshot.load(huge)


def test_diff_serialization_is_stable():
    first = _finding()
    second = _finding("medium", "id_like", "uid", "id")
    diff = compare_graph_snapshots(_snapshot([first]), _snapshot([second]))

    assert diff.to_dict() == diff.to_dict()
    assert diff.to_dict()["counts"] == diff.counts
    assert diff.to_dict()["warnings"] == list(diff.warnings)
    assert SNAPSHOT_SCHEMA == "leaklens-evidence-graph"
