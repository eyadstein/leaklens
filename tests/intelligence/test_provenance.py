"""Evidence-graph behaviour: anchors, clusters, links and measured impact."""

import pandas as pd
import pytest

from leaklens.evidence import Evidence as DetectorEvidence
from leaklens.intelligence import analyze_dataset_impact, analyze_findings
from leaklens.intelligence.context import build_context
from leaklens.intelligence.provenance import (
    _MAX_LINKS,
    _links_for,
    _render_graph,
    anchor_for,
    build_evidence_graph,
)
from leaklens.intelligence.schemas import (
    ConfidenceAssessment,
    Evidence,
    EvidenceCluster,
)
from leaklens.leakage import Finding


def _finding(kind, column, severity="high", evidence=()):
    return Finding(severity, kind, column, f"{kind} on {column}", evidence=evidence)


def _detector(kind="statistic", description="aggregate observation"):
    return DetectorEvidence(
        source="data",
        kind=kind,
        description=description,
        location="column",
        observed="observed value",
        expected="expected bound",
    )


def test_anchor_kinds_are_inspectable_and_stable():
    assert anchor_for(_finding("target_leakage", "amount")) == (
        "dataset:amount",
        "feature",
    )
    assert anchor_for(_finding("split_overlap", "*")) == ("split", "split")
    assert anchor_for(_finding("group_overlap", "patient")) == (
        "split:patient",
        "split",
    )
    assert anchor_for(_finding("preprocess_before_split", "train.py:10")) == (
        "file:train.py",
        "file",
    )
    assert anchor_for(_finding("notebook_out_of_order_execution", "nb.ipynb")) == (
        "file:nb.ipynb",
        "file",
    )
    assert anchor_for(_finding("image_overlap", "images/*")) == ("image", "image")
    assert anchor_for(_finding("some_new_kind", "value")) == (
        "finding:some_new_kind",
        "finding",
    )


def test_findings_on_the_same_feature_form_one_cluster():
    findings = [
        _finding("target_leakage", "amount", "high", (_detector(),)),
        _finding("id_like", "amount", "medium"),
    ]

    graph = build_evidence_graph(findings)

    assert len(graph.clusters) == 1
    cluster = graph.clusters[0]
    assert cluster.anchor == "dataset:amount"
    assert cluster.anchor_kind == "feature"
    assert cluster.finding_kinds == ("target_leakage", "id_like")
    assert cluster.finding_indices == (0, 1)
    assert cluster.severity == "high"
    assert cluster.priority == "high"
    assert len(cluster.evidence_ids) == 3


def test_findings_on_different_features_form_separate_clusters():
    graph = build_evidence_graph(
        [
            _finding("target_leakage", "amount"),
            _finding("target_leakage", "balance"),
        ]
    )

    assert {cluster.anchor for cluster in graph.clusters} == {
        "dataset:amount",
        "dataset:balance",
    }


def test_split_findings_share_the_split_anchor():
    graph = build_evidence_graph(
        [
            _finding("split_overlap", "*"),
            _finding("feature_overlap", "*", "medium"),
            _finding("near_duplicate_overlap", "*", "medium"),
        ]
    )

    assert len(graph.clusters) == 1
    assert graph.clusters[0].anchor == "split"
    assert graph.clusters[0].finding_kinds == (
        "split_overlap",
        "feature_overlap",
        "near_duplicate_overlap",
    )


def test_code_findings_group_by_file_not_by_line():
    graph = build_evidence_graph(
        [
            _finding("preprocess_before_split", "train.py:10"),
            _finding("fit_on_full_data", "train.py:42", "medium"),
            _finding("preprocess_before_split", "evaluate.py:8"),
        ]
    )

    anchors = {cluster.anchor: cluster.finding_kinds for cluster in graph.clusters}
    assert anchors["file:train.py"] == (
        "preprocess_before_split",
        "fit_on_full_data",
    )
    assert anchors["file:evaluate.py"] == ("preprocess_before_split",)


def test_independent_detectors_raise_cluster_confidence():
    single = build_evidence_graph([_finding("target_leakage", "amount")])
    corroborated = build_evidence_graph(
        [
            _finding("target_leakage", "amount", "high", (_detector(),)),
            _finding("id_like", "amount", "medium"),
        ]
    )

    assert (
        corroborated.clusters[0].confidence.score > single.clusters[0].confidence.score
    )
    assert corroborated.clusters[0].confidence.score <= 0.95
    assert any(
        "independent finding kinds" in reason
        for reason in corroborated.clusters[0].confidence.reasons
    )


def test_detector_evidence_records_are_linked_to_their_finding():
    graph = build_evidence_graph(
        [_finding("target_leakage", "amount", "high", (_detector(), _detector()))]
    )

    assert len(graph.clusters) == 1
    assert len(graph.links) == 2
    assert {link.relation for link in graph.links} == {"supports"}
    assert all(
        link.target_id == graph.clusters[0].evidence_ids[0] for link in graph.links
    )
    assert {link.source_id for link in graph.links} == {
        "scanner-000000-e01",
        "scanner-000000-e02",
    }


def test_detector_links_target_the_finding_they_explain():
    graph = build_evidence_graph(
        [
            _finding("target_leakage", "amount", "high", (_detector(),)),
            _finding("id_like", "amount", "medium", (_detector(),)),
        ]
    )

    supports = {
        link.source_id: link for link in graph.links if link.relation == "supports"
    }
    assert supports["scanner-000000-e01"].target_id == "scanner-000000"
    assert supports["scanner-000001-e01"].target_id == "scanner-000001"


@pytest.mark.parametrize("finding_index", [None, 7])
def test_detector_without_a_known_primary_is_not_linked(finding_index):
    confidence = ConfidenceAssessment(0.5, "moderate", ("reason",), 1, 0)
    primary = Evidence(
        "scanner-000000", "scanner", "target_leakage", "primary", finding_index=0
    )
    detector = Evidence(
        "scanner-000000-e01",
        "dataset",
        "target_leakage",
        "detector detail",
        finding_index=finding_index,
        metadata=(("role", "detector"),),
    )
    cluster = EvidenceCluster(
        "cluster-x",
        "dataset:amount",
        "feature",
        ("target_leakage",),
        (0,),
        ("scanner-000000", "scanner-000000-e01"),
        ("scanner", "dataset"),
        "high",
        "high",
        confidence,
    )

    links = _links_for(
        cluster,
        {"scanner-000000": primary, "scanner-000000-e01": detector},
    )

    assert links == []


def test_neighbors_and_lookup_return_consistent_stable_views():
    graph = build_evidence_graph(
        [
            _finding("target_leakage", "amount", "high", (_detector(),)),
            _finding("id_like", "amount", "medium"),
            _finding("categorical_target_leakage", "amount", "medium"),
        ]
    )

    assert set(graph.evidence_by_id) == {item.evidence_id for item in graph.evidence}
    assert graph.cluster_for(2) is graph.clusters[0]
    assert graph.cluster_for(99) is None
    assert graph.neighbors("missing") == ()

    neighbours = graph.neighbors("scanner-000000")
    assert [
        (relation, direction, item.evidence_id)
        for relation, direction, item in neighbours
    ] == [
        ("supports", "in", "scanner-000000-e01"),
        ("corroborates", "out", "scanner-000001"),
        ("corroborates", "out", "scanner-000002"),
    ]
    assert graph.relation_counts() == {
        "corroborates": 2,
        "same_anchor": 0,
        "supports": 1,
    }
    assert graph.summary() == {
        "evidence": 4,
        "clusters": 1,
        "links": 3,
        "relations": {"corroborates": 2, "same_anchor": 0, "supports": 1},
    }


def test_graph_exports_are_deterministic_and_inspectable():
    graph = build_evidence_graph(
        [
            _finding("target_leakage", "amount", "high", (_detector(),)),
            _finding("id_like", "amount", "medium"),
        ]
    )
    before = graph.to_dict()

    mermaid = graph.to_mermaid()
    dot = graph.to_dot()

    assert mermaid.startswith("flowchart LR\n")
    assert "subgraph c0" in mermaid
    assert "-->|corroborates|" in mermaid
    assert "-->|supports|" in mermaid
    assert dot.startswith("digraph leaklens {\n")
    assert dot.endswith("}\n")
    assert 'label="corroborates"' in dot
    assert 'label="supports"' in dot
    assert graph.to_mermaid() == mermaid
    assert graph.to_dot() == dot
    assert graph.to_dict() == before


def test_graph_exports_escape_user_controlled_labels():
    finding = Finding(
        "high",
        "target_leakage",
        "amount",
        'quoted "label" ] with \\ slash\nnewline',
    )
    graph = build_evidence_graph([finding])

    mermaid = graph.to_mermaid()
    dot = graph.to_dot()

    assert "quoted 'label' ) with / slash newline" in mermaid
    assert "\\" not in mermaid
    assert 'quoted \\"label\\" ] with \\\\ slash newline' in dot


def test_empty_graph_exports_are_valid_and_empty():
    graph = build_evidence_graph([])

    assert graph.to_mermaid() == "flowchart LR\n"
    assert graph.to_dot() == (
        "digraph leaklens {\n  rankdir=LR;\n  node [shape=box, fontsize=10];\n}\n"
    )


def test_unknown_export_format_is_rejected():
    graph = build_evidence_graph([_finding("target_leakage", "amount")])

    with pytest.raises(ValueError, match="unsupported graph format"):
        _render_graph(graph, "svg")


def test_report_graph_view_matches_the_analyzed_graph():
    report = analyze_findings([_finding("target_leakage", "amount")])
    graph = report.to_graph()

    assert graph.evidence == report.evidence
    assert graph.clusters == report.clusters
    assert graph.links == report.links


def test_links_are_bounded_and_reference_existing_evidence():
    findings = [
        _finding("id_like", f"column_{index}", "low", (_detector(), _detector()))
        for index in range(250)
    ]

    graph = build_evidence_graph(findings)

    assert len(graph.links) <= _MAX_LINKS
    assert graph.links_truncated is True
    evidence_ids = {item.evidence_id for item in graph.evidence}
    for link in graph.links:
        assert link.source_id in evidence_ids
        assert link.target_id in evidence_ids


def test_small_graphs_are_not_marked_truncated():
    graph = build_evidence_graph([_finding("id_like", "column", "low", (_detector(),))])

    assert graph.links
    assert graph.links_truncated is False


def test_measured_impact_is_attached_to_the_relevant_cluster():
    impact = analyze_dataset_impact(
        pd.DataFrame({"x": [1, 2, 3], "group": ["A", "B", "C"]}),
        pd.DataFrame({"x": [2, 3, 9], "group": ["B", "C", "D"]}),
        group_col="group",
    )

    graph = build_evidence_graph(
        [
            _finding("split_overlap", "*"),
            _finding("group_overlap", "group"),
            _finding("target_leakage", "x"),
        ],
        impact=impact,
    )

    by_anchor = {cluster.anchor: cluster for cluster in graph.clusters}
    assert any(
        "duplicate training rows" in note for note in by_anchor["split"].measured_impact
    )
    assert any(
        "group value(s) appear in both splits" in note
        for note in by_anchor["split:group"].measured_impact
    )
    assert by_anchor["dataset:x"].measured_impact == ()


def test_empty_findings_produce_an_empty_graph():
    graph = build_evidence_graph([])

    assert graph.evidence == ()
    assert graph.clusters == ()
    assert graph.links == ()


def test_graph_is_deterministic_across_calls():
    findings = [
        _finding("target_leakage", "amount", "high", (_detector(),)),
        _finding("id_like", "amount", "medium"),
        _finding("split_overlap", "*"),
    ]

    first = build_evidence_graph(findings).to_dict()
    second = build_evidence_graph(findings).to_dict()

    assert first == second


def test_graph_can_reuse_an_existing_context():
    findings = [_finding("target_leakage", "amount")]
    context = build_context(findings)

    graph = build_evidence_graph(findings, context=context)

    assert graph.evidence == context.evidence
    assert len(graph.clusters) == 1


def test_cluster_ids_stay_unique_when_anchors_slug_the_same():
    # Different anchors that normalise to the same slug still keep distinct ids.
    graph = build_evidence_graph(
        [
            _finding("target_leakage", "A B"),
            _finding("target_leakage", "a-b"),
        ]
    )

    assert len({cluster.cluster_id for cluster in graph.clusters}) == len(
        graph.clusters
    )


def test_cluster_rejects_unbounded_aggregates():
    from leaklens.intelligence.schemas import (
        ConfidenceAssessment,
        EvidenceCluster,
    )

    confidence = ConfidenceAssessment(0.5, "moderate", ("reason",), 1, 0)
    with pytest.raises(ValueError, match="anchor kind"):
        EvidenceCluster(
            "c",
            "anchor",
            "unknown",
            ("kind",),
            (0,),
            ("e",),
            ("scanner",),
            "high",
            "high",
            confidence,
        )
    with pytest.raises(ValueError, match="at least one evidence"):
        EvidenceCluster(
            "c",
            "anchor",
            "feature",
            ("kind",),
            (0,),
            (),
            ("scanner",),
            "high",
            "high",
            confidence,
        )
