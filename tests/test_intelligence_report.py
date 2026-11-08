from leaklens.report import render


def test_report_renders_intelligence_sections():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.85,
                    "level": "high",
                    "reasons": ["Two supporting observations"],
                },
                "root_causes": [
                    {
                        "category": "preprocessing_order",
                        "title": "Preprocessing before split",
                        "explanation": "The transformation may have seen held-out data.",
                    }
                ],
                "recommendations": [
                    {
                        "title": "Move preprocessing into pipeline",
                        "description": "Fit transformations only on training data.",
                        "priority": "high",
                        "actions": ["Split before fitting"],
                        "validation_steps": ["Check held-out data is never fitted"],
                    }
                ],
            },
        }
    )

    assert "Intelligence analysis" in report
    assert "Root-cause analysis" in report
    assert "Remediation recommendations" in report
    assert "Two supporting observations" in report
    assert "Move preprocessing into pipeline" in report


def test_intelligence_report_escapes_untrusted_text():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.5,
                    "level": "moderate",
                    "reasons": ["<script>alert(1)</script>"],
                },
                "root_causes": [
                    {
                        "category": "test",
                        "title": "<img src=x onerror=alert(1)>",
                        "explanation": "<script>bad()</script>",
                    }
                ],
                "recommendations": [
                    {
                        "title": "<b>Unsafe</b>",
                        "description": "Unsafe <tag>",
                        "priority": "high",
                        "actions": ["<script>bad()</script>"],
                        "validation_steps": ["<img src=x>"],
                    }
                ],
            },
        }
    )

    assert "<script>" not in report
    assert "<img src=x" not in report
    assert "&lt;script&gt;" in report
    assert "&lt;img" in report


def test_report_renders_dataset_impact_and_escapes_dataset_values():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "impact": {
                "train_rows": 100,
                "test_rows": 20,
                "duplicate_test_rows": 2,
                "duplicate_test_pct": 10.0,
                "group_column": "<script>bad()</script>",
                "shared_group_count": 1,
                "shared_group_test_rows": 3,
                "shared_group_test_pct": 15.0,
                "time_column": "event_time",
                "temporal_overlap": True,
                "train_max_time": "2025-01-05T00:00:00",
                "test_min_time": "2025-01-04T00:00:00",
            },
        }
    )

    assert "Dataset impact" in report
    assert "Duplicate test rows" in report
    assert "Shared group values" in report
    assert "Temporal boundary overlap" in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "<script>bad()</script>" not in report


def test_report_without_intelligence_remains_compatible():
    report = render({"trust_score": 100, "findings": []})

    assert "100/100" in report
    assert "Intelligence analysis" not in report


def test_report_renders_evidence_clusters_and_escapes_measurements():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.6,
                    "level": "moderate",
                    "reasons": ["Evidence present"],
                },
                "clusters": [
                    {
                        "cluster_id": "cluster-split",
                        "anchor": "<script>bad()</script>",
                        "anchor_kind": "split",
                        "finding_kinds": ["split_overlap", "feature_overlap"],
                        "finding_indices": [0, 1],
                        "evidence_ids": ["scanner-000000", "scanner-000001"],
                        "sources": ["scanner"],
                        "severity": "high",
                        "priority": "high",
                        "confidence": {
                            "score": 0.74,
                            "level": "high",
                            "reasons": ["Two findings"],
                        },
                        "measured_impact": ["2 of 5 test rows duplicate training rows"],
                    }
                ],
                "root_causes": [],
                "recommendations": [],
            },
        }
    )

    assert "Evidence clusters" in report
    assert "split_overlap, feature_overlap" in report
    assert "2 of 5 test rows duplicate training rows" in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "<script>bad()</script>" not in report


def test_report_renders_evidence_links_and_escapes_them():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "intelligence": {
                "confidence": {
                    "score": 0.6,
                    "level": "moderate",
                    "reasons": ["Evidence present"],
                },
                "clusters": [],
                "links": [
                    {
                        "source_id": "scanner-000000-e01",
                        "target_id": "<script>target</script>",
                        "relation": "supports",
                        "rationale": "<img src=x onerror=alert(1)>",
                    }
                ],
                "root_causes": [],
                "recommendations": [],
            },
        }
    )

    assert "Evidence links" in report
    assert "supports" in report
    assert "scanner-000000-e01" in report
    assert "&lt;script&gt;target&lt;/script&gt;" in report
    assert "&lt;img src=x" in report
    assert "<script>target</script>" not in report
    assert "<img src=x onerror" not in report


def test_report_renders_graph_baseline_and_escapes_it():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "graph_baseline": {
                "counts": {
                    "new": 1,
                    "unchanged": 2,
                    "changed": 1,
                    "resolved": 0,
                    "uncertain": 1,
                },
                "clusters": {
                    "new": 0,
                    "unchanged": 1,
                    "changed": 0,
                    "resolved": 0,
                    "uncertain": 0,
                },
                "relationships": {"added": 1, "removed": 0},
                "duplicates": 2,
                "resolution_complete": False,
                "warnings": ["<script>bad()</script>"],
                "evidence": [
                    {
                        "state": "new",
                        "identity": "primary:abc#0",
                        "kind": "<img src=x onerror=alert(1)>",
                        "fingerprint": "a" * 64,
                        "current_id": "scanner-000000",
                        "changes": [],
                    },
                    {
                        "state": "changed",
                        "identity": "primary:def#0",
                        "kind": "target_leakage",
                        "fingerprint": "b" * 64,
                        "evidence": "scanner-000001",
                        "changes": ["description", "relationships"],
                    },
                ],
                "cluster_changes": [
                    {
                        "state": "changed",
                        "anchor": "<script>anchor</script>",
                        "changes": ["membership"],
                    }
                ],
                "relationship_changes": [
                    {
                        "state": "added",
                        "source": "primary:abc#0",
                        "relation": "supports",
                        "target": "primary:def#0",
                    }
                ],
            },
        }
    )

    assert "Evidence-graph baseline comparison" in report
    assert "New:" in report and "Uncertain:" in report
    assert "One or both scans were incomplete" in report
    assert "Collapsed duplicate records" in report
    assert "Cluster changes" in report
    assert "Relationship changes" in report
    assert "not proof of leakage" in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "&lt;img src=x" in report
    assert "<script>" not in report
    assert "<img src=x onerror" not in report


def test_report_renders_location_moves_and_completeness_reasons():
    report = render(
        {
            "trust_score": 80,
            "findings": [],
            "graph_baseline": {
                "counts": {
                    "new": 0,
                    "unchanged": 1,
                    "changed": 0,
                    "resolved": 0,
                    "uncertain": 0,
                },
                "moved": 1,
                "resolution_complete": True,
                "completeness": {
                    "baseline": {"state": "complete", "reasons": []},
                    "current": {
                        "state": "incomplete",
                        "reasons": ["<script>bad()</script>"],
                    },
                    "context_compatible": True,
                },
                "evidence": [
                    {
                        "state": "unchanged",
                        "identity": "primary:abc#0",
                        "kind": "target_leakage",
                        "fingerprint": "a" * 64,
                        "baseline_id": "old",
                        "current_id": "new",
                        "changes": ["location"],
                        "baseline_location": "train.py:10",
                        "current_location": "train.py:42",
                        "location_changed": True,
                    }
                ],
                "cluster_changes": [],
                "relationship_changes": [],
            },
        }
    )

    assert "Current scan completeness:" in report
    assert "train.py:10 -&gt; train.py:42" in report
    assert "matched by content" in report
    assert "&lt;script&gt;bad()&lt;/script&gt;" in report
    assert "<script>bad()</script>" not in report


def test_report_omits_graph_baseline_section_by_default():
    report = render({"trust_score": 100, "findings": []})
    assert "Evidence-graph baseline comparison" not in report
