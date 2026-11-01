from leaklens.intelligence import DatasetImpact, analyze_findings
from leaklens.leakage import Finding


def test_analysis_generates_evidence_causes_and_recommendations():
    findings = [
        Finding(
            "high",
            "preprocess_before_split",
            "train.py:10",
            "StandardScaler fitted before splitting",
        ),
        Finding(
            "medium",
            "id_like",
            "customer_id",
            "98% unique values",
        ),
    ]

    report = analyze_findings(findings)

    assert report.finding_count == 2
    assert len(report.evidence) == 2
    assert len(report.root_causes) == 2
    assert len(report.recommendations) == 2
    assert report.engine == "deterministic"
    assert report.ai_enriched is False


def test_analysis_groups_findings_by_kind():
    findings = [
        Finding("high", "id_like", "id", "98% unique"),
        Finding("medium", "id_like", "record_id", "99% unique"),
    ]

    report = analyze_findings(findings)

    assert len(report.root_causes) == 1
    assert len(report.recommendations) == 1
    assert report.root_causes[0].related_finding_kinds == ("id_like",)


def test_unknown_finding_kind_gets_generic_explanation():
    report = analyze_findings(
        [Finding("low", "new_detector", "file.py:5", "New detector finding")]
    )

    assert report.root_causes[0].category == "unclassified_leakage"
    assert report.recommendations[0].title == "Investigate the detected leakage risk"


def test_empty_analysis_returns_valid_empty_report():
    report = analyze_findings([])

    assert report.finding_count == 0
    assert report.evidence == ()
    assert report.root_causes == ()
    assert report.recommendations == ()
    assert report.confidence.level == "very_low"


def test_analysis_can_include_dataset_impact():
    impact = DatasetImpact(
        train_rows=100,
        test_rows=20,
        duplicate_test_rows=2,
        duplicate_test_pct=10.0,
    )

    report = analyze_findings([], impact=impact)

    assert report.impact == impact
    assert report.to_dict()["impact"]["duplicate_test_rows"] == 2


def test_report_serializes_to_dictionary():
    report = analyze_findings(
        [Finding("high", "split_overlap", "*", "Duplicate rows found")]
    )

    data = report.to_dict()

    assert data["engine"] == "deterministic"
    assert data["ai_enriched"] is False
    assert data["finding_count"] == 1
    assert data["recommendations"][0]["priority"] == "high"
