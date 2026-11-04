import pandas as pd
import pytest

from leaklens.intelligence import (
    Evidence,
    analyze_dataset_impact,
    analyze_findings,
    assess_confidence,
    build_context,
)
from leaklens.intelligence.confidence import assess_confidence as _ac
from leaklens.intelligence.context import AnalysisContext, build_context as _bc
from leaklens.leakage import Finding


def _ev(kind, severity="high", description="detail", supports=True):
    return Evidence(
        evidence_id="scanner-000000",
        source="scanner",
        kind=kind,
        description=description,
        supports=supports,
        metadata=(("severity", severity),),
    )


# --- assess_confidence ---

def test_assess_confidence_empty():
    c = assess_confidence([])
    assert c.score == 0.0
    assert c.level == "very_low"
    assert c.supporting_evidence == 0
    assert c.contradicting_evidence == 0
    assert any("No supporting" in r for r in c.reasons)


def test_assess_confidence_levels_and_scores():
    # 0.25 + 1*0.15 = 0.40 → "low"
    c1 = assess_confidence([_ev("target_leakage")])
    assert c1.score == pytest.approx(0.40)
    assert c1.level == "low"

    # 0.25 + 2*0.15 + 0.10 (multi-kind bonus) = 0.65 → "moderate"
    c2 = assess_confidence([_ev("a"), _ev("b")])
    assert c2.score == pytest.approx(0.65)
    assert c2.level == "moderate"
    assert any("multiple finding categories" in r for r in c2.reasons)

    # 0.25 + 4*0.15 (capped) + 0.10 = 0.95 → very_high
    many = [_ev("a"), _ev("b"), _ev("c"), _ev("d"), _ev("e")]
    c3 = assess_confidence(many)
    assert c3.score == pytest.approx(0.95)
    assert c3.level == "very_high"

    # contradicting evidence subtracts
    c4 = assess_confidence([_ev("a"), _ev("b", supports=False)])
    assert c4.contradicting_evidence == 1
    assert c4.score == pytest.approx(0.28)
    assert c4.level == "very_low"
    assert any("contradicting" in r for r in c4.reasons)


def test_assess_confidence_high_boundary():
    # 0.25 + 3*0.15 + 0.10 (multi-kind) = 0.80 → "high"
    c = assess_confidence([_ev("a"), _ev("b"), _ev("c")])
    assert c.score == pytest.approx(0.80)
    assert c.level == "high"


def test_assess_confidence_no_supporting():
    c = assess_confidence([_ev("a", supports=False)])
    assert c.supporting_evidence == 0
    assert any("No supporting" in r for r in c.reasons)


def test_assess_confidence_score_clamped_negative():
    items = [_ev(f"k{i}", supports=False) for i in range(5)]
    c = assess_confidence(items)
    assert 0.0 <= c.score <= 1.0
    assert c.level == "very_low"


def test_confidence_reason_always_has_heuristic_warning():
    c = assess_confidence([_ev("a")])
    assert any("not a probability" in r for r in c.reasons)


# --- build_context ---

def test_build_context_preserves_order_and_metadata():
    findings = [
        Finding("high", "target_leakage", "col_x", "detail A"),
        Finding("medium", "id_like", "col_y", "detail B"),
    ]
    ctx = build_context(findings)
    assert isinstance(ctx, AnalysisContext)
    assert ctx.finding_count == 2
    assert ctx.finding_kinds == ("target_leakage", "id_like")
    ev = ctx.evidence
    assert ev[0].evidence_id == "scanner-000000"
    assert ev[0].kind == "target_leakage"
    assert ev[0].description == "detail A"
    assert ev[0].location == "col_x"
    assert ev[0].finding_index == 0
    assert dict(ev[0].metadata) == {"severity": "high"}
    assert ev[1].evidence_id == "scanner-000001"


def test_build_context_evidence_for():
    ctx = build_context([
        Finding("high", "target_leakage", "a", "1"),
        Finding("low", "target_leakage", "b", "2"),
        Finding("medium", "id_like", "c", "3"),
    ])
    assert len(ctx.evidence_for("target_leakage")) == 2
    assert ctx.evidence_for("nonexistent") == ()


def test_build_context_rejects_non_finding():
    with pytest.raises(TypeError):
        build_context(["not a finding"])


def test_build_context_empty():
    ctx = build_context([])
    assert ctx.finding_count == 0
    assert ctx.finding_kinds == ()


# --- analyze_findings ---

def test_analyze_findings_empty():
    report = analyze_findings([])
    assert report.finding_count == 0
    assert report.evidence == ()
    assert report.root_causes == ()
    assert report.recommendations == ()
    assert report.confidence.score == 0.0
    assert report.engine == "deterministic"
    assert report.ai_enriched is False
    assert report.impact is None


def test_analyze_findings_known_kinds():
    findings = [
        Finding("high", "target_leakage", "x", "detail"),
        Finding("medium", "id_like", "y", "detail"),
    ]
    report = analyze_findings(findings)
    assert report.finding_count == 2
    assert len(report.root_causes) == 2
    assert len(report.recommendations) == 2
    kinds = [r.category for r in report.root_causes]
    assert "target_contamination" in kinds
    assert "identifier_memorization" in kinds
    # recommendation priority matches severity
    prios = {r.related_finding_kinds: r.priority for r in report.recommendations}
    assert prios[("id_like",)] == "medium"
    # confidence is per-kind
    for rc in report.root_causes:
        assert rc.confidence.score > 0


def test_analyze_findings_unknown_kind_falls_back():
    report = analyze_findings([Finding("low", "some_new_kind", "c", "d")])
    assert len(report.root_causes) == 1
    assert report.root_causes[0].category == "unclassified_leakage"
    assert "some_new_kind" in report.root_causes[0].title
    rec = report.recommendations[0]
    assert rec.title == "Investigate the detected leakage risk"
    assert len(rec.actions) > 0


def test_analyze_findings_duplicate_kinds_single_root_cause():
    findings = [
        Finding("high", "target_leakage", "a", "1"),
        Finding("low", "target_leakage", "b", "2"),
    ]
    report = analyze_findings(findings)
    assert report.finding_count == 2
    assert len(report.root_causes) == 1  # deduped by kind
    assert len(report.recommendations) == 1
    # priority = max severity
    assert report.recommendations[0].priority == "high"


def test_analyze_findings_passes_impact_through():
    impact = analyze_dataset_impact(
        pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [3]})
    )
    report = analyze_findings([], impact=impact)
    assert report.impact is impact
    assert report.to_dict()["impact"]["train_rows"] == 2


def test_analyze_findings_to_dict_roundtrip():
    report = analyze_findings([Finding("high", "split_overlap", "*", "detail")])
    d = report.to_dict()
    assert d["finding_count"] == 1
    assert d["engine"] == "deterministic"
    assert d["confidence"]["level"] in ("very_low", "low", "moderate", "high", "very_high")


# --- analyze_dataset_impact ---

def test_analyze_dataset_impact_basic():
    train = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    test = pd.DataFrame({"a": [3, 4], "b": ["z", "w"]})
    impact = analyze_dataset_impact(train, test)
    assert impact.train_rows == 3
    assert impact.test_rows == 2
    assert impact.duplicate_test_rows == 1  # (3, "z") shared
    assert impact.duplicate_test_pct == pytest.approx(50.0)
    assert impact.group_column is None
    assert impact.temporal_overlap is None


def test_analyze_dataset_impact_no_common_columns():
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"b": [1, 2]})
    impact = analyze_dataset_impact(train, test)
    assert impact.duplicate_test_rows == 0
    assert impact.duplicate_test_pct == 0.0


def test_analyze_dataset_impact_empty_test():
    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": []}, dtype=float)
    impact = analyze_dataset_impact(train, test)
    assert impact.duplicate_test_rows == 0
    assert impact.duplicate_test_pct == 0.0


def test_analyze_dataset_impact_group_overlap():
    train = pd.DataFrame({"g": [1, 2, 3], "a": [10, 20, 30]})
    test = pd.DataFrame({"g": [2, 3, 4], "a": [1, 2, 3]})
    impact = analyze_dataset_impact(train, test, group_col="g")
    assert impact.group_column == "g"
    assert impact.shared_group_count == 2  # {2, 3}
    assert impact.shared_group_test_rows == 2
    assert impact.shared_group_test_pct == pytest.approx(66.67)


def test_analyze_dataset_impact_group_missing_raises():
    train = pd.DataFrame({"a": [1]})
    test = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="group column"):
        analyze_dataset_impact(train, test, group_col="missing")


def test_analyze_dataset_impact_temporal_overlap():
    train = pd.DataFrame({"t": ["2025-01-01", "2025-01-05"]})
    test = pd.DataFrame({"t": ["2025-01-03", "2025-01-10"]})
    impact = analyze_dataset_impact(train, test, time_col="t")
    assert impact.temporal_overlap is True  # train max (01-05) > test min (01-03)
    assert impact.train_max_time == "2025-01-05T00:00:00"
    assert impact.test_min_time == "2025-01-03T00:00:00"


def test_analyze_dataset_impact_temporal_no_overlap():
    train = pd.DataFrame({"t": ["2025-01-01", "2025-01-02"]})
    test = pd.DataFrame({"t": ["2025-01-03", "2025-01-04"]})
    impact = analyze_dataset_impact(train, test, time_col="t")
    assert impact.temporal_overlap is False


def test_analyze_dataset_impact_time_missing_raises():
    train = pd.DataFrame({"a": [1]})
    test = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="time column"):
        analyze_dataset_impact(train, test, time_col="missing")


def test_analyze_dataset_impact_type_error():
    with pytest.raises(TypeError):
        analyze_dataset_impact("not a df", pd.DataFrame({"a": [1]}))


def test_analyze_dataset_impact_unparseable_times():
    train = pd.DataFrame({"t": ["not-a-date", "also-bad"]})
    test = pd.DataFrame({"t": ["junk", "more junk"]})
    impact = analyze_dataset_impact(train, test, time_col="t")
    assert impact.temporal_overlap is None
    assert impact.train_max_time is None
    assert impact.test_min_time is None


def test_analyze_dataset_impact_rounds_percentages():
    train = pd.DataFrame({"a": [1]})
    test = pd.DataFrame({"a": [1, 2, 3]})
    impact = analyze_dataset_impact(train, test)
    assert impact.duplicate_test_pct == pytest.approx(33.33)


# --- schemas validation ---

def test_evidence_validation():
    with pytest.raises(ValueError):
        Evidence(evidence_id="", source="scanner", kind="x", description="d")
    with pytest.raises(ValueError):
        Evidence(evidence_id="e", source="bad_source", kind="x", description="d")
    with pytest.raises(ValueError):
        Evidence(evidence_id="e", source="scanner", kind="x", description="d", finding_index=-1)
    with pytest.raises(ValueError):
        Evidence(evidence_id="e", source="scanner", kind="x", description="d", location="   ")
    ok = Evidence(evidence_id="e", source="scanner", kind="x", description="d")
    assert ok.to_dict()["evidence_id"] == "e"


def test_intelligence_imports_are_consistent():
    # exported names resolve to the same functions
    assert assess_confidence is _ac
    assert build_context is _bc
