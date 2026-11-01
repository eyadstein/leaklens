from leaklens.intelligence.context import build_context
from leaklens.leakage import Finding


def test_build_context_preserves_finding_order_and_traceability():
    findings = [
        Finding(
            "high", "preprocess_before_split", "train.py:10", "Scaler fitted early"
        ),
        Finding("medium", "id_like", "customer_id", "98% unique"),
    ]

    context = build_context(findings)

    assert context.finding_count == 2
    assert context.finding_kinds == ("preprocess_before_split", "id_like")
    assert context.evidence[0].finding_index == 0
    assert context.evidence[1].finding_index == 1
    assert context.evidence[0].location == "train.py:10"


def test_build_context_handles_empty_findings():
    context = build_context([])

    assert context.finding_count == 0
    assert context.evidence == ()


def test_build_context_rejects_non_finding_values():
    import pytest

    with pytest.raises(TypeError, match="expected Finding"):
        build_context(["not a finding"])


def test_evidence_for_returns_only_matching_kind():
    findings = [
        Finding("high", "id_like", "id", "identifier"),
        Finding("medium", "id_like", "record_id", "another identifier"),
        Finding("low", "parse_error", "file.py", "parse issue"),
    ]

    context = build_context(findings)

    assert len(context.evidence_for("id_like")) == 2
    assert len(context.evidence_for("unknown")) == 0
