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


def test_detector_evidence_is_preserved_with_stable_ids():
    from leaklens.evidence import Evidence as DetectorEvidence

    finding = Finding(
        "high",
        "target_leakage",
        "amount",
        "alone predicts target",
        evidence=(
            DetectorEvidence(
                source="data",
                kind="statistic",
                description="one feature predicts the target",
                location="amount",
                observed="roc_auc=0.990",
                expected="roc_auc < 0.950",
            ),
        ),
    )

    context = build_context([finding])

    assert context.finding_count == 1
    assert context.finding_kinds == ("target_leakage",)
    assert len(context.evidence) == 2

    primary = context.primary_evidence[0]
    assert primary.evidence_id == "scanner-000000"
    assert dict(primary.metadata) == {"severity": "high"}

    detail = context.detector_evidence[0]
    assert detail.evidence_id == "scanner-000000-e01"
    assert detail.source == "dataset"
    assert detail.kind == "target_leakage"
    assert detail.location == "amount"
    assert dict(detail.metadata) == {
        "severity": "high",
        "role": "detector",
        "observation": "statistic",
        "observed": "roc_auc=0.990",
        "expected": "roc_auc < 0.950",
    }
    assert context.evidence_for_finding(0) == (primary, detail)


def test_split_detector_evidence_maps_to_the_dataset_source():
    from leaklens.evidence import Evidence as DetectorEvidence

    finding = Finding(
        "high",
        "split_overlap",
        "*",
        "overlap",
        evidence=(
            DetectorEvidence(
                source="split", kind="overlap", description="duplicate rows"
            ),
        ),
    )

    detail = build_context([finding]).detector_evidence[0]

    assert detail.source == "dataset"
    assert detail.location == "*"
    assert dict(detail.metadata)["observation"] == "overlap"
