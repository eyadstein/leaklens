import pytest

from leaklens.intelligence.schemas import (
    ConfidenceAssessment,
    Evidence,
    Recommendation,
    RootCause,
)


def test_evidence_serializes_and_preserves_traceability():
    evidence = Evidence(
        evidence_id="scanner-001",
        source="scanner",
        kind="preprocess_before_split",
        description="Scaler fitted before the train/test split",
        finding_index=2,
        location="pipeline.py:18",
    )

    result = evidence.to_dict()

    assert result["evidence_id"] == "scanner-001"
    assert result["finding_index"] == 2
    assert result["supports"] is True


def test_evidence_rejects_unknown_source():
    with pytest.raises(ValueError, match="unsupported evidence source"):
        Evidence("e-1", "external", "test", "description")


def test_evidence_rejects_negative_finding_index():
    with pytest.raises(ValueError, match="non-negative"):
        Evidence("e-1", "scanner", "test", "description", finding_index=-1)


def test_confidence_assessment_validates_score_and_counts():
    assessment = ConfidenceAssessment(
        score=0.85,
        level="high",
        reasons=("Two independent scanner findings support this assessment",),
        supporting_evidence=2,
        contradicting_evidence=0,
    )

    assert assessment.to_dict()["score"] == 0.85

    with pytest.raises(ValueError, match="between 0 and 1"):
        ConfidenceAssessment(1.5, "high", ("reason",), 1, 0)

    with pytest.raises(ValueError, match="non-negative"):
        ConfidenceAssessment(0.5, "moderate", ("reason",), -1, 0)


def test_root_cause_requires_related_findings():
    confidence = ConfidenceAssessment(0.9, "very_high", ("Direct evidence",), 1, 0)

    cause = RootCause(
        category="preprocessing_order",
        title="Preprocessing before splitting",
        explanation="The transformation learned from the complete dataset.",
        related_finding_kinds=("preprocess_before_split",),
        confidence=confidence,
    )

    assert cause.to_dict()["related_finding_kinds"] == ("preprocess_before_split",)

    with pytest.raises(ValueError, match="related finding"):
        RootCause("other", "Title", "Explanation", (), confidence)


def test_recommendation_requires_actions_and_validation():
    recommendation = Recommendation(
        title="Move preprocessing into a pipeline",
        description="Fit transformations only on training folds.",
        priority="high",
        actions=("Use sklearn.pipeline.Pipeline",),
        validation_steps=("Run cross-validation after the change",),
        related_finding_kinds=("preprocess_before_split",),
    )

    assert recommendation.automated is False
    assert recommendation.to_dict()["priority"] == "high"

    with pytest.raises(ValueError, match="at least one non-empty action"):
        Recommendation("Title", "Description", "high", (), ("Validate",), ("leak",))

    with pytest.raises(ValueError, match="at least one non-empty validation"):
        Recommendation("Title", "Description", "high", ("Fix",), (), ("leak",))


def test_schema_instances_are_immutable():
    evidence = Evidence("e-1", "scanner", "test", "description")

    with pytest.raises(AttributeError):
        evidence.kind = "changed"
