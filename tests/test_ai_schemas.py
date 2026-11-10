"""Strict contracts for the experimental AI layer."""

from __future__ import annotations

import pytest

from leaklens.ai.schemas import (
    ENGINE_NAME,
    SCHEMA_VERSION,
    AIReport,
    EvidenceSpan,
    Hypothesis,
    LLMNote,
    ModelCard,
    ReasoningPlan,
    Remediation,
    RootCauseGroup,
)


def evidence(evidence_id: str = "ev-abc", **overrides) -> EvidenceSpan:
    values = {
        "evidence_id": evidence_id,
        "source": "code",
        "kind": "data_flow",
        "location": "model.py:4",
        "excerpt": "X = scaler.fit_transform(X)",
    }
    values.update(overrides)
    return EvidenceSpan(**values)


def hypothesis(hypothesis_id: str = "hyp-abc", **overrides) -> Hypothesis:
    values = {
        "hypothesis_id": hypothesis_id,
        "concept": "concat_before_split",
        "title": "Test data may be concatenated",
        "explanation": "A held-out frame appears before the split.",
        "confidence": 0.7,
        "evidence": (evidence(),),
        "location": "model.py:4",
        "detector": "semantics",
        "severity_hint": "medium",
        "limitations": ("A heuristic can be wrong.",),
    }
    values.update(overrides)
    return Hypothesis(**values)


def model_card() -> ModelCard:
    return ModelCard(
        name="test-model",
        kind="gradient_boosting",
        features=10,
        training_examples=20,
        groups=4,
        threshold=0.5,
        precision=0.8,
        recall=0.7,
        f1=0.75,
        false_positive_rate=0.1,
        evaluation="Grouped test folds.",
        limitations=("Small synthetic corpus.",),
    )


def empty_plan() -> ReasoningPlan:
    return ReasoningPlan(
        root_causes=(),
        priority_order=(),
        summary="No observations.",
        verification=("Re-run the deterministic scan.",),
    )


def ai_report(**overrides) -> AIReport:
    values = {
        "scope": "code",
        "hypotheses": (),
        "plan": empty_plan(),
        "models": (model_card(),),
        "deterministic_findings": 0,
        "deterministic_trust_score": 100,
        "limitations": ("AI hypotheses are not findings.",),
    }
    values.update(overrides)
    return AIReport(**values)


def test_evidence_rejects_unknown_sources_and_control_characters():
    with pytest.raises(ValueError, match="unsupported evidence source"):
        evidence(source="internet")

    with pytest.raises(ValueError, match="control characters"):
        evidence(excerpt="hidden\x1b[31mtext")

    with pytest.raises(ValueError, match="non-empty"):
        evidence(source="finding", excerpt="")


def test_hypothesis_never_claims_certainty_and_needs_evidence():
    with pytest.raises(ValueError, match=r"0\.95"):
        hypothesis(confidence=0.96)

    with pytest.raises(ValueError, match="at least one evidence"):
        hypothesis(evidence=())

    with pytest.raises(ValueError, match="info, low or medium"):
        hypothesis(severity_hint="high")

    with pytest.raises(ValueError, match="at least one limitation"):
        hypothesis(limitations=())

    assert hypothesis(confidence=0.95).confidence == 0.95


def test_evidence_ids_must_be_unique_within_a_hypothesis():
    with pytest.raises(ValueError, match="unique"):
        hypothesis(evidence=(evidence(), evidence()))


def test_confidence_basis_defaults_keep_legacy_hypotheses_valid():
    item = hypothesis()

    assert item.confidence_basis == "unavailable"
    assert item.sufficiency == "weak"
    payload = item.to_dict()
    assert payload["confidence_basis"] == "unavailable"
    assert payload["sufficiency"] == "weak"


def test_confidence_basis_and_sufficiency_are_validated():
    assert (
        hypothesis(confidence_basis="heuristic_weight").confidence_basis
        == "heuristic_weight"
    )
    assert hypothesis(confidence_basis="unavailable").confidence_basis == "unavailable"
    assert (
        hypothesis(
            detector="learned", confidence_basis="calibrated_probability"
        ).confidence_basis
        == "calibrated_probability"
    )
    assert hypothesis(sufficiency="supported").sufficiency == "supported"

    with pytest.raises(ValueError, match="confidence basis"):
        hypothesis(confidence_basis="certainty")

    with pytest.raises(ValueError, match="sufficiency"):
        hypothesis(sufficiency="certain")

    # A calibrated probability may only be claimed when the learned detector
    # (the only producer with a fitted model) backs it.
    with pytest.raises(ValueError, match="calibrated_probability"):
        hypothesis(detector="semantics", confidence_basis="calibrated_probability")


def test_schema_version_carries_the_confidence_contract():
    assert SCHEMA_VERSION == "1.2"


def test_group_and_plan_must_stay_internally_consistent():
    remediation = Remediation(
        title="Fix the split",
        rationale="Split before fitting.",
        actions=("Split first.",),
        verification=("Re-scan.",),
        priority="high",
        evidence_ids=("ev-abc",),
    )
    group = RootCauseGroup(
        group_id="rc-overlap",
        category="overlap",
        title="Rows overlap",
        explanation="Rows appear on both sides.",
        priority="high",
        confidence=0.8,
        evidence_ids=("ev-abc",),
        hypothesis_ids=("hyp-abc",),
        remediation=remediation,
        source="combined",
    )

    with pytest.raises(ValueError, match="exactly the group ids"):
        ReasoningPlan(
            root_causes=(group,),
            priority_order=(),
            summary="One group.",
            verification=("Re-scan.",),
        )

    with pytest.raises(ValueError, match="unknown group"):
        ReasoningPlan(
            root_causes=(group,),
            priority_order=("rc-overlap",),
            summary="One group.",
            verification=("Re-scan.",),
            llm_summary="A summary.",
            llm_notes=(
                LLMNote(
                    group_id="rc-invented",
                    explanation="A rephrasing.",
                ),
            ),
        )


def test_ai_report_rejects_unknown_hypothesis_and_evidence_references():
    remediation = Remediation(
        title="Fix",
        rationale="Split first.",
        actions=("Split first.",),
        verification=("Re-scan.",),
        priority="high",
        evidence_ids=("ev-invented",),
    )
    group = RootCauseGroup(
        group_id="rc-overlap",
        category="overlap",
        title="Rows overlap",
        explanation="Rows appear on both sides.",
        priority="high",
        confidence=0.8,
        evidence_ids=("ev-invented",),
        hypothesis_ids=(),
        remediation=remediation,
        source="deterministic",
    )
    plan = ReasoningPlan(
        root_causes=(group,),
        priority_order=("rc-overlap",),
        summary="One group.",
        verification=("Re-scan.",),
    )

    with pytest.raises(ValueError, match="unknown evidence"):
        ai_report(plan=plan)

    with pytest.raises(ValueError, match="unknown hypotheses"):
        ai_report(
            plan=ReasoningPlan(
                root_causes=(
                    RootCauseGroup(
                        group_id="rc-overlap",
                        category="overlap",
                        title="Rows overlap",
                        explanation="Rows appear on both sides.",
                        priority="high",
                        confidence=0.8,
                        evidence_ids=("det-0",),
                        hypothesis_ids=("hyp-missing",),
                        remediation=Remediation(
                            title="Fix",
                            rationale="Split first.",
                            actions=("Split first.",),
                            verification=("Re-scan.",),
                            priority="high",
                            evidence_ids=("det-0",),
                        ),
                        source="combined",
                    ),
                ),
                priority_order=("rc-overlap",),
                summary="One group.",
                verification=("Re-scan.",),
            ),
            deterministic_findings=1,
        )


def test_ai_report_echoes_but_never_recomputes_the_trust_score():
    report = ai_report(
        hypotheses=(hypothesis(),),
        plan=ReasoningPlan(
            root_causes=(),
            priority_order=(),
            summary="One hypothesis.",
            verification=("Re-scan.",),
        ),
    )
    payload = report.to_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["engine"] == ENGINE_NAME
    assert payload["changes_deterministic_findings"] is False
    assert payload["deterministic_trust_score"] == 100
    assert "trust_score" not in payload
    assert payload["hypotheses"][0]["hypothesis_id"] == "hyp-abc"


def test_ai_report_refuses_to_claim_it_changed_deterministic_findings():
    with pytest.raises(ValueError, match="never claim"):
        ai_report(changes_deterministic_findings=True)

    with pytest.raises(ValueError, match="at least one limitation"):
        ai_report(limitations=())

    with pytest.raises(ValueError, match="at least one model card"):
        ai_report(models=())


def test_report_accepts_deterministic_finding_evidence_ids():
    remediation = Remediation(
        title="Fix the split",
        rationale="A finding was raised.",
        actions=("Split first.",),
        verification=("Re-scan.",),
        priority="high",
        evidence_ids=("det-0",),
    )
    group = RootCauseGroup(
        group_id="rc-overlap",
        category="overlap",
        title="Rows overlap",
        explanation="A deterministic finding was raised.",
        priority="high",
        confidence=0.85,
        evidence_ids=("det-0",),
        hypothesis_ids=(),
        remediation=remediation,
        source="deterministic",
    )
    payload = ai_report(
        plan=ReasoningPlan(
            root_causes=(group,),
            priority_order=("rc-overlap",),
            summary="One finding.",
            verification=("Re-scan.",),
        ),
        deterministic_findings=1,
    ).to_dict()

    assert payload["plan"]["root_causes"][0]["evidence_ids"] == ("det-0",)
