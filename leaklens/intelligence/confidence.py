"""Deterministic, explainable confidence assessment."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import ConfidenceAssessment, Evidence


def assess_confidence(evidence: Iterable[Evidence]) -> ConfidenceAssessment:
    """Estimate evidence strength using a transparent heuristic.

    This score measures the strength of the available evidence.
    It is not a calibrated probability that a finding is correct.
    """

    items = tuple(evidence)
    supporting = sum(item.supports for item in items)
    contradicting = len(items) - supporting

    if not items:
        return ConfidenceAssessment(
            score=0.0,
            level="very_low",
            reasons=("No supporting or contradicting evidence was provided.",),
            supporting_evidence=0,
            contradicting_evidence=0,
        )

    score = 0.25
    score += min(supporting, 4) * 0.15
    score -= min(contradicting, 4) * 0.12

    supporting_kinds = {item.kind for item in items if item.supports}
    if len(supporting_kinds) > 1:
        score += 0.10

    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.85:
        level = "very_high"
    elif score >= 0.70:
        level = "high"
    elif score >= 0.50:
        level = "moderate"
    elif score >= 0.30:
        level = "low"
    else:
        level = "very_low"

    reasons = []

    if supporting:
        reasons.append(f"{supporting} supporting evidence item(s) were identified.")
    if contradicting:
        reasons.append(
            f"{contradicting} contradicting evidence item(s) were identified."
        )
    if len(supporting_kinds) > 1:
        reasons.append("Supporting evidence covers multiple finding categories.")
    if supporting == 0:
        reasons.append("No supporting evidence was identified.")

    reasons.append(
        "The score is a deterministic evidence-strength heuristic, not a probability."
    )

    return ConfidenceAssessment(
        score=score,
        level=level,
        reasons=tuple(reasons),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
    )
