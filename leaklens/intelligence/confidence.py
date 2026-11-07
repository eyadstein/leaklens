"""Deterministic, explainable confidence assessment."""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import ConfidenceAssessment, Evidence


def _level(score: float) -> str:
    if score >= 0.85:
        return "very_high"
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "moderate"
    if score >= 0.30:
        return "low"
    return "very_low"


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
    level = _level(score)

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


def assess_cluster_confidence(evidence: Iterable[Evidence]) -> ConfidenceAssessment:
    """Assess a cluster of related findings with explicit corroboration bonuses.

    The base score is :func:`assess_confidence` over one primary record per
    finding. Small, capped bonuses reward genuinely independent signals:
    different detector kinds, detector-level evidence, and more than one
    source surface. The score is capped at 0.95 because a heuristic must not
    claim certainty.
    """

    items = tuple(evidence)
    primaries = tuple(
        item for item in items if dict(item.metadata).get("role") != "detector"
    )
    base = assess_confidence(primaries or items)

    if not items:
        return base

    kinds = {item.kind for item in primaries}
    detector_records = len(items) - len(primaries)
    sources = {item.source for item in items}

    score = base.score
    reasons = list(base.reasons)

    if len(kinds) > 1:
        score += 0.03 * min(len(kinds) - 1, 3)
        reasons.append(
            f"{len(kinds)} independent finding kinds point at the same anchor."
        )
    if detector_records:
        score += min(0.06, 0.02 * min(detector_records, 3))
        reasons.append(
            f"{detector_records} detector evidence record(s) describe why the "
            "detector fired."
        )
    if len(sources) > 1:
        score += 0.02
        reasons.append("Evidence comes from more than one detector surface.")

    score = round(min(0.95, max(0.0, score)), 3)
    return ConfidenceAssessment(
        score=score,
        level=_level(score),
        reasons=tuple(reasons),
        supporting_evidence=base.supporting_evidence,
        contradicting_evidence=base.contradicting_evidence,
    )
