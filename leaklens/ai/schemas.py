"""Validated contracts for the experimental AI intelligence layer.

Everything here is deliberately separate from :mod:`leaklens.leakage`.  The
deterministic ``Finding`` objects remain authoritative; these schemas describe
*additional hypotheses* that always point back to inspectable evidence.

Validation is strict on purpose: AI (including an optional LLM) must never be
able to invent evidence ids, escalate a severity, or smuggle control characters
into reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..intelligence.schemas import (
    EvidenceSpan,
    Hypothesis,
    LLMNote,
    ReasoningPlan,
    Remediation,
    RootCauseGroup,
    _text,
    _unit,
)

SCHEMA_VERSION = "1.2"
ENGINE_NAME = "leaklens.ai.experimental"

# 1.1 reports remain readable: M7 only adds an optional selection block.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.1", "1.2"})

# The exchangeability assumption a distribution-free selective-risk bound
# would need does not hold across repositories, so we publish the limit and
# never claim a guarantee.
EXCHANGEABILITY_LIMITATION = (
    "Repository-level shift violates the exchangeability assumption behind "
    "selective-risk bounds; no distribution-free guarantee is claimed."
)

SUFFICIENCY_LABELS = {
    "supported": "supported by calibration data",
    "weak": "weak (no fitted decision)",
    "insufficient": "insufficient calibration data (ranking only)",
}

# One canonical mapping from the machine value to the words a user reads.
# A calibrated model probability is exposed through a monotone ranking
# transform, so the displayed number is labelled as a ranking score even
# though its provenance is a calibrated probability.
CONFIDENCE_BASIS_LABELS = {
    "calibrated_probability": "uncalibrated ranking score",
    "heuristic_weight": "heuristic rule strength",
    "unavailable": "unavailable",
}


def confidence_label(basis: object) -> str:
    """Human label for a confidence basis; unknown values read as unavailable."""

    return CONFIDENCE_BASIS_LABELS.get(str(basis), "unavailable")


def sufficiency_label(value: object) -> str:
    """Human label for a sufficiency decision; unknown values read as weak."""

    return SUFFICIENCY_LABELS.get(str(value), SUFFICIENCY_LABELS["weak"])


_SCOPES = frozenset({"code", "dataset", "mixed"})
_ENRICHMENTS = frozenset({"deterministic", "llm", "deterministic_fallback"})
_SOURCES = frozenset({"deterministic", "ai", "combined"})


@dataclass(frozen=True, slots=True)
class LLMEnrichment:
    """Validated LLM output.  It cannot add groups or evidence."""

    summary: str
    notes: tuple[LLMNote, ...]

    def __post_init__(self) -> None:
        _text(self.summary, "summary", 600)
        if not isinstance(self.notes, tuple):
            raise TypeError("notes must be a tuple")
        ids = [note.group_id for note in self.notes]
        if len(ids) != len(set(ids)):
            raise ValueError("LLM notes must not repeat a group id")
        if self.notes and not all(isinstance(note, LLMNote) for note in self.notes):
            raise ValueError("notes must contain LLMNote objects")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Measured facts about the small local model, including its limits."""

    name: str
    kind: str
    features: int
    training_examples: int
    groups: int
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    evaluation: str
    limitations: tuple[str, ...]
    external: tuple[tuple[str, float], ...] = ()
    version: str = ""
    validation_examples: int = 0
    calibration: tuple[tuple[str, float], ...] = ()
    candidates: tuple[tuple[str, float], ...] = ()
    protocol: str = ""
    artifact_digest: str = ""
    corpus_digest: str = ""
    feature_catalogue_digest: str = ""
    library_version: str = ""

    def __post_init__(self) -> None:
        _text(self.name, "name", 120)
        _text(self.kind, "kind", 40)
        _count(self.features, "features")
        _count(self.training_examples, "training_examples")
        _count(self.groups, "groups")
        _unit(self.threshold, "threshold")
        for name in ("precision", "recall", "f1", "false_positive_rate"):
            _unit(getattr(self, name), name)
        _text(self.evaluation, "evaluation", 300)
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("a model card must state at least one limitation")
        for limitation in self.limitations:
            _text(limitation, "limitation", 300)
        for key, value in self.external:
            _text(key, "external metric", 80)
            _unit(value, "external metric value")
        if not isinstance(self.version, str):
            raise TypeError("model version must be a string")
        if self.version:
            _text(self.version, "version", 80)
        _count(self.validation_examples, "validation_examples")
        for key, value in self.calibration:
            _text(key, "calibration metric", 80)
            _unit(value, "calibration metric value")
        for key, value in self.candidates:
            _text(key, "candidate", 80)
            _unit(value, "candidate score")
        if not isinstance(self.protocol, str):
            raise TypeError("protocol must be a string")
        if self.protocol:
            _text(self.protocol, "protocol", 600)
        for name in (
            "artifact_digest",
            "corpus_digest",
            "feature_catalogue_digest",
            "library_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if value:
                _text(value, name, 120)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Fit-time selective-reporting decision and its measured limits."""

    applied: bool
    threshold: float
    coverage: float
    unit_coverage: float
    accepted: int
    emitted: int
    examples: int
    selective_risk: float
    risk_interval: tuple[float, float]
    unselective_error_rate: float
    protocol: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.applied, bool):
            raise TypeError("applied must be a boolean")
        _unit(self.threshold, "threshold")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("selection threshold must be a probability")
        _unit(self.coverage, "coverage")
        _unit(self.unit_coverage, "unit_coverage")
        _count(self.accepted, "accepted")
        _count(self.emitted, "emitted")
        _count(self.examples, "examples")
        if not 0 < self.accepted <= self.emitted <= self.examples:
            raise ValueError(
                "selection counts must satisfy accepted <= emitted <= examples"
            )
        _unit(self.selective_risk, "selective_risk")
        _unit(self.unselective_error_rate, "unselective_error_rate")
        if not isinstance(self.risk_interval, tuple) or len(self.risk_interval) != 2:
            raise ValueError("risk_interval must be a (low, high) pair")
        low, high = self.risk_interval
        _unit(low, "risk_interval")
        _unit(high, "risk_interval")
        if low > high:
            raise ValueError("risk_interval must be ordered low to high")
        _text(self.protocol, "protocol", 600)
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("a selection decision must state its limitations")
        for limitation in self.limitations:
            _text(limitation, "limitation", 300)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AIReport:
    """Top-level AI payload.  It never contains or recalculates a trust score."""

    scope: str
    hypotheses: tuple[Hypothesis, ...]
    plan: ReasoningPlan
    models: tuple[ModelCard, ...]
    deterministic_findings: int
    target: str | None = None
    deterministic_trust_score: int | None = None
    schema_version: str = SCHEMA_VERSION
    engine: str = ENGINE_NAME
    limitations: tuple[str, ...] = ()
    changes_deterministic_findings: bool = False
    selection: SelectionReport | None = None

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"unsupported scope: {self.scope}")
        if self.changes_deterministic_findings is not False:
            raise ValueError("AI must never claim to change deterministic findings")
        if self.schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.engine != ENGINE_NAME:
            raise ValueError(f"unsupported engine: {self.engine}")
        if self.selection is not None and not isinstance(
            self.selection, SelectionReport
        ):
            raise TypeError("selection must be a SelectionReport")
        ids = [item.hypothesis_id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis ids must be unique")
        known_hypotheses = set(ids)
        known_evidence = {
            span.evidence_id
            for hypothesis in self.hypotheses
            for span in hypothesis.evidence
        }
        known_evidence.update(
            f"det-{index}" for index in range(self.deterministic_findings)
        )
        for group in self.plan.root_causes:
            unknown_hypotheses = set(group.hypothesis_ids) - known_hypotheses
            if unknown_hypotheses:
                raise ValueError(
                    "root-cause group references unknown hypotheses: "
                    + ", ".join(sorted(unknown_hypotheses))
                )
            unknown_evidence = set(group.evidence_ids) - known_evidence
            if unknown_evidence:
                raise ValueError(
                    "root-cause group references unknown evidence: "
                    + ", ".join(sorted(unknown_evidence))
                )
        if not isinstance(self.models, tuple) or not self.models:
            raise ValueError("at least one model card is required")
        if not all(isinstance(model, ModelCard) for model in self.models):
            raise ValueError("models must contain ModelCard objects")
        _count(self.deterministic_findings, "deterministic_findings")
        if self.deterministic_trust_score is not None and (
            not isinstance(self.deterministic_trust_score, int)
            or not 0 <= self.deterministic_trust_score <= 100
        ):
            raise ValueError("deterministic_trust_score must be between 0 and 100")
        if self.target is not None:
            _text(self.target, "target", 200)
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("an AI report must state at least one limitation")
        for limitation in self.limitations:
            _text(limitation, "limitation", 300)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "scope": self.scope,
            "target": self.target,
            "deterministic_findings": self.deterministic_findings,
            "deterministic_trust_score": self.deterministic_trust_score,
            "changes_deterministic_findings": self.changes_deterministic_findings,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "plan": self.plan.to_dict(),
            "models": [model.to_dict() for model in self.models],
            "limitations": list(self.limitations),
        }
        if self.selection is not None:
            payload["selection"] = self.selection.to_dict()
        return payload


GLOBAL_LIMITATIONS = (
    "AI hypotheses are experimental and are not verified leakage findings.",
    "They never change deterministic findings, severity or the trust score.",
    "A clean scan does not prove a dataset or pipeline is risk-free.",
)


__all__ = [
    "CONFIDENCE_BASIS_LABELS",
    "ENGINE_NAME",
    "EXCHANGEABILITY_LIMITATION",
    "GLOBAL_LIMITATIONS",
    "SCHEMA_VERSION",
    "SUFFICIENCY_LABELS",
    "AIReport",
    "EvidenceSpan",
    "Hypothesis",
    "LLMEnrichment",
    "LLMNote",
    "ModelCard",
    "ReasoningPlan",
    "Remediation",
    "RootCauseGroup",
    "SelectionReport",
    "confidence_label",
    "sufficiency_label",
]
