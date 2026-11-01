"""Validated, immutable data contracts for intelligence analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
_VALID_CONFIDENCE_LEVELS = frozenset(
    {"very_high", "high", "moderate", "low", "very_low"}
)
_VALID_EVIDENCE_SOURCES = frozenset({"scanner", "dataset", "pipeline", "user"})


@dataclass(frozen=True, slots=True)
class Evidence:
    """A traceable observation supporting or contradicting an assessment."""

    evidence_id: str
    source: str
    kind: str
    description: str
    finding_index: int | None = None
    location: str | None = None
    supports: bool = True
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source", "kind", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.source not in _VALID_EVIDENCE_SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source}")
        if self.finding_index is not None and self.finding_index < 0:
            raise ValueError("finding_index must be non-negative")
        if self.location is not None and not self.location.strip():
            raise ValueError("location must be non-empty when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """An explainable confidence assessment, not a probability of correctness."""

    score: float
    level: str
    reasons: tuple[str, ...]
    supporting_evidence: int
    contradicting_evidence: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")
        if self.level not in _VALID_CONFIDENCE_LEVELS:
            raise ValueError(f"unsupported confidence level: {self.level}")
        if self.supporting_evidence < 0 or self.contradicting_evidence < 0:
            raise ValueError("evidence counts must be non-negative")
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("at least one non-empty reason is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RootCause:
    """A proposed explanation of the mechanism behind a leakage finding."""

    category: str
    title: str
    explanation: str
    related_finding_kinds: tuple[str, ...]
    confidence: ConfidenceAssessment

    def __post_init__(self) -> None:
        for name in ("category", "title", "explanation"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if not self.related_finding_kinds:
            raise ValueError("at least one related finding kind is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """An actionable remediation proposal with explicit validation criteria."""

    title: str
    description: str
    priority: str
    actions: tuple[str, ...]
    validation_steps: tuple[str, ...]
    related_finding_kinds: tuple[str, ...]
    automated: bool = False

    def __post_init__(self) -> None:
        for name in ("title", "description", "priority"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.priority not in _VALID_SEVERITIES:
            raise ValueError(f"unsupported priority: {self.priority}")
        if not self.actions or any(not item.strip() for item in self.actions):
            raise ValueError("at least one non-empty action is required")
        if not self.validation_steps or any(
            not item.strip() for item in self.validation_steps
        ):
            raise ValueError("at least one non-empty validation step is required")
        if not self.related_finding_kinds:
            raise ValueError("at least one related finding kind is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
