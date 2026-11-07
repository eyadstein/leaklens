"""Validated, immutable data contracts for intelligence analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_VALID_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
_VALID_CONFIDENCE_LEVELS = frozenset(
    {"very_high", "high", "moderate", "low", "very_low"}
)
_VALID_EVIDENCE_SOURCES = frozenset({"scanner", "dataset", "pipeline", "user"})
_VALID_ANCHOR_KINDS = frozenset(
    {"feature", "file", "split", "dataset", "image", "finding"}
)
_VALID_RELATIONS = frozenset({"supports", "corroborates", "same_anchor"})


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
class EvidenceLink:
    """A deterministic, inspectable relationship between two evidence records.

    Links never add claims: they only state that two existing observations share
    an anchor or that one observation is detector-level detail for another.
    """

    source_id: str
    target_id: str
    relation: str
    rationale: str

    def __post_init__(self) -> None:
        for name in ("source_id", "target_id", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.source_id == self.target_id:
            raise ValueError("an evidence link must connect two different records")
        if self.relation not in _VALID_RELATIONS:
            raise ValueError(f"unsupported evidence relation: {self.relation}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceCluster:
    """Evidence records that describe one inspectable dataset, split or file.

    A cluster is grouping, not a new finding: severity and priority are copied
    from the underlying deterministic findings, and every member has an id.
    """

    cluster_id: str
    anchor: str
    anchor_kind: str
    finding_kinds: tuple[str, ...]
    finding_indices: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    sources: tuple[str, ...]
    severity: str
    priority: str
    confidence: ConfidenceAssessment
    measured_impact: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("cluster_id", "anchor"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must not be empty")
        if self.anchor_kind not in _VALID_ANCHOR_KINDS:
            raise ValueError(f"unsupported anchor kind: {self.anchor_kind}")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"unsupported severity: {self.severity}")
        if self.priority not in _VALID_SEVERITIES:
            raise ValueError(f"unsupported priority: {self.priority}")
        if not self.finding_kinds or any(
            not kind.strip() for kind in self.finding_kinds
        ):
            raise ValueError("at least one finding kind is required")
        if not self.finding_indices or any(index < 0 for index in self.finding_indices):
            raise ValueError("finding indices must be non-negative")
        if not self.evidence_ids:
            raise ValueError("a cluster requires at least one evidence id")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence ids must be unique within a cluster")

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
