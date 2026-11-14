"""Validated, immutable data contracts for intelligence analysis."""

from __future__ import annotations

import math
import re
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


# --- Investigation plan contracts (Phase 18 M1, ADR-025) -------------------
#
# These types lived in ``leaklens.ai.schemas`` until Phase 18. The intelligence
# layer owns the plan because an investigation must be buildable with the AI
# layer unimportable; ``leaklens.ai.schemas`` re-exports them unchanged for
# compatibility. Validation stays strict for the same reason it always was:
# nothing may invent evidence ids or escalate a severity.

_PRIORITIES = frozenset({"high", "medium", "low", "info"})
_SOURCES = frozenset({"deterministic", "ai", "combined"})
_ENRICHMENTS = frozenset({"deterministic", "llm", "deterministic_fallback"})
_SEVERITY_HINTS = frozenset({"info", "low", "medium"})
_CONFIDENCE_BASES = frozenset(
    {"calibrated_probability", "heuristic_weight", "unavailable"}
)
_SUFFICIENCY = frozenset({"supported", "weak", "insufficient"})
_DETECTORS = frozenset({"semantics", "learned"})
_EVIDENCE_SOURCES = frozenset(
    {"code", "schema", "relationship", "statistic", "model", "finding"}
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_.-]{0,63}$")


def _text(value: Any, name: str, limit: int, *, allow_newlines: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    if _CONTROL.search(value):
        raise ValueError(f"{name} must not contain control characters")
    if not allow_newlines and ("\n" in value or "\r" in value):
        raise ValueError(f"{name} must be a single line")
    return value


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, 64)
    if not _ID.match(text):
        raise ValueError(f"{name} contains unsupported characters")
    return text


def _unit(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One inspectable observation behind a hypothesis.

    ``excerpt`` is generated from the user's own code or schema.  It is never
    accepted from an LLM and is redacted before any remote provider sees it.
    """

    evidence_id: str
    source: str
    kind: str
    location: str
    excerpt: str
    supports: bool = True
    graph_node_id: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        if self.source not in _EVIDENCE_SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source}")
        _text(self.kind, "kind", 64)
        _text(self.location, "location", 300)
        _text(self.excerpt, "excerpt", 400)
        if not isinstance(self.supports, bool):
            raise TypeError("supports must be a boolean")
        if self.graph_node_id is not None:
            _identifier(self.graph_node_id, "graph_node_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """An AI-generated, evidence-bound suspicion.  Never a verified finding."""

    hypothesis_id: str
    concept: str
    title: str
    explanation: str
    confidence: float
    evidence: tuple[EvidenceSpan, ...]
    location: str
    detector: str
    severity_hint: str = "medium"
    confidence_basis: str = "unavailable"
    sufficiency: str = "weak"
    column: str | None = None
    corroborates: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.hypothesis_id, "hypothesis_id")
        _text(self.concept, "concept", 80)
        _text(self.title, "title", 160)
        _text(self.explanation, "explanation", 1200)
        _unit(self.confidence, "confidence")
        if self.confidence > 0.95:
            raise ValueError("confidence must not exceed 0.95 (never certainty)")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("a hypothesis requires at least one evidence span")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique within a hypothesis")
        if not all(isinstance(item, EvidenceSpan) for item in self.evidence):
            raise ValueError("evidence must contain EvidenceSpan objects")
        _text(self.location, "location", 300)
        if self.detector not in _DETECTORS:
            raise ValueError(f"unsupported detector: {self.detector}")
        if self.severity_hint not in _SEVERITY_HINTS:
            raise ValueError("AI hypotheses may only use info, low or medium severity")
        if self.confidence_basis not in _CONFIDENCE_BASES:
            raise ValueError(f"unsupported confidence basis: {self.confidence_basis}")
        if self.sufficiency not in _SUFFICIENCY:
            raise ValueError(f"unsupported sufficiency: {self.sufficiency}")
        if self.confidence_basis == "calibrated_probability" and (
            self.detector != "learned"
        ):
            raise ValueError("calibrated_probability requires a fitted model behind it")
        if self.column is not None:
            _text(self.column, "column", 200)
        if self.corroborates is not None:
            _identifier(self.corroborates, "corroborates")
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("a hypothesis must state at least one limitation")
        for limitation in self.limitations:
            _text(limitation, "limitation", 300)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Remediation:
    """A suggested, never automatically executed, remediation and check."""

    title: str
    rationale: str
    actions: tuple[str, ...]
    verification: tuple[str, ...]
    priority: str
    evidence_ids: tuple[str, ...]
    code_hint: str | None = None

    def __post_init__(self) -> None:
        _text(self.title, "title", 160)
        _text(self.rationale, "rationale", 600)
        for name, items, limit in (
            ("actions", self.actions, 300),
            ("verification", self.verification, 300),
        ):
            if not isinstance(items, tuple) or not items:
                raise ValueError(f"{name} must contain at least one item")
            for item in items:
                _text(item, name, limit)
        if self.priority not in _PRIORITIES:
            raise ValueError(f"unsupported priority: {self.priority}")
        if not isinstance(self.evidence_ids, tuple) or not self.evidence_ids:
            raise ValueError("remediation requires at least one evidence id")
        for evidence_id in self.evidence_ids:
            _identifier(evidence_id, "evidence_id")
        if self.code_hint is not None:
            _text(self.code_hint, "code_hint", 600, allow_newlines=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RootCauseGroup:
    """Related deterministic findings and hypotheses with one remediation plan."""

    group_id: str
    category: str
    title: str
    explanation: str
    priority: str
    confidence: float
    evidence_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    remediation: Remediation
    source: str

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        _text(self.category, "category", 80)
        _text(self.title, "title", 160)
        _text(self.explanation, "explanation", 1000)
        if self.priority not in _PRIORITIES:
            raise ValueError(f"unsupported priority: {self.priority}")
        _unit(self.confidence, "confidence")
        if self.confidence > 0.95:
            raise ValueError("confidence must not exceed 0.95")
        if not self.evidence_ids:
            raise ValueError("a root-cause group requires evidence")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence ids must be unique within a group")
        for evidence_id in self.evidence_ids:
            _identifier(evidence_id, "evidence_id")
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("hypothesis ids must be unique within a group")
        for hypothesis_id in self.hypothesis_ids:
            _identifier(hypothesis_id, "hypothesis_id")
        if not isinstance(self.remediation, Remediation):
            raise TypeError("remediation must be a Remediation object")
        if self.source not in _SOURCES:
            raise ValueError(f"unsupported source: {self.source}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMNote:
    """An LLM rephrasing of one deterministic group, bound to an existing id."""

    group_id: str
    explanation: str
    actions: tuple[str, ...] = ()
    verification: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.group_id, "group_id")
        _text(self.explanation, "explanation", 800)
        for name, items in (
            ("actions", self.actions),
            ("verification", self.verification),
        ):
            if not isinstance(items, tuple):
                raise TypeError(f"{name} must be a tuple")
            for item in items:
                _text(item, name, 240)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReasoningPlan:
    """Deterministic root-cause plan, optionally enriched by an LLM."""

    root_causes: tuple[RootCauseGroup, ...]
    priority_order: tuple[str, ...]
    summary: str
    verification: tuple[str, ...]
    enrichment: str = "deterministic"
    fallback_reason: str | None = None
    llm_summary: str | None = None
    llm_notes: tuple[LLMNote, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.root_causes, tuple):
            raise TypeError("root_causes must be a tuple")
        ids = [group.group_id for group in self.root_causes]
        if len(ids) != len(set(ids)):
            raise ValueError("root-cause group ids must be unique")
        if len(self.priority_order) != len(set(self.priority_order)):
            raise ValueError("priority_order must not contain duplicates")
        if set(self.priority_order) != set(ids):
            raise ValueError("priority_order must contain exactly the group ids")
        _text(self.summary, "summary", 800)
        if not isinstance(self.verification, tuple) or not self.verification:
            raise ValueError("verification must contain at least one step")
        for step in self.verification:
            _text(step, "verification", 300)
        if self.enrichment not in _ENRICHMENTS:
            raise ValueError(f"unsupported enrichment: {self.enrichment}")
        if self.fallback_reason is not None:
            _text(self.fallback_reason, "fallback_reason", 300)
        if self.llm_summary is not None:
            _text(self.llm_summary, "llm_summary", 600)
        known = set(ids)
        for note in self.llm_notes:
            if note.group_id not in known:
                raise ValueError(f"LLM note references unknown group: {note.group_id}")

    def to_dict(self) -> dict[str, Any]:
        if self.llm_notes:
            data = {
                "root_causes": [group.to_dict() for group in self.root_causes],
                "priority_order": list(self.priority_order),
                "summary": self.summary,
                "verification": list(self.verification),
                "enrichment": self.enrichment,
                "fallback_reason": self.fallback_reason,
                "llm_summary": self.llm_summary,
                "llm_notes": [note.to_dict() for note in self.llm_notes],
            }
            return data
        return asdict(self)
