"""Validated contracts for the experimental AI intelligence layer.

Everything here is deliberately separate from :mod:`leaklens.leakage`.  The
deterministic ``Finding`` objects remain authoritative; these schemas describe
*additional hypotheses* that always point back to inspectable evidence.

Validation is strict on purpose: AI (including an optional LLM) must never be
able to invent evidence ids, escalate a severity, or smuggle control characters
into reports.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "1.0"
ENGINE_NAME = "leaklens.ai.experimental"

_EVIDENCE_SOURCES = frozenset(
    {"code", "schema", "relationship", "statistic", "model", "finding"}
)
_SEVERITY_HINTS = frozenset({"info", "low", "medium"})
_PRIORITIES = frozenset({"high", "medium", "low", "info"})
_DETECTORS = frozenset({"semantics", "learned"})
_SCOPES = frozenset({"code", "dataset", "mixed"})
_ENRICHMENTS = frozenset({"deterministic", "llm", "deterministic_fallback"})
_SOURCES = frozenset({"deterministic", "ai", "combined"})

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


def _count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence_id")
        if self.source not in _EVIDENCE_SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source}")
        _text(self.kind, "kind", 64)
        _text(self.location, "location", 300)
        _text(self.excerpt, "excerpt", 400)
        if not isinstance(self.supports, bool):
            raise TypeError("supports must be a boolean")

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

    def __post_init__(self) -> None:
        if self.scope not in _SCOPES:
            raise ValueError(f"unsupported scope: {self.scope}")
        if self.changes_deterministic_findings is not False:
            raise ValueError("AI must never claim to change deterministic findings")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.engine != ENGINE_NAME:
            raise ValueError(f"unsupported engine: {self.engine}")
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
        return {
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


GLOBAL_LIMITATIONS = (
    "AI hypotheses are experimental and are not verified leakage findings.",
    "They never change deterministic findings, severity or the trust score.",
    "A clean scan does not prove a dataset or pipeline is risk-free.",
)


__all__ = [
    "ENGINE_NAME",
    "GLOBAL_LIMITATIONS",
    "SCHEMA_VERSION",
    "AIReport",
    "EvidenceSpan",
    "Hypothesis",
    "LLMEnrichment",
    "LLMNote",
    "ModelCard",
    "ReasoningPlan",
    "Remediation",
    "RootCauseGroup",
]
