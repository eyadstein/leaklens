"""Deterministic investigation plans (Phase 18 M1, ADR-025).

The plan is built from the evidence graph and never imports ``leaklens.ai``: a
complete plan exists with the AI layer absent. ``ai.reasoning`` may only add
validated narration afterwards (M4). Every step cites evidence ids that are
already in the graph and finding positions that already exist, so an
ungrounded step cannot be serialized.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from leaklens import manifest
from leaklens.leakage import Finding

from .context import AnalysisContext, build_context
from .provenance import EvidenceGraph, build_evidence_graph

ENGINE = "leaklens.intelligence.investigation"
MAX_STEPS = 200
MAX_RATIONALE = 300
MAX_QUESTION = 300
MAX_NARRATION = 800
NARRATION_STATES = (
    "unavailable",
    "llm",
    "deterministic_fallback",
    "ai_unavailable",
)

_NARRATION_UNAVAILABLE = "unavailable"
_UNREGISTERED_DETECTOR = "unregistered_detector"
_ANCHOR_PREPOSITION = {
    "feature": "on feature",
    "split": "on split",
    "dataset": "on dataset",
    "file": "in file",
    "image": "in image",
}


@dataclass(frozen=True, slots=True)
class _Step:
    """One ordered, grounded investigation action."""

    ordinal: int
    evidence_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    rationale: str
    step_type: ClassVar[str] = "step"

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise ValueError("ordinal must be a positive integer")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale is required")
        if len(self.rationale) > MAX_RATIONALE:
            raise ValueError(f"rationale must be at most {MAX_RATIONALE} characters")
        for name, ids in (
            ("evidence_ids", self.evidence_ids),
            ("finding_ids", self.finding_ids),
        ):
            if not isinstance(ids, tuple) or not ids:
                raise ValueError(f"{name} must be a non-empty tuple")
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} must not repeat an id")
            for value in ids:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type,
            "ordinal": self.ordinal,
            "evidence_ids": list(self.evidence_ids),
            "finding_ids": list(self.finding_ids),
            "rationale": self.rationale,
        }


class InspectEvidence(_Step):
    """Look at the evidence a cluster already carries."""

    step_type: ClassVar[str] = "inspect_evidence"


@dataclass(frozen=True, slots=True)
class SuggestedTest(_Step):
    """A deterministic follow-up check mapped to one registered detector.

    The check is never free text: ``detector`` must exist in
    ``manifest.DETECTORS``. The builder omits an unknown kind at runtime rather
    than inventing a check, so constructing this class with an unregistered
    kind is a development-time error.
    """

    step_type: ClassVar[str] = "suggested_test"
    detector: str = ""
    question: str = ""

    def __post_init__(self) -> None:
        _Step.__post_init__(self)
        if self.detector not in manifest.BY_KIND:
            raise ValueError(f"unregistered detector kind: {self.detector!r}")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question is required")
        if len(self.question) > MAX_QUESTION:
            raise ValueError(f"question must be at most {MAX_QUESTION} characters")

    def to_dict(self) -> dict[str, Any]:
        data = _Step.to_dict(self)
        data["detector"] = self.detector
        data["question"] = self.question
        return data


@dataclass(frozen=True, slots=True)
class OmittedTest:
    """A cluster whose test was not proposed because no kind was registered."""

    kinds: tuple[str, ...]
    reason: str
    finding_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, ids in (("kinds", self.kinds), ("finding_ids", self.finding_ids)):
            if not isinstance(ids, tuple) or not ids:
                raise ValueError(f"{name} must be a non-empty tuple")
            if any(not isinstance(value, str) or not value.strip() for value in ids):
                raise ValueError(f"{name} must contain non-empty strings")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinds": list(self.kinds),
            "reason": self.reason,
            "finding_ids": list(self.finding_ids),
        }


class MissingEvidence(_Step):
    """A finding fired without structured detector evidence to inspect."""

    step_type: ClassVar[str] = "missing_evidence"


@dataclass(frozen=True, slots=True)
class NarrationSlot(_Step):
    """Optional rephrasing of one cluster; it cannot add evidence.

    ``text`` stays empty until the AI layer fills it (M4). An empty slot is
    omitted from the serialized plan, so a plan whose narration is removed is
    byte-identical to one built with narration disabled.
    """

    step_type: ClassVar[str] = "narration_slot"
    text: str = ""

    def __post_init__(self) -> None:
        _Step.__post_init__(self)
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if len(self.text) > MAX_NARRATION:
            raise ValueError(f"text must be at most {MAX_NARRATION} characters")

    def to_dict(self) -> dict[str, Any]:
        data = _Step.to_dict(self)
        if self.text:
            data["text"] = self.text
        return data


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    """Ordered, grounded plan; deterministic and AI-independent."""

    steps: tuple[_Step, ...]
    finding_count: int
    engine: str = ENGINE
    narration: str = _NARRATION_UNAVAILABLE
    omitted_tests: tuple[OmittedTest, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "narration": self.narration,
            "finding_count": self.finding_count,
            "steps": [step.to_dict() for step in self.steps],
            "omitted_tests": [item.to_dict() for item in self.omitted_tests],
        }

    def narration_slots(self) -> tuple[NarrationSlot, ...]:
        return tuple(step for step in self.steps if isinstance(step, NarrationSlot))

    def with_narration(
        self,
        state: str,
        texts: Mapping[int, str] | None = None,
    ) -> InvestigationPlan:
        """Return a copy carrying narration text; no other field may change."""

        if state not in NARRATION_STATES:
            raise ValueError(f"unsupported narration state: {state}")
        texts = texts or {}
        for ordinal, text in texts.items():
            if not isinstance(ordinal, int) or not isinstance(text, str):
                raise TypeError("narration texts must map a step ordinal to a string")
            if len(text) > MAX_NARRATION:
                raise ValueError(f"text must be at most {MAX_NARRATION} characters")
        steps = tuple(
            replace(step, text=texts.get(step.ordinal, ""))
            if isinstance(step, NarrationSlot)
            else step
            for step in self.steps
        )
        return replace(self, steps=steps, narration=state)

    def without_narration(self) -> InvestigationPlan:
        """Drop every narration field, restoring the AI-free plan."""

        return self.with_narration(_NARRATION_UNAVAILABLE)


def _finding_id(index: int) -> str:
    return f"det-{index}"


def _kinds(cluster) -> str:
    return ", ".join(cluster.finding_kinds) or "unknown"


def _rationale(cluster, step: str) -> str:
    kinds = _kinds(cluster)
    if step == "inspect":
        return (
            f"Inspect the evidence behind the {kinds} finding(s) on {cluster.anchor!r}."
        )
    if step == "test":
        return (
            f"Re-run the {kinds} check after remediation and confirm the cited "
            "evidence no longer reproduces."
        )
    if step == "missing":
        return (
            f"The {kinds} finding(s) on {cluster.anchor!r} carry no structured "
            "detector evidence; add a reproducible check before relying on it."
        )
    return (
        "Optional narration may rephrase this group; it cannot add evidence or "
        "change a finding."
    )


def _detector_indices(evidence: Iterable) -> set[int]:
    indices: set[int] = set()
    for item in evidence:
        if (
            dict(item.metadata).get("role") == "detector"
            and item.finding_index is not None
        ):
            indices.add(item.finding_index)
    return indices


def _test_kinds(cluster) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the cluster's finding kinds into registered and unknown ones."""
    registered = tuple(
        kind for kind in cluster.finding_kinds if kind in manifest.BY_KIND
    )
    unknown = tuple(
        kind for kind in cluster.finding_kinds if kind not in manifest.BY_KIND
    )
    return registered, unknown


def _question(cluster, detector: str) -> str:
    where = _ANCHOR_PREPOSITION.get(cluster.anchor_kind, "at")
    text = (
        f"Does re-running the {detector} detector still reproduce the "
        f"{cluster.severity} finding {where} {cluster.anchor!r}?"
    )
    return text[:MAX_QUESTION]


def _ground(steps: list[_Step], findings: tuple[Finding, ...], known: set[str]) -> None:
    """Reject a step that cites an id outside the payload."""

    for step in steps:
        for evidence_id in step.evidence_ids:
            if evidence_id not in known:
                raise ValueError(f"ungrounded evidence id: {evidence_id!r}")
        for finding_id in step.finding_ids:
            index = int(finding_id.removeprefix("det-"))
            if not 0 <= index < len(findings):
                raise ValueError(f"ungrounded finding id: {finding_id!r}")


def build_investigation(
    findings: Iterable[Finding],
    graph: EvidenceGraph | None = None,
    context: AnalysisContext | None = None,
) -> InvestigationPlan:
    """Build the ordered plan from existing findings and their evidence graph."""

    findings = tuple(findings)
    context = context if context is not None else build_context(findings)
    graph = (
        graph if graph is not None else build_evidence_graph(findings, context=context)
    )

    with_evidence = _detector_indices(context.evidence)
    steps: list[_Step] = []
    omitted: list[OmittedTest] = []

    for cluster in graph.clusters:
        evidence_ids = tuple(cluster.evidence_ids)
        finding_ids = tuple(_finding_id(index) for index in cluster.finding_indices)
        builders: tuple[tuple[type[_Step], str], ...] = (
            (InspectEvidence, "inspect"),
            (SuggestedTest, "test"),
        )
        if not set(cluster.finding_indices) & with_evidence:
            builders += ((MissingEvidence, "missing"),)
        builders += ((NarrationSlot, "narration"),)
        for step_type, rationale_kind in builders:
            if len(steps) >= MAX_STEPS:
                break
            kwargs: dict[str, Any] = {}
            if step_type is SuggestedTest:
                registered, unknown = _test_kinds(cluster)
                if unknown:
                    omitted.append(
                        OmittedTest(
                            kinds=unknown,
                            reason=_UNREGISTERED_DETECTOR,
                            finding_ids=finding_ids,
                        )
                    )
                if not registered:
                    continue
                detector = registered[0]
                kwargs = {
                    "detector": detector,
                    "question": _question(cluster, detector),
                }
            steps.append(
                step_type(
                    ordinal=len(steps) + 1,
                    evidence_ids=evidence_ids,
                    finding_ids=finding_ids,
                    rationale=_rationale(cluster, rationale_kind),
                    **kwargs,
                )
            )

    _ground(steps, findings, set(graph.evidence_by_id))
    return InvestigationPlan(
        steps=tuple(steps),
        finding_count=len(findings),
        omitted_tests=tuple(omitted),
    )
