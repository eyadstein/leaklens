"""Provenance-labelled governance fields (Phase 18 M7).

A governance pack is a fixed set of card fields, each labelled ``asserted``
(supplied by the caller and not verified), ``measured`` (computed by this audit
run) or ``unknown`` (not supplied and not verified). The pack records what was
observed; it is never a compliance claim or a certification. This module uses
the standard library only and never imports ``leaklens.ai``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ENGINE = "leaklens.governance"
CARD_VERSION = 1
ASSERTED = "asserted"
MEASURED = "measured"
UNKNOWN = "unknown"
LABELS = (ASSERTED, MEASURED, UNKNOWN)
MAX_SOURCE = 300
MAX_FIELDS = 100

NON_CLAIMS = (
    "This pack records what one audit observed; it is not a certification, "
    "an attestation, a guarantee or legal advice.",
    "A clean or partial pack is not proof that a dataset or model is safe, "
    "lawful or compliant.",
    "An unknown field was not supplied to LeakLens and was not verified.",
)

_SEVERITIES = ("high", "medium", "low")


class GovernanceError(ValueError):
    """A governance pack could not be built or written from the supplied data."""


@dataclass(frozen=True, slots=True)
class Field:
    """One labelled governance field and the source that produced it.

    An ``unknown`` field carries no value: the absence is the statement. A
    non-unknown field must carry one, so a reader never sees an empty label.
    """

    name: str
    label: str
    source: str
    value: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field name must be a non-empty string")
        if not all(part.isidentifier() for part in self.name.split(".")):
            raise ValueError(f"field name must be dotted identifiers: {self.name!r}")
        if self.label not in LABELS:
            raise ValueError(f"unsupported provenance label: {self.label!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if len(self.source) > MAX_SOURCE:
            raise ValueError(f"source must be at most {MAX_SOURCE} characters")
        if self.label == UNKNOWN:
            if self.value is not None:
                raise ValueError("an unknown field cannot carry a value")
            return
        if self.value is None:
            raise ValueError(f"{self.label} field {self.name!r} needs a value")
        if isinstance(self.value, str):
            if not self.value.strip():
                raise ValueError(f"field {self.name!r} has an empty value")
        elif isinstance(self.value, bool):
            pass
        elif not isinstance(self.value, (int, float)):
            raise TypeError("value must be a string, number, bool or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.name,
            "label": self.label,
            "value": self.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class GovernanceCard:
    """Ordered fields plus the non-claims; one payload for every renderer."""

    fields: tuple[Field, ...]
    engine: str = ENGINE
    card_version: int = CARD_VERSION
    non_claims: tuple[str, ...] = NON_CLAIMS

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a governance card needs at least one field")
        if len(self.fields) > MAX_FIELDS:
            raise ValueError(f"a governance card has at most {MAX_FIELDS} fields")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("field names must be unique")
        if self.engine != ENGINE:
            raise ValueError(f"unsupported engine: {self.engine!r}")
        if self.card_version != CARD_VERSION:
            raise ValueError(f"unsupported card version: {self.card_version!r}")
        if not self.non_claims:
            raise ValueError("a governance card must state its non-claims")

    def unknowns(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields if field.label == UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "card_version": self.card_version,
            "fields": [field.to_dict() for field in self.fields],
            "unknowns": list(self.unknowns()),
            "non_claims": list(self.non_claims),
        }


def _declared(name: str, source: str, value: str | None) -> Field:
    """A caller-supplied input, or an explicit unknown when it was not given."""
    if isinstance(value, str) and value.strip():
        return Field(name, ASSERTED, source, value)
    return Field(name, UNKNOWN, f"{source}: not supplied to this invocation")


def _severity_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    """Count payload findings exactly as ``assemble.severity_counts`` does."""
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise GovernanceError("the audit payload has no findings list")
    counts = dict.fromkeys(_SEVERITIES, 0)
    for finding in findings:
        severity = finding.get("severity") if isinstance(finding, Mapping) else None
        if severity in counts:
            counts[severity] += 1
    counts["total"] = len(findings)
    return counts


def _trust_score(payload: Mapping[str, Any]) -> int:
    value = payload.get("trust_score")
    if type(value) is not int or not 0 <= value <= 100:
        raise GovernanceError("the audit payload has no integer trust_score")
    return value


def _plan_fields(payload: Mapping[str, Any]) -> tuple[Field, ...]:
    plan = payload.get("investigation")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("steps"), list):
        reason = "investigation: absent because --intelligence was not enabled"
        return (
            Field("investigation.steps", UNKNOWN, reason),
            Field("investigation.narration", UNKNOWN, reason),
        )
    narration = plan.get("narration")
    narration_field = (
        Field("investigation.narration", MEASURED, "investigation.narration", narration)
        if isinstance(narration, str) and narration
        else Field(
            "investigation.narration",
            UNKNOWN,
            "investigation.narration: absent from the plan",
        )
    )
    return (
        Field(
            "investigation.steps",
            MEASURED,
            "investigation.steps",
            len(plan["steps"]),
        ),
        narration_field,
    )


def _gap_field(payload: Mapping[str, Any]) -> Field:
    report = payload.get("missing_evidence")
    if not isinstance(report, Mapping) or type(report.get("total")) is not int:
        return Field(
            "audit.missing_evidence",
            UNKNOWN,
            "missing_evidence: the key is absent when nothing was skipped",
        )
    return Field(
        "audit.missing_evidence", MEASURED, "missing_evidence.total", report["total"]
    )


def build_card(
    payload: Mapping[str, Any],
    *,
    train: str | None = None,
    test: str | None = None,
    target: str | None = None,
) -> GovernanceCard:
    """Build the pack from one assembled audit payload and the CLI inputs."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    counts = _severity_counts(payload)
    measured: tuple[Field, ...] = (
        Field("audit.trust_score", MEASURED, "leakage.score()", _trust_score(payload)),
        *(
            Field(
                f"audit.findings.{severity}",
                MEASURED,
                "assemble.severity_counts",
                counts[severity],
            )
            for severity in _SEVERITIES
        ),
        Field(
            "audit.findings.total",
            MEASURED,
            "assemble.severity_counts",
            counts["total"],
        ),
        _gap_field(payload),
        *_plan_fields(payload),
    )
    unknown: tuple[Field, ...] = (
        Field(
            "model.name",
            UNKNOWN,
            "not supplied: LeakLens audits data and code, not a model",
        ),
        Field(
            "model.version",
            UNKNOWN,
            "not supplied: LeakLens audits data and code, not a model",
        ),
        Field(
            "data.license", UNKNOWN, "not supplied: LeakLens does not inspect licensing"
        ),
        Field(
            "data.collection",
            UNKNOWN,
            "not supplied: LeakLens does not inspect collection",
        ),
        Field(
            "evaluation.protocol",
            UNKNOWN,
            "not supplied: LeakLens does not inspect the evaluation protocol",
        ),
    )
    return GovernanceCard(
        fields=(
            _declared("dataset.train", "cli.dataset", train),
            _declared("dataset.test", "cli.test", test),
            _declared("dataset.target", "cli.target", target),
            *measured,
            *unknown,
        )
    )


# The label table Markdown renders in its header; kept next to the vocabulary.
LABEL_MEANINGS = MappingProxyType(
    {
        ASSERTED: "supplied by the caller and not verified by this audit",
        MEASURED: "computed by this audit run",
        UNKNOWN: "not supplied and not verified",
    }
)
