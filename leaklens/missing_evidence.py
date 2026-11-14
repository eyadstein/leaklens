"""One central missing-evidence record for assembled payloads (Phase 18 M3).

The record lists what an audit could not check and why: engine ``skipped``
maps, split ``unresolved`` counters and limitations, and tests the
investigation plan could not propose. Each entry keeps its own reason and
count instead of collapsing into "some checks were skipped". The record never
creates, raises or scores a finding and never changes the exit gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .availability import EngineUnavailable

MAX_ENTRIES = 200
MAX_TEXT = 300

# What a skip means for the reader, by area. The reason code stays per entry.
_AREA_EFFECTS = {
    "missingness": (
        "The missingness audit did not evaluate the affected feature(s), so a "
        "missingness link to the target may be unreported."
    ),
    "drift": (
        "The drift diagnostic did not evaluate the affected column(s); the trust "
        "score and exit gate are unchanged."
    ),
    "pipeline": (
        "The pipeline audit did not examine the affected metadata, so pipeline "
        "lineage or parameter problems may be unreported."
    ),
    "monitoring": (
        "The monitoring report was not evaluated; its drift or skew metrics are "
        "absent from the plan."
    ),
    "text": (
        "Documents were not compared across text splits; text leakage may be "
        "unreported."
    ),
    "dl": (
        "Some deep-learning sources or artifacts were not inspected; related "
        "risks may be unreported."
    ),
    "splits": (
        "Some declared images were not verified; declared-split overlap may be "
        "unreported."
    ),
    "investigation": (
        "A proposed test was omitted because no registered detector can run it."
    ),
}
_DEFAULT_EFFECT = "This check did not run; its evidence is absent from the plan."

# Payload sections this collector reads. Anything else is ignored.
_SECTIONS = ("missingness", "drift", "pipeline", "monitoring", "text", "dl", "splits")


def _clip(value: str) -> str:
    return value[:MAX_TEXT]


def effect_for(area: str) -> str:
    """Return what a skip in ``area`` means, without collapsing the reason."""
    return _AREA_EFFECTS.get(area, _DEFAULT_EFFECT)


@dataclass(frozen=True, slots=True)
class MissingRecord:
    """One enumerated gap, with its own reason and count."""

    area: str
    reason: str
    effect: str
    source: str
    count: int = 1
    finding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("area", "reason", "effect", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
            if len(value) > MAX_TEXT:
                raise ValueError(f"{name} must be at most {MAX_TEXT} characters")
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(self.finding_ids, tuple):
            raise TypeError("finding_ids must be a tuple")
        if any(
            not isinstance(item, str) or not item.strip() for item in self.finding_ids
        ):
            raise ValueError("finding_ids must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "area": self.area,
            "reason": self.reason,
            "effect": self.effect,
            "source": self.source,
            "count": self.count,
        }
        if self.finding_ids:
            data["finding_ids"] = list(self.finding_ids)
        return data


@dataclass(frozen=True, slots=True)
class MissingEvidenceReport:
    """Deterministically ordered entries plus per-reason counts."""

    entries: tuple[MissingRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.reason] = counts.get(entry.reason, 0) + entry.count
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "counts": dict(sorted(counts.items())),
            "total": sum(counts.values()),
        }


def from_skips(
    area: str, source: str, skipped: Mapping[str, int] | None
) -> list[MissingRecord]:
    """Turn one engine's ``{reason: count}`` map into one record per reason."""
    if not isinstance(skipped, Mapping):
        return []
    records: list[MissingRecord] = []
    for reason in sorted(skipped):
        count = skipped[reason]
        if type(count) is not int or count < 1:
            continue
        records.append(
            MissingRecord(
                area=area,
                reason=str(reason),
                effect=effect_for(area),
                source=source,
                count=count,
            )
        )
    return records


def from_document_counts(
    area: str, counts: Mapping[str, int] | None, *, source: str
) -> list[MissingRecord]:
    """Turn per-split document counts into one ``document_skipped`` record each."""
    if not isinstance(counts, Mapping):
        return []
    records: list[MissingRecord] = []
    for split in sorted(counts):
        count = counts[split]
        if type(count) is not int or count < 1:
            continue
        records.append(
            MissingRecord(
                area=area,
                reason="document_skipped",
                effect=effect_for(area),
                source=f"{source}.{split}",
                count=count,
            )
        )
    return records


def from_limitations(
    area: str, source: str, limitations: Iterable[str] | None
) -> list[MissingRecord]:
    """Keep each distinct engine limitation as its own entry."""
    if not limitations:
        return []
    records: list[MissingRecord] = []
    for message in limitations:
        text = str(message).strip()
        if not text:
            continue
        records.append(
            MissingRecord(
                area=area,
                reason="limitation",
                effect=_clip(text),
                source=source,
                count=1,
            )
        )
    return records


def engine_unavailable(error: EngineUnavailable, area: str) -> MissingRecord:
    """Record a caught optional-engine failure, including the extra it names."""
    if not isinstance(error, EngineUnavailable):
        raise TypeError("expected an EngineUnavailable error")
    return MissingRecord(
        area=area,
        reason="engine_unavailable",
        effect=_clip(str(error)),
        source=f"{area}.availability",
        count=1,
    )


def _records_from_report(
    area: str, source: str, report: Mapping[str, Any]
) -> list[MissingRecord]:
    skipped = report.get("skipped")
    if area == "text":
        records = from_document_counts(area, skipped, source=source)
    else:
        records = from_skips(area, f"{source}.skipped", skipped)
    records.extend(from_skips(area, f"{source}.unresolved", report.get("unresolved")))
    records.extend(
        from_limitations(area, f"{source}.limitations", report.get("limitations"))
    )
    return records


def from_payload(payload: Mapping[str, Any]) -> list[MissingRecord]:
    """Collect records from the engine sections already present in a payload."""
    records: list[MissingRecord] = []
    for area in _SECTIONS:
        if area not in payload:
            continue
        value = payload[area]
        if not isinstance(value, Mapping):
            continue
        nested = value.get("reports")
        if area == "pipeline" and isinstance(nested, list):
            for index, report in enumerate(nested):
                if isinstance(report, Mapping):
                    ident = str(report.get("id") or index)
                    records.extend(
                        _records_from_report(area, f"{area}:{ident}", report)
                    )
            continue
        records.extend(_records_from_report(area, area, value))
    return records


def build(records: Iterable[MissingRecord]) -> MissingEvidenceReport:
    """Order entries deterministically and cap them with a visible reason."""
    entries = sorted(records, key=lambda item: (item.area, item.source, item.reason))
    if len(entries) > MAX_ENTRIES:
        dropped = entries[MAX_ENTRIES - 1 :]
        entries = entries[: MAX_ENTRIES - 1]
        entries.append(
            MissingRecord(
                area="missing_evidence",
                reason="entry_cap",
                effect=f"{len(dropped)} additional entries were not recorded.",
                source="missing_evidence",
                count=len(dropped),
            )
        )
    return MissingEvidenceReport(entries=tuple(entries))
