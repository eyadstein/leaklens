"""Conversion of scanner findings into traceable analysis evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from leaklens.leakage import Finding

from .schemas import Evidence

# Detector evidence uses the shared provenance vocabulary; the intelligence
# layer keeps its smaller source set so existing consumers keep working.
_SOURCE_MAP = {
    "data": "dataset",
    "split": "dataset",
    "code": "scanner",
    "notebook": "scanner",
    "image": "scanner",
}


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Immutable input context for deterministic and AI-assisted analysis."""

    evidence: tuple[Evidence, ...]

    @property
    def primary_evidence(self) -> tuple[Evidence, ...]:
        """One record per finding, in original order."""
        return tuple(
            item
            for item in self.evidence
            if dict(item.metadata).get("role") != "detector"
        )

    @property
    def detector_evidence(self) -> tuple[Evidence, ...]:
        """Detector-supplied detail records, linked to their primary record."""
        return tuple(
            item
            for item in self.evidence
            if dict(item.metadata).get("role") == "detector"
        )

    @property
    def finding_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.kind for item in self.primary_evidence))

    @property
    def finding_count(self) -> int:
        return len({item.finding_index for item in self.primary_evidence})

    def evidence_for(self, kind: str) -> tuple[Evidence, ...]:
        return tuple(item for item in self.evidence if item.kind == kind)

    def evidence_for_finding(self, index: int) -> tuple[Evidence, ...]:
        return tuple(item for item in self.evidence if item.finding_index == index)


def _detector_metadata(finding: Finding, item) -> tuple[tuple[str, str], ...]:
    metadata = [
        ("severity", finding.severity),
        ("role", "detector"),
        ("observation", item.kind),
    ]
    if item.observed is not None:
        metadata.append(("observed", item.observed))
    if item.expected is not None:
        metadata.append(("expected", item.expected))
    return tuple(metadata)


def build_context(findings: Iterable[Finding]) -> AnalysisContext:
    """Build stable evidence records while preserving original finding order.

    Every finding keeps one primary record with the historical
    ``scanner-<index>`` id and metadata. Detector evidence attached to a
    finding is preserved as additional ``-e<ordinal>`` records, so confidence
    and reports can see *why* a detector fired instead of only its message.
    """

    evidence = []

    for index, finding in enumerate(findings):
        if not isinstance(finding, Finding):
            raise TypeError(f"expected Finding at index {index}")

        evidence.append(
            Evidence(
                evidence_id=f"scanner-{index:06d}",
                source="scanner",
                kind=finding.kind,
                description=finding.detail,
                finding_index=index,
                location=finding.column,
                metadata=(("severity", finding.severity),),
            )
        )

        for ordinal, item in enumerate(finding.evidence, start=1):
            evidence.append(
                Evidence(
                    evidence_id=f"scanner-{index:06d}-e{ordinal:02d}",
                    source=_SOURCE_MAP.get(item.source, "scanner"),
                    kind=finding.kind,
                    description=item.description,
                    finding_index=index,
                    location=item.location or finding.column,
                    metadata=_detector_metadata(finding, item),
                )
            )

    return AnalysisContext(evidence=tuple(evidence))
