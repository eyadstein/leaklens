"""Conversion of scanner findings into traceable analysis evidence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from leaklens.leakage import Finding

from .schemas import Evidence


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Immutable input context for deterministic and AI-assisted analysis."""

    evidence: tuple[Evidence, ...]

    @property
    def finding_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.kind for item in self.evidence))

    @property
    def finding_count(self) -> int:
        return len(self.evidence)

    def evidence_for(self, kind: str) -> tuple[Evidence, ...]:
        return tuple(item for item in self.evidence if item.kind == kind)


def build_context(findings: Iterable[Finding]) -> AnalysisContext:
    """Build stable evidence records while preserving original finding order."""

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

    return AnalysisContext(evidence=tuple(evidence))
