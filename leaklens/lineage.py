"""Dependency-free pipeline lineage contract (ADR-021).

``LineageEdge`` is not the evidence-to-evidence ``EvidenceLink``: it names a
pipeline stage, the column/artifact that reaches it, the transformation, the
metadata locator that proves it and whether that link was resolved. Edges are
immutable and always sorted, so two runs over the same repository produce the
same graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CONFIDENCES = frozenset({"confirmed", "unresolved"})
MAX_FIELD = 300


def _text(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    if len(text) > MAX_FIELD:
        raise ValueError(f"{field} exceeds {MAX_FIELD} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{field} contains control characters")
    return text


@dataclass(frozen=True, slots=True)
class LineageEdge:
    """One stage-to-artifact link: ``(stage, column, transform, locator, confidence)``."""

    stage: str
    column: str
    transform: str
    locator: str
    confidence: str = "unresolved"

    def __post_init__(self) -> None:
        for name in ("stage", "column", "transform", "locator"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"unsupported lineage confidence: {self.confidence!r}")

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "column": self.column,
            "transform": self.transform,
            "locator": self.locator,
            "confidence": self.confidence,
        }


def sort_edges(edges: Iterable[LineageEdge]) -> tuple[LineageEdge, ...]:
    """Deterministic order; the graph never depends on filesystem iteration."""
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.stage,
                edge.column,
                edge.transform,
                edge.locator,
                edge.confidence,
            ),
        )
    )
