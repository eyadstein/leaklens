"""Structured, privacy-safe provenance for deterministic findings.

An evidence record explains *why* a detector fired. It carries detector
metadata, aggregate counts/ratios and expected bounds — never raw dataset
values, credentials or image bytes. Records are immutable, JSON-safe and
serialized in a fixed key order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SOURCES = frozenset({"data", "split", "code", "notebook", "image"})
KINDS = frozenset({"statistic", "overlap", "ordering", "duplicate", "parse", "pattern"})

MAX_TEXT = 500
MAX_FIELD = 300
MAX_ITEMS = 16


def _text(value: Any, field: str, limit: int, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{field} must not be empty")
    if not text:
        return None
    if len(text) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    # Control characters and unpaired surrogates are not JSON-safe and can
    # smuggle terminal escapes into reports.
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{field} contains control characters")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{field} is not valid UTF-8") from None
    return text


@dataclass(frozen=True, slots=True)
class Evidence:
    """One reason a detector fired, with bounded and validated fields."""

    source: str
    kind: str
    description: str
    location: str | None = None
    observed: str | None = None
    expected: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source!r}")
        if self.kind not in KINDS:
            raise ValueError(f"unsupported evidence kind: {self.kind!r}")
        _text(self.description, "description", MAX_TEXT, required=True)
        _text(self.location, "location", MAX_FIELD, required=False)
        _text(self.observed, "observed", MAX_FIELD, required=False)
        _text(self.expected, "expected", MAX_FIELD, required=False)

    def to_dict(self) -> dict[str, str]:
        """Deterministic JSON-safe representation; absent optionals are omitted."""
        payload = {
            "source": self.source,
            "kind": self.kind,
            "description": self.description,
        }
        for name in ("location", "observed", "expected"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


def coerce_evidence(value: Any) -> tuple[Evidence, ...]:
    """Normalize detector input into an immutable, bounded evidence tuple."""
    if value is None:
        return ()
    if isinstance(value, Evidence):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("evidence must be an Evidence or a sequence of Evidence")
    if len(value) > MAX_ITEMS:
        raise ValueError(f"evidence exceeds {MAX_ITEMS} records")
    for item in value:
        if not isinstance(item, Evidence):
            raise TypeError("every evidence item must be an Evidence")
    return tuple(value)
