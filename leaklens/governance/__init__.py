"""Governance pack (Phase 18 M7): provenance-labelled JSON and Markdown.

``cards.py`` owns the field contracts and labels every field ``asserted``,
``measured`` or ``unknown``; ``export.py`` renders one payload into both
formats. The package imports the standard library and ``leaklens.paths`` only
and never imports ``leaklens.ai``.
"""

from .cards import (
    ASSERTED,
    CARD_VERSION,
    ENGINE,
    LABEL_MEANINGS,
    LABELS,
    MEASURED,
    NON_CLAIMS,
    UNKNOWN,
    Field,
    GovernanceCard,
    GovernanceError,
    build_card,
)
from .export import to_json, to_markdown, write

__all__ = [
    "ASSERTED",
    "CARD_VERSION",
    "ENGINE",
    "LABELS",
    "LABEL_MEANINGS",
    "MEASURED",
    "NON_CLAIMS",
    "UNKNOWN",
    "Field",
    "GovernanceCard",
    "GovernanceError",
    "build_card",
    "to_json",
    "to_markdown",
    "write",
]
