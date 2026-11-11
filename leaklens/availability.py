"""Optional-engine availability: one base error and one import probe (ADR-023).

``EngineUnavailable`` subclasses ``ImportError`` so every existing caller that
catches ``ImportError`` keeps working. Assembly catches only this type and
records a skip; any other exception is an engine error, never a silent skip.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class EngineUnavailable(ImportError):
    """An optional dependency an engine requires is not installed."""


def probe(
    module: str,
    extra: str,
    *,
    purpose: str,
    label: str | None = None,
    error: type[EngineUnavailable] = EngineUnavailable,
) -> ModuleType:
    """Import ``module`` or raise ``error`` naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise error(
            f'{purpose} need {label or module}: pip install "leaklens-ml[{extra}]"'
        ) from None
