"""Shared types for the labeled corpus modules."""

from __future__ import annotations

from dataclasses import dataclass

TRAIN = "train"
VALIDATION = "validation"
LOCKED_TEST = "test"
HOLDOUT = "holdout"


@dataclass(frozen=True, slots=True)
class CorpusCase:
    group: str
    name: str
    code: str
    leaky: bool
    category: str
    split: str = TRAIN
    source: str = "phase4-core"


__all__ = ["HOLDOUT", "LOCKED_TEST", "TRAIN", "VALIDATION", "CorpusCase"]
