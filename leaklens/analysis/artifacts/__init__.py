"""Typed, no-execution model-artifact intake (Phase 16 M2/M-S1).

Only metadata is ever read: safetensors headers, JSON documents and ONNX
protobuf fields. Pickle-based checkpoints are refused by magic and extension
without deserializing a byte, and archives are refused without extraction. Every
refusal is a typed reason (a skip) or a first-class finding, never a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ArtifactParseError(ValueError):
    """An artifact's bytes could not be parsed within the intake budgets."""


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """Bounded metadata for one accepted artifact; never weights or bytes."""

    format: str
    digest: str
    summary: dict[str, object] = field(default_factory=dict)
