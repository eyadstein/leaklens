"""Checkpoint and artifact metadata intake (Phase 16 M2/M6).

Delegates every byte-level decision to :mod:`leaklens.analysis.artifacts`: this
module only walks a declared file list, records bounded metadata and keeps the
refusal findings and typed skips separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...leakage import Finding
from ..artifacts import formats


@dataclass(frozen=True, slots=True)
class CheckpointReport:
    artifacts: tuple[dict[str, Any], ...]
    findings: tuple[Finding, ...]
    skipped: dict[str, int]


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return path.name


def inspect_artifacts(
    root: str | Path,
    files: list[Path],
    *,
    max_bytes: int = formats.MAX_ARTIFACT_BYTES,
) -> CheckpointReport:
    """Inspect each artifact once; never raise for hostile input."""
    base = Path(root).resolve()
    artifacts: list[dict[str, Any]] = []
    findings: list[Finding] = []
    skipped: dict[str, int] = {}
    for path in sorted(files):
        meta, finding, reason = formats.inspect_artifact(
            base, path, max_bytes=max_bytes
        )
        if finding is not None:
            findings.append(finding)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        if meta is not None:
            artifacts.append(
                {
                    "path": _rel(base, path),
                    "format": meta.format,
                    "digest": meta.digest,
                    **meta.summary,
                }
            )
    return CheckpointReport(
        tuple(artifacts), tuple(findings), dict(sorted(skipped.items()))
    )
