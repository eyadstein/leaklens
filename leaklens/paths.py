"""One containment helper shared by the workspace, the API and Phase 19 (ADR-024).

``resolve_within`` refuses a path whose drive/anchor differs from the root
*before* calling ``Path.resolve()``, so a UNC or device path cannot trigger
network resolution (SMB/NTLM) just to be rejected afterwards. A UNC workspace
root still works: its children share the same anchor.
"""

from __future__ import annotations

from pathlib import Path


class PathOutsideRoot(ValueError):
    """A supplied path resolves outside its containment root."""


def contained(root: Path, target: Path) -> bool:
    """True when ``target`` is ``root`` or below it (both already resolved)."""
    return target == root or target.is_relative_to(root)


def _same(value: str, other: str) -> bool:
    return value.replace("\\", "/").lower() == other.replace("\\", "/").lower()


def resolve_within(root: Path, value: str) -> Path:
    """Resolve an operator/request path and require it to stay inside ``root``."""
    text = (value or "").strip()
    if not text:
        raise PathOutsideRoot("a path is required")
    # POSIX reads "..\outside.csv" as one filename; normalise before parsing.
    # shortcut: a POSIX file whose name contains "\" is unreachable from the root;
    # upgrade if a workspace ever has to expose such names.
    candidate = Path(text.replace("\\", "/"))
    if candidate.drive and not _same(candidate.drive, root.drive):
        raise PathOutsideRoot("path is outside the workspace root")
    if candidate.anchor and not _same(candidate.anchor, root.anchor):
        raise PathOutsideRoot("path is outside the workspace root")
    target = candidate if candidate.is_absolute() else root / candidate
    target = target.resolve()
    if not contained(root, target):
        raise PathOutsideRoot("path is outside the workspace root")
    return target
