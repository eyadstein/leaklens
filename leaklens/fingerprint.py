"""Deterministic, domain-separated finding fingerprints.

Canonical form (schema 1)::

    leaklens:finding:v1\\0
    schema=1
    kind=<detector kind>
    surface=<detector family>
    path=<repo-relative POSIX path or empty>
    feature=<dataset column/feature or empty>
    key=<detector-supplied stable key or empty>

Line numbers, notebook cells, explanatory detail, timestamps and observed
values are deliberately excluded: moving a finding or changing a message must
not change its identity. The path is relative to the repository root when one
is supplied (or discoverable from a ``.git`` ancestor); absolute paths without
a discoverable root are flattened to a stable separator-normalized form.
Dataset and split detectors put a column name in ``column``, so it is always
treated as a feature there and never parsed as a file location (``*`` means
"no single column"); this keeps columns that contain ``/`` or ``:1`` distinct.

The hash is ``SHA-256(domain || canonical_form)``. Moving a file, changing the
detector kind, changing the affected feature/path, or changing the optional
stable key changes the fingerprint. Detectors that split one kind into
semantically different issues must supply distinct stable keys. The bundled
code scanners derive that key from the offending construct's AST shape and
enclosing function/class, so repeated findings of one kind in a file stay
distinct and the key never contains a line number or message text.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .manifest import surface_for

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from .leakage import Finding

SCHEMA_VERSION = 1
DOMAIN = b"leaklens:finding:v1\0"
LOCATION = re.compile(r"^(?P<path>.+?)(?:\[cell \d+\])?:(?P<line>\d+)$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_MAX_KEY = 200


def _safe(value: Any, limit: int = 500) -> str:
    return _CONTROL.sub(" ", str(value)).strip()[:limit]


def _split_path(value: str) -> tuple[str, bool]:
    text = str(value).replace("\\", "/")
    absolute = bool(_DRIVE.match(text)) or text.startswith("/")
    if _DRIVE.match(text):
        text = text[2:]
    text = posixpath.normpath(text)
    return text, absolute


def _repo_root(path: str) -> Path | None:
    candidate = Path(path)
    if not candidate.is_absolute():
        return None
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def normalize_path(path: str, root: str | Path | None = None) -> str:
    """Normalize separators/drive prefixes to a repository-relative POSIX path."""
    cleaned, absolute = _split_path(path)
    if not absolute:
        return "" if cleaned == "." else cleaned.removeprefix("./")
    if root is not None:
        root_clean, _ = _split_path(str(root))
        parts = [p for p in cleaned.split("/") if p not in ("", ".")]
        root_parts = [p for p in root_clean.split("/") if p not in ("", ".")]
        folded = [p.casefold() for p in parts]
        root_folded = [p.casefold() for p in root_parts]
        if len(parts) >= len(root_parts) and folded[: len(root_parts)] == root_folded:
            cleaned = "/".join(parts[len(root_parts) :])
        else:
            cleaned = posixpath.relpath(f"/{cleaned}", f"/{root_clean}")
    else:
        repo = _repo_root(str(path))
        if repo is not None:
            return normalize_path(cleaned, root=repo)
    cleaned = cleaned.lstrip("/")
    return "" if cleaned == "." else cleaned


def _looks_like_path(value: str) -> bool:
    return bool(
        "/" in value
        or "\\" in value
        or _DRIVE.match(value)
        or value.endswith((".py", ".ipynb"))
    )


_FEATURE_SURFACES = frozenset({"dataset", "split"})


def normalize_location(
    column: str,
    root: str | Path | None = None,
    *,
    surface: str | None = None,
) -> tuple[str, str]:
    """Return ``(path, feature)`` for a finding's location column."""
    text = str(column)
    if surface in _FEATURE_SURFACES:
        text = text.strip()
        return "", "" if text in {"", "*"} else _safe(text, limit=300)
    match = LOCATION.match(text)
    if match:
        return normalize_path(match.group("path"), root=root), ""
    text = text.strip()
    if text in {"", "*"}:
        return "", ""
    if _looks_like_path(text):
        return normalize_path(text, root=root), ""
    return "", _safe(text, limit=300)


def normalize_stable_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("stable_key must be a string or None")
    return _safe(value, limit=_MAX_KEY)


def canonical_form(finding: Finding, root: str | Path | None = None) -> str:
    """Versioned canonical representation that the digest is computed over."""
    surface = surface_for(finding.kind)
    path, feature = normalize_location(str(finding.column), root=root, surface=surface)
    fields = (
        f"schema={SCHEMA_VERSION}",
        f"kind={_safe(finding.kind, limit=120)}",
        f"surface={_safe(surface, limit=40)}",
        f"path={_safe(path)}",
        f"feature={_safe(feature, limit=300)}",
        f"key={normalize_stable_key(getattr(finding, 'stable_key', None))}",
    )
    return "\n".join(fields)


def fingerprint_for(finding: Finding, root: str | Path | None = None) -> str:
    payload = DOMAIN + canonical_form(finding, root=root).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
