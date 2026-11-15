"""Render one governance payload as JSON or Markdown and write it atomically.

Both renderers take the same payload produced by ``GovernanceCard.to_dict``, so
the two formats cannot drift apart. Output paths resolve through
``leaklens.paths.resolve_within`` and are replaced atomically, so a failed write
never leaves a half-written pack.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..paths import resolve_within
from .cards import GovernanceError

_FORMATS = {".json": "json", ".md": "markdown", ".markdown": "markdown"}


def to_json(payload: Mapping[str, Any]) -> str:
    """Serialize the payload as the canonical JSON document."""
    return json.dumps(payload, indent=2) + "\n"


def _cell(value: Any) -> str:
    """One Markdown table cell; strings stay unambiguous and pipes cannot split."""
    if value is None:
        return "-"
    text = json.dumps(value) if isinstance(value, str) else str(value)
    return text.replace("|", "\\|")


def to_markdown(payload: Mapping[str, Any]) -> str:
    """Render the same payload as a Markdown pack."""
    lines = [
        "# LeakLens governance pack",
        "",
        "Every field is labelled `asserted` (supplied by the caller, not "
        "verified), `measured` (computed by this audit run) or `unknown` (not "
        "supplied and not verified). This pack is not a compliance claim.",
        "",
        "| Field | Label | Value | Source |",
        "|---|---|---|---|",
    ]
    for field in payload["fields"]:
        lines.append(
            f"| `{field['field']}` | `{field['label']}` | "
            f"{_cell(field['value'])} | {_cell(field['source'])} |"
        )
    unknowns = list(payload.get("unknowns", ()))
    lines += ["", f"## Unknowns ({len(unknowns)})", ""]
    if unknowns:
        lines += [f"- `{name}`" for name in unknowns]
    else:
        lines.append("- None.")
    lines += ["", "## Non-claims", ""]
    lines += [f"- {claim}" for claim in payload.get("non_claims", ())]
    return "\n".join(lines) + "\n"


def write(
    payload: Mapping[str, Any], path: str | Path, *, root: str | Path | None = None
) -> Path:
    """Write the pack to ``path`` (``.json``, ``.md`` or ``.markdown``)."""
    base = Path.cwd() if root is None else Path(root)
    target = resolve_within(base, str(path))
    chosen = _FORMATS.get(target.suffix.lower())
    if chosen is None:
        raise GovernanceError("output path needs a .json, .md or .markdown suffix")
    text = to_json(payload) if chosen == "json" else to_markdown(payload)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=".governance-", suffix=target.suffix
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    except OSError as exc:
        raise GovernanceError(f"cannot write {str(path)!r}: {exc}") from exc
    return target
