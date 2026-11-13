"""Text intake and deterministic normalisation (Phase 15 M1).

Normalisation is NFKC, case folding and whitespace collapse. A document's
digest is the first 16 hex characters of the SHA-256 of its normalised text, so
case, Unicode form and whitespace cannot hide an exact duplicate. Nothing here
keeps raw source text in a finding.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_DOCS = 10_000
DEFAULT_MIN_CHARS = 32
MAX_CHARS = 2_000_000
MAX_FILE_BYTES = 256 * 1024 * 1024
TEXT_SUFFIXES = frozenset({".txt", ".md", ".text"})
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
SNAPSHOT_SUFFIXES = frozenset({".parquet", ".feather", ".arrow"})


@dataclass(frozen=True, slots=True)
class Document:
    """One normalised document; ``digest`` identifies its normalised text."""

    id: str
    split: str
    text: str
    digest: str
    chars: int


def normalize_text(value: str) -> str:
    """NFKC, case-fold and collapse every whitespace run to one space."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def text_digest(value: str) -> str:
    """Stable 16-hex digest of the normalised text."""
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()[:16]


def _strip_boilerplate(value: str, boilerplate: frozenset[str]) -> str:
    if not boilerplate:
        return value
    kept = [
        line for line in value.splitlines() if normalize_text(line) not in boilerplate
    ]
    return "\n".join(kept)


def documents_from_texts(
    texts: Iterable[object],
    split: str,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars: int = MAX_CHARS,
    boilerplate: frozenset[str] = frozenset(),
) -> tuple[list[Document], int]:
    """Normalise texts into documents, counting every skipped input.

    Ids use the original position (``train:4``) so a skipped row never shifts
    the identity of a later document.
    """
    if not split:
        raise ValueError("split must be a non-empty name")
    if min_chars < 0 or max_docs <= 0 or max_chars <= 0:
        raise ValueError("invalid intake limits")
    documents: list[Document] = []
    skipped = 0
    for index, value in enumerate(texts):
        if not isinstance(value, str) or len(documents) >= max_docs:
            skipped += 1
            continue
        text = normalize_text(_strip_boilerplate(value, boilerplate))
        if not text or len(text) < min_chars or len(text) > max_chars:
            skipped += 1
            continue
        documents.append(
            Document(f"{split}:{index}", split, text, text_digest(text), len(text))
        )
    return documents, skipped


def _field_value(item: object, field: str | None) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get(field or "text")
        return value if isinstance(value, str) else None
    return None


def _jsonl_rows(path: Path, field: str | None, max_rows: int) -> list[str | None]:
    rows: list[str | None] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_field_value(json.loads(line), field))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def _json_rows(path: Path, field: str | None, max_rows: int) -> list[str | None]:
    data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    items = data if isinstance(data, list) else [data]
    return [_field_value(item, field) for item in items[:max_rows]]


def load_documents(
    path: str | Path,
    split: str,
    *,
    field: str | None = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_chars: int = MAX_CHARS,
    boilerplate: frozenset[str] = frozenset(),
) -> tuple[list[Document], int]:
    """Read one text corpus file and normalise it.

    Supported: ``.jsonl``/``.ndjson``, ``.json``, ``.txt``/``.md`` and
    ``.parquet``/``.feather``/``.arrow`` (through the local snapshot reader).
    Malformed JSONL rows are counted as skipped rather than crashing the audit.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"text corpus not found: {source}")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("text corpus exceeds the 256 MiB intake budget")
    suffix = source.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        rows = _jsonl_rows(source, field, max_docs)
    elif suffix == ".json":
        rows = _json_rows(source, field, max_docs)
    elif suffix in TEXT_SUFFIXES:
        rows = [source.read_text(encoding="utf-8-sig", errors="replace")]
    elif suffix in SNAPSHOT_SUFFIXES:
        from ...integrations.hf import read_snapshot

        rows = list(read_snapshot(source, field=field, max_rows=max_docs))
    else:
        raise ValueError(f"unsupported text format: {suffix or 'no suffix'}")

    if rows and all(row is None for row in rows):
        raise ValueError(f"no readable documents in {source}")
    return documents_from_texts(
        rows,
        split,
        min_chars=min_chars,
        max_docs=max_docs,
        max_chars=max_chars,
        boilerplate=boilerplate,
    )
