"""Benchmark contamination against user-supplied eval fingerprints (Phase 15 M4).

Matching is hash-only: the caller supplies digests produced by
``normalize.text_digest`` (or ``write_fingerprints``). No evaluation corpus ships
with LeakLens and no eval text is stored, logged or echoed in evidence.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from pathlib import Path

from ...evidence import Evidence
from ...leakage import Finding
from .normalize import Document

FINGERPRINT = re.compile(r"^[0-9a-f]{8,64}$")
MAX_EXAMPLES = 8


def load_fingerprints(path: str | Path) -> frozenset[str]:
    """Read one hex fingerprint per line; ``#`` comments and blanks are ignored."""
    source = Path(path)
    raw = source.read_text(encoding="utf-8-sig", errors="replace")
    values: set[str] = set()
    for number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        if not FINGERPRINT.match(line):
            raise ValueError(f"invalid fingerprint on line {number}")
        values.add(line)
    return frozenset(values)


def write_fingerprints(path: str | Path, documents: Sequence[Document]) -> int:
    """Write sorted unique digests, the M4 companion for building a fingerprint file."""
    values = sorted({document.digest for document in documents})
    Path(path).write_text(
        "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
    )
    return len(values)


def contamination_findings(
    documents: Sequence[Document],
    fingerprints: Collection[str],
    field: str,
) -> list[Finding]:
    """One high finding when a training document matches a supplied eval digest."""
    matched = [document for document in documents if document.digest in fingerprints]
    if not matched:
        return []
    examples = [
        Evidence(
            source="split",
            kind="duplicate",
            description=f"training document {document.id} matches an eval fingerprint",
            observed=f"digest={document.digest}",
        )
        for document in matched[:MAX_EXAMPLES]
    ]
    summary = Evidence(
        source="split",
        kind="overlap",
        description="training documents match user-supplied eval fingerprints",
        location=f"field={field}",
        observed=f"matched={len(matched)} train_docs={len(documents)}",
        expected="0 eval fingerprints in train",
    )
    return [
        Finding(
            severity="high",
            kind="text_eval_contamination",
            column=field,
            detail=(
                f"{len(matched)} training documents match the supplied evaluation "
                f"fingerprints (field {field})."
            ),
            evidence=(summary, *examples),
            stable_key=field,
            diagnostic=True,
        )
    ]
