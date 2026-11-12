"""Exact and near-duplicate text overlap across declared splits (Phase 15 M2/M3).

Matching is deterministic: exact overlap is a digest intersection; near overlap
is a bottom-k word-shingle sketch fed to the shared ``near_duplicate`` primitive
(LSH banding, threshold check, stable cluster key). Findings are aggregated per
field and carry counts plus 16-hex digests, never document text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ...evidence import Evidence
from ...leakage import Finding
from ...near_duplicate import DEFAULT_THRESHOLD, cluster, jaccard
from .normalize import Document

DEFAULT_SHINGLE_SIZE = 5
DEFAULT_HASHES = 64
MAX_EXAMPLES = 8
# Boilerplate-like bands (licences, headers) produce quadratic candidate pairs
# and mostly noise; exact duplicates are already covered by M2.
MAX_BUCKET = 64


def _shingles(text: str, size: int) -> set[str]:
    words = text.split()
    if len(words) < size:
        return set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def signatures(
    documents: Sequence[Document],
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_HASHES,
) -> dict[str, tuple[int, ...]]:
    """Bottom-k MinHash sketch per document id."""
    if shingle_size < 1 or num_hashes < 1:
        raise ValueError("sketch parameters must be positive")
    sketches: dict[str, tuple[int, ...]] = {}
    for document in documents:
        hashed = sorted(
            {
                int.from_bytes(
                    hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
                    "big",
                )
                for shingle in _shingles(document.text, shingle_size)
            }
        )
        sketches[document.id] = tuple(hashed[:num_hashes])
    return sketches


def exact_matches(train: Sequence[Document], test: Sequence[Document]) -> set[str]:
    """Ids of test documents whose normalised digest also appears in train."""
    train_digests = {document.digest for document in train}
    return {document.id for document in test if document.digest in train_digests}


def near_matches(
    train: Sequence[Document],
    test: Sequence[Document],
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_HASHES,
    threshold: float = DEFAULT_THRESHOLD,
) -> set[str]:
    """Ids of test documents that share a near-duplicate cluster with train."""
    if not train or not test:
        return set()
    sketches = signatures(
        [*train, *test], shingle_size=shingle_size, num_hashes=num_hashes
    )
    test_ids = {document.id for document in test}
    matched: set[str] = set()
    for group in cluster(sketches, threshold=threshold, rows=1, max_bucket=MAX_BUCKET):
        members = set(group.members)
        if members & test_ids and members - test_ids:
            matched |= members & test_ids
    return matched


def _summary(
    kind: str,
    field: str,
    description: str,
    observed: str,
    expected: str,
) -> Evidence:
    return Evidence(
        source="split",
        kind=kind,
        description=description,
        location=f"field={field}",
        observed=observed,
        expected=expected,
    )


def exact_findings(
    train: Sequence[Document], test: Sequence[Document], field: str
) -> list[Finding]:
    """One high finding per field when normalised text repeats across splits."""
    matched = exact_matches(train, test)
    if not matched:
        return []
    unique: set[str] = set()
    examples: list[Evidence] = []
    for document in sorted(test, key=lambda item: item.id):
        if document.id not in matched or document.digest in unique:
            continue
        unique.add(document.digest)
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                Evidence(
                    source="split",
                    kind="duplicate",
                    description=f"test document {document.id} repeats a train document",
                    observed=f"digest={document.digest}",
                )
            )
    summary = _summary(
        "overlap",
        field,
        f"{len(matched)} test documents duplicate train text exactly",
        f"matched={len(matched)} test_docs={len(test)} unique_digests={len(unique)}",
        "0 exact duplicates across splits",
    )
    return [
        Finding(
            severity="high",
            kind="text_exact_duplicate",
            column=field,
            detail=(
                f"{len(matched)} test documents repeat train documents exactly "
                f"(field {field})."
            ),
            evidence=(summary, *examples),
            stable_key=field,
            diagnostic=True,
        )
    ]


def near_findings(
    train: Sequence[Document],
    test: Sequence[Document],
    field: str,
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    num_hashes: int = DEFAULT_HASHES,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Finding]:
    """One medium finding per field when near-duplicates cross the split."""
    matched = near_matches(
        train,
        test,
        shingle_size=shingle_size,
        num_hashes=num_hashes,
        threshold=threshold,
    )
    if not matched:
        return []
    sketches = signatures(
        [*train, *test], shingle_size=shingle_size, num_hashes=num_hashes
    )
    train_by_id = {document.id: document for document in train}
    examples: list[Evidence] = []
    for document in sorted(test, key=lambda item: item.id):
        if document.id not in matched or len(examples) >= MAX_EXAMPLES:
            continue
        best_id, best = "", 0.0
        for train_id in sorted(train_by_id):
            score = jaccard(sketches[document.id], sketches[train_id])
            if score > best:
                best_id, best = train_id, score
        # A transitive cluster member can reach the cluster through another test
        # document and share no shingle with any train document.
        if not best_id:
            continue
        examples.append(
            Evidence(
                source="split",
                kind="duplicate",
                description=f"test document {document.id} is near a train document",
                observed=(
                    f"test_digest={document.digest} "
                    f"train_digest={train_by_id[best_id].digest} "
                    f"similarity={best:.2f}"
                ),
            )
        )
    summary = _summary(
        "overlap",
        field,
        f"{len(matched)} test documents are near-duplicates of train text",
        (
            f"matched={len(matched)} test_docs={len(test)} "
            f"shingles={shingle_size} hashes={num_hashes} threshold={threshold:.2f}"
        ),
        "0 near-duplicates across splits",
    )
    return [
        Finding(
            severity="medium",
            kind="text_near_duplicate",
            column=field,
            detail=(
                f"{len(matched)} test documents are near-duplicates of train "
                f"documents (field {field})."
            ),
            evidence=(summary, *examples),
            stable_key=field,
            diagnostic=True,
        )
    ]
