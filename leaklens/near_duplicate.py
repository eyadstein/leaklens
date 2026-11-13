"""Shared near-duplicate primitive: bucket -> threshold -> stable cluster key (ADR-022).

Domain modules supply integer signatures (a bottom-k sketch for text, a hash for
images) and may supply their own similarity function. This module owns LSH
banding, the threshold check, union-find and the deterministic cluster key. It
does no I/O, loads no model and makes no embeddings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_ROWS = 4
DEFAULT_THRESHOLD = 0.8
KEY_LENGTH = 16


def band_key(band: Sequence[int]) -> str:
    """Stable, process-independent key for one LSH band."""
    payload = ",".join(str(int(value)) for value in band).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _check_geometry(rows: int, length: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if length % rows:
        raise ValueError("signature length must be a multiple of rows")


def bucket(
    signatures: Mapping[str, Sequence[int]], *, rows: int = DEFAULT_ROWS
) -> dict[str, tuple[str, ...]]:
    """Group ids that share at least one band. Keys and members are sorted."""
    groups: dict[str, list[str]] = {}
    for name in sorted(signatures):
        signature = signatures[name]
        _check_geometry(rows, len(signature))
        for start in range(0, len(signature), rows):
            groups.setdefault(band_key(signature[start : start + rows]), []).append(
                name
            )
    return {key: tuple(sorted(names)) for key, names in sorted(groups.items())}


def candidate_pairs(
    buckets: Mapping[str, Sequence[str]], *, max_bucket: int | None = None
) -> list[tuple[str, str]]:
    """Sorted unique id pairs that share a band; never a self-pair.

    ``max_bucket`` skips boilerplate-like buckets whose size makes every pair
    both expensive and uninformative.
    """
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        ordered = sorted(set(members))
        if max_bucket is not None and len(ordered) > max_bucket:
            continue
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pairs.add((left, right))
    return sorted(pairs)


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """Set-overlap estimate between two sketches in ``[0, 1]``."""
    if not left or not right:
        return 0.0
    return len(set(left) & set(right)) / max(len(left), len(right))


def cluster_key(members: Sequence[str]) -> str:
    """Stable cluster identity derived only from its sorted members."""
    payload = "\n".join(sorted(members)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True)
class Cluster:
    key: str
    members: tuple[str, ...]


def cluster(
    signatures: Mapping[str, Sequence[int]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    rows: int = DEFAULT_ROWS,
    similarity: Callable[[Sequence[int], Sequence[int]], float] = jaccard,
    max_bucket: int | None = None,
) -> list[Cluster]:
    """Union candidate pairs whose similarity reaches ``threshold``.

    Clusters are returned in stable key order and never contain a singleton.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    pairs = candidate_pairs(bucket(signatures, rows=rows), max_bucket=max_bucket)

    parent = {name: name for name in signatures}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for left, right in pairs:
        if similarity(signatures[left], signatures[right]) < threshold:
            continue
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    groups: dict[str, list[str]] = {}
    for name in sorted(signatures):
        groups.setdefault(find(name), []).append(name)
    found = [
        Cluster(cluster_key(members), tuple(members))
        for members in groups.values()
        if len(members) > 1
    ]
    return sorted(found, key=lambda item: item.key)
