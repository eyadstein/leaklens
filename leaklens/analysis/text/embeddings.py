"""Optional embedding similarity from user-supplied vectors (Phase 15 M6).

LeakLens never loads a model and never computes an embedding. The operator
supplies two ``.npy`` matrices row-aligned with the kept documents (in file
order) and the model id that produced them. Findings state the model identity
and that results are not reproducible across model versions; this path is never
enabled by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...evidence import Evidence
from ...leakage import Finding
from .normalize import Document

MAX_ROWS = 20_000
MAX_DIMS = 4096
DEFAULT_THRESHOLD = 0.95
MAX_EXAMPLES = 8
CHUNK = 256


def load_vectors(path: str | Path) -> Any:
    """Read a numeric 2-D ``.npy`` matrix; pickle is never allowed."""
    source = Path(path)
    try:
        data = np.load(source, allow_pickle=False)
    except ValueError:
        raise ValueError(f"{source.name} must be a numeric .npy array") from None
    except OSError:
        raise ValueError(f"cannot read vectors from {source.name}") from None
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("vectors must be a 2-D array")
    if data.shape[0] > MAX_ROWS or data.shape[1] > MAX_DIMS:
        raise ValueError("vector matrix exceeds the 20000 x 4096 limit")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        raise ValueError("vectors must be real numbers")
    if not np.isfinite(data).all():
        raise ValueError("vectors must be finite")
    return data.astype(np.float32, copy=False)


def _unit(vectors: Any) -> Any:
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / np.clip(norms, 1e-12, None)[:, None]


def embedding_findings(
    train: list[Document],
    test: list[Document],
    train_vectors: Any,
    test_vectors: Any,
    *,
    model: str,
    threshold: float = DEFAULT_THRESHOLD,
    field: str = "text",
) -> list[Finding]:
    """Report test documents whose cosine similarity to a train document passes."""
    if not model.strip():
        raise ValueError("an embedding model id is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if train_vectors.shape[0] != len(train):
        raise ValueError("train vector rows must match the kept train documents")
    if test_vectors.shape[0] != len(test):
        raise ValueError("test vector rows must match the kept test documents")
    if train_vectors.shape[1] != test_vectors.shape[1]:
        raise ValueError("train and test vectors must share one dimension")
    if not train or not test:
        return []

    unit_train = _unit(train_vectors)
    unit_test = _unit(test_vectors)
    matches: list[tuple[int, int, float]] = []
    for start in range(0, len(test), CHUNK):
        scores = unit_test[start : start + CHUNK] @ unit_train.T
        for offset, row in enumerate(scores):
            best = int(row.argmax())
            value = float(row[best])
            if value >= threshold:
                matches.append((start + offset, best, value))
    if not matches:
        return []

    label = model.strip()[:80]
    summary = Evidence(
        source="split",
        kind="statistic",
        description="test documents are embedding-similar to train documents",
        location=f"field={field}",
        observed=(
            f"model={label} threshold={threshold:.2f} matched={len(matches)} "
            f"test_docs={len(test)} not reproducible across model versions"
        ),
        expected="0 cross-split embedding matches",
    )
    examples = [
        Evidence(
            source="split",
            kind="duplicate",
            description=f"test document {test[index].id} is embedding-similar to train",
            observed=(
                f"test_digest={test[index].digest} "
                f"train_digest={train[train_index].digest} cosine={value:.3f} "
                f"model={label}"
            ),
        )
        for index, train_index, value in matches[:MAX_EXAMPLES]
    ]
    return [
        Finding(
            severity="medium",
            kind="text_embedding_similarity",
            column=field,
            detail=(
                f"{len(matches)} test documents are embedding-similar to train "
                f"documents (model {label}); results are not reproducible across "
                "model versions."
            ),
            evidence=(summary, *examples),
            stable_key=field,
            diagnostic=True,
        )
    ]
