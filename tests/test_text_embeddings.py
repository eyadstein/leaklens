"""Optional embedding similarity from user-supplied vectors (M6, non-reproducible)."""

from __future__ import annotations

import numpy as np
import pytest

from leaklens.analysis.text.embeddings import (
    embedding_findings,
    load_vectors,
)
from leaklens.analysis.text.normalize import documents_from_texts
from leaklens.leakage import score

BASE = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
)
OTHER = "one two three four five six seven eight nine ten eleven twelve thirteen"


def _docs(texts, split):
    return documents_from_texts(texts, split, min_chars=10)[0]


def test_load_vectors_reads_numeric_npy_files(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    vectors = load_vectors(path)
    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32


def test_object_arrays_are_refused_without_unpickling(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, np.array([{"x": 1}], dtype=object))
    with pytest.raises(ValueError, match="numeric"):
        load_vectors(path)


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([[np.nan, 0.0]], dtype=np.float32),
        np.zeros((1, 4097), dtype=np.float32),
    ],
)
def test_bad_vector_shapes_are_refused(tmp_path, array):
    path = tmp_path / "vectors.npy"
    np.save(path, array)
    with pytest.raises(ValueError):
        load_vectors(path)


def test_embedding_matches_need_an_explicit_model_and_aligned_rows():
    train = _docs([BASE], "train")
    test = _docs([BASE], "test")
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="model"):
        embedding_findings(train, test, vectors, vectors, model="")
    with pytest.raises(ValueError, match="rows"):
        embedding_findings(
            train, test, vectors, np.zeros((2, 2), np.float32), model="m"
        )
    with pytest.raises(ValueError, match="dimension"):
        embedding_findings(
            train, test, vectors, np.zeros((1, 3), np.float32), model="m"
        )


def test_similar_vectors_are_reported_without_model_execution():
    train = _docs([BASE], "train")
    test = _docs([BASE, OTHER], "test")
    train_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    test_vectors = np.array([[1.0, 0.01], [0.0, 1.0]], dtype=np.float32)
    findings = embedding_findings(
        train, test, train_vectors, test_vectors, model="all-MiniLM-L6-v2"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == "text_embedding_similarity"
    assert finding.severity == "medium"
    assert finding.diagnostic is True
    assert "matched=1" in finding.evidence[0].observed
    assert "all-MiniLM-L6-v2" in finding.detail
    assert score(findings) == 100


def test_zero_vectors_do_not_crash_or_match():
    train = _docs([BASE], "train")
    test = _docs([BASE], "test")
    zeros = np.zeros((1, 2), dtype=np.float32)
    assert embedding_findings(train, test, zeros, zeros, model="m") == []
