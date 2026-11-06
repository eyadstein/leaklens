"""CPU and memory measurements for the optional learned detector.

Everything here is measured on the machine running the benchmark.  Numbers are
reported as observed, with the Python and scikit-learn versions, because latency
and memory depend heavily on both.  The goal is to show that the detector stays
usable on a laptop, not to claim a production SLA.
"""

from __future__ import annotations

import platform
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from .features import model_features
from .learned import get_model, reset_model_cache


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def _model_footprint(model: Any) -> dict[str, int]:
    estimator = model.model.estimator
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    result: dict[str, int] = {}
    if hasattr(final, "coef_"):
        result["coefficients"] = int(np.asarray(final.coef_).size)
    if hasattr(final, "n_estimators"):
        result["trees"] = int(final.n_estimators)
        nodes = 0
        for tree in getattr(final, "estimators_", ()):
            nodes += int(getattr(tree, "tree_", tree).node_count)
        result["tree_nodes"] = nodes
    return result


def profile_holdout(bootstrap: int = 5) -> dict[str, object]:
    """Time cold training, feature extraction and inference on the holdout."""

    from .corpus_realworld import holdout_cases

    cases = holdout_cases()
    reset_model_cache()

    tracemalloc.start()
    started = time.perf_counter()
    evaluation, model = get_model()
    training_seconds = time.perf_counter() - started
    training_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    extraction: list[float] = []
    matrices = []
    tracemalloc.start()
    started = time.perf_counter()
    for case in cases:
        begin = time.perf_counter()
        matrices.append(model_features(case.code))
        extraction.append(time.perf_counter() - begin)
    extraction_seconds = time.perf_counter() - started
    extraction_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    matrix = np.vstack(matrices)
    inference: list[float] = []
    tracemalloc.start()
    for _ in range(bootstrap):
        for row in matrix:
            begin = time.perf_counter()
            model.predict_matrix(row.reshape(1, -1))
            inference.append(time.perf_counter() - begin)
    inference_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    total_bytes = sum(row.nbytes for row in matrices)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": len(cases),
        "features": len(matrix[0]),
        "cold_training_seconds": round(training_seconds, 4),
        "cold_training_peak_bytes": int(training_peak),
        "extraction_total_seconds": round(extraction_seconds, 6),
        "extraction_median_ms": round(statistics.median(extraction) * 1000, 4),
        "extraction_p95_ms": round(_percentile(extraction, 0.95) * 1000, 4),
        "extraction_max_ms": round(max(extraction) * 1000, 4),
        "extraction_peak_bytes": int(extraction_peak),
        "feature_matrix_bytes": int(total_bytes),
        "inference_median_ms": round(statistics.median(inference) * 1000, 6),
        "inference_p95_ms": round(_percentile(inference, 0.95) * 1000, 6),
        "inference_peak_bytes": int(inference_peak),
        "model_candidates": len(evaluation.candidates),
        "model_threshold": evaluation.threshold,
        "model_name": evaluation.model_name,
        "footprint": _model_footprint(model),
    }


def profile_project(root: Path | None = None, *, limit: int = 400) -> dict[str, object]:
    """Measure scanning a real project tree (default: this package)."""

    base = root or Path(__file__).resolve().parents[1]
    files = sorted(base.rglob("*.py"))[:limit]
    sources: list[str] = []
    for path in files:
        try:
            sources.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:  # pragma: no cover - unreadable file
            continue
    if not sources:
        return {"files": 0}

    _evaluation, model = get_model()
    per_file: list[float] = []
    tracemalloc.start()
    started = time.perf_counter()
    rows = []
    for source in sources:
        begin = time.perf_counter()
        rows.append(model_features(source))
        per_file.append(time.perf_counter() - begin)
    matrix_seconds = time.perf_counter() - started
    extraction_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    matrix = np.vstack(rows)
    tracemalloc.start()
    started = time.perf_counter()
    model.predict_matrix(matrix)
    inference_seconds = time.perf_counter() - started
    inference_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    return {
        "root": str(base),
        "files": len(sources),
        "total_python_bytes": sum(
            len(source.encode("utf-8", "replace")) for source in sources
        ),
        "extraction_total_seconds": round(matrix_seconds, 4),
        "extraction_median_ms": round(statistics.median(per_file) * 1000, 4),
        "extraction_p95_ms": round(_percentile(per_file, 0.95) * 1000, 4),
        "extraction_peak_bytes": int(extraction_peak),
        "inference_total_seconds": round(inference_seconds, 6),
        "inference_peak_bytes": int(inference_peak),
        "matrix_bytes": int(matrix.nbytes),
    }


__all__ = ["profile_holdout", "profile_project"]
