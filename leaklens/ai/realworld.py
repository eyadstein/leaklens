"""Single-pass evaluation of the frozen Phase 6 model on the real-world holdout.

This module is the only place that scores ``corpus_realworld``.  It is imported
by the benchmark and the regression tests, never by training.  The holdout is
applied exactly as frozen: the model family, calibration and threshold come from
the development corpus, and no holdout outcome is fed back into them.

Reported quantities
-------------------
* overall precision, recall, F1, FPR and accuracy with a confusion matrix;
* the same metrics by category, repository, family and provenance class;
* Brier score and expected calibration error;
* cluster-bootstrap confidence intervals that resample **repositories**, not
  rows, because rows inside a repository share an idiom;
* every mistake with its probability, active features, traceable code lines and
  the held-out label rationale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .corpus_realworld import (
    HOLDOUT_ID,
    RealWorldCase,
    corpus_stats,
    holdout_cases,
    manifest_digest,
    repositories,
)
from .features import MODEL_FEATURE_NAMES, model_features
from .learned import _FEATURE_EVIDENCE, MODEL_VERSION, _calibration, _metrics, get_model

# Frozen at the first Phase 6 commit, before any model change.
FROZEN_DIGEST = "7fde39ce94d34ae7f05243163e2761d5410f4b3a01f9770eabc2a24ea45d5742"

_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "specificity",
)


def _subset_metrics(
    cases: tuple[RealWorldCase, ...],
    labels: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    indexes: list[int],
) -> dict[str, float]:
    return _metrics(labels[indexes], predicted[indexes])


def _grouped(cases: tuple[RealWorldCase, ...], attribute: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        key = str(getattr(case, attribute))
        groups.setdefault(key, []).append(index)
    return dict(sorted(groups.items()))


def _cluster_bootstrap(
    cases: tuple[RealWorldCase, ...],
    labels: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals from resampling repositories with replacement."""

    by_repository = _grouped(cases, "repository")
    names = list(by_repository)
    if len(names) < 2 or iterations <= 0:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    for _ in range(iterations):
        chosen = rng.choice(len(names), size=len(names), replace=True)
        indexes: list[int] = []
        for position in chosen:
            indexes.extend(by_repository[names[int(position)]])
        metrics = _metrics(labels[indexes], predicted[indexes])
        for key in _METRIC_KEYS:
            samples[key].append(metrics[key])
    intervals: dict[str, tuple[float, float]] = {}
    for key, values in samples.items():
        array = np.asarray(values, dtype=float)
        intervals[key] = (
            round(float(np.percentile(array, 2.5)), 4),
            round(float(np.percentile(array, 97.5)), 4),
        )
    return intervals


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single proportion."""

    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z
        * float(np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)))
        / denominator
    )
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def _mistake(
    case: RealWorldCase,
    probability: float,
    model,
) -> dict[str, object]:
    contributions = model.top_contributions(case.code, limit=6)
    lines = case.code.splitlines()
    evidence = []
    seen_lines: set[int] = set()
    for name, _value, _importance in contributions:
        pattern = _FEATURE_EVIDENCE.get(name)
        if pattern is None:
            continue
        for index, line in enumerate(lines, start=1):
            if index in seen_lines or line.lstrip().startswith(("import ", "from ")):
                continue
            if pattern.search(line):
                seen_lines.add(index)
                evidence.append(
                    {"line": index, "code": line.strip()[:200], "feature": name}
                )
                break
        if len(evidence) >= 3:
            break
    return {
        "case": case.name,
        "repository": case.repository,
        "family": case.family,
        "category": case.category,
        "provenance": case.provenance,
        "probability": round(float(probability), 4),
        "rationale": case.label_rationale,
        "source_url": case.source_url,
        "signals": [
            {"feature": name, "value": round(value, 2), "importance": round(imp, 4)}
            for name, value, imp in contributions
        ],
        "evidence": evidence,
    }


_NOTEBOOK_ROOT = Path(__file__).resolve().parents[2] / "realworld"
_NOTEBOOK_LABELS = {
    "confirmed_leakage": True,
    "legitimate": False,
}


def _read_notebook_code(path: Path) -> str:
    """Return the concatenated code cells of a notebook without executing it."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            parts.append("".join(str(part) for part in source))
        elif isinstance(source, str):
            parts.append(source)
    return "\n".join(parts)


def public_notebook_cases() -> list[dict[str, object]]:
    """Load the hand-reviewed public notebooks from ``realworld/``.

    These are real, fetched notebooks, not reconstructions.  Only the
    ``confirmed_leakage`` and ``legitimate`` verdicts are scored; the remaining
    ``potential_leakage`` and ``uncertain`` verdicts stay excluded, exactly as
    in the deterministic scanner's evaluation.  The AI layer never trains on
    this directory.
    """

    labels_path = _NOTEBOOK_ROOT / "labels.json"
    manifest_path = _NOTEBOOK_ROOT / "manifest.jsonl"
    if not labels_path.is_file() or not manifest_path.is_file():
        return []

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            manifest[str(record["id"])] = record

    cases: list[dict[str, object]] = []
    for name in sorted(labels):
        verdict = labels[name].get("verdict")
        if verdict not in _NOTEBOOK_LABELS:
            continue
        record = manifest.get(name, {})
        path = _NOTEBOOK_ROOT / "notebooks" / f"{name}.ipynb"
        if not path.is_file():
            continue
        try:
            code = _read_notebook_code(path)
        except (OSError, ValueError, TypeError):
            continue
        if not code.strip():
            continue
        cases.append(
            {
                "name": name,
                "code": code,
                "leaky": _NOTEBOOK_LABELS[verdict],
                "verdict": verdict,
                "reason": str(labels[name].get("reason", "")),
                "repository": str(record.get("source_repo", "")),
                "path": str(record.get("source_path", "")),
                "source_url": str(record.get("source_url", "")),
                "sha256": str(record.get("sha256", "")),
            }
        )
    return cases


def evaluate_public_notebooks() -> dict[str, object]:
    """Score the frozen model on real public notebooks labelled by hand.

    This is deliberately a **smoke check**, not a holdout: it has eight scored
    units, the sample is not random, and the notebooks belong to the project's
    earlier deterministic-scanner corpus.  It is reported because it uses real
    fetched code instead of reconstructions, which makes it complementary to
    ``evaluate_realworld``.
    """

    cases = public_notebook_cases()
    if not cases:
        return {"available": False, "cases": 0}

    matrix = np.vstack([model_features(str(case["code"])) for case in cases])
    labels = np.asarray([int(bool(case["leaky"])) for case in cases], dtype=int)
    _evaluation, model = get_model(compare_candidates=False)
    probabilities = model.predict_matrix(matrix)
    predicted = (probabilities >= model.threshold).astype(int)
    metrics = _metrics(labels, predicted)

    scored = []
    for case, probability, guess in zip(cases, probabilities, predicted):
        scored.append(
            {
                "name": case["name"],
                "repository": case["repository"],
                "path": case["path"],
                "source_url": case["source_url"],
                "sha256": case["sha256"],
                "verdict": case["verdict"],
                "leaky": case["leaky"],
                "probability": round(float(probability), 4),
                "predicted": bool(guess),
                "correct": bool(guess) == case["leaky"],
                "reason": case["reason"],
            }
        )

    return {
        "available": True,
        "source": "realworld/ (fetched public notebooks, hand-reviewed)",
        "case_count": len(cases),
        "excluded_verdicts": ["potential_leakage", "uncertain"],
        "metrics": metrics,
        "calibration": _calibration(labels, probabilities),
        "accuracy_interval": _wilson(
            int(metrics["true_positives"] + metrics["true_negatives"]), len(cases)
        ),
        "cases": scored,
        "mistakes": [item for item in scored if not item["correct"]],
    }


def evaluate_realworld(*, bootstrap: int = 2000, seed: int = 0) -> dict[str, object]:
    """Score the frozen model on the frozen holdout once."""

    cases = holdout_cases()
    matrix = np.vstack([model_features(case.code) for case in cases])
    labels = np.asarray([int(case.leaky) for case in cases], dtype=int)
    _evaluation, model = get_model(compare_candidates=False)
    probabilities = model.predict_matrix(matrix)
    predicted = (probabilities >= model.threshold).astype(int)
    metrics = _metrics(labels, predicted)
    confusion = {
        "true_positives": int(metrics["true_positives"]),
        "false_positives": int(metrics["false_positives"]),
        "true_negatives": int(metrics["true_negatives"]),
        "false_negatives": int(metrics["false_negatives"]),
    }

    per_category = {
        key: _subset_metrics(cases, labels, predicted, probabilities, indexes)
        for key, indexes in _grouped(cases, "category").items()
    }
    per_repository = {
        key: _subset_metrics(cases, labels, predicted, probabilities, indexes)
        for key, indexes in _grouped(cases, "repository").items()
    }
    per_family = {
        key: _subset_metrics(cases, labels, predicted, probabilities, indexes)
        for key, indexes in _grouped(cases, "family").items()
    }
    per_provenance = {
        key: _subset_metrics(cases, labels, predicted, probabilities, indexes)
        for key, indexes in _grouped(cases, "provenance").items()
    }

    false_positives = []
    false_negatives = []
    for case, probability, guess in zip(cases, probabilities, predicted):
        if bool(guess) == case.leaky:
            continue
        item = _mistake(case, float(probability), model)
        if case.leaky:
            false_negatives.append(item)
        else:
            false_positives.append(item)

    return {
        "holdout_id": HOLDOUT_ID,
        "digest": manifest_digest(),
        "digest_matches_freeze": manifest_digest() == FROZEN_DIGEST,
        "cases": len(cases),
        "leaky": int(labels.sum()),
        "legitimate": int(len(labels) - labels.sum()),
        "repositories": len(repositories()),
        "model": {
            "name": model.card.name,
            "version": MODEL_VERSION,
            "kind": model.card.kind,
            "features": len(MODEL_FEATURE_NAMES),
            "threshold": model.threshold,
        },
        "metrics": metrics,
        "confusion": confusion,
        "accuracy_interval": _wilson(
            confusion["true_positives"] + confusion["true_negatives"], len(cases)
        ),
        "precision_interval": _wilson(
            confusion["true_positives"],
            confusion["true_positives"] + confusion["false_positives"],
        ),
        "recall_interval": _wilson(
            confusion["true_positives"],
            confusion["true_positives"] + confusion["false_negatives"],
        ),
        "false_positive_rate_interval": _wilson(
            confusion["false_positives"],
            confusion["false_positives"] + confusion["true_negatives"],
        ),
        "bootstrap_intervals": _cluster_bootstrap(
            cases,
            labels,
            predicted,
            probabilities,
            iterations=bootstrap,
            seed=seed,
        ),
        "calibration": _calibration(labels, probabilities),
        "per_category": per_category,
        "per_repository": per_repository,
        "per_family": per_family,
        "per_provenance": per_provenance,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "corpus": corpus_stats(),
    }


__all__ = [
    "FROZEN_DIGEST",
    "evaluate_public_notebooks",
    "evaluate_realworld",
    "public_notebook_cases",
]
