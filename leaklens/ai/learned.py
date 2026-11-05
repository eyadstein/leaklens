"""Optional CPU-only learned detector for leakage-like code patterns.

The model is a small logistic regression trained from the grouped synthetic
corpus in :mod:`leaklens.ai.corpus`.  It is intentionally modest:

* no GPU, no network, no downloaded weights, no pickle loading;
* grouped cross-validation so a template family never spans folds;
* honest external transfer metrics on the existing scanner benchmark;
* every prediction is reported as an uncalibrated hypothesis, never a finding.

The deterministic scanner remains authoritative.  This detector only adds a
separate, clearly marked signal when a caller opts in.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from leaklens.leakage import Finding

from .corpus import CorpusCase, corpus_cases, corpus_stats
from .features import MODEL_FEATURE_NAMES, model_features
from .schemas import EvidenceSpan, Hypothesis, ModelCard
from .semantics import CodeSource

_RANDOM_STATE = 0
# Fixed after a one-time grouped sweep so the shipped detector behaves the same
# on every supported scikit-learn version. The benchmark reports out-of-fold
# metrics at this fixed threshold instead of re-selecting it at import time.
_DEFAULT_THRESHOLD = 0.65

MODEL_LIMITATIONS = (
    "Trained on a small synthetic corpus of Python patterns, not production code.",
    "Learns surface syntax and English identifiers; other languages and naming styles may fail.",
    "The probability is uncalibrated and should not be read as a false-positive guarantee.",
    "External transfer is measured on the existing scanner benchmark, not claimed as an improvement.",
    "Applies to Python code only; dataset metadata is handled by deterministic semantic rules.",
)


@dataclass(frozen=True, slots=True)
class LearnedModel:
    model: GradientBoostingClassifier
    card: ModelCard
    feature_importances: tuple[float, ...]
    threshold: float

    def predict_proba(self, source: str) -> float:
        vector = model_features(source).reshape(1, -1)
        return float(self.model.predict_proba(vector)[0][1])

    def top_contributions(
        self, source: str, limit: int = 6
    ) -> tuple[tuple[str, float, float], ...]:
        vector = model_features(source)
        order = np.argsort(self.feature_importances)[::-1]
        result = []
        for index in order:
            if vector[int(index)] == 0.0:
                continue
            result.append(
                (
                    MODEL_FEATURE_NAMES[int(index)],
                    float(vector[int(index)]),
                    float(self.feature_importances[int(index)]),
                )
            )
            if len(result) >= limit:
                break
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    metrics: dict[str, float]
    threshold: float
    folds: int
    examples: int
    groups: int
    external: dict[str, float]


def _metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    true_positive = int(np.sum((labels == 1) & (predicted == 1)))
    false_positive = int(np.sum((labels == 0) & (predicted == 1)))
    true_negative = int(np.sum((labels == 0) & (predicted == 0)))
    false_negative = int(np.sum((labels == 1) & (predicted == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative
        else 0.0
    )
    accuracy = (true_positive + true_negative) / len(labels) if len(labels) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "true_negatives": float(true_negative),
        "false_negatives": float(false_negative),
    }


def _new_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=120,
        max_depth=1,
        learning_rate=0.08,
        random_state=_RANDOM_STATE,
    )


def _grouped_folds(groups: np.ndarray, labels: np.ndarray, n_splits: int = 5):
    unique_groups = len(set(groups.tolist()))
    splits = min(n_splits, unique_groups)
    if splits < 2:
        raise RuntimeError("the learning corpus needs at least two groups")
    try:
        splitter = StratifiedGroupKFold(
            n_splits=splits, shuffle=True, random_state=_RANDOM_STATE
        )
        return list(splitter.split(np.zeros(len(labels)), labels, groups)), splits
    except (TypeError, ValueError):  # pragma: no cover - older sklearn fallback
        splitter = GroupKFold(n_splits=splits)
        return list(splitter.split(np.zeros(len(labels)), labels, groups)), splits


def _external_cases() -> list[tuple[str, str, bool]]:
    """Existing scanner benchmarks, used only for post-training transfer tests."""

    from leaklens.bench import CASES
    from leaklens.bench_advanced import ADVANCED_CASES
    from leaklens.bench_cases import EXTRA_CASES

    cases: list[tuple[str, str, bool]] = list(CASES)
    cases.extend((case.name, case.code, case.leaky) for case in EXTRA_CASES)
    cases.extend((case.name, case.code, case.leaky) for case in ADVANCED_CASES)
    return cases


def _evaluate(
    cases: Sequence[CorpusCase],
    *,
    include_external: bool = True,
) -> tuple[ModelEvaluation, LearnedModel]:
    matrix = np.vstack([model_features(case.code) for case in cases])
    labels = np.asarray([int(case.leaky) for case in cases], dtype=int)
    groups = np.asarray([case.group for case in cases], dtype=object)

    folds, splits = _grouped_folds(groups, labels)
    out_of_fold = np.zeros(len(labels), dtype=float)
    covered = np.zeros(len(labels), dtype=bool)
    for train_index, test_index in folds:
        fold_model = _new_model()
        fold_model.fit(matrix[train_index], labels[train_index])
        probabilities = fold_model.predict_proba(matrix[test_index])[:, 1]
        out_of_fold[test_index] = probabilities
        covered[test_index] = True
    if not covered.all():  # pragma: no cover - defensive
        raise RuntimeError("grouped cross-validation left examples uncovered")

    threshold = _DEFAULT_THRESHOLD
    metrics = _metrics(labels, (out_of_fold >= threshold).astype(int))
    final_model = _new_model()
    final_model.fit(matrix, labels)
    importances = tuple(float(value) for value in final_model.feature_importances_)

    external: dict[str, float] = {}
    if include_external:
        external_cases = _external_cases()
        if external_cases:
            external_matrix = np.vstack(
                [model_features(code) for _name, code, _leaky in external_cases]
            )
            external_labels = np.asarray(
                [int(leaky) for _name, _code, leaky in external_cases], dtype=int
            )
            probabilities = final_model.predict_proba(external_matrix)[:, 1]
            external = _metrics(
                external_labels, (probabilities >= threshold).astype(int)
            )

    stats = corpus_stats()
    external_rates = {
        key: value
        for key, value in external.items()
        if key in {"precision", "recall", "f1", "false_positive_rate", "accuracy"}
    }
    card = ModelCard(
        name="leaklens-ai-gb",
        kind="gradient_boosting",
        features=len(MODEL_FEATURE_NAMES),
        training_examples=len(cases),
        groups=int(stats["groups"]),
        threshold=threshold,
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        false_positive_rate=metrics["false_positive_rate"],
        evaluation=(
            f"Grouped {splits}-fold cross-validation over synthetic template "
            "families; no family spans a fold."
        ),
        limitations=MODEL_LIMITATIONS,
        external=tuple(
            sorted((f"external_{key}", value) for key, value in external_rates.items())
        ),
    )
    evaluation = ModelEvaluation(
        metrics=metrics,
        threshold=threshold,
        folds=splits,
        examples=len(cases),
        groups=int(stats["groups"]),
        external=external,
    )
    return evaluation, LearnedModel(
        model=final_model,
        card=card,
        feature_importances=importances,
        threshold=threshold,
    )


_lock = threading.Lock()
_cached: tuple[ModelEvaluation, LearnedModel] | None = None


def evaluate_model(*, include_external: bool = True) -> ModelEvaluation:
    return get_model(include_external=include_external)[0]


def get_model(*, include_external: bool = True) -> tuple[ModelEvaluation, LearnedModel]:
    """Train once, cache in memory, and return the evaluation plus the model.

    Nothing is written to disk and no serialized model is loaded, so a tampered
    file cannot execute code.  The cache is guarded because the public API may
    serve concurrent requests.
    """

    global _cached
    with _lock:
        if _cached is None or (include_external and not _cached[0].external):
            _cached = _evaluate(corpus_cases(), include_external=include_external)
        return _cached


def reset_model_cache() -> None:
    """Testing helper.  Never needed at runtime."""

    global _cached
    with _lock:
        _cached = None


def _evidence(source: str, kind: str, location: str, excerpt: str) -> EvidenceSpan:
    clean = re.sub(r"\s+", " ", excerpt).strip()[:320]
    digest = hashlib.sha1(
        f"{source}|{kind}|{location}|{clean}".encode("utf-8", "replace"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return EvidenceSpan(
        evidence_id=f"ev-{digest}",
        source=source,
        kind=kind,
        location=location[:280],
        excerpt=clean,
    )


def _evidence_line(lines: Sequence[str]) -> int:
    keywords = (
        "fit",
        "split",
        "target",
        "label",
        "test",
        "holdout",
        "time",
        "date",
        "group",
        "customer",
        "shift",
        "rolling",
        "fillna",
        "concat",
        "resample",
        "scale",
        "encode",
    )
    for index, line in enumerate(lines, start=1):
        lowered = line.lower()
        if any(
            word in lowered for word in keywords
        ) and not lowered.lstrip().startswith(("import ", "from ")):
            return index
    return 1


def learned_hypotheses(
    code: Sequence[CodeSource],
    findings: Sequence[Finding] = (),
    *,
    threshold: float | None = None,
) -> tuple[Hypothesis, ...]:
    """Return at most one learned hypothesis per source unit above threshold."""

    _evaluation, model = get_model()
    cutoff = model.threshold if threshold is None else float(threshold)
    known_kinds = {finding.kind for finding in findings}
    corroborating = next(
        (
            kind
            for kind in (
                "preprocess_before_split",
                "resample_before_split",
                "feature_selection_before_split",
                "target_encode_before_split",
                "split_overlap",
                "feature_overlap",
                "near_duplicate_overlap",
                "temporal_leakage",
                "target_leakage",
            )
            if kind in known_kinds
        ),
        None,
    )

    hypotheses: list[Hypothesis] = []
    for source in code:
        for label, text in source.chunks():
            probability = model.predict_proba(text)
            if probability < cutoff:
                continue
            contributions = model.top_contributions(text, limit=5)
            line_number = _evidence_line(text.splitlines())
            first_line = text.splitlines()[line_number - 1] if text.splitlines() else ""
            evidence = [
                _evidence(
                    "model",
                    "probability",
                    f"{label}:{line_number}",
                    f"p(leaky pattern)={probability:.3f}; threshold={cutoff:.2f}",
                ),
                _evidence(
                    "code",
                    "pattern",
                    f"{label}:{line_number}",
                    first_line or f"code unit {label}",
                ),
            ]
            for name, value, importance in contributions:
                evidence.append(
                    _evidence(
                        "model",
                        "feature",
                        f"{label}:{line_number}",
                        f"{name}={value:.2f} (global importance {importance:.3f})",
                    )
                )
            confidence = min(0.9, 0.45 + 0.40 * probability)
            signal_text = ", ".join(name for name, _value, _c in contributions[:3])
            digest = hashlib.sha1(
                f"{label}|{line_number}|{probability:.6f}".encode("utf-8", "replace"),
                usedforsecurity=False,
            ).hexdigest()[:12]
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp-learned-{digest}",
                    concept="learned_leakage_pattern",
                    title="Learned model sees a leakage-like code pattern",
                    explanation=(
                        f"A local gradient-boosted model trained on labeled synthetic "
                        f"pipelines gives this unit p={probability:.3f} for a leakage-like "
                        f"pattern (threshold {cutoff:.2f}). Active signals: "
                        f"{signal_text or 'none'}. Feature importances are global to the "
                        "model, not a causal explanation. This is a statistical hint, not "
                        "a verified finding."
                    ),
                    confidence=round(confidence, 4),
                    evidence=tuple(evidence),
                    location=f"{label}:{line_number}",
                    detector="learned",
                    severity_hint="medium" if probability >= 0.85 else "low",
                    corroborates=corroborating,
                    limitations=MODEL_LIMITATIONS[:3],
                )
            )
    return tuple(hypotheses)


__all__ = [
    "MODEL_LIMITATIONS",
    "LearnedModel",
    "ModelEvaluation",
    "evaluate_model",
    "get_model",
    "learned_hypotheses",
    "reset_model_cache",
]
