"""Optional CPU-only learned detector for leakage-like code patterns.

The detector is a small, calibrated lightweight model (tree ensemble or
logistic regression, chosen on validation) trained from the authored
development corpus in :mod:`leaklens.ai.corpus`:

* no GPU, no network, no downloaded weights, no pickle loading;
* scenario sources never span train, validation and the locked test set;
* grouped cross-validation so a template family never spans folds;
* architecture selection on grouped CV + validation, calibration from grouped
  out-of-fold probabilities, and threshold selection on validation only;
* the locked test set is imported by the benchmark, never by training;
* every prediction is reported as a calibrated hypothesis, never a finding.

The deterministic scanner remains authoritative.  This detector only adds a
separate, clearly marked signal when a caller opts in.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from leaklens.leakage import Finding

from .corpus import (
    CorpusCase,
    corpus_stats,
    development_cases,
    training_cases,
    validation_cases,
)
from .features import FEATURE_NAMES, MODEL_FEATURE_NAMES, model_features
from .schemas import EvidenceSpan, Hypothesis, ModelCard
from .semantics import CodeSource

_RANDOM_STATE = 0
MODEL_VERSION = "leaklens-ai-p7-2026.11"
_MAX_FALSE_POSITIVE_RATE = 0.05

# Selection frozen after Phase 6 validation.  The runtime path fits only this
# candidate so a cold API request never trains all eleven; the benchmark still
# compares every candidate and reports when a scikit-learn release ranks a tied
# variant differently.
_FROZEN_CANDIDATE = "logistic_regression_balanced"

MODEL_LIMITATIONS = (
    "Trained on an authored synthetic corpus: Phase 5/6 families plus 40 Phase 7 "
    "hard-positive/hard-negative cases and 34 Phase 6 idiom cases.",
    "The Phase 5 locked test is a small authored corpus of 47 cases, not real repositories.",
    "The Phase 7 holdout is 12 authored synthetic units frozen before the final "
    "fit; it is a regression check, not production evidence.",
    "The Phase 6 real-world holdout is 48 reviewed units from 12 public repositories; "
    "44 of them are reconstructions, not verbatim upstream files.",
    "Probabilities are calibrated on 40 validation cases; treat them as approximate.",
    "English identifiers and Python control-flow cues may not transfer to other styles.",
    "External scanner transfer and the real-world holdout are checks, not improvement claims.",
)


@dataclass(frozen=True, slots=True)
class _PlattCalibrator:
    """One-dimensional logistic calibration fitted on grouped OOF scores."""

    coefficient: float
    intercept: float

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> _PlattCalibrator:
        logits = _logit(scores).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits, labels)
        return cls(
            coefficient=float(model.coef_[0][0]),
            intercept=float(model.intercept_[0]),
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        logits = _logit(scores)
        values = 1.0 / (1.0 + np.exp(-(self.coefficient * logits + self.intercept)))
        return np.clip(values, 0.001, 0.999)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


@dataclass(frozen=True, slots=True)
class _CalibratedModel:
    estimator: Any
    calibrator: _PlattCalibrator

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        raw = self.estimator.predict_proba(matrix)[:, 1]
        return self.calibrator.predict(raw)

    def raw_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(matrix)[:, 1]


@dataclass(frozen=True, slots=True)
class LearnedModel:
    model: _CalibratedModel
    card: ModelCard
    feature_importances: tuple[float, ...]
    threshold: float

    def predict_proba(self, source: str) -> float:
        vector = model_features(source).reshape(1, -1)
        return float(self.model.predict_proba(vector)[0])

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(matrix)

    def local_contributions(
        self, source: str, limit: int = 6
    ) -> tuple[tuple[str, float, float], ...]:
        """Return the per-instance evidence behind this prediction.

        Each item is ``(feature, value, contribution)`` where ``contribution``
        is signed: positive values push the unit toward "leaky", negative values
        push it toward "legitimate".  For the frozen linear model this is the
        exact log-odds contribution (standardised value x coefficient).  For a
        tree fallback it is an occlusion delta in calibrated probability.

        This is deliberately *local*: a feature that is globally important but
        absent from this unit is never cited.
        """

        vector = model_features(source)
        deltas = self._local_deltas(vector)
        order = np.argsort(-np.abs(deltas))
        result: list[tuple[str, float, float]] = []
        for index in order:
            if vector[int(index)] == 0.0:
                continue
            result.append(
                (
                    MODEL_FEATURE_NAMES[int(index)],
                    float(vector[int(index)]),
                    float(deltas[int(index)]),
                )
            )
            if len(result) >= limit:
                break
        return tuple(result)

    def top_contributions(
        self, source: str, limit: int = 6
    ) -> tuple[tuple[str, float, float], ...]:
        """Backwards-compatible alias for :meth:`local_contributions`."""

        return self.local_contributions(source, limit=limit)

    def _local_deltas(self, vector: np.ndarray) -> np.ndarray:
        estimator = self.model.estimator
        linear = None
        scaled_vector = vector
        if isinstance(estimator, Pipeline):
            first = estimator.steps[0][1]
            final = estimator.steps[-1][1]
            if hasattr(first, "mean_") and hasattr(first, "scale_"):
                scaled_vector = np.asarray(
                    first.transform(vector.reshape(1, -1)), dtype=float
                ).ravel()
            if hasattr(final, "coef_"):
                linear = np.asarray(final.coef_, dtype=float).ravel()
        elif hasattr(estimator, "coef_"):
            linear = np.asarray(estimator.coef_, dtype=float).ravel()
        if linear is not None and linear.shape[0] == vector.shape[0]:
            return linear * scaled_vector

        # Generic fallback: zero each active feature and measure the change in
        # the calibrated probability.  Slower, but it works for tree ensembles
        # and keeps the explanation local rather than global.
        base = float(self.model.predict_proba(vector.reshape(1, -1))[0])
        deltas = np.zeros(vector.shape[0], dtype=float)
        for index, value in enumerate(vector):
            if value == 0.0:
                continue
            perturbed = vector.copy()
            perturbed[index] = 0.0
            deltas[index] = base - float(
                self.model.predict_proba(perturbed.reshape(1, -1))[0]
            )
        return deltas


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    metrics: dict[str, float]
    threshold: float
    folds: int
    examples: int
    groups: int
    external: dict[str, float]
    train_examples: int = 0
    validation_examples: int = 0
    train_metrics: dict[str, float] | None = None
    calibration: dict[str, float] | None = None
    candidates: tuple[dict[str, float], ...] = ()
    model_name: str = ""
    protocol: str = ""


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
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    accuracy = (true_positive + true_negative) / len(labels) if len(labels) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "true_negatives": float(true_negative),
        "false_negatives": float(false_negative),
    }


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    brier = float(np.mean((probabilities - labels) ** 2)) if len(labels) else 0.0
    bins = np.linspace(0.0, 1.0, 11)
    expected = 0.0
    for lower, upper in itertools.pairwise(bins):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        weight = float(np.count_nonzero(mask)) / len(labels)
        gap = abs(float(np.mean(probabilities[mask])) - float(np.mean(labels[mask])))
        expected += weight * gap
    return {
        "brier": round(brier, 4),
        "expected_calibration_error": round(expected, 4),
        "mean_probability": round(float(np.mean(probabilities)), 4)
        if len(probabilities)
        else 0.0,
        "observed_rate": round(float(np.mean(labels)), 4) if len(labels) else 0.0,
    }


def _candidate_specs() -> tuple[tuple[str, Any], ...]:
    return (
        (
            "gradient_boosting_depth2",
            GradientBoostingClassifier(
                n_estimators=250,
                max_depth=2,
                learning_rate=0.05,
                random_state=_RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting_depth1",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=1,
                learning_rate=0.06,
                random_state=_RANDOM_STATE,
            ),
        ),
        (
            "random_forest_depth6",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=6,
                random_state=_RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "random_forest_unbounded",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=_RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "extra_trees_depth8",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=8,
                random_state=_RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "extra_trees_unbounded",
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                random_state=_RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=250,
                max_depth=4,
                learning_rate=0.06,
                early_stopping=False,
                random_state=_RANDOM_STATE,
            ),
        ),
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=500, random_state=_RANDOM_STATE),
            ),
        ),
        (
            "logistic_regression_l1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.5,
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=_RANDOM_STATE,
                ),
            ),
        ),
        (
            "logistic_regression_balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=500,
                    random_state=_RANDOM_STATE,
                ),
            ),
        ),
        (
            "random_forest_depth4",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=4,
                min_samples_leaf=2,
                random_state=_RANDOM_STATE,
                n_jobs=1,
            ),
        ),
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


def _oof_probabilities(
    estimator: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out_of_fold = np.zeros(len(labels), dtype=float)
    covered = np.zeros(len(labels), dtype=bool)
    for train_index, test_index in folds:
        fold_model = clone(estimator)
        fold_model.fit(matrix[train_index], labels[train_index])
        out_of_fold[test_index] = fold_model.predict_proba(matrix[test_index])[:, 1]
        covered[test_index] = True
    if not covered.all():  # pragma: no cover - defensive
        raise RuntimeError("grouped cross-validation left examples uncovered")
    return out_of_fold


def _fit_calibrated(
    estimator: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    out_of_fold: np.ndarray,
) -> _CalibratedModel:
    calibrator = _PlattCalibrator.fit(out_of_fold, labels)
    fitted = clone(estimator)
    fitted.fit(matrix, labels)
    return _CalibratedModel(estimator=fitted, calibrator=calibrator)


def _candidate_results(
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[list[dict[str, float]], str, dict[str, np.ndarray]]:
    rows: list[dict[str, float]] = []
    out_of_fold: dict[str, np.ndarray] = {}
    for name, estimator in _candidate_specs():
        oof = _oof_probabilities(estimator, matrix, labels, folds)
        out_of_fold[name] = oof
        cv = _metrics(labels, (oof >= 0.5).astype(int))
        calibrated = _fit_calibrated(estimator, matrix, labels, oof)
        validation_probabilities = calibrated.predict_proba(validation_matrix)
        validation = _metrics(
            validation_labels, (validation_probabilities >= 0.5).astype(int)
        )
        rows.append(
            {
                "name": name,
                "cv_accuracy": cv["accuracy"],
                "cv_precision": cv["precision"],
                "cv_recall": cv["recall"],
                "cv_f1": cv["f1"],
                "cv_false_positive_rate": cv["false_positive_rate"],
                "validation_accuracy": validation["accuracy"],
                "validation_precision": validation["precision"],
                "validation_recall": validation["recall"],
                "validation_f1": validation["f1"],
                "validation_false_positive_rate": validation["false_positive_rate"],
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["validation_f1"],
            row["validation_precision"],
            row["validation_recall"],
            row["cv_f1"],
        ),
    )
    return rows, str(best["name"]), out_of_fold


def _choose_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, str]:
    """Choose on validation only; prefer a margin when classes are separable."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]
    if (
        len(positive)
        and len(negative)
        and float(np.max(negative)) < float(np.min(positive))
    ):
        midpoint = (float(np.max(negative)) + float(np.min(positive))) / 2.0
        return round(midpoint, 2), "validation_midpoint"
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in np.arange(0.05, 0.951, 0.01):
        metrics = _metrics(labels, (probabilities >= threshold).astype(int))
        feasible = metrics["false_positive_rate"] <= _MAX_FALSE_POSITIVE_RATE
        key = (
            1.0 if feasible else 0.0,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:  # pragma: no cover - defensive
        return 0.5, "default"
    return round(best[1], 2), "validation_f1"


def _external_cases() -> list[tuple[str, str, bool]]:
    """Existing scanner benchmarks, used only for post-training transfer tests."""

    from leaklens.bench import CASES
    from leaklens.bench_advanced import ADVANCED_CASES
    from leaklens.bench_cases import EXTRA_CASES

    cases: list[tuple[str, str, bool]] = list(CASES)
    cases.extend((case.name, case.code, case.leaky) for case in EXTRA_CASES)
    cases.extend((case.name, case.code, case.leaky) for case in ADVANCED_CASES)
    return cases


def _feature_importances(estimator: Any) -> tuple[float, ...]:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif isinstance(estimator, Pipeline):
        final = estimator.steps[-1][1]
        if hasattr(final, "coef_"):
            values = np.abs(np.asarray(final.coef_, dtype=float)).ravel()
        elif hasattr(final, "feature_importances_"):
            values = np.asarray(final.feature_importances_, dtype=float)
        else:  # pragma: no cover - defensive
            values = np.ones(len(MODEL_FEATURE_NAMES), dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:  # pragma: no cover - defensive
        values = np.ones(len(MODEL_FEATURE_NAMES), dtype=float)
    if values.shape[0] != len(MODEL_FEATURE_NAMES):  # pragma: no cover
        return tuple(1.0 / len(MODEL_FEATURE_NAMES) for _ in MODEL_FEATURE_NAMES)
    total = float(values.sum())
    if total > 0:
        values = values / total
    return tuple(float(value) for value in values)


def _matrix(cases: Sequence[CorpusCase]) -> np.ndarray:
    return np.vstack([model_features(case.code) for case in cases])


def _labels(cases: Sequence[CorpusCase]) -> np.ndarray:
    return np.asarray([int(case.leaky) for case in cases], dtype=int)


def _groups(cases: Sequence[CorpusCase]) -> np.ndarray:
    return np.asarray([case.group for case in cases], dtype=object)


def _evaluate(
    *, include_external: bool = True, candidate: str | None = None
) -> tuple[ModelEvaluation, LearnedModel]:
    train = training_cases()
    validation = validation_cases()
    development = development_cases()

    train_matrix = _matrix(train)
    train_labels = _labels(train)
    validation_matrix = _matrix(validation)
    validation_labels = _labels(validation)
    development_matrix = _matrix(development)
    development_labels = _labels(development)

    folds, splits = _grouped_folds(_groups(train), train_labels)
    if candidate is None:
        candidate_rows, selected_name, train_oof = _candidate_results(
            train_matrix,
            train_labels,
            folds,
            validation_matrix,
            validation_labels,
        )
    else:
        # Runtime path: fit the frozen winner only.  The remaining candidates are
        # still compared by the benchmark, never by a user-facing request.
        specs = dict(_candidate_specs())
        if candidate not in specs:
            raise ValueError(f"unknown candidate {candidate!r}")
        selected_name = candidate
        oof = _oof_probabilities(specs[candidate], train_matrix, train_labels, folds)
        train_oof = {candidate: oof}
        cv = _metrics(train_labels, (oof >= 0.5).astype(int))
        calibrated = _fit_calibrated(specs[candidate], train_matrix, train_labels, oof)
        validation_row = _metrics(
            validation_labels,
            (calibrated.predict_proba(validation_matrix) >= 0.5).astype(int),
        )
        candidate_rows = [
            {
                "name": candidate,
                "cv_accuracy": cv["accuracy"],
                "cv_precision": cv["precision"],
                "cv_recall": cv["recall"],
                "cv_f1": cv["f1"],
                "cv_false_positive_rate": cv["false_positive_rate"],
                "validation_accuracy": validation_row["accuracy"],
                "validation_precision": validation_row["precision"],
                "validation_recall": validation_row["recall"],
                "validation_f1": validation_row["f1"],
                "validation_false_positive_rate": validation_row["false_positive_rate"],
            }
        ]
    selected = dict(_candidate_specs())[selected_name]

    calibrated_train = _fit_calibrated(
        selected, train_matrix, train_labels, train_oof[selected_name]
    )
    validation_probabilities = calibrated_train.predict_proba(validation_matrix)
    threshold, threshold_rule = _choose_threshold(
        validation_labels, validation_probabilities
    )
    validation_metrics = _metrics(
        validation_labels, (validation_probabilities >= threshold).astype(int)
    )
    train_calibrated = calibrated_train.calibrator.predict(train_oof[selected_name])
    train_metrics = _metrics(train_labels, (train_calibrated >= threshold).astype(int))
    calibration = _calibration(validation_labels, validation_probabilities)

    development_folds, _ = _grouped_folds(_groups(development), development_labels)
    development_oof = _oof_probabilities(
        selected, development_matrix, development_labels, development_folds
    )
    final = _fit_calibrated(
        selected, development_matrix, development_labels, development_oof
    )
    importances = _feature_importances(final.estimator)

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
            probabilities = final.predict_proba(external_matrix)
            external = _metrics(
                external_labels, (probabilities >= threshold).astype(int)
            )

    stats = corpus_stats()
    selection = (
        "candidate selection frozen after validation"
        if candidate is not None
        else "candidate comparison repeated on grouped CV + validation"
    )
    protocol = (
        f"train={stats['train_cases']} cases/{stats['train_sources']} sources; "
        f"validation={stats['validation_cases']} cases/{stats['validation_sources']} "
        "sources; locked test imported only by the benchmark; "
        f"{selection}; threshold rule={threshold_rule}; runtime hypotheses also "
        "require at least one semantic risk indicator."
    )
    external_rates = {
        key: value
        for key, value in external.items()
        if key in {"precision", "recall", "f1", "false_positive_rate", "accuracy"}
    }
    card = ModelCard(
        name="leaklens-ai-p7",
        kind="calibrated_ensemble"
        if "forest" in selected_name or "trees" in selected_name
        else selected_name,
        features=len(MODEL_FEATURE_NAMES),
        training_examples=len(development),
        groups=int(stats["groups"]),
        threshold=threshold,
        precision=validation_metrics["precision"],
        recall=validation_metrics["recall"],
        f1=validation_metrics["f1"],
        false_positive_rate=validation_metrics["false_positive_rate"],
        evaluation=(
            f"Grouped {splits}-fold CV on train; validation-only threshold; "
            "locked test never used for fitting or tuning."
        ),
        limitations=MODEL_LIMITATIONS,
        external=tuple(
            sorted((f"external_{key}", value) for key, value in external_rates.items())
        ),
        version=MODEL_VERSION,
        validation_examples=len(validation),
        calibration=tuple(sorted(calibration.items())),
        candidates=tuple((row["name"], row["validation_f1"]) for row in candidate_rows),
        protocol=protocol,
    )
    evaluation = ModelEvaluation(
        metrics=validation_metrics,
        threshold=threshold,
        folds=splits,
        examples=len(validation),
        groups=int(stats["groups"]),
        external=external,
        train_examples=len(train),
        validation_examples=len(validation),
        train_metrics=train_metrics,
        calibration=calibration,
        candidates=tuple(candidate_rows),
        model_name=selected_name,
        protocol=protocol,
    )
    return evaluation, LearnedModel(
        model=final,
        card=card,
        feature_importances=importances,
        threshold=threshold,
    )


_lock = threading.Lock()
_cached: tuple[ModelEvaluation, LearnedModel] | None = None
_runtime_cached: tuple[ModelEvaluation, LearnedModel] | None = None


def evaluate_model(*, include_external: bool = True) -> ModelEvaluation:
    return get_model(include_external=include_external)[0]


def get_model(
    *, include_external: bool = True, compare_candidates: bool = True
) -> tuple[ModelEvaluation, LearnedModel]:
    """Train once, cache in memory, and return the evaluation plus the model.

    ``compare_candidates=False`` is the runtime path used by the public API: it
    fits only the candidate frozen after validation instead of retraining all
    eleven, which keeps a cold opt-in AI request inside a small server's time
    budget.  Both paths use the same corpus, grouped folds, calibration and
    threshold rule; where the full comparison picks the same winner they are
    numerically identical.  The benchmark always asks for the full comparison.

    Nothing is written to disk and no serialized model is loaded, so a tampered
    file cannot execute code.  The cache is guarded because the public API may
    serve concurrent requests.
    """

    global _cached, _runtime_cached
    with _lock:
        if not compare_candidates:
            if _runtime_cached is None or (
                include_external and not _runtime_cached[0].external
            ):
                _runtime_cached = _evaluate(
                    include_external=include_external, candidate=_FROZEN_CANDIDATE
                )
            return _runtime_cached
        if _cached is None or (include_external and not _cached[0].external):
            _cached = _evaluate(include_external=include_external)
        return _cached


def reset_model_cache() -> None:
    """Testing helper.  Never needed at runtime."""

    global _cached, _runtime_cached
    with _lock:
        _cached = None
        _runtime_cached = None


def evaluate_locked_test() -> dict[str, object]:
    """Evaluate the frozen shipped model on the locked independent test set.

    Only the benchmark and its regression test call this.  Training, feature
    selection, calibration and threshold tuning never import the test cases.
    """

    from .corpus_test import LOCKED_TEST_CASES

    _evaluation, model = get_model(compare_candidates=False)
    matrix = _matrix(LOCKED_TEST_CASES)
    labels = _labels(LOCKED_TEST_CASES)
    probabilities = model.predict_matrix(matrix)
    predicted = (probabilities >= model.threshold).astype(int)
    metrics = _metrics(labels, predicted)

    per_category: dict[str, dict[str, float]] = {}
    for category in sorted({case.category for case in LOCKED_TEST_CASES}):
        indexes = [
            index
            for index, case in enumerate(LOCKED_TEST_CASES)
            if case.category == category
        ]
        per_category[category] = _metrics(labels[indexes], predicted[indexes])
    per_source: dict[str, dict[str, float]] = {}
    for source in sorted({case.source for case in LOCKED_TEST_CASES}):
        indexes = [
            index
            for index, case in enumerate(LOCKED_TEST_CASES)
            if case.source == source
        ]
        per_source[source] = _metrics(labels[indexes], predicted[indexes])

    false_positives = []
    false_negatives = []
    for case, probability, guess in zip(LOCKED_TEST_CASES, probabilities, predicted):
        if bool(guess) == case.leaky:
            continue
        contributions = model.top_contributions(case.code, limit=4)
        item = {
            "case": case.name,
            "group": case.group,
            "category": case.category,
            "source": case.source,
            "probability": round(float(probability), 4),
            "signals": [name for name, _value, _importance in contributions],
        }
        if case.leaky:
            false_negatives.append(item)
        else:
            false_positives.append(item)

    return {
        "cases": len(LOCKED_TEST_CASES),
        "leaky": int(np.sum(labels)),
        "legitimate": int(len(labels) - np.sum(labels)),
        "threshold": model.threshold,
        "metrics": metrics,
        "calibration": _calibration(labels, probabilities),
        "per_category": per_category,
        "per_source": per_source,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def evaluate_phase7_holdout() -> dict[str, object]:
    """Score the frozen model on the independently authored Phase 7 holdout.

    Only the benchmark and its regression test call this.  The holdout was
    written and frozen before the final model fit; it is never imported by the
    training path and no threshold, feature or candidate was selected on it.
    """

    from .corpus_phase7 import PHASE7_HOLDOUT_CASES

    _evaluation, model = get_model(compare_candidates=False)
    matrix = _matrix(PHASE7_HOLDOUT_CASES)
    labels = _labels(PHASE7_HOLDOUT_CASES)
    probabilities = model.predict_matrix(matrix)
    predicted = (probabilities >= model.threshold).astype(int)
    metrics = _metrics(labels, predicted)

    per_family: dict[str, dict[str, float]] = {}
    for family in sorted({case.source for case in PHASE7_HOLDOUT_CASES}):
        indexes = [
            index
            for index, case in enumerate(PHASE7_HOLDOUT_CASES)
            if case.source == family
        ]
        per_family[family] = _metrics(labels[indexes], predicted[indexes])

    false_positives = []
    false_negatives = []
    for case, probability, guess in zip(PHASE7_HOLDOUT_CASES, probabilities, predicted):
        if bool(guess) == case.leaky:
            continue
        contributions = model.local_contributions(case.code, limit=4)
        item = {
            "case": case.name,
            "group": case.group,
            "source": case.source,
            "probability": round(float(probability), 4),
            "signals": [
                f"{name} ({contribution:+.2f})"
                for name, _value, contribution in contributions
            ],
        }
        if case.leaky:
            false_negatives.append(item)
        else:
            false_positives.append(item)

    return {
        "cases": len(PHASE7_HOLDOUT_CASES),
        "leaky": int(np.sum(labels)),
        "legitimate": int(len(labels) - np.sum(labels)),
        "threshold": model.threshold,
        "metrics": metrics,
        "calibration": _calibration(labels, probabilities),
        "per_family": per_family,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "provenance": (
            "Authored synthetic pairs, frozen before the final fit; grouped by "
            "template pair and source family; never used for training or tuning."
        ),
    }


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


# Feature -> the line-level construct that activates it.  The mapping is
# deliberately narrow: it is used to point a reviewer at code, not to explain
# the model causally.  Unknown features fall back to the keyword heuristic.
_FEATURE_EVIDENCE: dict[str, re.Pattern[str]] = {
    "fit_before_split": re.compile(r"\.(?:fit|fit_transform)\s*\("),
    "fit_transform_before_split": re.compile(r"\.fit_transform\s*\("),
    "transformer_before_split": re.compile(
        r"\b(?:StandardScaler|MinMaxScaler|RobustScaler|Normalizer|"
        r"PowerTransformer|QuantileTransformer|SimpleImputer|KNNImputer|PCA)"
    ),
    "encoder_before_split": re.compile(
        r"\b(?:OneHotEncoder|OrdinalEncoder|TargetEncoder)\b|get_dummies"
    ),
    "unsafe_target_encoding": re.compile(
        r"\b(?:TargetEncoder|WOEEncoder|CatBoostEncoder)\b"
    ),
    "target_mean_before_split": re.compile(r"groupby\([^)]*\).*(?:mean|agg)\("),
    "functional_scaling_before_split": re.compile(
        r"\b(?:scale|normalize|power_transform|quantile_transform|"
        r"robust_scale|minmax_scale|maxabs_scale)\s*\("
    ),
    "manual_scaling_before_split": re.compile(
        r"[/][^=\n]*\.(?:max|min|std|mean)\s*\(|"
        r"\([^)]*\.(?:mean|min)\s*\([^)]*\)\s*\)\s*/"
    ),
    "full_data_imputation_before_split": re.compile(
        r"\.fillna\([^)]*\.(?:mode|mean|median|value_counts)\("
    ),
    "get_dummies_before_split": re.compile(r"get_dummies\s*\("),
    "cv_on_test_partition": re.compile(
        r"cross_val_(?:score|validate|predict)\([^)]*test"
    ),
    "group_kfold_without_groups": re.compile(r"GroupKFold\s*\("),
    "group_boundary_ignored": re.compile(r"train_test_split\s*\("),
    "repeated_test_scoring": re.compile(r"\.score\(\s*(?:X_)?test"),
    "frequency_encoding_before_split": re.compile(r"value_counts\(\)"),
    "default_shuffle_after_sort": re.compile(r"\.sort_values\s*\("),
    "temporal_order_risk": re.compile(r"train_test_split\s*\("),
    "future_feature_risk": re.compile(r"(?:shift\s*\(\s*-|bfill|center\s*=\s*True)"),
    "target_feature_risk": re.compile(r"\[['\"][^'\"]+['\"]\]\s*="),
    "duplicates_after_split": re.compile(r"drop_duplicates\s*\("),
    "search_over_pipeline_safe": re.compile(
        r"(?:GridSearchCV|RandomizedSearchCV)\s*\("
    ),
    "pipeline_boundary": re.compile(r"(?:Pipeline|make_pipeline)\s*\("),
    # Coarse features that still deserve a pointer to code.  These are only
    # used to cite a line for a reviewer; they do not affect the prediction.
    "has_fit": re.compile(r"\.(?:fit|fit_transform)\s*\("),
    "has_fit_transform": re.compile(r"\.fit_transform\s*\("),
    "fit_without_split": re.compile(r"\.(?:fit|fit_transform)\s*\("),
    "fit_on_train_frame": re.compile(r"\.(?:fit|fit_transform)\s*\("),
    "fit_on_test_frame": re.compile(r"\.(?:fit|fit_transform)\s*\(\s*(?:X_)?test"),
    "cv_on_full_data": re.compile(r"cross_val_|\.split\s*\("),
    "cross_validation": re.compile(r"cross_val_|\.split\s*\("),
    "plain_kfold": re.compile(r"\bKFold\s*\("),
    "has_shuffle": re.compile(r"shuffle\s*=\s*True"),
    "target_in_assignment": re.compile(r"\b(?:target|label|outcome)\b"),
    "target_reference": re.compile(r"\b(?:target|label|outcome)\b"),
    "target_relabel_safe": re.compile(r"\b(?:target|label|outcome)\b"),
    "time_words": re.compile(r"\b(?:timestamp|datetime|date|time)\b"),
    "sort_values": re.compile(r"\.sort_values\s*\("),
    "train_only_marker": re.compile(r"\btrain(?:ing)?\b"),
    "log_split_position": re.compile(r"train_test_split\s*\(|\.split\s*\("),
    "test_concat": re.compile(r"\.concat\s*\("),
}


# Features that describe a concrete leakage path rather than mere script shape.
# A statistical score is not enough on its own: the runtime hypothesis gate
# requires at least one of these to be active.  They are all context-aware
# (position relative to the split, frame identity, target flow), so a safe
# script that merely *contains* a scaler, a concat or a sort keeps its score
# but does not produce a false alarm.
_SEMANTIC_RISK_FEATURES = frozenset(
    {
        "unsafe_preprocessing",
        "unsafe_resampling",
        "unsafe_feature_selection",
        "unsafe_target_encoding",
        "transformer_before_split",
        "encoder_before_split",
        "target_mean_before_split",
        "manual_full_data_stats",
        "functional_scaling_before_split",
        "manual_scaling_before_split",
        "full_data_imputation_before_split",
        "get_dummies_before_split",
        "frequency_encoding_before_split",
        "target_feature_risk",
        "future_feature_risk",
        "temporal_order_risk",
        "default_shuffle_after_sort",
        "group_boundary_ignored",
        "identifier_feature_risk",
        "test_rows_before_split",
        "test_concat",
        "cv_on_test_partition",
        "repeated_test_scoring",
        "tuning_on_full_data",
        "group_kfold_without_groups",
        "onehot_only_full_data",
        "group_aggregate_before_split",
        "pivot_before_split",
        "fit_on_test_frame",
        "fit_on_all_data_after_split",
        "duplicates_after_split",
        "leak_column_name",
    }
)


def feature_metadata_errors() -> tuple[str, ...]:
    """Return explanation metadata that cannot describe an emitted feature.

    ``FEATURE_NAMES`` is the single authoritative catalogue: the vector, the
    model view and every explanation table are keyed by it.  Anything else -
    typically a Phase 5 intermediate that never became an emitted indicator -
    is dead metadata: it can never fire, and it hides the missing evidence
    pattern for the indicator that replaced it.
    """

    emitted = set(FEATURE_NAMES)
    eligible = set(MODEL_FEATURE_NAMES)
    errors: list[str] = []
    for name in sorted(set(_FEATURE_EVIDENCE) - emitted):
        errors.append(f"evidence pattern for unemitted feature: {name}")
    for name in sorted(set(_FEATURE_EVIDENCE) - eligible):
        errors.append(f"evidence pattern for model-excluded feature: {name}")
    for name in sorted(set(_SEMANTIC_RISK_FEATURES) - emitted):
        errors.append(f"risk indicator is not an emitted feature: {name}")
    for name in sorted(set(_SEMANTIC_RISK_FEATURES) - eligible):
        errors.append(f"risk indicator is excluded from the model: {name}")
    return tuple(errors)


_METADATA_ERRORS = feature_metadata_errors()
if _METADATA_ERRORS:  # pragma: no cover - import guard
    raise RuntimeError(
        "feature metadata drifted from FEATURE_NAMES: " + "; ".join(_METADATA_ERRORS)
    )


def _active_risk_features(source: str) -> tuple[str, ...]:
    """Return the semantic risk indicators active for one source unit."""

    vector = model_features(source)
    return tuple(
        name
        for name, value in zip(MODEL_FEATURE_NAMES, vector, strict=True)
        if value and name in _SEMANTIC_RISK_FEATURES
    )


def _signal_lines(
    lines: Sequence[str],
    contributions: Sequence[tuple[str, float, float]],
    *,
    limit: int = 2,
) -> list[tuple[int, str]]:
    """Return distinct code lines that activate the strongest contributions."""

    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for name, _value, _importance in contributions:
        pattern = _FEATURE_EVIDENCE.get(name)
        if pattern is None:
            continue
        for index, line in enumerate(lines, start=1):
            if index in seen:
                continue
            if pattern.search(line) and not line.lstrip().startswith(
                ("import ", "from ")
            ):
                seen.add(index)
                found.append((index, line.strip()))
                break
        if len(found) >= limit:
            break
    return found


def learned_hypotheses(
    code: Sequence[CodeSource],
    findings: Sequence[Finding] = (),
    *,
    threshold: float | None = None,
) -> tuple[Hypothesis, ...]:
    """Return at most one learned hypothesis per source unit above threshold."""

    _evaluation, model = get_model(compare_candidates=False)
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
        chunks = source.chunks()
        texts = [text for _label, text in chunks]
        for index, (label, text) in enumerate(chunks):
            # Notebook cells are judged in execution order, so an earlier
            # preprocessing cell can be seen by a later split cell.
            context = "\n".join(texts[: index + 1]) if len(texts) > 1 else text
            probability = model.predict_proba(context)
            if probability < cutoff:
                continue
            risk_features = _active_risk_features(context)
            if not risk_features and corroborating is None:
                # A score driven only by script shape (has_split, concat,
                # scaler presence) is not actionable evidence.  The unit is
                # still scored by the model; it simply does not raise a hint.
                continue
            contributions = model.local_contributions(context, limit=5)
            supporting = tuple(item for item in contributions if item[2] > 0)
            opposing = tuple(item for item in contributions if item[2] < 0)
            line_number = _evidence_line(text.splitlines())
            first_line = text.splitlines()[line_number - 1] if text.splitlines() else ""
            signal_lines = _signal_lines(context.splitlines(), supporting)
            primary_line = signal_lines[0][0] if signal_lines else line_number
            margin = probability - cutoff
            if margin >= 0.20:
                uncertainty = "clear"
            elif margin >= 0.05:
                uncertainty = "moderate"
            else:
                uncertainty = "uncertain"
            evidence = [
                _evidence(
                    "model",
                    "probability",
                    f"{label}:{primary_line}",
                    f"calibrated p(leaky pattern)={probability:.3f}; "
                    f"threshold={cutoff:.2f}; margin={margin:+.3f}; "
                    f"uncertainty={uncertainty}; model={MODEL_VERSION}",
                )
            ]
            evidence.append(
                _evidence(
                    "model",
                    "indicator",
                    f"{label}:{primary_line}",
                    "semantic risk indicators: "
                    + (", ".join(risk_features) if risk_features else "none"),
                )
            )
            if signal_lines:
                for index, snippet in signal_lines:
                    evidence.append(
                        _evidence("code", "pattern", f"{label}:{index}", snippet)
                    )
            else:
                evidence.append(
                    _evidence(
                        "code",
                        "pattern",
                        f"{label}:{line_number}",
                        first_line or f"code unit {label}",
                    )
                )
            for name, value, contribution in contributions:
                direction = "supports" if contribution > 0 else "opposes"
                evidence.append(
                    _evidence(
                        "model",
                        "feature",
                        f"{label}:{primary_line}",
                        f"{name}={value:.2f}; local {direction} "
                        f"log-odds {contribution:+.3f}",
                    )
                )
            confidence = (
                min(0.95, 0.35 + 0.55 * probability)
                if uncertainty == "clear"
                else min(0.70, 0.30 + 0.40 * probability)
            )
            signal_text = ", ".join(
                f"{name} ({contribution:+.2f})"
                for name, _value, contribution in supporting[:3]
            )
            counter_text = ", ".join(
                name for name, _value, _contribution in opposing[:2]
            )
            top_signal = supporting[0][0] if supporting else "none"
            excerpt = signal_lines[0][1] if signal_lines else first_line
            digest = hashlib.sha1(
                f"{label}|{primary_line}|{probability:.6f}".encode("utf-8", "replace"),
                usedforsecurity=False,
            ).hexdigest()[:12]
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=f"hyp-learned-{digest}",
                    concept="learned_leakage_pattern",
                    title=(
                        "Learned model sees a leakage-like code pattern"
                        + (f" ({top_signal})" if contributions else "")
                        + ("" if uncertainty == "clear" else " [needs review]")
                    ),
                    explanation=(
                        f"A calibrated local model trained on labeled authored "
                        f"pipelines gives this unit p={probability:.3f} for a "
                        f"leakage-like pattern (threshold {cutoff:.2f}, "
                        f"margin {margin:+.3f}, {uncertainty}). Supporting local "
                        f"signals: {signal_text or 'none'}."
                        + (
                            f" Signals pointing the other way: {counter_text}."
                            if counter_text
                            else ""
                        )
                        + f" The strongest cited line is `{excerpt[:160]}`. "
                        "Contributions are local log-odds for this unit, not a "
                        "causal explanation. The hint is emitted only when a "
                        "semantic risk indicator is active (score alone is not "
                        "enough). This is a statistical hint, not a verified "
                        "finding."
                    ),
                    confidence=round(confidence, 4),
                    evidence=tuple(evidence),
                    location=f"{label}:{primary_line}",
                    detector="learned",
                    severity_hint=(
                        "medium" if probability >= 0.85 and margin >= 0.20 else "low"
                    ),
                    corroborates=corroborating,
                    limitations=MODEL_LIMITATIONS[:3],
                )
            )
    return tuple(hypotheses)


__all__ = [
    "MODEL_LIMITATIONS",
    "MODEL_VERSION",
    "LearnedModel",
    "ModelEvaluation",
    "evaluate_locked_test",
    "evaluate_model",
    "evaluate_phase7_holdout",
    "get_model",
    "learned_hypotheses",
    "reset_model_cache",
]
