"""Learned detector: corpus discipline, grouped CV, transfer and safety."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from leaklens.ai import learned
from leaklens.ai.corpus import corpus_cases, corpus_stats
from leaklens.ai.features import (
    EXCLUDED_MODEL_FEATURES,
    FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    extract_features,
    model_features,
)
from leaklens.ai.learned import (
    _external_cases,
    _grouped_folds,
    evaluate_model,
    get_model,
    learned_hypotheses,
    reset_model_cache,
)
from leaklens.ai.semantics import CodeSource

LEAKY = (
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "X = StandardScaler().fit_transform(X)\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
)
SAFE = (
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "scaler = StandardScaler().fit(X_train)\n"
    "X_train = scaler.transform(X_train)\n"
    "X_test = scaler.transform(X_test)\n"
)


def test_corpus_is_grouped_and_labeled():
    stats = corpus_stats()
    cases = corpus_cases()

    assert stats == {
        "cases": 72,
        "leaky": 43,
        "legitimate": 29,
        "groups": 26,
        "leaky_groups": 15,
        "legitimate_groups": 11,
    }
    assert len(cases) == 72
    assert all(case.name and case.code and case.category for case in cases)


def test_training_and_external_cases_do_not_overlap():
    training = {case.code for case in corpus_cases()}
    external = {code for _name, code, _leaky in _external_cases()}

    assert training
    assert external
    assert training.isdisjoint(external), (
        "external benchmark cases leaked into training"
    )


def test_grouped_folds_never_split_a_template_family():
    cases = corpus_cases()
    labels = np.asarray([int(case.leaky) for case in cases])
    groups = np.asarray([case.group for case in cases], dtype=object)
    folds, splits = _grouped_folds(groups, labels)

    assert splits == 5
    seen = []
    for train_index, test_index in folds:
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        assert train_groups.isdisjoint(test_groups)
        seen.extend(test_index.tolist())
    assert sorted(seen) == list(range(len(cases)))


def test_model_features_exclude_shortcuts_but_keep_indicators():
    assert len(FEATURE_NAMES) == 64
    assert EXCLUDED_MODEL_FEATURES == {"log_lines", "log_bytes", "has_random_state"}
    assert len(MODEL_FEATURE_NAMES) == 61
    assert not set(MODEL_FEATURE_NAMES) & EXCLUDED_MODEL_FEATURES
    assert extract_features(LEAKY).shape == (64,)
    assert model_features(LEAKY).shape == (61,)


def test_measured_metrics_are_reported_with_a_model_card():
    reset_model_cache()
    evaluation = evaluate_model()
    _evaluation, model = get_model()

    assert evaluation.examples == 72
    assert evaluation.groups == 26
    assert evaluation.folds == 5
    assert 0.0 <= evaluation.threshold <= 1.0
    assert evaluation.metrics["precision"] >= 0.6
    assert evaluation.metrics["recall"] >= 0.6
    assert evaluation.metrics["f1"] >= 0.7
    assert evaluation.metrics["false_positive_rate"] <= 0.25
    assert model.card.name == "leaklens-ai-gb"
    assert model.card.kind == "gradient_boosting"
    assert model.card.features == 61
    assert model.card.training_examples == 72
    assert model.card.limitations
    assert "Grouped" in model.card.evaluation


def test_external_transfer_is_measured_but_not_claimed_as_an_improvement():
    evaluation = evaluate_model()
    external = evaluation.external

    assert external["precision"] >= 0.6
    assert external["recall"] >= 0.5
    assert external["f1"] >= 0.6
    assert external["false_positive_rate"] <= 0.3
    _evaluation, model = get_model()
    metrics = dict(model.card.external)
    assert metrics["external_precision"] == external["precision"]
    assert metrics["external_false_positive_rate"] == external["false_positive_rate"]


def test_obvious_leak_and_safe_pipeline_are_separated():
    _evaluation, model = get_model()
    leaky = model.predict_proba(LEAKY)
    safe = model.predict_proba(SAFE)

    assert leaky >= model.threshold
    assert safe < model.threshold
    assert leaky > safe


def test_learned_hypotheses_are_optional_evidence_bound_and_not_certain():
    hypotheses = learned_hypotheses((CodeSource(label="model.py", source=LEAKY),))
    safe_hypotheses = learned_hypotheses((CodeSource(label="safe.py", source=SAFE),))

    assert len(hypotheses) == 1
    assert not safe_hypotheses
    item = hypotheses[0]
    assert item.detector == "learned"
    assert item.concept == "learned_leakage_pattern"
    assert item.severity_hint in {"low", "medium"}
    assert item.confidence <= 0.95
    assert item.limitations
    assert {span.source for span in item.evidence} <= {"model", "code"}
    assert any(span.kind == "probability" for span in item.evidence)
    assert "not a verified finding" in item.explanation


def test_learned_hypotheses_can_corroborate_a_deterministic_finding():
    from leaklens.leakage import Finding

    finding = Finding(
        "high", "preprocess_before_split", "model.py:3", "fit before split"
    )
    hypotheses = learned_hypotheses(
        (CodeSource(label="model.py", source=LEAKY),),
        (finding,),
    )

    assert hypotheses
    assert hypotheses[0].corroborates == "preprocess_before_split"
    assert finding.severity == "high"


def test_training_is_deterministic_and_offline(monkeypatch):
    reset_model_cache()
    first = evaluate_model()
    _evaluation, first_model = get_model()
    first_prediction = first_model.predict_proba(LEAKY)

    def blocked(*_args, **_kwargs):
        raise AssertionError("the AI layer must not open a network connection")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

    reset_model_cache()
    second = evaluate_model()
    _evaluation, second_model = get_model()
    second_prediction = second_model.predict_proba(LEAKY)

    assert first.metrics == second.metrics
    assert first.threshold == second.threshold
    assert first_prediction == pytest.approx(second_prediction, abs=1e-12)


def test_training_never_serializes_the_model(tmp_path, monkeypatch):
    import pickle

    reset_model_cache()

    def blocked(*_args, **_kwargs):
        raise AssertionError("the model must stay in memory")

    monkeypatch.setattr(pickle, "dump", blocked)
    monkeypatch.setattr(pickle, "load", blocked)
    monkeypatch.setattr(learned, "_cached", None)
    evaluate_model()

    assert not list(tmp_path.iterdir())
