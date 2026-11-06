"""Learned detector: split discipline, calibration, locked test and safety."""

from __future__ import annotations

import re
import socket

import numpy as np
import pytest

from leaklens.ai import learned
from leaklens.ai.corpus import (
    corpus_cases,
    corpus_stats,
    development_cases,
    training_cases,
    validation_cases,
)
from leaklens.ai.corpus_test import LOCKED_TEST_CASES
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
    evaluate_locked_test,
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


def test_corpus_is_grouped_labeled_and_split_by_source():
    stats = corpus_stats()
    cases = corpus_cases()

    assert stats == {
        "cases": 219,
        "leaky": 116,
        "legitimate": 103,
        "groups": 154,
        "leaky_groups": 87,
        "legitimate_groups": 84,
        "sources": 14,
        "train_cases": 179,
        "train_leaky": 95,
        "train_legitimate": 84,
        "train_groups": 114,
        "validation_cases": 40,
        "validation_leaky": 21,
        "validation_legitimate": 19,
        "validation_groups": 40,
        "train_sources": 10,
        "validation_sources": 4,
    }
    assert len(cases) == 219
    assert all(case.name and case.code and case.category for case in cases)
    assert all(case.split in {"train", "validation"} for case in cases)
    assert {case.split for case in training_cases()} == {"train"}
    assert {case.split for case in validation_cases()} == {"validation"}
    assert {case.split for case in LOCKED_TEST_CASES} == {"test"}


def test_locked_test_is_disjoint_from_development_by_source_and_group():
    development = development_cases()
    development_sources = {case.source for case in development}
    development_groups = {case.group for case in development}
    test_sources = {case.source for case in LOCKED_TEST_CASES}
    test_groups = {case.group for case in LOCKED_TEST_CASES}

    assert test_sources
    assert test_groups
    assert development_sources.isdisjoint(test_sources), (
        "a scenario source spans development and the locked test"
    )
    assert development_groups.isdisjoint(test_groups), (
        "a template family spans development and the locked test"
    )


def test_locked_test_has_no_near_duplicate_of_a_development_case():
    def tokens(code: str) -> set[str]:
        return set(re.findall(r"[a-z_][a-z0-9_]*", code.lower()))

    development = [(case.name, tokens(case.code)) for case in development_cases()]
    maximum = 0.0
    for case in LOCKED_TEST_CASES:
        test_tokens = tokens(case.code)
        for _name, other in development:
            union = test_tokens | other
            if union:
                maximum = max(maximum, len(test_tokens & other) / len(union))

    assert maximum < 0.8, "locked test contains a near-duplicate development case"
    assert not {case.code for case in LOCKED_TEST_CASES} & {
        case.code for case in development_cases()
    }


def test_duplicate_development_examples_share_a_group():
    cases = development_cases()
    by_code: dict[str, set[str]] = {}
    by_features: dict[tuple[float, ...], set[str]] = {}
    feature_groups: dict[tuple[float, ...], set[bool]] = {}
    for case in cases:
        by_code.setdefault(case.code.strip(), set()).add(case.group)
        vector = tuple(round(float(value), 6) for value in extract_features(case.code))
        by_features.setdefault(vector, set()).add(case.group)
        feature_groups.setdefault(vector, set()).add(case.leaky)

    assert all(len(groups) == 1 for groups in by_code.values()), (
        "an exact duplicate development case spans template groups"
    )
    assert all(len(groups) == 1 for groups in by_features.values()), (
        "feature-identical development cases span template groups"
    )
    assert all(len(labels) == 1 for labels in feature_groups.values()), (
        "feature-identical development cases have conflicting labels"
    )


def test_training_and_external_cases_do_not_overlap():
    training = {case.code for case in development_cases()}
    external = {code for _name, code, _leaky in _external_cases()}

    assert training
    assert external
    assert training.isdisjoint(external), (
        "external benchmark cases leaked into training"
    )


def test_grouped_folds_never_split_a_template_family():
    cases = training_cases()
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
    assert len(FEATURE_NAMES) == 112
    assert EXCLUDED_MODEL_FEATURES == {
        "log_lines",
        "log_bytes",
        "has_random_state",
        "target_column_passed",
    }
    assert len(MODEL_FEATURE_NAMES) == 108
    assert not set(MODEL_FEATURE_NAMES) & EXCLUDED_MODEL_FEATURES
    assert extract_features(LEAKY).shape == (112,)
    assert model_features(LEAKY).shape == (108,)
    values = model_features(LEAKY)
    assert np.isfinite(values).all()


def test_ordering_and_frame_identity_features_separate_common_confusions():
    safe_relabel = (
        "from sklearn.model_selection import train_test_split\n"
        "frame['target'] = frame['target'].map({'yes': 1, 'no': 0})\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "frame.drop(columns=['target']), frame['target'])\n"
    )
    full_stats = (
        "from sklearn.model_selection import train_test_split\n"
        "frame['x'] = (frame['x'] - frame['x'].mean()) / frame['x'].std()\n"
        "X_train, X_test, y_train, y_test = train_test_split(frame, frame['y'])\n"
    )
    train_stats = (
        "from sklearn.model_selection import train_test_split\n"
        "train, test = train_test_split(frame, random_state=0)\n"
        "train['x'] = (train['x'] - train['x'].mean()) / train['x'].std()\n"
    )
    test_fit = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
        "scaler = StandardScaler().fit(X_test)\n"
    )
    shuffled_window = (
        "from sklearn.model_selection import train_test_split\n"
        "frame['next'] = frame['sales'].shift(-1)\n"
        "frame = frame.sort_values('month')\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "frame, frame['sales'], shuffle=True)\n"
    )

    def active(code: str) -> set[str]:
        values = extract_features(code)
        return {FEATURE_NAMES[i] for i, value in enumerate(values) if value}

    relabel = active(safe_relabel)
    assert "target_relabel_safe" in relabel
    assert "target_feature_risk" not in relabel

    assert "manual_full_data_stats" in active(full_stats)
    assert "manual_train_stats" in active(train_stats)
    assert "fit_on_test_frame" in active(test_fit)
    shuffled = active(shuffled_window)
    assert "future_feature_risk" in shuffled
    assert "temporal_order_risk" in shuffled


def test_measured_metrics_are_reported_with_a_model_card():
    reset_model_cache()
    evaluation = evaluate_model()
    _evaluation, model = get_model()

    assert evaluation.examples == 40
    assert evaluation.train_examples == 179
    assert evaluation.validation_examples == 40
    assert evaluation.groups == 154
    assert evaluation.folds == 5
    assert 0.05 <= evaluation.threshold <= 0.95
    assert evaluation.model_name in {row["name"]: row for row in evaluation.candidates}
    assert len(evaluation.candidates) >= 6
    assert all(
        {"cv_f1", "validation_f1", "validation_false_positive_rate"} <= set(row)
        for row in evaluation.candidates
    )
    assert evaluation.metrics["precision"] >= 0.9
    assert evaluation.metrics["recall"] >= 0.9
    assert evaluation.metrics["f1"] >= 0.9
    assert evaluation.metrics["false_positive_rate"] <= 0.05
    assert evaluation.train_metrics is not None
    assert evaluation.train_metrics["f1"] <= 1.0
    assert evaluation.calibration is not None
    assert evaluation.calibration["brier"] <= 0.1
    assert "validation" in evaluation.protocol
    assert "locked test" in evaluation.protocol

    assert model.card.name == "leaklens-ai-p6"
    assert model.card.version == learned.MODEL_VERSION
    assert model.card.features == 108
    assert model.card.training_examples == 219
    assert model.card.validation_examples == 40
    assert model.card.calibration
    assert model.card.candidates
    assert model.card.protocol
    assert model.card.limitations
    assert "Grouped" in model.card.evaluation


def test_runtime_path_fits_the_frozen_candidate_and_matches_the_benchmark():
    reset_model_cache()
    runtime_evaluation, runtime_model = get_model(compare_candidates=False)
    reset_model_cache()
    full_evaluation, full_model = get_model()

    # Only the benchmark pays for the full comparison.
    assert len(runtime_evaluation.candidates) == 1
    assert runtime_evaluation.candidates[0]["name"] == learned._FROZEN_CANDIDATE
    assert len(full_evaluation.candidates) == 11

    cases = (
        list(development_cases())
        + list(LOCKED_TEST_CASES)
        + list(learned._external_cases())
    )
    matrix = np.vstack(
        [
            model_features(case.code if hasattr(case, "code") else case[1])
            for case in cases
        ]
    )
    if full_evaluation.model_name == learned._FROZEN_CANDIDATE:
        # On the reference scikit-learn release the frozen winner is also the
        # validation winner, so both paths must be numerically identical.
        np.testing.assert_array_equal(
            runtime_model.predict_matrix(matrix), full_model.predict_matrix(matrix)
        )
        assert runtime_model.threshold == full_model.threshold
        assert runtime_model.card.external == full_model.card.external

    # The frozen model must be reproducible from a clean cache even when a
    # scikit-learn release ranks a tied candidate differently.
    reset_model_cache()
    _second_evaluation, second_model = get_model(compare_candidates=False)
    np.testing.assert_array_equal(
        runtime_model.predict_matrix(matrix), second_model.predict_matrix(matrix)
    )
    assert second_model.threshold == runtime_model.threshold


def test_locked_independent_test_meets_the_phase5_targets():
    reset_model_cache()
    data = evaluate_locked_test()
    metrics = data["metrics"]

    assert data["cases"] == 47
    assert data["leaky"] == 25
    assert data["legitimate"] == 22
    assert metrics["accuracy"] >= 0.94
    assert metrics["precision"] >= 0.94
    assert metrics["recall"] >= 0.90
    assert metrics["f1"] >= 0.92
    assert metrics["false_positive_rate"] <= 0.05
    assert (
        metrics["true_positives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_negatives"]
        == 47
    )
    assert data["per_category"]
    assert data["per_source"]
    assert set(data["per_source"]) == {
        "test:credit",
        "test:marketing",
        "test:sensor",
        "test:support",
    }
    assert data["calibration"]["brier"] <= 0.1
    assert all(
        item["probability"] >= data["threshold"] for item in data["false_positives"]
    )
    assert all(
        item["probability"] < data["threshold"] for item in data["false_negatives"]
    )


def test_external_transfer_is_measured_but_not_claimed_as_an_improvement():
    evaluation = evaluate_model()
    external = evaluation.external

    assert external["precision"] >= 0.7
    assert external["recall"] >= 0.7
    assert external["f1"] >= 0.7
    assert external["false_positive_rate"] <= 0.2
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
    assert learned.MODEL_VERSION in item.explanation or any(
        learned.MODEL_VERSION in span.excerpt for span in item.evidence
    )
    assert "not a verified finding" in item.explanation
    assert any(span.source == "code" for span in item.evidence)
    for span in item.evidence:
        if span.source != "code":
            continue
        line_number = int(span.location.rsplit(":", 1)[1])
        assert LEAKY.splitlines()[line_number - 1].strip() == span.excerpt
    assert "(" in item.title and item.title.endswith(")")


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


def test_notebook_cells_are_judged_in_execution_order():
    source = CodeSource(
        label="notebook.ipynb",
        source=(
            "# In[1]:\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "scaled = StandardScaler().fit_transform(X)\n"
            "# In[2]:\n"
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(scaled, y)\n"
        ),
        cells=(
            "from sklearn.preprocessing import StandardScaler\n"
            "scaled = StandardScaler().fit_transform(X)\n",
            "from sklearn.model_selection import train_test_split\n"
            "X_train, X_test, y_train, y_test = train_test_split(scaled, y)\n",
        ),
    )

    hypotheses = learned_hypotheses((source,))
    assert hypotheses
    assert any(
        item.location.startswith("notebook.ipynb[cell 2]:") for item in hypotheses
    )


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
    assert first.model_name == second.model_name
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
