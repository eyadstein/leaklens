"""Learned detector: split discipline, calibration, locked test and safety."""

from __future__ import annotations

import re
import socket

import numpy as np
import pytest

from leaklens.ai import features as feature_module
from leaklens.ai import learned
from leaklens.ai.corpus import (
    corpus_cases,
    corpus_stats,
    development_cases,
    phase7_holdout_cases,
    training_cases,
    validation_cases,
)
from leaklens.ai.corpus_test import LOCKED_TEST_CASES
from leaklens.ai.features import (
    EXCLUDED_MODEL_FEATURES,
    FEATURE_NAMES,
    MODEL_FEATURE_INDICES,
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
        "cases": 259,
        "leaky": 136,
        "legitimate": 123,
        "groups": 174,
        "leaky_groups": 107,
        "legitimate_groups": 104,
        "sources": 19,
        "train_cases": 219,
        "train_leaky": 115,
        "train_legitimate": 104,
        "train_groups": 134,
        "validation_cases": 40,
        "validation_leaky": 21,
        "validation_legitimate": 19,
        "validation_groups": 40,
        "train_sources": 15,
        "validation_sources": 4,
        "holdout_cases": 12,
        "holdout_leaky": 6,
        "holdout_legitimate": 6,
        "holdout_groups": 6,
        "holdout_sources": 5,
    }
    assert len(cases) == 259
    assert all(case.name and case.code and case.category for case in cases)
    assert all(case.split in {"train", "validation"} for case in cases)
    assert {case.split for case in training_cases()} == {"train"}
    assert {case.split for case in validation_cases()} == {"validation"}
    assert {case.split for case in LOCKED_TEST_CASES} == {"test"}


def test_phase7_holdout_is_frozen_and_disjoint_from_development():
    development = development_cases()
    holdout = phase7_holdout_cases()

    assert len(holdout) == 12
    assert {case.split for case in holdout} == {"holdout"}
    assert {case.source for case in holdout}.isdisjoint(
        {case.source for case in development}
    )
    assert {case.group for case in holdout}.isdisjoint(
        {case.group for case in development}
    )
    # The holdout must not leak into the development corpus or its stats.
    assert {case.code for case in holdout}.isdisjoint(
        {case.code for case in development}
    )
    assert corpus_stats()["holdout_cases"] == 12
    assert all(case.name and case.code and case.category for case in holdout)


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


def test_feature_metadata_references_only_emitted_features():
    """Every name in the explanation tables must exist in the emitted vector.

    ``target_encoder_before_split`` was a local intermediate that never became an
    emitted feature, yet both tables below treated it as one.  A key that cannot
    be produced is dead metadata: it can never fire, and it hides the missing
    evidence pattern for the indicator that replaced it.
    """

    emitted = set(FEATURE_NAMES)
    model_eligible = set(MODEL_FEATURE_NAMES)

    assert sorted(set(learned._FEATURE_EVIDENCE) - emitted) == []
    assert sorted(set(learned._FEATURE_EVIDENCE) - model_eligible) == []
    assert sorted(set(learned._SEMANTIC_RISK_FEATURES) - emitted) == []
    assert sorted(set(learned._SEMANTIC_RISK_FEATURES) - model_eligible) == []
    assert learned.feature_metadata_errors() == ()


def test_feature_metadata_errors_are_reported_for_unknown_and_excluded_names(
    monkeypatch,
):
    monkeypatch.setitem(learned._FEATURE_EVIDENCE, "mystery_indicator", re.compile("x"))
    errors = learned.feature_metadata_errors()
    assert any("mystery_indicator" in error for error in errors)

    monkeypatch.undo()
    monkeypatch.setattr(learned, "_SEMANTIC_RISK_FEATURES", frozenset({"log_lines"}))
    errors = learned.feature_metadata_errors()
    assert any("log_lines" in error for error in errors)

    monkeypatch.undo()
    monkeypatch.setitem(learned._FEATURE_EVIDENCE, "log_lines", re.compile("x"))
    errors = learned.feature_metadata_errors()
    assert any("model-excluded feature: log_lines" in error for error in errors)


def test_feature_names_are_unique_and_model_indices_match_their_names():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert set(MODEL_FEATURE_NAMES) == set(FEATURE_NAMES) - EXCLUDED_MODEL_FEATURES
    assert [FEATURE_NAMES[index] for index in MODEL_FEATURE_INDICES] == list(
        MODEL_FEATURE_NAMES
    )


def test_extractor_rejects_a_feature_that_was_not_registered(monkeypatch):
    """A produced-but-unregistered indicator must fail loudly, not vanish.

    The vector is built by iterating ``FEATURE_NAMES``, so an extra key in the
    extractor's dict used to be dropped without a trace and the model silently
    trained on a narrower vector.
    """

    monkeypatch.setattr(feature_module, "FEATURE_NAMES", FEATURE_NAMES[:-1])
    with pytest.raises(RuntimeError, match="not registered"):
        extract_features(LEAKY)


def test_extractor_rejects_a_registered_feature_it_never_produces(monkeypatch):
    """A catalogue entry without a value must fail, not raise a bare KeyError."""

    monkeypatch.setattr(
        feature_module, "FEATURE_NAMES", (*FEATURE_NAMES, "phantom_indicator")
    )
    with pytest.raises(RuntimeError, match="did not produce"):
        extract_features(LEAKY)


def test_reordering_the_catalogue_reorders_the_vector(monkeypatch):
    """Order is the contract: renaming must never silently follow a sort.

    The model consumes position, not names, so a refactor that sorts or
    re-inserts names would keep every set-based check green while feeding the
    model different columns under the same weights.
    """

    baseline = extract_features(LEAKY)
    reordered = (FEATURE_NAMES[1], FEATURE_NAMES[0], *FEATURE_NAMES[2:])
    monkeypatch.setattr(feature_module, "FEATURE_NAMES", reordered)
    moved = extract_features(LEAKY)
    assert moved[0] == baseline[1]
    assert moved[1] == baseline[0]
    assert np.array_equal(np.delete(moved, [0, 1]), np.delete(baseline, [0, 1]))


def test_local_contributions_name_the_value_stored_at_their_own_index():
    """Pins positional integrity: a name must always describe its own value."""

    reset_model_cache()
    _evaluation, model = get_model(compare_candidates=False)
    stored = dict(zip(FEATURE_NAMES, extract_features(LEAKY), strict=True))
    contributions = model.local_contributions(LEAKY, limit=len(MODEL_FEATURE_NAMES))
    assert contributions
    for name, value, _contribution in contributions:
        assert name in stored
        assert value == pytest.approx(stored[name])


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
    assert evaluation.train_examples == 219
    assert evaluation.validation_examples == 40
    assert evaluation.groups == 174
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

    assert model.card.name == "leaklens-ai-p7"
    assert model.card.version == learned.MODEL_VERSION
    assert model.card.features == 108
    assert model.card.training_examples == 259
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
    assert "(" in item.title
    assert item.title.endswith(")") or item.title.endswith("[needs review]")


def test_learned_hypotheses_record_a_calibrated_probability_basis():
    item = learned_hypotheses((CodeSource(label="model.py", source=LEAKY),))[0]

    assert item.detector == "learned"
    assert item.confidence_basis == "calibrated_probability"
    assert item.sufficiency == "supported"


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


def test_local_contributions_are_signed_and_only_cite_active_features():
    _evaluation, model = get_model()
    leaky_contributions = model.local_contributions(LEAKY, limit=8)
    safe_contributions = model.local_contributions(SAFE, limit=8)

    assert leaky_contributions
    assert any(contribution > 0 for _name, _value, contribution in leaky_contributions)
    assert any(contribution < 0 for _name, _value, contribution in safe_contributions)
    # Local attribution skips features that are zero for this unit, so a
    # globally important feature absent here is never cited.
    assert all(value != 0.0 for _name, value, _contribution in leaky_contributions)
    assert all(value != 0.0 for _name, value, _contribution in safe_contributions)

    # The alias kept for callers returns the same local evidence.
    assert model.top_contributions(LEAKY, limit=8) == leaky_contributions


def test_hypotheses_cite_local_not_global_evidence():
    hypotheses = learned_hypotheses((CodeSource(label="model.py", source=LEAKY),))
    item = hypotheses[0]

    feature_spans = [span for span in item.evidence if span.kind == "feature"]
    assert feature_spans
    assert any(
        "local supports log-odds" in span.excerpt
        or "local opposes log-odds" in span.excerpt
        for span in feature_spans
    )
    assert all("global importance" not in span.excerpt for span in item.evidence)
    assert any(span.kind == "indicator" for span in item.evidence)
    assert "local log-odds" in item.explanation


def test_uncertain_units_are_labelled_for_review_with_lower_confidence():
    _evaluation, model = get_model()
    probability = model.predict_proba(LEAKY)
    item = learned_hypotheses(
        (CodeSource(label="model.py", source=LEAKY),),
        threshold=probability - 0.01,
    )[0]

    assert item.title.endswith("[needs review]")
    assert "uncertain" in item.explanation
    assert "margin" in item.explanation
    assert item.confidence_basis == "calibrated_probability"
    assert item.sufficiency == "weak"
    assert item.confidence <= 0.70
    assert item.severity_hint == "low"


def test_runtime_gate_suppresses_shape_only_scores(monkeypatch):
    class _HighScoreModel:
        threshold = 0.5

        def predict_proba(self, _source):
            return 0.99

        def local_contributions(self, _source, limit=5):
            return ()

    monkeypatch.setattr(
        learned, "get_model", lambda **_kwargs: (None, _HighScoreModel())
    )

    shape_only = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "scaled = StandardScaler()\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y)\n"
    )
    assert learned._active_risk_features(shape_only) == ()
    assert (
        learned.learned_hypotheses((CodeSource(label="shape.py", source=shape_only),))
        == ()
    )

    # A deterministic corroborating finding overrides the gate, because the
    # deterministic scanner is authoritative.
    from leaklens.leakage import Finding

    finding = Finding(
        "high", "preprocess_before_split", "shape.py:3", "fit before split"
    )
    hypotheses = learned.learned_hypotheses(
        (CodeSource(label="shape.py", source=shape_only),), (finding,)
    )
    assert hypotheses
    assert hypotheses[0].corroborates == "preprocess_before_split"


def test_phase7_guards_recognise_lookalikes():
    by_group: dict[str, dict[bool, object]] = {}
    for case in development_cases():
        if case.group.startswith("p7_"):
            by_group.setdefault(case.group, {})[case.leaky] = case

    def active(case) -> set[str]:
        return {
            FEATURE_NAMES[index]
            for index, value in enumerate(extract_features(case.code))
            if value
        }

    fold_safe = active(by_group["p7_scaler_fold_loop"][False])
    assert "fold_local_fit" in fold_safe
    assert "unsafe_preprocessing" not in fold_safe
    assert "unsafe_preprocessing" in active(by_group["p7_scaler_fold_loop"][True])

    target_safe = active(by_group["p7_label_encode_target"][False])
    assert "target_relabel_safe" in target_safe
    assert "encoder_before_split" not in target_safe
    assert "encoder_before_split" in active(by_group["p7_label_encode_target"][True])

    sort_safe = active(by_group["p7_sort_report"][False])
    assert "temporal_order_risk" not in sort_safe
    assert "default_shuffle_after_sort" not in sort_safe
    sort_leaky = active(by_group["p7_sort_report"][True])
    assert "default_shuffle_after_sort" in sort_leaky

    lag_safe = active(by_group["p7_forward_lag"][False])
    assert "future_feature_risk" not in lag_safe
    assert "future_feature_risk" in active(by_group["p7_forward_lag"][True])


def test_phase7_holdout_reports_a_frozen_protocol():
    reset_model_cache()
    data = learned.evaluate_phase7_holdout()

    assert data["cases"] == 12
    assert data["leaky"] == 6
    assert data["legitimate"] == 6
    assert data["per_family"]
    assert "frozen before the final fit" in data["provenance"]
    assert (
        data["metrics"]["true_positives"]
        + data["metrics"]["false_positives"]
        + (data["metrics"]["false_negatives"] + data["metrics"]["true_negatives"])
        == 12
    )
    assert all(
        item["probability"] >= data["threshold"] for item in data["false_positives"]
    )
    assert all(
        item["probability"] < data["threshold"] for item in data["false_negatives"]
    )
