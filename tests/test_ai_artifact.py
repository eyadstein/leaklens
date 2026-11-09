"""Model artifact: identity, provenance, parity and safe fallback."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from leaklens.ai import artifact, learned
from leaklens.ai.corpus_realworld import holdout_cases
from leaklens.ai.corpus_test import LOCKED_TEST_CASES
from leaklens.ai.features import MODEL_FEATURE_NAMES, model_features
from leaklens.ai.schemas import ModelCard


@pytest.fixture(autouse=True)
def _clear_model_cache():
    learned.reset_model_cache()
    yield
    learned.reset_model_cache()


def _matrix(cases) -> np.ndarray:
    return np.vstack([model_features(case.code) for case in cases])


def _checked_in() -> dict:
    return json.loads(artifact.ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_checked_in_artifact_is_inspectable_and_matches_this_build():
    path = artifact.ARTIFACT_PATH
    assert path.is_file()
    assert path.stat().st_size < artifact.MAX_ARTIFACT_BYTES
    raw = path.read_text(encoding="utf-8")
    assert "pickle" not in raw

    payload = _checked_in()
    assert payload["schema_version"] == artifact.SCHEMA_VERSION
    assert payload["model"]["version"] == learned.MODEL_VERSION
    assert payload["model"]["features"] == list(MODEL_FEATURE_NAMES)
    assert payload["model"]["corpus_digest"] == artifact.corpus_digest()
    assert (
        payload["model"]["feature_catalogue_digest"]
        == artifact.feature_catalogue_digest()
    )
    assert payload["model"]["library_version"] == artifact.library_version()

    evaluation, model = artifact.load_artifact()
    assert model.card.artifact_digest == payload["artifact_digest"]
    assert evaluation.candidates[0]["name"] == learned._FROZEN_CANDIDATE
    assert evaluation.examples == 40


def test_a_fresh_fit_produces_the_checked_in_payload():
    evaluation, model = learned._evaluate(
        include_external=True, candidate=learned._FROZEN_CANDIDATE
    )

    assert artifact.build_payload(evaluation, model) == _checked_in()


def test_runtime_serving_path_loads_the_artifact_without_fitting(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("a digest-valid artifact must not refit")

    monkeypatch.setattr(learned, "_evaluate", blocked)

    evaluation, model = learned.get_model(compare_candidates=False)

    assert learned.artifact_fallback_reason() is None
    assert len(evaluation.candidates) == 1
    assert model.card.artifact_digest.startswith("sha256:")
    assert model.card.corpus_digest == artifact.corpus_digest()
    assert model.card.feature_catalogue_digest == artifact.feature_catalogue_digest()
    assert model.card.library_version == artifact.library_version()
    assert np.isfinite(model.predict_proba("X_train, X_test, y_train, y_test"))


def test_runtime_cache_loads_the_artifact_once(monkeypatch):
    real_load = artifact.load_artifact
    calls: list[int] = []

    def counted():
        calls.append(1)
        return real_load()

    monkeypatch.setattr(artifact, "load_artifact", counted)

    first = learned.get_model(compare_candidates=False)[1]
    second = learned.get_model(compare_candidates=False)[1]

    assert first is second
    assert calls == [1]


def test_artifact_predictions_match_a_fresh_fit_for_the_holdouts():
    artifact_evaluation, artifact_model = artifact.load_artifact()
    fit_evaluation, fit_model = learned._evaluate(
        include_external=True, candidate=learned._FROZEN_CANDIDATE
    )
    cases = list(holdout_cases()) + list(LOCKED_TEST_CASES)
    matrix = _matrix(cases)

    assert len(cases) == 48 + 47
    np.testing.assert_allclose(
        artifact_model.predict_matrix(matrix),
        fit_model.predict_matrix(matrix),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        artifact_model.local_contributions(cases[0].code),
        fit_model.local_contributions(cases[0].code),
    )
    assert artifact_model.threshold == fit_model.threshold
    assert artifact_model.feature_importances == fit_model.feature_importances
    assert artifact_evaluation.metrics == fit_evaluation.metrics
    assert artifact_evaluation.calibration == fit_evaluation.calibration
    assert artifact_model.card.corpus_digest == fit_model.card.corpus_digest
    assert artifact_model.card.artifact_digest
    assert fit_model.card.artifact_digest == ""


def test_model_card_provenance_round_trips_and_old_cards_still_validate():
    _evaluation, model = artifact.load_artifact()
    round_trip = ModelCard(**model.card.to_dict())

    assert round_trip == model.card
    assert round_trip.artifact_digest
    assert round_trip.corpus_digest
    assert round_trip.feature_catalogue_digest
    assert round_trip.library_version

    legacy = {
        key: value
        for key, value in model.card.to_dict().items()
        if key
        not in {
            "artifact_digest",
            "corpus_digest",
            "feature_catalogue_digest",
            "library_version",
        }
    }
    old_card = ModelCard(**legacy)
    assert old_card.artifact_digest == ""
    assert old_card.corpus_digest == ""
    assert old_card.feature_catalogue_digest == ""
    assert old_card.library_version == ""


def test_provenance_digests_are_content_addressed(monkeypatch):
    assert artifact.corpus_digest() == artifact.corpus_digest()
    assert artifact.feature_catalogue_digest().startswith("sha256:")

    monkeypatch.setattr(
        artifact, "MODEL_FEATURE_NAMES", tuple(reversed(MODEL_FEATURE_NAMES))
    )
    assert (
        artifact.feature_catalogue_digest()
        != _checked_in()["model"]["feature_catalogue_digest"]
    )


def _write_variant(tmp_path, mutate, *, restamp: bool) -> Path:
    payload = _checked_in()
    mutate(payload)
    if restamp:
        payload.pop("artifact_digest", None)
        payload["artifact_digest"] = artifact._digest(payload)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _edit(payload, section, key, value):
    payload[section][key] = value


CORRUPTIONS = (
    (
        "edited coefficient",
        lambda payload: payload["model"]["coefficients"].__setitem__(
            0, payload["model"]["coefficients"][0] + 0.5
        ),
        False,
        "digest mismatch",
    ),
    (
        "edited digest",
        lambda payload: payload.__setitem__("artifact_digest", "sha256:" + "0" * 64),
        False,
        "digest mismatch",
    ),
    (
        "unsupported schema",
        lambda payload: payload.__setitem__("schema_version", "leaklens-ai-artifact-0"),
        True,
        "schema",
    ),
    (
        "stale model version",
        lambda payload: _edit(payload, "model", "version", "leaklens-ai-p6"),
        True,
        "model",
    ),
    (
        "stale corpus",
        lambda payload: _edit(payload, "model", "corpus_digest", "sha256:" + "1" * 64),
        True,
        "corpus digest",
    ),
    (
        "stale feature catalogue",
        lambda payload: _edit(
            payload, "model", "feature_catalogue_digest", "sha256:" + "2" * 64
        ),
        True,
        "feature catalogue digest",
    ),
    (
        "incompatible library",
        lambda payload: _edit(payload, "model", "library_version", "0.1.0"),
        True,
        "scikit-learn",
    ),
    (
        "renamed feature",
        lambda payload: payload["model"]["features"].__setitem__(0, "renamed"),
        True,
        "feature catalogue",
    ),
    (
        "invalid threshold",
        lambda payload: _edit(payload, "model", "threshold", 1.5),
        True,
        "threshold",
    ),
    (
        "wrong vector length",
        lambda payload: payload["model"]["coefficients"].pop(),
        True,
        "wrong length",
    ),
)


@pytest.mark.parametrize("name, mutate, restamp, reason", CORRUPTIONS)
def test_corrupt_or_stale_artifacts_fall_back_to_fit_with_a_reason(
    tmp_path, monkeypatch, name, mutate, restamp, reason
):
    good_evaluation, good_model = artifact.load_artifact()
    path = _write_variant(tmp_path, mutate, restamp=restamp)
    monkeypatch.setattr(artifact, "ARTIFACT_PATH", path)

    fit_calls: list[object] = []

    def fake_fit(*, include_external: bool, candidate=None):
        fit_calls.append(candidate)
        return good_evaluation, replace(
            good_model, card=replace(good_model.card, artifact_digest="")
        )

    monkeypatch.setattr(learned, "_evaluate", fake_fit)

    _evaluation, model = learned.get_model(compare_candidates=False)

    assert fit_calls == [learned._FROZEN_CANDIDATE]
    assert reason in (learned.artifact_fallback_reason() or "")
    assert model.card.artifact_digest == ""


def test_non_finite_artifact_values_are_rejected(tmp_path):
    payload = _checked_in()
    payload["model"]["coefficients"][0] = float("nan")
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(artifact.ArtifactError, match="non-finite"):
        artifact.load_artifact(path)


def test_missing_truncated_and_oversized_artifacts_fall_back(tmp_path, monkeypatch):
    good_evaluation, good_model = artifact.load_artifact()
    missing = tmp_path / "absent.json"
    truncated = tmp_path / "truncated.json"
    truncated.write_text(
        artifact.ARTIFACT_PATH.read_text(encoding="utf-8")[:200], encoding="utf-8"
    )
    oversized = tmp_path / "oversized.json"
    oversized.write_text(" " * artifact.MAX_ARTIFACT_BYTES + "{}", encoding="utf-8")

    def fake_fit(*, include_external: bool, candidate=None):
        return good_evaluation, replace(
            good_model, card=replace(good_model.card, artifact_digest="")
        )

    monkeypatch.setattr(learned, "_evaluate", fake_fit)

    cases = (
        (missing, "missing"),
        (truncated, "not valid JSON"),
        (oversized, "budget"),
    )
    for path, reason in cases:
        learned.reset_model_cache()
        monkeypatch.setattr(artifact, "ARTIFACT_PATH", path)
        _evaluation, model = learned.get_model(compare_candidates=False)
        assert reason in (learned.artifact_fallback_reason() or "")
        assert model.card.artifact_digest == ""
