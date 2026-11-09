"""Model artifact: identity, provenance, parity and safe fallback."""

from __future__ import annotations

import json
import math
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


def _compatible() -> bool:
    """Whether the checked-in artifact targets this scikit-learn minor."""

    return artifact._minor(
        _checked_in()["model"]["library_version"]
    ) == artifact._minor(artifact.library_version())


def _comparable(payload: dict) -> dict:
    """Drop fields that legitimately differ between fitted environments."""

    comparable = json.loads(json.dumps(payload))
    comparable.pop("artifact_digest", None)
    for section in ("model", "card"):
        comparable[section].pop("library_version", None)
    return comparable


# A fit on another OS/BLAS build lands within a few ulp of the checked-in one
# (CI observed ~2e-16 relative).  This bound is the same order as the existing
# prediction-parity tolerance and still rejects any real weight change.
_FLOAT_REL_TOL = 1e-9
_FLOAT_ABS_TOL = 1e-12


def _assert_payload_close(
    actual, expected, path: str = "payload", *, numeric: bool = True
) -> None:
    """Compare structure exactly and floating-point leaves numerically."""

    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_payload_close(
                actual[key], expected[key], f"{path}.{key}", numeric=numeric
            )
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_payload_close(left, right, f"{path}[{index}]", numeric=numeric)
    elif isinstance(expected, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), path
        if numeric:
            assert math.isclose(
                actual, expected, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL
            ), f"{path}: {actual!r} != {expected!r}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.fixture(scope="module")
def fresh_fit():
    """Fit once and serialize it with the library available in this environment."""

    evaluation, model = learned._evaluate(
        include_external=True, candidate=learned._FROZEN_CANDIDATE
    )
    return evaluation, model, artifact.build_payload(evaluation, model)


def _load_payload(payload: dict, tmp_path: Path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return artifact.load_artifact(path)


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
    assert payload["model"]["library_version"]

    if _compatible():
        evaluation, model = artifact.load_artifact()
        assert model.card.artifact_digest == payload["artifact_digest"]
        assert evaluation.candidates[0]["name"] == learned._FROZEN_CANDIDATE
        assert evaluation.examples == 40
    else:
        # Python 3.10 resolves an older scikit-learn minor, so the loader must
        # refuse the artifact instead of serving numbers fitted elsewhere.
        with pytest.raises(artifact.ArtifactError, match="scikit-learn"):
            artifact.load_artifact()


def test_a_fresh_fit_matches_the_checked_in_payload(fresh_fit):
    payload = _checked_in()
    # Integrity is exact and independent of the machine: the recorded digest
    # covers the canonical serialized body.
    body = {key: value for key, value in payload.items() if key != "artifact_digest"}
    assert payload["artifact_digest"] == artifact._digest(body)

    _evaluation, _model, fresh = fresh_fit

    # Cross-environment fits differ only in floating-point leaves; names,
    # ordering, protocol and version metadata stay exact.  Numbers are only
    # comparable across a matching scikit-learn minor.
    _assert_payload_close(
        _comparable(fresh), _comparable(payload), numeric=_compatible()
    )
    assert fresh["model"]["library_version"] == artifact.library_version()
    # Serialization itself is deterministic for a given fit.
    assert fresh == artifact.build_payload(_evaluation, _model)
    assert fresh["artifact_digest"] == artifact._digest(
        {key: value for key, value in fresh.items() if key != "artifact_digest"}
    )


def test_runtime_serving_path_loads_a_valid_artifact_without_fitting(
    tmp_path, monkeypatch, fresh_fit
):
    _evaluation, _model, payload = fresh_fit
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(artifact, "ARTIFACT_PATH", path)

    def blocked(*_args, **_kwargs):
        raise AssertionError("a digest-valid artifact must not refit")

    monkeypatch.setattr(learned, "_evaluate", blocked)

    evaluation, model = learned.get_model(compare_candidates=False)

    assert learned.artifact_fallback_reason() is None
    assert len(evaluation.candidates) == 1
    assert model.card.artifact_digest == payload["artifact_digest"]
    assert model.card.corpus_digest == artifact.corpus_digest()
    assert model.card.feature_catalogue_digest == artifact.feature_catalogue_digest()
    assert model.card.library_version
    assert artifact._minor(model.card.library_version) == artifact._minor(
        artifact.library_version()
    )
    assert np.isfinite(model.predict_proba("X_train, X_test, y_train, y_test"))


def test_runtime_serving_path_uses_the_checked_in_artifact_when_compatible(monkeypatch):
    if not _compatible():
        # Another scikit-learn minor fitted it: the loader must record why it
        # fell back and fit in memory instead of serving foreign numbers.
        evaluation, model = learned.get_model(compare_candidates=False)
        assert "scikit-learn" in (learned.artifact_fallback_reason() or "")
        assert model.card.artifact_digest == ""
    else:

        def blocked(*_args, **_kwargs):
            raise AssertionError("a digest-valid artifact must not refit")

        monkeypatch.setattr(learned, "_evaluate", blocked)
        evaluation, model = learned.get_model(compare_candidates=False)
        assert learned.artifact_fallback_reason() is None
        assert model.card.artifact_digest.startswith("sha256:")

    assert len(evaluation.candidates) == 1
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


def test_artifact_predictions_match_a_fresh_fit_for_the_holdouts(tmp_path, fresh_fit):
    fit_evaluation, fit_model, payload = fresh_fit
    artifact_evaluation, artifact_model = _load_payload(payload, tmp_path)
    cases = list(holdout_cases()) + list(LOCKED_TEST_CASES)
    matrix = _matrix(cases)

    assert len(cases) == 48 + 47
    np.testing.assert_allclose(
        artifact_model.predict_matrix(matrix),
        fit_model.predict_matrix(matrix),
        rtol=0.0,
        atol=1e-9,
    )
    artifact_contributions = artifact_model.local_contributions(cases[0].code)
    fit_contributions = fit_model.local_contributions(cases[0].code)
    assert [name for name, _value, _delta in artifact_contributions] == [
        name for name, _value, _delta in fit_contributions
    ]
    np.testing.assert_array_equal(
        [value for _name, value, _delta in artifact_contributions],
        [value for _name, value, _delta in fit_contributions],
    )
    np.testing.assert_allclose(
        [delta for _name, _value, delta in artifact_contributions],
        [delta for _name, _value, delta in fit_contributions],
        rtol=_FLOAT_REL_TOL,
        atol=_FLOAT_ABS_TOL,
    )
    assert math.isclose(
        artifact_model.threshold,
        fit_model.threshold,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )
    np.testing.assert_allclose(
        artifact_model.feature_importances,
        fit_model.feature_importances,
        rtol=_FLOAT_REL_TOL,
        atol=_FLOAT_ABS_TOL,
    )
    _assert_payload_close(
        dict(artifact_evaluation.metrics), dict(fit_evaluation.metrics)
    )
    assert (artifact_evaluation.calibration is None) == (
        fit_evaluation.calibration is None
    )
    if artifact_evaluation.calibration is not None:
        _assert_payload_close(
            dict(artifact_evaluation.calibration),
            dict(fit_evaluation.calibration),
        )
    assert artifact_model.card.corpus_digest == fit_model.card.corpus_digest
    assert artifact_model.card.artifact_digest
    assert fit_model.card.artifact_digest == ""

    if _compatible():
        # The checked-in artifact must serve predictions consistent with a fit
        # in the same library minor.
        _checked_evaluation, checked_model = artifact.load_artifact()
        np.testing.assert_allclose(
            checked_model.predict_matrix(matrix),
            fit_model.predict_matrix(matrix),
            rtol=0.0,
            atol=1e-9,
        )


def test_model_card_provenance_round_trips_and_old_cards_still_validate(
    tmp_path, fresh_fit
):
    _fit_evaluation, _fit_model, payload = fresh_fit
    _evaluation, model = _load_payload(payload, tmp_path)
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


def test_payload_comparison_accepts_ulp_noise_but_not_real_change():
    payload = _checked_in()
    noisy = json.loads(json.dumps(payload))
    for values in (noisy["model"]["coefficients"], noisy["model"]["scaler_mean"]):
        values[0] = math.nextafter(values[0], math.inf)

    _assert_payload_close(noisy, payload)

    changed = json.loads(json.dumps(payload))
    changed["model"]["coefficients"][0] += 1e-3
    with pytest.raises(AssertionError):
        _assert_payload_close(changed, payload)
    # Structural-only comparison ignores numbers but still checks shape.
    _assert_payload_close(changed, payload, numeric=False)


def _write_variant(tmp_path, base, mutate, *, restamp: bool) -> Path:
    payload = json.loads(json.dumps(base))
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
    tmp_path, monkeypatch, fresh_fit, name, mutate, restamp, reason
):
    good_evaluation, good_model, base = fresh_fit
    path = _write_variant(tmp_path, base, mutate, restamp=restamp)
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


def test_non_finite_artifact_values_are_rejected(tmp_path, fresh_fit):
    _evaluation, _model, base = fresh_fit
    payload = json.loads(json.dumps(base))
    payload["model"]["coefficients"][0] = float("nan")
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(artifact.ArtifactError, match="non-finite"):
        artifact.load_artifact(path)


def test_missing_truncated_and_oversized_artifacts_fall_back(
    tmp_path, monkeypatch, fresh_fit
):
    good_evaluation, good_model, _base = fresh_fit
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
