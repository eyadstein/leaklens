"""Versioned JSON artifact for the frozen AI model.

The serving model is a small linear pipeline, so the artifact stores its fitted
numbers instead of a pickled estimator.  Plain JSON cannot execute code on load
and stays inspectable, while rebuilding scikit-learn objects from the recorded
parameters keeps prediction and local attribution identical to an in-process
fit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .corpus import training_cases, validation_cases
from .features import MODEL_FEATURE_NAMES
from .schemas import ModelCard

SCHEMA_VERSION = "leaklens-ai-artifact-1"
ARTIFACT_PATH = Path(__file__).with_name("model.json")
MAX_ARTIFACT_BYTES = 64 * 1024


class ArtifactError(ValueError):
    """An artifact is missing, corrupt or was built for another model/corpus."""


def _reject_constant(value: str) -> None:
    raise ArtifactError(f"artifact contains the non-finite value {value!r}")


def _digest(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def corpus_digest() -> str:
    """Digest the development corpus (train + validation) that produced the model."""

    cases = [
        [
            case.split,
            case.name,
            case.group,
            case.source,
            case.category,
            case.leaky,
            case.code,
        ]
        for split in (training_cases(), validation_cases())
        for case in split
    ]
    return _digest(cases)


def feature_catalogue_digest() -> str:
    """Digest the ordered model feature catalogue."""

    return _digest(list(MODEL_FEATURE_NAMES))


def library_version() -> str:
    return sklearn.__version__


def _minor(version: object) -> str:
    return ".".join(str(version).split(".")[:2])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def _floats(section: dict[str, Any], key: str, length: int) -> np.ndarray:
    values = section.get(key)
    _require(isinstance(values, list), f"artifact field {key!r} is missing")
    _require(len(values) == length, f"artifact field {key!r} has the wrong length")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact field {key!r} is not numeric: {exc}") from exc
    _require(bool(np.all(np.isfinite(array))), f"artifact field {key!r} is not finite")
    return array


def _float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"artifact field {key!r} is not numeric",
    )
    number = float(value)
    _require(np.isfinite(number), f"artifact field {key!r} is not finite")
    return number


def _evaluation_dict(evaluation: Any) -> dict[str, Any]:
    data = asdict(evaluation)
    data["candidates"] = [dict(row) for row in evaluation.candidates]
    return data


def build_payload(evaluation: Any, model: Any) -> dict[str, Any]:
    """Serialize a fitted model and its evaluation into a plain JSON payload."""

    estimator = model.model.estimator
    _require(
        isinstance(estimator, Pipeline) and len(estimator.steps) == 2,
        "the serving artifact requires the frozen two-step pipeline",
    )
    scaler = estimator.steps[0][1]
    classifier = estimator.steps[-1][1]
    coefficients = np.asarray(classifier.coef_, dtype=float).ravel()
    _require(
        coefficients.shape[0] == len(MODEL_FEATURE_NAMES),
        "the fitted coefficient vector does not match the feature catalogue",
    )

    corpus = corpus_digest()
    catalogue = feature_catalogue_digest()
    library = library_version()
    card = model.card.to_dict()
    # The digest covers the card too, so the redundant copy is omitted.
    card.pop("artifact_digest", None)
    card.update(
        {
            "corpus_digest": corpus,
            "feature_catalogue_digest": catalogue,
            "library_version": library,
        }
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": model.card.name,
            "version": model.card.version,
            "kind": model.card.kind,
            "threshold": float(model.threshold),
            "features": list(MODEL_FEATURE_NAMES),
            "feature_catalogue_digest": catalogue,
            "corpus_digest": corpus,
            "library_version": library,
            "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
            "coefficients": coefficients.tolist(),
            "intercept": float(
                np.asarray(classifier.intercept_, dtype=float).ravel()[0]
            ),
            "calibrator": {
                "coefficient": float(model.model.calibrator.coefficient),
                "intercept": float(model.model.calibrator.intercept),
            },
            "feature_importances": list(model.feature_importances),
        },
        "evaluation": _evaluation_dict(evaluation),
        "card": card,
    }
    if model.selection is not None:
        payload["model"]["selection"] = model.selection.to_dict()
    # Round-trip so a freshly built payload compares equal to the written file.
    payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    payload["artifact_digest"] = _digest(payload)
    return payload


def write_artifact(
    evaluation: Any, model: Any, path: Path | None = None
) -> tuple[Path, str]:
    """Write the artifact and return its path and digest."""

    target = ARTIFACT_PATH if path is None else Path(path)
    payload = build_payload(evaluation, model)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target, str(payload["artifact_digest"])


def _card(data: dict[str, Any], digest: str) -> ModelCard:
    values = dict(data)
    values["limitations"] = tuple(values.get("limitations", ()))
    for key in ("external", "calibration", "candidates"):
        values[key] = tuple(tuple(item) for item in values.get(key, ()))
    try:
        return replace(ModelCard(**values), artifact_digest=digest)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact model card is invalid: {exc}") from exc


def _from_payload(payload: object) -> tuple[Any, Any]:
    from .learned import (
        MODEL_VERSION,
        LearnedModel,
        ModelEvaluation,
        _CalibratedModel,
        _PlattCalibrator,
    )
    from .schemas import SelectionReport

    _require(isinstance(payload, dict), "artifact payload is not an object")
    payload = cast(dict[str, Any], payload)
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        f"artifact schema {payload.get('schema_version')!r} is not supported",
    )
    recorded = payload.get("artifact_digest")
    body = dict(payload)
    body.pop("artifact_digest", None)
    _require(
        isinstance(recorded, str) and recorded == _digest(body),
        "artifact digest mismatch",
    )

    section = payload.get("model")
    _require(isinstance(section, dict), "artifact model section is missing")
    section = cast(dict[str, Any], section)
    _require(
        section.get("version") == MODEL_VERSION,
        f"artifact model {section.get('version')!r} does not match {MODEL_VERSION!r}",
    )
    _require(
        section.get("feature_catalogue_digest") == feature_catalogue_digest(),
        "artifact feature catalogue digest does not match this build",
    )
    _require(
        section.get("corpus_digest") == corpus_digest(),
        "artifact corpus digest does not match this corpus",
    )
    _require(
        _minor(section.get("library_version")) == _minor(library_version()),
        "artifact was fitted with scikit-learn "
        f"{section.get('library_version')!r}; this build uses {library_version()!r}",
    )

    count = len(MODEL_FEATURE_NAMES)
    _require(
        tuple(section.get("features", ())) == MODEL_FEATURE_NAMES,
        "artifact feature catalogue does not match this build",
    )
    threshold = section.get("threshold")
    _require(
        isinstance(threshold, (int, float))
        and not isinstance(threshold, bool)
        and 0.0 < float(threshold) < 1.0,
        "artifact threshold is not a probability",
    )
    mean = _floats(section, "scaler_mean", count)
    scale = _floats(section, "scaler_scale", count)
    _require(bool(np.all(scale > 0)), "artifact scaler scale is not positive")
    coefficients = _floats(section, "coefficients", count)
    importances = _floats(section, "feature_importances", count)
    calibrator_section = section.get("calibrator")
    _require(
        isinstance(calibrator_section, dict),
        "artifact calibrator is missing",
    )
    calibrator_section = cast(dict[str, Any], calibrator_section)
    calibrator = _PlattCalibrator(
        coefficient=_float(calibrator_section, "coefficient"),
        intercept=_float(calibrator_section, "intercept"),
    )
    intercept = _float(section, "intercept")

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = scale
    scaler.n_features_in_ = count
    classifier = LogisticRegression()
    classifier.coef_ = coefficients.reshape(1, -1)
    classifier.intercept_ = np.asarray([intercept], dtype=float)
    classifier.classes_ = np.asarray([0, 1])
    classifier.n_features_in_ = count
    estimator = Pipeline(
        [("standardscaler", scaler), ("logisticregression", classifier)]
    )

    card_data = payload.get("card")
    _require(isinstance(card_data, dict), "artifact model card is missing")
    card_data = cast(dict[str, Any], card_data)
    card = _card(card_data, str(recorded))

    selection = None
    selection_data = section.get("selection")
    if selection_data is not None:
        _require(
            isinstance(selection_data, dict), "artifact selection decision is invalid"
        )
        values = dict(cast(dict[str, Any], selection_data))
        values["risk_interval"] = tuple(values.get("risk_interval", ()))
        values["limitations"] = tuple(values.get("limitations", ()))
        try:
            selection = SelectionReport(**values)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"artifact selection decision is invalid: {exc}"
            ) from exc

    evaluation_data = payload.get("evaluation")
    _require(isinstance(evaluation_data, dict), "artifact evaluation is missing")
    evaluation_data = cast(dict[str, Any], evaluation_data)
    values = dict(evaluation_data)
    values["candidates"] = tuple(dict(row) for row in values.get("candidates", ()))
    try:
        evaluation = ModelEvaluation(**values)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"artifact evaluation is invalid: {exc}") from exc

    model = LearnedModel(
        model=_CalibratedModel(estimator=estimator, calibrator=calibrator),
        card=card,
        feature_importances=tuple(float(value) for value in importances),
        threshold=float(threshold),
        selection=selection,
    )
    return evaluation, model


def load_artifact(
    path: Path | None = None,
) -> tuple[Any, Any]:
    """Load and verify the artifact, or raise :class:`ArtifactError` with a reason."""

    target = ARTIFACT_PATH if path is None else Path(path)
    try:
        raw = target.read_bytes()
    except FileNotFoundError:
        raise ArtifactError(f"artifact file is missing: {target}") from None
    except OSError as exc:
        raise ArtifactError(f"artifact file could not be read: {exc}") from exc
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"artifact is larger than the {MAX_ARTIFACT_BYTES // 1024} KiB budget"
        )
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"artifact is not valid JSON: {exc}") from exc
    return _from_payload(payload)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the checked-in artifact from the current corpus and features."""

    import argparse

    parser = argparse.ArgumentParser(prog="python -m leaklens.ai.artifact")
    parser.add_argument("--path", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args(argv)

    from .learned import _FROZEN_CANDIDATE, _evaluate

    evaluation, model = _evaluate(include_external=True, candidate=_FROZEN_CANDIDATE)
    path, digest = write_artifact(evaluation, model, args.path)
    print(f"wrote {path} ({path.stat().st_size} bytes, {digest})")
    return 0


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    raise SystemExit(main())
