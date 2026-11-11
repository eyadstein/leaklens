"""The prediction endpoint deserializes only the server-fixed model (M-S0 G1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from leaklens import api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def _clear_model_cache():
    api._load_model.cache_clear()
    yield
    api._load_model.cache_clear()


def test_request_selected_model_is_refused_without_opening_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAKLENS_MODEL_PATH", raising=False)

    def _must_not_load(*args, **kwargs):
        raise AssertionError("a request-selected path reached joblib.load")

    monkeypatch.setattr(api.joblib, "load", _must_not_load)
    for value in ("../../evil.joblib", "models/other.joblib", "C:/Windows/x.joblib"):
        response = client.post(
            "/predict", json={"rows": [{"x": 1}], "model_path": value}
        )
        assert response.status_code == 422, response.text
        assert "fixed by the server" in response.text


def test_fixed_model_is_loaded_and_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAKLENS_MODEL_PATH", raising=False)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "best.joblib").write_bytes(b"placeholder")

    class _Prep:
        transformers_ = (("num", object(), ["x"]), ("cat", object(), []))

    class _Pipe:
        def __init__(self):
            self.named_steps = {"prep": _Prep()}

        def predict(self, frame):
            return np.asarray(frame["x"], dtype=float) * 2

    bundle = {
        "features": ["x"],
        "pipeline": _Pipe(),
        "task": "regression",
        "model": "stub",
    }
    monkeypatch.setattr(api.joblib, "load", lambda path: bundle)

    for _ in range(2):
        response = client.post("/predict", json={"rows": [{"x": 2.0}]})
        assert response.status_code == 200, response.text
        assert response.json()["predictions"] == [{"value": 4.0}]


def test_error_bodies_never_echo_the_model_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    secret = tmp_path / "private" / "model.joblib"
    monkeypatch.setenv("LEAKLENS_MODEL_PATH", str(secret))
    response = client.post(
        "/predict", json={"rows": [{"x": 1}], "model_path": str(secret)}
    )
    assert response.status_code == 404
    assert str(secret) not in response.text
