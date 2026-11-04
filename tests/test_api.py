import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from leaklens.api import app

c = TestClient(app)


def _csv():
    r = np.random.default_rng(0)
    n = 300
    x = r.normal(size=n)
    d = pd.DataFrame({"x": x, "y": (x + r.normal(size=n) > 0).astype(int)})
    d["leak"] = d["y"] * 2.0
    return d.to_csv(index=False).encode()


def test_health():
    assert c.get("/health").json()["status"] == "ok"


def test_audit():
    r = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "y"}
    )
    assert r.status_code == 200 and r.json()["trust_score"] < 100


def test_audit_bad_target():
    r = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "nope"}
    )
    assert r.status_code == 422


def test_scan():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
    )
    assert c.post("/scan", json={"source": src}).json()["findings"]


def test_predict_missing_model(tmp_path, monkeypatch):
    # Isolated sandbox: the API only accepts models under ./models/ relative to
    # CWD, so chdir there and request a nonexistent file inside models/.
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(tmp_path)
    r = c.post("/predict", json={"rows": [{"x": 1}], "model_path": "models/x.joblib"})
    assert r.status_code == 404


def test_predict_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from leaklens.arena import run

    monkeypatch.chdir(tmp_path)
    r = np.random.default_rng(0)
    n = 400
    x = r.normal(size=n)
    d = pd.DataFrame(
        {
            "x": x,
            "c": r.choice(["a", "b"], n),
            "y": (x + r.normal(size=n) > 0).astype(int),
        }
    )
    models = tmp_path / "models"
    models.mkdir()
    run(
        d,
        "y",
        out_dir=str(models),
        tracking_uri=f"sqlite:///{(tmp_path / 'm.db').as_posix()}",
    )
    # The API resolves relative to CWD (tmp_path); the model sits in ./models/.
    resp = c.post(
        "/predict",
        json={
            "rows": [{"x": 2.0, "c": "a"}, {"x": -2.0, "c": "b"}],
            "model_path": "models/best.joblib",
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2
