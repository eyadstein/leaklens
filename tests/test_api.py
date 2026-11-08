import json

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


def _assert_additive_finding(finding):
    assert set(finding) == {
        "severity",
        "kind",
        "column",
        "detail",
        "evidence",
        "fingerprint",
    }
    for field in ("severity", "kind", "column", "detail", "fingerprint"):
        assert isinstance(finding[field], str)
    assert isinstance(finding["evidence"], list)
    assert len(finding["fingerprint"]) == 64


def test_scan_payload_keeps_legacy_fields_and_adds_optional_ones():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
    )
    body = c.post("/scan", json={"source": src}).json()
    assert set(body) == {"trust_score", "findings"}
    assert body["findings"]
    for finding in body["findings"]:
        _assert_additive_finding(finding)


def test_scan_intelligence_is_opt_in_and_additive():
    plain = c.post("/scan", json={"source": BAD_SOURCE}).json()
    response = c.post("/scan", json={"source": BAD_SOURCE, "intelligence": True}).json()

    assert set(response) == {"trust_score", "findings", "intelligence"}
    assert response["trust_score"] == plain["trust_score"]
    assert response["findings"] == plain["findings"]
    graph = response["intelligence"]
    assert set(graph) == {
        "evidence",
        "clusters",
        "links",
        "confidence",
        "root_causes",
        "recommendations",
        "finding_count",
        "impact",
        "engine",
        "ai_enriched",
    }


def test_audit_intelligence_is_opt_in_and_additive():
    plain = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "y"}
    ).json()
    response = c.post(
        "/audit",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={"target": "y", "intelligence": "true"},
    ).json()

    assert "intelligence" not in plain
    assert response["trust_score"] == plain["trust_score"]
    assert response["findings"] == plain["findings"]
    assert response["intelligence"]["clusters"]
    assert response["intelligence"]["links"]


def test_scan_graph_baseline_is_opt_in_and_compatible_with_defaults():
    plain = c.post("/scan", json={"source": BAD_SOURCE}).json()
    first = c.post("/scan", json={"source": BAD_SOURCE, "include_graph_snapshot": True})
    snapshot = first.json()["graph_snapshot"]
    compared = c.post(
        "/scan", json={"source": BAD_SOURCE, "graph_baseline": snapshot}
    ).json()

    assert set(first.json()) == {"trust_score", "findings", "graph_snapshot"}
    assert set(compared) == {"trust_score", "findings", "graph_baseline"}
    assert compared["trust_score"] == plain["trust_score"]
    assert compared["findings"] == plain["findings"]
    assert compared["graph_baseline"]["counts"]["new"] == 0
    assert compared["graph_baseline"]["counts"]["resolved"] == 0
    assert compared["graph_baseline"]["resolution_complete"] is True


def test_scan_rejects_invalid_graph_baseline():
    response = c.post("/scan", json={"source": BAD_SOURCE, "graph_baseline": {"x": 1}})
    assert response.status_code == 422
    assert "invalid graph baseline" in response.json()["detail"]


def test_audit_graph_baseline_is_opt_in_and_additive():
    plain = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "y"}
    ).json()
    first = c.post(
        "/audit",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={"target": "y", "include_graph_snapshot": "true"},
    ).json()
    compared = c.post(
        "/audit",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={
            "target": "y",
            "graph_baseline": json.dumps(first["graph_snapshot"]),
        },
    ).json()

    assert "graph_snapshot" not in plain
    assert "graph_baseline" not in plain
    assert compared["trust_score"] == plain["trust_score"]
    assert compared["findings"] == plain["findings"]
    assert compared["graph_baseline"]["counts"]["new"] == 0


def test_audit_payload_keeps_legacy_fields_and_adds_optional_ones():
    body = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "y"}
    ).json()
    assert {"trust_score", "audit", "findings"} <= set(body)
    assert body["findings"]
    for finding in body["findings"]:
        _assert_additive_finding(finding)
    assert body["missingness"]["status"] in {"ok", "insufficient_data"}
    assert body["missingness"]["findings"] == len(
        [f for f in body["findings"] if f["kind"].startswith("missingness_")]
    )


def _baseline_from(findings):
    return {
        "schema": "leaklens-baseline",
        "version": 1,
        "fingerprint_schema": 1,
        "tool_version": "0.13.0",
        "findings": [
            {
                "fingerprint": f["fingerprint"],
                "kind": f["kind"],
                "severity": f["severity"],
                "column": f["column"],
            }
            for f in findings
        ],
    }


BAD_SOURCE = (
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.model_selection import train_test_split\n"
    "sc = StandardScaler()\nX = sc.fit_transform(X)\na = train_test_split(X)\n"
)


def test_scan_policy_is_additive():
    body = c.post("/scan", json={"source": BAD_SOURCE}).json()
    baseline = _baseline_from(body["findings"])
    baseline["findings"] = baseline["findings"][:1]
    response = c.post("/scan", json={"source": BAD_SOURCE, "baseline": baseline}).json()
    assert set(response) == {"trust_score", "findings", "policy"}
    assert response["policy"]["baseline"]["new"] >= 1
    assert response["policy"]["baseline"]["unchanged"] >= 1
    statuses = {f.get("status") for f in response["findings"]}
    assert statuses <= {"new", "unchanged"}


def test_scan_suppression_is_reported_without_removing_the_finding():
    body = c.post("/scan", json={"source": BAD_SOURCE}).json()
    target = body["findings"][0]
    response = c.post(
        "/scan",
        json={
            "source": BAD_SOURCE,
            "suppressions": [
                {"fingerprint": target["fingerprint"], "reason": "reviewed"}
            ],
        },
    ).json()
    suppressed = [f for f in response["findings"] if f.get("status") == "suppressed"]
    assert [f["fingerprint"] for f in suppressed] == [target["fingerprint"]]
    assert suppressed[0]["severity"] == target["severity"]
    assert suppressed[0]["suppression"]["reason"] == "reviewed"
    assert response["trust_score"] > body["trust_score"]
    assert response["policy"]["suppressions"]["suppressed"] == 1


def test_scan_rejects_invalid_policy():
    baseline = {"schema": "leaklens-baseline", "version": 99}
    response = c.post("/scan", json={"source": BAD_SOURCE, "baseline": baseline})
    assert response.status_code == 422
    response = c.post(
        "/scan",
        json={
            "source": BAD_SOURCE,
            "suppressions": [{"fingerprint": "abc", "reason": "x"}],
        },
    )
    assert response.status_code == 422


def test_audit_accepts_policy_form_fields():
    body = c.post(
        "/audit", files={"file": ("d.csv", _csv(), "text/csv")}, data={"target": "y"}
    ).json()
    baseline = _baseline_from(body["findings"])
    response = c.post(
        "/audit",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={"target": "y", "baseline": json.dumps(baseline)},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["policy"]["baseline"]["resolved"] == 0
    assert "missingness" in payload

    bad = c.post(
        "/audit",
        files={"file": ("d.csv", _csv(), "text/csv")},
        data={"target": "y", "baseline": "{not json"},
    )
    assert bad.status_code == 422


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
