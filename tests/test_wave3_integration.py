"""Wave 3 integration: config, CLI isolation, API, SARIF and HTML."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from leaklens.cli import main
from leaklens.config import ConfigError, load_config
from leaklens.drift import DriftSettings, audit_drift
from leaklens.leakage import Finding
from leaklens.report import render
from leaklens.sarif import to_sarif

HAS_SQLGLOT = importlib.util.find_spec("sqlglot") is not None


def _config(tmp_path, text):
    path = tmp_path / ".leaklens.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_sql_and_drift_config_are_parsed_strictly(tmp_path):
    cfg = _config(
        tmp_path,
        '[sql]\nenabled = true\ndialect = "postgres"\nsplit_column = "split"\n\n'
        '[sql.schema.orders]\nid = "INT"\nlabel = "INT"\n\n'
        "[drift]\nenabled = true\nmin_rows = 300\nmin_effect = 0.3\n"
        "max_columns = 5\nmax_categories = 4\n",
    )
    assert cfg.sql.enabled is True
    assert cfg.sql.dialect == "postgres"
    assert cfg.sql.split_column == "split"
    assert cfg.sql.schema == {"orders": {"id": "INT", "label": "INT"}}
    assert cfg.drift.enabled is True
    assert cfg.drift.min_rows == 300
    assert cfg.drift.min_effect == 0.3
    assert cfg.drift.max_columns == 5


@pytest.mark.parametrize(
    "text",
    [
        '[sql]\ndialect = "mysql"\n',
        "[sql]\nunknown = true\n",
        "[sql]\nschema = {}\n",
        '[sql]\nschema = { orders = "id INT" }\n',
        "[drift]\nmin_effect = 0\n",
        "[drift]\nmin_rows = 5\n",
        "[drift]\nunknown = 1\n",
    ],
)
def test_invalid_sql_and_drift_config_fail_closed(tmp_path, text):
    with pytest.raises(ConfigError):
        _config(tmp_path, text)


def _frames():
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, n)})
    test = pd.DataFrame({"x": rng.normal(3.0, 1.0, n), "y": rng.integers(0, 2, n)})
    return train, test


def _run_cli(tmp_path, capsys, *extra):
    train, test = _frames()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    code = main(
        [
            str(train_path),
            "--target",
            "y",
            "--test",
            str(test_path),
            "--json",
            *extra,
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def test_cli_drift_is_opt_in_and_does_not_change_the_score(tmp_path, capsys):
    base_code, base = _run_cli(tmp_path, capsys)
    drift_code, drifted = _run_cli(tmp_path, capsys, "--drift")
    assert base_code == drift_code
    assert "drift" not in base
    assert drifted["drift"]["status"] == "ok"
    assert drifted["trust_score"] == base["trust_score"]
    shifted = [f for f in drifted["findings"] if f["kind"] == "distribution_shift"]
    assert shifted and all(f["diagnostic"] is True for f in shifted)


def test_cli_drift_without_test_file_is_reported_as_skipped(tmp_path, capsys):
    train, _ = _frames()
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    assert main([str(path), "--target", "y", "--drift", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["drift"]["status"] == "skipped"


def test_sarif_marks_diagnostics_additively():
    finding = Finding(
        "low",
        "distribution_shift",
        "x",
        "shifted",
        diagnostic=True,
    )
    payload = to_sarif([finding])
    result = payload["runs"][0]["results"][0]
    assert result["properties"]["leaklens"]["diagnostic"] is True
    normal = to_sarif([Finding("low", "id_like", "x", "identifier")])
    assert "diagnostic" not in normal["runs"][0]["results"][0]["properties"]["leaklens"]


def test_html_report_renders_drift_only_when_present():
    base = render({"trust_score": 100, "findings": []})
    assert "Distribution drift" not in base
    report = render(
        {
            "trust_score": 100,
            "findings": [
                {
                    "severity": "low",
                    "kind": "distribution_shift",
                    "column": "x",
                    "detail": "shifted",
                    "diagnostic": True,
                }
            ],
            "drift": {
                "status": "ok",
                "checked_columns": 2,
                "skipped": {"small_sample": 1},
            },
        }
    )
    assert "Distribution drift" in report
    assert "ignored by the trust score" in report


def test_api_audit_drift_matches_the_non_drift_score():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from leaklens.api import app

    train, test = _frames()
    client = TestClient(app)
    files = {
        "file": ("train.csv", train.to_csv(index=False).encode(), "text/csv"),
        "test": ("test.csv", test.to_csv(index=False).encode(), "text/csv"),
    }
    plain = client.post("/audit", files=files, data={"target": "y"})
    drifted = client.post("/audit", files=files, data={"target": "y", "drift": "true"})
    assert plain.status_code == 200 and drifted.status_code == 200
    assert plain.json()["trust_score"] == drifted.json()["trust_score"]
    assert "drift" not in plain.json()
    assert drifted.json()["drift"]["status"] == "ok"


@pytest.mark.skipif(not HAS_SQLGLOT, reason="sqlglot extra not installed")
def test_api_sql_scan_is_optional_and_conservative():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from leaklens.api import app

    client = TestClient(app)
    response = client.post(
        "/scan",
        json={
            "source": "SELECT id, label * 2 AS score FROM orders",
            "name": "query.sql",
            "dialect": "postgres",
            "target": "label",
        },
    )
    assert response.status_code == 200
    kinds = {f["kind"] for f in response.json()["findings"]}
    assert "sql_target_as_feature" in kinds
    bad = client.post(
        "/scan",
        json={"source": "SELECT 1", "name": "query.sql", "dialect": "mysql"},
    )
    assert bad.status_code == 422


def test_drift_settings_are_immutable_defaults():
    settings = DriftSettings()
    assert settings.enabled is False
    report = audit_drift(pd.DataFrame({"x": [1.0] * 300}), settings=settings)
    assert report.status == "disabled"
