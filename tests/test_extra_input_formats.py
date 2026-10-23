import pandas as pd
import pytest
from fastapi.testclient import TestClient

from leaklens.api import app
from leaklens.cli import main
from leaklens.readers import read_dataset


def test_json_input_supported(tmp_path, capsys):
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    json_path = tmp_path / "data.json"
    df.to_json(json_path, orient="records")

    response = TestClient(app).post(
        "/audit",
        files={"file": ("data.json", json_path.read_bytes(), "application/octet-stream")},
        data={"target": "y"},
    )
    assert response.status_code == 200, response.text

    assert main([str(json_path), "--target", "y", "--json"]) == 0
    assert '"trust_score"' in capsys.readouterr().out


def test_excel_input_supported(tmp_path, capsys):
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    xlsx_path = tmp_path / "data.xlsx"
    df.to_excel(xlsx_path, index=False)

    response = TestClient(app).post(
        "/audit",
        files={"file": ("data.xlsx", xlsx_path.read_bytes(), "application/octet-stream")},
        data={"target": "y"},
    )
    assert response.status_code == 200, response.text

    assert main([str(xlsx_path), "--target", "y", "--json"]) == 0
    assert '"trust_score"' in capsys.readouterr().out


def test_feather_input_supported(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    feather_path = tmp_path / "data.feather"
    df.to_feather(feather_path)

    response = TestClient(app).post(
        "/audit",
        files={"file": ("data.feather", feather_path.read_bytes(), "application/octet-stream")},
        data={"target": "y"},
    )
    assert response.status_code == 200, response.text

    assert main([str(feather_path), "--target", "y", "--json"]) == 0
    assert '"trust_score"' in capsys.readouterr().out


def test_unknown_suffix_falls_back_to_csv(tmp_path, capsys):
    df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    path = tmp_path / "extensionless"
    df.to_csv(path, index=False)

    assert read_dataset(path).equals(df)
    assert main([str(path), "--target", "y", "--json"]) == 0
    assert '"trust_score"' in capsys.readouterr().out


def test_missing_optional_dependency_is_actionable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("leaklens.readers._available", lambda module: False)
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"not really a spreadsheet")

    with pytest.raises(ImportError, match=r'pip install "leaklens-ml\[excel\]"'):
        read_dataset(path)

    with pytest.raises(SystemExit) as exc:
        main([str(path), "--target", "y"])
    assert exc.value.code == 2
    assert "openpyxl" in capsys.readouterr().err
