"""Tests for `leaklens doctor` (Phase 1: installation & reliability).

No network access, no secrets, no mutation of the real repo (tmp_path only).
"""

import json

from leaklens.cli import doctor, main


def test_doctor_text_exit_ok(capsys):
    assert doctor([]) == 0
    out = capsys.readouterr().out
    assert "Python" in out
    assert "LeakLens" in out
    assert "Core deps" in out
    assert "DOCTOR OK" in out


def test_doctor_json_shape(capsys):
    assert doctor(["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["python_ok"] is True
    assert data["required_ok"] is True
    assert set(data["extras"]) >= {
        "arena",
        "parquet",
        "feather",
        "excel",
        "images",
        "app",
    }
    assert "extras_available" in data
    assert data["config_valid"] is True
    assert data["entry_point"] == "leaklens:main"


def test_doctor_help(capsys):
    assert doctor(["--help"]) == 0
    assert "usage: leaklens doctor" in capsys.readouterr().out


def test_doctor_via_main_dispatch(capsys):
    assert main(["doctor", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "leaklens_version" in data


def test_doctor_validates_an_explicit_config(tmp_path, capsys):
    valid = tmp_path / "ok.toml"
    valid.write_text('target = "y"\n[missingness]\nmin_rows = 100\n', encoding="utf-8")
    assert doctor(["--config", str(valid), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["config_valid"] is True
    assert data["toml_config_present"] == str(valid)

    bad = tmp_path / "bad.toml"
    bad.write_text("unknown = true\n", encoding="utf-8")
    assert doctor(["--config", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "INVALID" in out and "unknown configuration keys" in out

    assert doctor(["--config"]) == 1


def test_doctor_no_secrets(monkeypatch, capsys):
    monkeypatch.setenv("LEAKLENS_LLM_API_KEY", "supersecretvalue")
    doctor([])
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "supersecretvalue" not in out
