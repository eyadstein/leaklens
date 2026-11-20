"""SBOM output-path contract: safe default, custom path, never inside dist/."""

import json

from scripts import generate_sbom


def test_default_output_is_outside_dist(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    assert generate_sbom.main(["generate_sbom.py"]) == 0
    out = tmp_path / "sbom" / "sbom.json"
    assert out.is_file()
    assert not (tmp_path / "dist").exists()


def test_default_output_keeps_cyclonedx_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    assert generate_sbom.main(["generate_sbom.py"]) == 0
    bom = json.loads((tmp_path / "sbom" / "sbom.json").read_text(encoding="utf-8"))
    assert bom["bomFormat"] == "CycloneDX"
    # CycloneDX JSON uses camelCase; the XML-only hyphenated key must not return.
    assert "bom-format" not in bom
    assert {"bomFormat", "specVersion", "version"} <= bom.keys()
    assert bom["specVersion"] == "1.5"
    assert bom["metadata"]["component"]["name"] == "leaklens-ml"
    assert bom["components"]


def test_explicit_output_path_still_supported(tmp_path):
    out = tmp_path / "out" / "custom.json"
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 0
    assert out.is_file()


def test_output_inside_dist_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    out = tmp_path / "dist" / "sbom.json"
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 1
    assert not out.exists()
    assert "outside dist/" in capsys.readouterr().err


def test_relative_dist_path_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert generate_sbom.main(["generate_sbom.py", "dist/sbom.json"]) == 1
    assert not (tmp_path / "dist" / "sbom.json").exists()
    assert "outside dist/" in capsys.readouterr().err
