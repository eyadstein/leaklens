"""SBOM output-path contract: safe default, custom path, never inside dist/."""

import json

import pytest

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


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_writes_nothing(tmp_path, monkeypatch, capsys, flag):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert generate_sbom.main(["generate_sbom.py", flag]) == 0
    out = capsys.readouterr().out
    assert "usage: generate_sbom.py" in out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("flag", ["--nope", "-o"])
def test_unknown_option_exits_two_and_writes_nothing(
    tmp_path, monkeypatch, capsys, flag
):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert generate_sbom.main(["generate_sbom.py", flag]) == 2
    err = capsys.readouterr().err
    assert "usage: generate_sbom.py" in err and flag in err
    assert list(tmp_path.iterdir()) == []


def test_two_output_paths_are_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert generate_sbom.main(["generate_sbom.py", str(first), str(second)]) == 2
    assert not first.exists() and not second.exists()
    assert "only one output path" in capsys.readouterr().err


FIXTURE_PROJECT = """\
[project]
name = "leaklens-ml"
version = "9.9.9"
description = "fixture"
requires-python = ">=3.10"
dependencies = ["alpha>=1", "missing-pkg"]

[project.optional-dependencies]
app = ["beta"]
dev = ["pytest"]
"""

# alpha -> gamma -> shared -> alpha is a cycle; beta joins at gamma and shared.
FIXTURE_REQUIRES = {
    "alpha": ["gamma>=3"],
    "beta": ["gamma>=3", "shared"],
    "gamma": ["shared", 'tomli; python_version < "3.11"'],
    "shared": ["alpha"],
}
FIXTURE_VERSIONS = {"alpha": "1.0", "beta": "2.0", "gamma": "3.0", "shared": "4.0"}


def _fixture(tmp_path, monkeypatch, *, versions=FIXTURE_VERSIONS):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(FIXTURE_PROJECT, encoding="utf-8")
    monkeypatch.setattr(generate_sbom, "ROOT", tmp_path)
    monkeypatch.setattr(generate_sbom, "PYPROJECT", pyproject)

    def version(name):
        if name not in versions:
            raise generate_sbom.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(generate_sbom.metadata, "version", version)
    monkeypatch.setattr(
        generate_sbom.metadata,
        "requires",
        lambda name: FIXTURE_REQUIRES.get(name, []),
    )
    return pyproject


def _by_name(bom):
    return {component["name"]: component for component in bom["components"]}


def _edges(bom):
    return {
        dependency["ref"]: dependency["dependsOn"] for dependency in bom["dependencies"]
    }


def test_transitive_dependencies_become_components_and_edges(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    bom = generate_sbom.build_sbom()
    components = _by_name(bom)

    # gamma and shared are not declared anywhere in pyproject; they exist only
    # in installed metadata, and the marker-guarded tomli is not graphed.
    assert sorted(components) == ["alpha", "beta", "gamma", "missing-pkg", "shared"]
    assert "tomli" not in components
    assert all(
        component["purl"].startswith("pkg:pypi/") for component in components.values()
    )
    assert components["gamma"]["version"] == "3.0"

    # beta is only reachable through the app extra; the rest is runtime.
    assert components["alpha"]["scope"] == "required"
    assert components["gamma"]["scope"] == "required"
    assert components["shared"]["scope"] == "required"
    assert components["missing-pkg"]["scope"] == "required"
    assert components["beta"]["scope"] == "optional"

    edges = _edges(bom)
    root = bom["metadata"]["component"]["bom-ref"]
    assert edges[root] == sorted(
        [components["alpha"]["bom-ref"], components["missing-pkg"]["bom-ref"]]
    )
    assert edges[components["alpha"]["bom-ref"]] == [components["gamma"]["bom-ref"]]
    assert edges[components["beta"]["bom-ref"]] == sorted(
        [components["gamma"]["bom-ref"], components["shared"]["bom-ref"]]
    )
    # The cycle terminates instead of repeating, and every node has one entry.
    assert edges[components["shared"]["bom-ref"]] == [components["alpha"]["bom-ref"]]
    assert edges[components["missing-pkg"]["bom-ref"]] == []
    assert len(edges) == len(components) + 1


def test_graph_completeness_is_declared_not_assumed(tmp_path, monkeypatch, capsys):
    _fixture(tmp_path, monkeypatch)
    out = tmp_path / "sbom.json"
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 0
    properties = {
        item["name"]: item["value"]
        for item in json.loads(out.read_text(encoding="utf-8"))["metadata"][
            "properties"
        ]
    }
    assert properties["sbom-graph"] == "declared-only"
    assert properties["sbom-conditional-requirements"] == "1"

    # With every declared package installed the same graph reads complete.
    _fixture(
        tmp_path,
        monkeypatch,
        versions={**FIXTURE_VERSIONS, "missing-pkg": "0.1"},
    )
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 0
    properties = {
        item["name"]: item["value"]
        for item in json.loads(out.read_text(encoding="utf-8"))["metadata"][
            "properties"
        ]
    }
    assert properties["sbom-graph"] == "complete"


def test_generation_is_byte_identical_across_runs(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert generate_sbom.main(["generate_sbom.py", str(first)]) == 0
    assert generate_sbom.main(["generate_sbom.py", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_the_node_cap_fails_loudly_instead_of_truncating(tmp_path, monkeypatch, capsys):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(generate_sbom, "MAX_COMPONENTS", 3)
    out = tmp_path / "sbom.json"
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 1
    assert not out.exists()
    assert "exceeds 3 components" in capsys.readouterr().err


def test_the_depth_cap_fails_loudly_instead_of_truncating(
    tmp_path, monkeypatch, capsys
):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(generate_sbom, "MAX_DEPTH", 0)
    out = tmp_path / "sbom.json"
    assert generate_sbom.main(["generate_sbom.py", str(out)]) == 1
    assert not out.exists()
    assert "exceeds 0 levels" in capsys.readouterr().err
