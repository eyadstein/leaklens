"""Tests for release tag/version, distribution completeness and workflow refs."""

import io
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.validate_release import (
    expected_members,
    main,
    package_version,
    sdist_version,
    validate,
    validate_metadata,
    validate_sdist,
    validate_wheel,
    wheel_version,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _wheel_names(version="0.13.0"):
    return [
        "leaklens/__init__.py",
        "leaklens/leakage.py",
        "leaklens/ai/semantics.py",
        f"leaklens_ml-{version}.dist-info/METADATA",
        f"leaklens_ml-{version}.dist-info/WHEEL",
        f"leaklens_ml-{version}.dist-info/RECORD",
        f"leaklens_ml-{version}.dist-info/licenses/LICENSE",
    ]


def _source_tree(root: Path, *, version="0.13.0", prompt=True):
    """A miniature package tree with one module, one data file and one prompt."""
    (root / "leaklens" / "ai").mkdir(parents=True)
    (root / "leaklens" / "__init__.py").write_text("", encoding="utf-8")
    (root / "leaklens" / "leakage.py").write_text("", encoding="utf-8")
    (root / "leaklens" / "ai" / "semantics.py").write_text("", encoding="utf-8")
    (root / "leaklens" / "ai" / "model.json").write_text("{}", encoding="utf-8")
    (root / "leaklens" / "prompts").mkdir()
    if prompt:
        (root / "leaklens" / "prompts" / "explain.txt").write_text(
            "x", encoding="utf-8"
        )
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "leaklens-ml"\nversion = "{version}"\n'
        '[tool.setuptools.package-data]\n"leaklens.ai" = ["*.json"]\n',
        encoding="utf-8",
    )
    (root / "MANIFEST.in").write_text(
        "recursive-include leaklens/prompts *.txt\n", encoding="utf-8"
    )
    return root


def _metadata(version="0.13.0", name="leaklens-ml"):
    return f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"


def _complete_wheel(root: Path, version="0.13.0"):
    return [
        *expected_members(root),
        *(
            f"leaklens_ml-{version}.dist-info/{meta}"
            for meta in ("METADATA", "WHEEL", "RECORD")
        ),
    ]


def _write_wheel(path: Path, names, *, root: Path | None = None, metadata=None):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            if metadata and name.endswith("/METADATA"):
                archive.writestr(name, metadata)
            elif root is not None and (root / name).is_file():
                archive.writestr(name, (root / name).read_bytes())
            else:
                archive.writestr(name, "")
    return path


def _sdist_names(root: Path, version="0.13.0"):
    prefix = f"leaklens_ml-{version}/"
    return [
        f"leaklens_ml-{version}",
        *(prefix + meta for meta in ("PKG-INFO", "pyproject.toml", "MANIFEST.in")),
        *(prefix + name for name in expected_members(root, sdist=True)),
    ]


def _write_sdist(path: Path, names, *, root: Path | None = None, version="0.13.0"):
    prefix = f"leaklens_ml-{version}/"
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            if name == f"leaklens_ml-{version}":
                archive.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
                continue
            relative = name[len(prefix) :] if name.startswith(prefix) else ""
            if name.endswith("/PKG-INFO"):
                payload = _metadata(version).encode("utf-8")
            elif root is not None and relative and (root / relative).is_file():
                payload = (root / relative).read_bytes()
            else:
                payload = b""
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _pyproject(tmp_path, version="0.13.0"):
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    return path


def test_valid_release_tag_matches_package_version(tmp_path):
    path = _pyproject(tmp_path)
    assert package_version(path) == "0.13.0"
    validate("v0.13.0", "0.13.0", "release", "refs/tags/v0.13.0")


def test_mismatched_tag_version_rejected():
    with pytest.raises(ValueError, match="does not match"):
        validate("v0.12.0", "0.13.0")


@pytest.mark.parametrize(
    "tag",
    ["", "0.13.0", "v0.13", "v0.13.0.0", "v0.13.0-rc1", "release-0.13.0"],
)
def test_invalid_or_missing_tag_rejected(tag):
    with pytest.raises(ValueError):
        validate(tag, "0.13.0")


def test_manual_dispatch_from_main_allowed():
    validate("v0.13.0", "0.13.0", "workflow_dispatch", "refs/heads/main")


def test_manual_dispatch_from_branch_rejected():
    with pytest.raises(ValueError, match="refs/heads/main"):
        validate("v0.13.0", "0.13.0", "workflow_dispatch", "refs/heads/feature")


def test_release_event_ref_must_match_tag():
    with pytest.raises(ValueError, match="does not match tag"):
        validate("v0.13.0", "0.13.0", "release", "refs/tags/v0.12.9")


def test_cli_accepts_valid_tag(tmp_path, capsys):
    path = _pyproject(tmp_path)
    argv = [
        "--tag",
        "v0.13.0",
        "--pyproject",
        str(path),
        "--event",
        "release",
        "--ref",
        "refs/tags/v0.13.0",
    ]
    assert main(argv) == 0
    assert "matches package version 0.13.0" in capsys.readouterr().out


def test_cli_rejects_mismatch_and_unauthorized_dispatch(tmp_path, capsys):
    path = _pyproject(tmp_path)
    assert main(["--tag", "v0.12.0", "--pyproject", str(path)]) == 1
    assert "does not match" in capsys.readouterr().err

    argv = [
        "--tag",
        "v0.13.0",
        "--pyproject",
        str(path),
        "--event",
        "workflow_dispatch",
        "--ref",
        "refs/heads/feature",
    ]
    assert main(argv) == 1
    assert "refs/heads/main" in capsys.readouterr().err


def test_wheel_allowlist_accepts_package_and_dist_info():
    validate_wheel(_wheel_names(), "0.13.0")
    validate_wheel(_wheel_names("0.14.0"), "0.14.0")


def test_wheel_allowlist_rejects_anything_else():
    names = [*_wheel_names(), "tests/secret.csv"]
    with pytest.raises(ValueError, match=r"tests/secret\.csv"):
        validate_wheel(names, "0.13.0")


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "README.md",
        "leaklens/../evil.py",
        "/etc/passwd",
        "leaklens_ml-0.13.0.dist-info/../../evil.py",
        "leaklens_other-0.13.0.dist-info/METADATA",
    ],
)
def test_wheel_allowlist_rejects_odd_entries(name):
    with pytest.raises(ValueError):
        validate_wheel([*_wheel_names(), name], "0.13.0")


def test_wheel_allowlist_rejects_an_empty_archive():
    with pytest.raises(ValueError, match="empty"):
        validate_wheel([], "0.13.0")


def test_wheel_version_comes_from_the_filename(tmp_path):
    wheel = tmp_path / "leaklens_ml-0.14.0-py3-none-any.whl"
    assert wheel_version(wheel) == "0.14.0"
    with pytest.raises(ValueError, match="not a leaklens wheel"):
        wheel_version(tmp_path / "other-0.14.0-py3-none-any.whl")


def test_cli_wheel_mode_is_independent_of_the_tag(tmp_path, capsys):
    root = _source_tree(tmp_path / "src")
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    _write_wheel(wheel, _complete_wheel(root), root=root, metadata=_metadata())
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 0
    assert "expected package files" in capsys.readouterr().out

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tests/secret.csv", "x")
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 1
    assert "tests/secret.csv" in capsys.readouterr().err


def test_cli_wheel_mode_fails_on_an_incomplete_distribution(tmp_path, capsys):
    root = _source_tree(tmp_path / "src")
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    names = [n for n in _complete_wheel(root) if n != "leaklens/leakage.py"]
    _write_wheel(wheel, names, root=root, metadata=_metadata())
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "missing required file" in err
    assert "leaklens/leakage.py" in err


def test_expected_members_follow_the_tree_and_packaging_config(tmp_path):
    root = _source_tree(tmp_path / "src")
    cache = root / "leaklens" / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("", encoding="utf-8")

    assert expected_members(root) == [
        "leaklens/__init__.py",
        "leaklens/ai/model.json",
        "leaklens/ai/semantics.py",
        "leaklens/leakage.py",
        "leaklens/prompts/explain.txt",
    ]


def test_expected_members_rejects_a_tree_without_the_package(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a package directory"):
        expected_members(tmp_path)


def test_wheel_requires_every_source_module(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [n for n in _complete_wheel(root) if n != "leaklens/ai/semantics.py"]
    with pytest.raises(ValueError, match=re.escape("leaklens/ai/semantics.py")):
        validate_wheel(names, "0.13.0", required=expected_members(root))


def test_wheel_requires_package_resources(tmp_path):
    root = _source_tree(tmp_path / "src")
    resource = "leaklens/prompts/explain.txt"
    names = [n for n in _complete_wheel(root) if n != resource]
    with pytest.raises(ValueError, match=re.escape("leaklens/prompts/explain.txt")):
        validate_wheel(names, "0.13.0", required=expected_members(root))


def test_wheel_rejects_duplicate_members(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [*_complete_wheel(root), "leaklens/leakage.py"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_wheel(names, "0.13.0", required=expected_members(root))


@pytest.mark.parametrize("meta", ["METADATA", "WHEEL", "RECORD"])
def test_wheel_requires_dist_info_metadata(meta):
    names = [n for n in _wheel_names() if not n.endswith(f"/{meta}")]
    with pytest.raises(ValueError, match=meta):
        validate_wheel(names, "0.13.0")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_metadata(name="other"), "does not match package"),
        (_metadata(version="9.9.9"), "does not match"),
        ("Metadata-Version: 2.1\n", "does not match package"),
    ],
)
def test_metadata_must_match_name_and_version(text, match):
    with pytest.raises(ValueError, match=match):
        validate_metadata(text, "0.13.0")


def test_validation_never_imports_archive_members(tmp_path):
    root = _source_tree(tmp_path / "src")
    (root / "leaklens" / "__init__.py").write_text(
        "raise RuntimeError('must never be imported')\n", encoding="utf-8"
    )
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    _write_wheel(wheel, _complete_wheel(root), root=root, metadata=_metadata())
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 0


def test_cli_wheel_rejects_a_version_mismatch_with_the_source(tmp_path, capsys):
    root = _source_tree(tmp_path / "src", version="0.14.0")
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    _write_wheel(
        wheel, _complete_wheel(root, "0.13.0"), root=root, metadata=_metadata()
    )
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 1
    assert "does not match package version" in capsys.readouterr().err


def test_sdist_accepts_a_complete_archive(tmp_path, capsys):
    root = _source_tree(tmp_path / "src")
    sdist = tmp_path / "leaklens_ml-0.13.0.tar.gz"
    _write_sdist(sdist, _sdist_names(root), root=root)
    assert main(["--sdist", str(sdist), "--source-root", str(root)]) == 0
    assert "no build junk" in capsys.readouterr().out


def test_sdist_requires_every_source_module(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [n for n in _sdist_names(root) if not n.endswith("leaklens/leakage.py")]
    with pytest.raises(ValueError, match=re.escape("leaklens/leakage.py")):
        validate_sdist(names, "0.13.0", required=expected_members(root))


@pytest.mark.parametrize(
    "junk",
    [
        "leaklens_ml-0.13.0/graphify-out/GRAPH_REPORT.md",
        "leaklens_ml-0.13.0/.phase5-bench.json",
        "leaklens_ml-0.13.0/dist/leaklens_ml-0.13.0-py3-none-any.whl",
        "leaklens_ml-0.13.0/build/lib/leaklens/__init__.py",
        "leaklens_ml-0.13.0/leaklens/__pycache__/cli.cpython-311.pyc",
        "leaklens_ml-0.13.0/leaklens/stale.pyc",
    ],
)
def test_sdist_rejects_build_junk(tmp_path, junk):
    root = _source_tree(tmp_path / "src")
    names = [*_sdist_names(root), junk]
    with pytest.raises(ValueError, match="unexpected"):
        validate_sdist(names, "0.13.0", required=expected_members(root))


@pytest.mark.parametrize("meta", ["PKG-INFO", "pyproject.toml", "MANIFEST.in"])
def test_sdist_requires_metadata(tmp_path, meta):
    root = _source_tree(tmp_path / "src")
    names = [n for n in _sdist_names(root) if not n.endswith(f"/{meta}")]
    with pytest.raises(ValueError, match=meta):
        validate_sdist(names, "0.13.0", required=expected_members(root))


def test_sdist_rejects_a_foreign_root(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [*_sdist_names(root), "other_pkg-0.13.0/evil.py"]
    with pytest.raises(ValueError, match="unexpected"):
        validate_sdist(names, "0.13.0", required=expected_members(root))


def test_sdist_rejects_duplicate_members(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [*_sdist_names(root), "leaklens_ml-0.13.0/PKG-INFO"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_sdist(names, "0.13.0", required=expected_members(root))


def test_sdist_version_comes_from_the_filename(tmp_path):
    sdist = tmp_path / "leaklens_ml-0.14.0.tar.gz"
    assert sdist_version(sdist) == "0.14.0"
    with pytest.raises(ValueError, match="not a leaklens sdist"):
        sdist_version(tmp_path / "other-0.14.0.tar.gz")


def test_wheel_rejects_a_package_member_missing_from_the_tree(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [*_complete_wheel(root), "leaklens/legacy_detector.py"]
    with pytest.raises(ValueError, match=re.escape("leaklens/legacy_detector.py")):
        validate_wheel(names, "0.13.0", required=expected_members(root))


def test_sdist_rejects_a_package_member_missing_from_the_tree(tmp_path):
    root = _source_tree(tmp_path / "src")
    names = [*_sdist_names(root), "leaklens_ml-0.13.0/leaklens/legacy_detector.py"]
    with pytest.raises(ValueError, match=re.escape("leaklens/legacy_detector.py")):
        validate_sdist(names, "0.13.0", required=expected_members(root, sdist=True))


@pytest.mark.parametrize(
    "junk",
    [".env", "models/best.joblib", "mlflow.db", "demo.csv", "tests/secret.csv"],
)
def test_sdist_rejects_local_data_and_secret_files(tmp_path, junk):
    root = _source_tree(tmp_path / "src")
    names = [*_sdist_names(root), f"leaklens_ml-0.13.0/{junk}"]
    with pytest.raises(ValueError, match="unexpected"):
        validate_sdist(names, "0.13.0", required=expected_members(root, sdist=True))


def test_wheel_content_must_match_the_source_tree(tmp_path, capsys):
    root = _source_tree(tmp_path / "src")
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    _write_wheel(wheel, _complete_wheel(root), root=root, metadata=_metadata())
    (root / "leaklens" / "leakage.py").write_text("stale\n", encoding="utf-8")
    assert main(["--wheel", str(wheel), "--source-root", str(root)]) == 1
    assert "does not match the source tree" in capsys.readouterr().err


def test_sdist_content_must_match_the_source_tree(tmp_path, capsys):
    root = _source_tree(tmp_path / "src")
    sdist = tmp_path / "leaklens_ml-0.13.0.tar.gz"
    _write_sdist(sdist, _sdist_names(root), root=root)
    (root / "leaklens" / "leakage.py").write_text("stale\n", encoding="utf-8")
    assert main(["--sdist", str(sdist), "--source-root", str(root)]) == 1
    assert "does not match the source tree" in capsys.readouterr().err


def test_metadata_body_lines_do_not_override_header_fields():
    validate_metadata(_metadata() + "\nName: other\nVersion: 9.9.9\n", "0.13.0")


def test_manifest_include_accepts_multiple_patterns(tmp_path):
    root = _source_tree(tmp_path / "src")
    (root / "leaklens" / "data").mkdir()
    (root / "leaklens" / "data" / "a.bin").write_text("a", encoding="utf-8")
    (root / "leaklens" / "data" / "b.bin").write_text("b", encoding="utf-8")
    (root / "MANIFEST.in").write_text("include leaklens/data/*.bin\n", encoding="utf-8")
    members = set(expected_members(root))
    assert {"leaklens/data/a.bin", "leaklens/data/b.bin"} <= members


def test_manifest_graft_pulls_a_whole_directory(tmp_path):
    root = _source_tree(tmp_path / "src")
    (root / "leaklens" / "data" / "deep").mkdir(parents=True)
    (root / "leaklens" / "data" / "deep" / "note.rst").write_text("n", encoding="utf-8")
    (root / "MANIFEST.in").write_text("graft leaklens/data\n", encoding="utf-8")
    assert "leaklens/data/deep/note.rst" in expected_members(root)


def test_manifest_global_exclude_removes_expected_members(tmp_path):
    root = _source_tree(tmp_path / "src")
    (root / "MANIFEST.in").write_text(
        "recursive-include leaklens/prompts *.txt\nglobal-exclude explain.txt\n",
        encoding="utf-8",
    )
    assert "leaklens/prompts/explain.txt" not in expected_members(root)


def test_root_manifest_include_is_sdist_only(tmp_path):
    root = _source_tree(tmp_path / "src")
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "MANIFEST.in").write_text(
        "recursive-include leaklens/prompts *.txt\ninclude README.md\n",
        encoding="utf-8",
    )
    assert "README.md" in expected_members(root, sdist=True)
    assert "README.md" not in expected_members(root)


def test_every_workflow_action_is_sha_pinned():
    pattern = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+?)(?:\s*#.*)?$", re.MULTILINE)
    unpinned = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for ref in pattern.findall(path.read_text(encoding="utf-8")):
            if "@" not in ref or not re.search(r"@[0-9a-f]{40}$", ref):
                unpinned.append(f"{path.name}: {ref}")
    assert not unpinned, "unpinned action(s):\n" + "\n".join(unpinned)
