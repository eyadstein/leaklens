"""Tests for release tag/version and workflow ref validation."""

import re
import zipfile
from pathlib import Path

import pytest

from scripts.validate_release import (
    main,
    package_version,
    validate,
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
        f"leaklens_ml-{version}.dist-info/RECORD",
        f"leaklens_ml-{version}.dist-info/licenses/LICENSE",
    ]


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
    wheel = tmp_path / "leaklens_ml-0.13.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in _wheel_names():
            archive.writestr(name, "")
    assert main(["--wheel", str(wheel)]) == 0
    assert "only package and dist-info" in capsys.readouterr().out

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tests/secret.csv", "x")
    assert main(["--wheel", str(wheel)]) == 1
    assert "tests/secret.csv" in capsys.readouterr().err


def test_every_workflow_action_is_sha_pinned():
    pattern = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+?)(?:\s*#.*)?$", re.MULTILINE)
    unpinned = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for ref in pattern.findall(path.read_text(encoding="utf-8")):
            if "@" not in ref or not re.search(r"@[0-9a-f]{40}$", ref):
                unpinned.append(f"{path.name}: {ref}")
    assert not unpinned, "unpinned action(s):\n" + "\n".join(unpinned)
