"""Tests for release tag/version and workflow ref validation."""

import pytest

from scripts.validate_release import main, package_version, validate


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
