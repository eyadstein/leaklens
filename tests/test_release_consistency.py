"""Version consistency across pyproject, runtime, citation and changelog."""

from pathlib import Path

import pytest

from scripts.validate_release import check_versions, main, package_version

ROOT = Path(__file__).resolve().parents[1]


def _tree(
    tmp_path,
    *,
    version="9.9.9",
    runtime=None,
    citation=None,
    heading=None,
    released="2026-01-01",
):
    (tmp_path / "leaklens").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "leaklens" / "__init__.py").write_text(
        f'__version__ = "{runtime or version}"\n', encoding="utf-8"
    )
    date_line = f"date-released: {released}\n" if released else ""
    (tmp_path / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {citation or version}\n{date_line}",
        encoding="utf-8",
    )
    heading = version if heading is None else heading
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## {heading}\n\n- note\n", encoding="utf-8"
    )
    return tmp_path


def test_repository_versions_agree():
    assert check_versions(ROOT) == package_version(ROOT / "pyproject.toml")


def test_matching_tree_returns_version(tmp_path):
    assert check_versions(_tree(tmp_path)) == "9.9.9"


def test_unreleased_heading_is_valid(tmp_path):
    assert check_versions(_tree(tmp_path, heading="Unreleased")) == "9.9.9"


@pytest.mark.parametrize("heading", ["[9.9.9] - 2026-01-01", "9.9.9 - 2026-01-01"])
def test_release_heading_must_name_the_version(tmp_path, heading):
    assert check_versions(_tree(tmp_path, heading=heading)) == "9.9.9"


def test_runtime_mismatch_names_both_files_and_values(tmp_path):
    _tree(tmp_path, runtime="1.2.3")
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    message = str(err.value)
    assert "__init__.py" in message and "1.2.3" in message and "9.9.9" in message


def test_citation_mismatch_names_both_files_and_values(tmp_path):
    _tree(tmp_path, citation="1.2.3")
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    message = str(err.value)
    assert "CITATION.cff" in message and "1.2.3" in message and "9.9.9" in message


def test_malformed_pyproject_version_is_rejected(tmp_path):
    _tree(tmp_path, version="9.9")
    with pytest.raises(ValueError, match=r"not a valid X\.Y\.Z version"):
        check_versions(tmp_path)


def test_malformed_runtime_version_names_the_runtime_file(tmp_path):
    _tree(tmp_path, runtime="9.9")
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    assert "__init__.py" in str(err.value) and "9.9" in str(err.value)


def test_malformed_citation_version_is_rejected(tmp_path):
    _tree(tmp_path, citation="not-a-version")
    with pytest.raises(ValueError, match=r"not a valid X\.Y\.Z version"):
        check_versions(tmp_path)


def test_changelog_heading_must_match_or_be_unreleased(tmp_path):
    _tree(tmp_path, heading="1.2.3")
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    assert "1.2.3" in str(err.value) and "9.9.9" in str(err.value)


def test_missing_changelog_heading_is_rejected(tmp_path):
    _tree(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
    with pytest.raises(ValueError, match="release heading"):
        check_versions(tmp_path)


def test_cli_versions_mode_passes(tmp_path, capsys):
    _tree(tmp_path)
    assert main(["--versions", "--source-root", str(tmp_path)]) == 0
    assert "is consistent" in capsys.readouterr().out


def test_cli_versions_mode_fails_loudly(tmp_path, capsys):
    _tree(tmp_path, runtime="1.2.3")
    assert main(["--versions", "--source-root", str(tmp_path)]) == 1
    assert "::error::" in capsys.readouterr().err


def test_repository_citation_date_is_valid():
    from scripts.validate_release import citation_date

    assert citation_date(ROOT / "CITATION.cff") == "2026-11-20"


def test_missing_citation_date_is_required(tmp_path):
    _tree(tmp_path, released="")
    with pytest.raises(ValueError, match="no date-released field"):
        check_versions(tmp_path)


@pytest.mark.parametrize(
    "released", ["2026-02-30", "20-11-2026", "2026-11-20T00:00:00", "soon"]
)
def test_malformed_citation_date_is_rejected(tmp_path, released):
    _tree(tmp_path, released=released)
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    assert "date-released is invalid" in str(err.value)
    assert "CITATION.cff" in str(err.value)


def test_changelog_date_must_agree_with_citation_date(tmp_path):
    _tree(tmp_path, heading="9.9.9 - 2026-02-03")
    with pytest.raises(ValueError) as err:
        check_versions(tmp_path)
    message = str(err.value)
    assert "2026-02-03" in message and "2026-01-01" in message
    assert "does not match" in message


def test_release_date_argument_must_agree_with_citation_date(tmp_path):
    _tree(tmp_path)
    assert check_versions(tmp_path, release_date="2026-01-01") == "9.9.9"
    with pytest.raises(ValueError, match="does not match"):
        check_versions(tmp_path, release_date="2026-01-02")


def test_malformed_release_date_argument_is_rejected(tmp_path):
    _tree(tmp_path)
    with pytest.raises(ValueError, match="is not a YYYY-MM-DD date"):
        check_versions(tmp_path, release_date="01/01/2026")


def test_cli_release_date_flag_matches_and_mismatches(tmp_path, capsys):
    _tree(tmp_path)
    assert (
        main(
            [
                "--versions",
                "--source-root",
                str(tmp_path),
                "--release-date",
                "2026-01-01",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--versions",
                "--source-root",
                str(tmp_path),
                "--release-date",
                "2026-01-02",
            ]
        )
        == 1
    )
    assert "date-released" in capsys.readouterr().err
