from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".github" / "ISSUE_TEMPLATE" / "release_checklist.md"


def test_release_checklist_template_has_valid_front_matter():
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "---"
    assert lines.count("---") == 2
    metadata = lines[1:lines.index("---", 1)]
    assert any(line.startswith("name: ") for line in metadata)
    assert any(line.startswith("about: ") for line in metadata)
    assert any(line.startswith("title: ") for line in metadata)


def test_release_checklist_covers_publish_and_verification():
    text = TEMPLATE.read_text(encoding="utf-8")
    checks = [line for line in text.splitlines() if line.startswith("- [ ] ")]

    assert len(checks) >= 12
    for topic in (
        "pyproject.toml",
        "CHANGELOG.md",
        "pytest",
        "leaklens.bench",
        "python -m build",
        "GitHub Release",
        ".github/workflows/release.yml",
        "PyPI",
        "leaklens --version",
    ):
        assert topic in text
