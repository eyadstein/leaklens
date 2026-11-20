"""The PR bot must send contract files to the owner instead of auto-merging.

The matcher is read from the workflow and executed here, so these tests exercise
the same regex the bot runs rather than asserting that a string exists in YAML.
"""

import re
from pathlib import Path

import pytest

BOT = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pr-bot.yml"


def _sensitive_pattern() -> re.Pattern:
    text = BOT.read_text(encoding="utf-8")
    match = re.search(r"filter\(f =>\s*/(.+?)/\.test\(f\)\)", text, re.DOTALL)
    assert match, "sensitive-file matcher not found in pr-bot.yml"
    return re.compile(match.group(1))


def _sensitive(files):
    pattern = _sensitive_pattern()
    return [name for name in files if pattern.search(name)]


@pytest.mark.parametrize(
    "path",
    [
        "leaklens/schemas/report-1.schema.json",
        "leaklens/schemas/nested/deep.json",
        "leaklens/manifest.py",
        "leaklens/scanner.py",
        "leaklens/cli.py",
        "leaklens/temporal.py",
        "leaklens/sqlscan.py",
        "leaklens/api.py",
        "leaklens/report.py",
        "leaklens/partition.py",
        "leaklens/ai/learned.py",
        "leaklens/intelligence/orchestrator.py",
        "leaklens/analysis/text/dedup.py",
    ],
)
def test_runtime_and_contract_files_need_owner_review(path):
    assert _sensitive([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "setup.cfg",
        "requirements-dev.txt",
        "Dockerfile",
        "Dockerfile.api",
        "LICENSE",
        "action.yml",
        "MANIFEST.in",
        ".pre-commit-hooks.yaml",
        ".pre-commit-config.yaml",
        "scripts/build_site.py",
    ],
)
def test_existing_protections_are_preserved(path):
    assert _sensitive([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_cli.py",
        "README.md",
        "docs/CONFIGURATION.md",
        "examples/demo.py",
        "leaklens_extra/scanner.py",
        "docs/leaklens/scanner.py",
        "src/leaklens/scanner.py",
        "notleaklens/scanner.py",
        "LEAKLENS/scanner.py",
        "leaklensx.py",
        "leaklens.md",
    ],
)
def test_ordinary_project_files_still_auto_merge(path):
    assert _sensitive([path]) == []


def test_a_mixed_change_is_sensitive():
    files = [
        "leaklens/cli.py",
        "tests/test_cli.py",
        "leaklens/schemas/report-1.schema.json",
        "README.md",
    ]
    assert _sensitive(files) == [
        "leaklens/cli.py",
        "leaklens/schemas/report-1.schema.json",
    ]


def test_the_sensitive_check_stops_before_the_auto_merge():
    text = BOT.read_text(encoding="utf-8")
    guard = text.index("if (sensitive.length)")
    merge = text.index("pulls.merge(")
    assert guard < merge
    block = text[guard:merge]
    assert "needs-owner-review" in block
    assert "return;" in block
