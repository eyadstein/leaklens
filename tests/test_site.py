"""Static website checks: fixtures, links, accessibility landmarks and JS syntax."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from leaklens import __version__

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = ["index.html", "demo.html", "live.html", "docs.html", "report.html", "404.html"]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_site = _load_script("build_site")
build_fixtures = _load_script("build_site_fixtures")


def load_fixtures() -> dict:
    text = (DOCS / "assets" / "fixtures.mjs").read_text(encoding="utf-8")
    return json.loads(text[text.index("{") : text.rindex(";")])


@pytest.fixture(scope="module")
def fixtures() -> dict:
    return load_fixtures()


def test_fixtures_match_package_version(fixtures):
    assert fixtures["meta"]["version"] == __version__
    assert fixtures["meta"]["prerecorded"] is True
    assert fixtures["meta"]["scanner"] == "leaklens.scanner.scan_source"
    assert "leaklens.leakage" in fixtures["meta"]["dataset"]


def test_fixture_scores_follow_the_central_rule(fixtures):
    penalties = {"high": 35, "medium": 15, "low": 5}
    for group in ("code", "datasets", "overlaps"):
        for fixture in fixtures[group]:
            expected = max(
                0, 100 - sum(penalties[f["severity"]] for f in fixture["findings"])
            )
            assert fixture["trustScore"] == expected, fixture["id"]
            counts = {
                level: sum(f["severity"] == level for f in fixture["findings"])
                for level in penalties
            }
            assert fixture["counts"] == counts
            assert fixture["clean"] is (not fixture["findings"])


def test_every_finding_has_evidence_and_a_fix(fixtures):
    seen = 0
    for group in ("code", "datasets", "overlaps"):
        for fixture in fixtures[group]:
            for finding in fixture["findings"]:
                seen += 1
                assert finding["severity"] in {"high", "medium", "low"}
                assert finding["kind"]
                assert finding["detail"].strip()
                assert finding["fix"].strip()
                if fixture["group"] == "code":
                    assert isinstance(finding["line"], int)
    assert seen >= 10


def test_demo_covers_leaky_and_clean_paths(fixtures):
    ids = {
        f["id"] for group in ("code", "datasets", "overlaps") for f in fixtures[group]
    }
    assert {
        "advanced",
        "unsafe",
        "safe",
        "leaky_transactions",
        "clean_transactions",
    } <= ids
    assert {"overlap_exact", "overlap_near"} <= ids
    clean = [f for f in fixtures["datasets"] if f["clean"]]
    leaky = [f for f in fixtures["datasets"] if not f["clean"]]
    assert clean and leaky
    assert clean[0]["trustScore"] == 100
    assert leaky[0]["trustScore"] < 80


def test_exact_overlap_plot_uses_real_duplicated_rows(fixtures):
    exact = next(f for f in fixtures["overlaps"] if f["id"] == "overlap_exact")
    plot = exact["plot"]
    assert plot["counts"]["duplicates"] > 0
    duplicates = [p for p in plot["test"] if p["dup"]]
    assert len(duplicates) == plot["counts"]["duplicates"]
    assert all(p["pair"] is not None for p in duplicates)
    assert all(0 <= p["pair"] < len(plot["train"]) for p in duplicates)
    assert any(p["dup"] for p in plot["train"])


def test_site_build_validation_passes(capsys):
    assert build_site.run(check_only=True) == 0, capsys.readouterr().err
    output = capsys.readouterr().out
    assert "site ok" in output


def test_pages_have_landmarks_and_no_duplicate_ids():
    pages = {name: build_site.parse_page(name) for name in PAGES}
    assert set(pages) == set(PAGES)
    for name, page in pages.items():
        assert page.has_doctype, name
        assert page.lang == "en", name
        assert page.meta_viewport, name
        assert page.meta_description, name
        assert page.has_main and page.has_header and page.has_nav, name
        assert not page.duplicate_ids, (name, page.duplicate_ids)
        h1s = [text for tag, text in page.headings if tag == "h1"]
        assert len(h1s) == 1 and h1s[0], name


def test_demo_page_is_explicitly_prerecorded():
    text = (DOCS / "demo.html").read_text(encoding="utf-8")
    assert "prerecorded" in text.lower()
    assert "No Python runs in your browser" in text
    assert "Replaying recorded output" in text
    assert "build_site_fixtures.py" in text


def test_live_page_is_explicitly_not_a_mock():
    text = (DOCS / "live.html").read_text(encoding="utf-8")
    assert "never replays a recorded result" in text
    assert "No backend connected" in text
    assert "GitHub Pages cannot execute Python" in text
    assert "data-connection" in text
    assert "data-file-input" in text
    assert "data-progress" in text
    assert "data-results" in text
    assert "Live scan" in text
    # The live page must not import the recorded fixtures.
    assert "fixtures.mjs" not in text


def test_live_javascript_has_no_hard_coded_backend():
    for name in ("live.js", "live-engine.mjs"):
        text = (DOCS / "assets" / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), name
    config = (DOCS / "assets" / "live-config.js").read_text(encoding="utf-8")
    assert 'apiBase: ""' in config


def test_landing_page_no_longer_contains_the_old_generated_report():
    text = (DOCS / "index.html").read_text(encoding="utf-8")
    assert "LeakLens report</title>" not in text
    assert "docs/assets/site.css" not in text
    assert 'href="assets/site.css"' in text
    assert "Your model isn't smart" in text


def test_frontend_assets_are_local_and_motion_is_respected():
    css = (DOCS / "assets" / "site.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css
    for width in ("1020px", "820px", "560px"):
        assert f"max-width: {width}" in css
    assert ":focus-visible" in css
    for name in PAGES:
        text = (DOCS / name).read_text(encoding="utf-8")
        assert not re.search(r'<script[^>]+src="https?://', text, re.IGNORECASE), name
        assert not re.search(r'<link[^>]+href="https?://', text, re.IGNORECASE), name


def test_safe_pipeline_example_matches_the_recorded_fixture(fixtures):
    source = (ROOT / "examples" / "safe_pipeline.py").read_text(encoding="utf-8")
    safe = next(f for f in fixtures["code"] if f["id"] == "safe")
    assert safe["source"] == source
    assert safe["findings"] == []
    assert safe["trustScore"] == 100


@pytest.mark.parametrize(
    "script",
    [
        "site.js",
        "demo.js",
        "hero3d.js",
        "demo-engine.mjs",
        "live.js",
        "live-engine.mjs",
        "live-config.js",
    ],
)
def test_frontend_javascript_parses(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    path = DOCS / "assets" / script
    if path.suffix == ".mjs":
        result = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    else:
        result = subprocess.run(
            [node, "--input-type=module", "--check"],
            input=path.read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    assert result.returncode == 0, result.stderr


def test_fixture_generator_is_deterministic():
    first = build_fixtures.render(build_fixtures.build())
    second = build_fixtures.render(build_fixtures.build())
    assert first == second
    assert first == (DOCS / "assets" / "fixtures.mjs").read_text(encoding="utf-8")
    assert sys.version_info >= (3, 10)
