"""Static website checks: fixtures, links, accessibility landmarks and JS syntax."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

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
    # The only allowed place for a backend URL is live-config.js. It ships
    # empty; after a verified deployment it may hold a clean HTTPS origin.
    config = (DOCS / "assets" / "live-config.js").read_text(encoding="utf-8")
    match = re.search(r'^[ \t]*apiBase:[ \t]*"([^"]*)"', config, re.MULTILINE)
    assert match, "live-config.js must define apiBase"
    api = match.group(1)
    if not api:
        assert 'apiBase: ""' in config
        return
    parsed = urlsplit(api)
    assert parsed.scheme == "https", api
    assert parsed.hostname, api
    assert not parsed.username and not parsed.password, api
    assert not parsed.query and not parsed.fragment, api
    assert parsed.path in {"", "/"}, api


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


def test_recorded_ai_output_is_real_bounded_and_opt_in(fixtures):
    code = [fixture for fixture in fixtures["code"] if fixture.get("ai")]
    assert {fixture["id"] for fixture in code} == {"advanced", "unsafe", "safe"}

    for fixture in code:
        ai = fixture["ai"]
        assert ai["engine"] == "leaklens.ai.experimental"
        assert ai["changesDeterministicFindings"] is False
        assert ai["model"]["name"] == "leaklens-ai-p7"
        assert ai["model"]["version"].startswith("leaklens-ai-p7")
        assert 0.05 <= ai["model"]["threshold"] <= 0.95
        assert ai["limitations"]
        for hypothesis in ai["hypotheses"]:
            assert 0 <= hypothesis["confidence"] <= 0.95
            assert hypothesis["severityHint"] in {"low", "medium"}
            assert hypothesis["title"] and hypothesis["explanation"]
            assert hypothesis["evidence"]
            assert {span["source"] for span in hypothesis["evidence"]} <= {
                "model",
                "code",
                "schema",
                "relationship",
                "statistic",
                "finding",
            }

    advanced = next(fixture for fixture in code if fixture["id"] == "advanced")
    assert advanced["ai"]["hypotheses"]
    assert any(
        hypothesis["concept"] == "target_derived_feature"
        for hypothesis in advanced["ai"]["hypotheses"]
    )
    # The AI layer is additive: the deterministic findings are unchanged by it.
    assert advanced["findings"]
    assert advanced["ai"]["remediation"]

    safe = next(fixture for fixture in code if fixture["id"] == "safe")
    assert safe["ai"]["hypotheses"] == []
    assert safe["ai"]["changesDeterministicFindings"] is False


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


def test_live_page_has_an_explicit_opt_in_ai_panel():
    text = (DOCS / "live.html").read_text(encoding="utf-8")
    script = (DOCS / "assets" / "live.js").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "data-ai-toggle" in text
    assert "data-ai-result" in text
    assert "experimental ai hypotheses" in normalized.lower()
    assert "never changes deterministic findings" in normalized
    assert "include_ai" in script
    assert "renderAI" in script
    assert "AI intelligence" in text
    assert "innerHTML" not in script
    assert "insertAdjacentHTML" not in script


def test_landing_page_integrity_and_accessibility_fixes():
    text = (DOCS / "index.html").read_text(encoding="utf-8")
    script = (DOCS / "assets" / "site.js").read_text(encoding="utf-8")
    css = (DOCS / "assets" / "site.css").read_text(encoding="utf-8")

    # The primary hero action works offline.
    assert '<a class="button" href="demo.html">Open the interactive demo</a>' in text
    assert 'href="live.html">Live scan API' in text
    assert "Scan your data live" not in text

    # The recorded audit card shows a proposed change, not a simulated fix.
    assert "Apply the fix" not in text
    assert "Preview the proposed change" in text
    assert 'aria-labelledby="audit-heading"' in text
    assert "data-audit-diff" in text
    assert "data-audit-diff-state" not in text
    assert 'aria-label="Trust score 50 out of 100"' in text
    assert 'aria-label="Trust score 100 out of 100"' in text
    assert "LeakLens reports the leak; you change the code." in text
    assert "Clean audit" not in text + script
    assert "ring.style.setProperty" not in script

    # Interactive states survive clipboard denial and throttled reveal observers.
    assert "showAllReveals" in script
    assert "if (!sawIntersection) showAllReveals()" in script
    # The label is captured before the await, or a second click snapshots "Copied".
    assert script.index("button.dataset.original") < script.index(
        "const ok = await copyText"
    )
    assert (
        "catch {"
        in script[
            script.index("const copyText") : script.index(
                'doc.addEventListener("click", async'
            )
        ]
    )

    # Empty filter state is announced and offers a way back.
    assert "data-detect-empty" in text
    assert "data-filter-reset" in text
    assert 'role="status" aria-live="polite">14 shown' in text

    # Benchmark evidence remains measured, caveated, and tied to its source.
    assert "<strong>109 / 109</strong>" not in text
    assert "<strong>32 / 32</strong>" not in text
    assert "<strong>12 / 12</strong>" not in text
    assert "docs/REALWORLD.md" in text
    assert "only 8 labels" in text

    # Small controls and focus states meet the web target-size guideline.
    assert "min-height: 44px" in css
    assert css.count("min-height: 44px") >= 3
    assert "--faint: #7d92a9" in css
    print_block = css[css.index("@media print") :]
    assert "body * {" in print_block
    assert "background: #fff !important" in print_block
    assert "color: #111 !important" in print_block
    assert ".detection-grid" in css
    assert "overflow-x: auto" in css
    assert "grid-template-columns: minmax(0, 1fr)" in css
    assert "min-width: 0" in css
    report = (DOCS / "report.html").read_text(encoding="utf-8")
    assert report.count('<div class="table-wrap">') == 2
