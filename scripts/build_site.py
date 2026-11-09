"""Regenerate the website fixtures and validate the static GitHub Pages build.

The site is plain HTML/CSS/JS served from ``docs/``.  There is no bundler, so
this script is the "production build": it regenerates the recorded scanner
output, then checks every page for missing assets, broken fragments, external
dependencies and basic accessibility requirements.

Usage:
    python scripts/build_site.py           # regenerate fixtures + validate
    python scripts/build_site.py --check   # validate only; fail if fixtures are stale
"""

from __future__ import annotations

import importlib.util
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaklens import __version__  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_site_fixtures", ROOT / "scripts" / "build_site_fixtures.py"
)
build_site_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_site_fixtures)

DOCS = ROOT / "docs"
FIXTURES = DOCS / "assets" / "fixtures.mjs"
PAGES = ["index.html", "demo.html", "live.html", "docs.html", "report.html", "404.html"]
REQUIRED_ASSETS = [
    "assets/site.css",
    "assets/site.js",
    "assets/demo.js",
    "assets/demo-engine.mjs",
    "assets/hero3d.js",
    "assets/fixtures.mjs",
    "assets/live.js",
    "assets/live-engine.mjs",
    "assets/live-config.js",
    "assets/logo.svg",
    "assets/favicon.svg",
]
EXTERNAL_OK = re.compile(r"^(https?:|mailto:|tel:|data:|#)")
ALLOWED_EXTERNAL_HOSTS = {
    "github.com",
    "pypi.org",
    "eyadstein.github.io",
}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
ERRORS: list[str] = []


def fail(message: str) -> None:
    ERRORS.append(message)


class Page(HTMLParser):
    """Collect the parts of a page that the static checks need."""

    def __init__(self, path: Path, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.text = text
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.stack: list[str] = []
        self.links: list[dict] = []
        self.assets: list[dict] = []
        self.images: list[dict] = []
        self.buttons: list[dict] = []
        self.headings: list[tuple[str, str]] = []
        self.script_srcs: list[str] = []
        self.stylesheets: list[str] = []
        self.base_href: str | None = None
        self.has_main = False
        self.has_header = False
        self.has_nav = False
        self.meta_description = False
        self.meta_viewport = False
        self.has_doctype = text.lstrip().lower().startswith("<!doctype html>")
        self.lang = ""
        self._capture: dict | None = None
        self._capture_text: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {key: (value or "") for key, value in attrs}
        if tag not in VOID_TAGS:
            self._depth += 1
            self.stack.append(tag)
        if tag == "html":
            self.lang = data.get("lang", "")
        if "id" in data:
            if data["id"] in self.ids:
                self.duplicate_ids.add(data["id"])
            self.ids.add(data["id"])
        if tag == "main":
            self.has_main = True
        if tag == "header":
            self.has_header = True
        if tag == "nav":
            self.has_nav = True
        if tag == "meta":
            if data.get("name") == "description" and data.get("content", "").strip():
                self.meta_description = True
            if data.get("name") == "viewport" and "width=device-width" in data.get(
                "content", ""
            ):
                self.meta_viewport = True
        if tag == "base" and data.get("href"):
            self.base_href = data["href"]
        if tag == "a":
            self.links.append(
                {
                    "href": data.get("href", ""),
                    "label": data.get("aria-label", ""),
                    "text": "",
                }
            )
            self._begin_capture(self.links[-1])
        elif tag == "img":
            self.images.append({"src": data.get("src", ""), "alt": data.get("alt")})
            if data.get("src"):
                self.assets.append({"src": data["src"], "tag": "img"})
        elif tag == "button":
            self.buttons.append(
                {
                    "label": data.get("aria-label", ""),
                    "text": "",
                    "type": data.get("type", ""),
                }
            )
            self._begin_capture(self.buttons[-1])
        elif tag == "script" and data.get("src"):
            self.script_srcs.append(data["src"])
            self.assets.append({"src": data["src"], "tag": "script"})
        elif tag == "link" and data.get("href"):
            rel = data.get("rel", "")
            if "stylesheet" in rel:
                self.stylesheets.append(data["href"])
            self.assets.append({"src": data["href"], "tag": f"link:{rel}"})
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.headings.append((tag, ""))
            self._begin_capture({"heading": len(self.headings) - 1})

    def _begin_capture(self, target: dict) -> None:
        if self._capture is not None:
            return
        self._capture = target
        self._capture["_depth"] = self._depth
        self._capture_text = []

    def _finish_capture(self) -> None:
        if self._capture is None:
            return
        text = " ".join("".join(self._capture_text).split())
        if "heading" in self._capture:
            index = self._capture["heading"]
            heading_tag, _ = self.headings[index]
            self.headings[index] = (heading_tag, text)
        else:
            self._capture["text"] = text
        self._capture = None
        self._capture_text = []

    def handle_endtag(self, tag: str) -> None:
        if self._capture is not None and self._depth <= self._capture["_depth"]:
            self._finish_capture()
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack.pop() != tag:
                pass
        if tag not in VOID_TAGS:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._capture_text.append(data)


def parse_page(name: str) -> Page:
    path = DOCS / name
    page = Page(path, path.read_text(encoding="utf-8"))
    page.feed(page.text)
    page.close()
    return page


def resolve(page: Page, raw: str) -> tuple[Path | None, str]:
    """Resolve a site-relative URL to a file under docs/ and return its fragment."""
    parts = urlsplit(raw)
    fragment = unquote(parts.fragment)
    target = unquote(parts.path)
    if not target:
        return page.path, fragment
    if target.startswith("/"):
        prefix = "/leaklens/"
        if page.base_href and page.base_href.startswith("/"):
            prefix = page.base_href
        if not target.startswith(prefix):
            fail(f"{page.path.name}: absolute URL outside the Pages root: {raw}")
            return None, fragment
        target = target[len(prefix) :]
    base = DOCS
    if page.base_href and page.base_href.startswith("/"):
        base = DOCS
    resolved = (base / target).resolve()
    if resolved.is_dir():
        resolved = resolved / "index.html"
    return resolved, fragment


def check_links(
    page: Page, fragments: dict[str, set[str]], fixture_ids: set[str]
) -> None:
    for link in page.links:
        href = link["href"].strip()
        label = link["label"] or link["text"]
        if not href:
            fail(f"{page.path.name}: empty href")
            continue
        if not label:
            fail(f"{page.path.name}: link without text or aria-label ({href})")
        if EXTERNAL_OK.match(href):
            if href.startswith("#"):
                if href[1:] and href[1:] not in page.ids:
                    fail(f"{page.path.name}: missing fragment target {href}")
            elif href.startswith(("http://", "https://")):
                host = urlsplit(href).hostname or ""
                if host not in ALLOWED_EXTERNAL_HOSTS:
                    fail(f"{page.path.name}: unexpected external host {host}")
            continue
        target, fragment = resolve(page, href)
        if target is None:
            continue
        if not target.exists():
            fail(f"{page.path.name}: broken link {href}")
            continue
        if fragment:
            known = (
                page.ids if target == page.path else fragments.get(target.name, set())
            )
            js_managed = target.name == "demo.html" and fragment in fixture_ids
            if fragment not in known and not js_managed:
                fail(f"{page.path.name}: missing fragment {fragment} in {target.name}")


def check_assets(page: Page) -> None:
    for asset in page.assets:
        source = asset["src"].strip()
        if not source:
            fail(f"{page.path.name}: {asset['tag']} without a source")
            continue
        if source.startswith(("http://", "https://", "data:")):
            fail(f"{page.path.name}: external asset {source}")
            continue
        target, _ = resolve(page, source)
        if target is None:
            continue
        if not target.exists():
            fail(f"{page.path.name}: missing asset {source}")
    for image in page.images:
        if image["alt"] is None:
            fail(f"{page.path.name}: image without alt attribute ({image['src']})")
    for button in page.buttons:
        if not (button["label"] or button["text"]):
            fail(f"{page.path.name}: button without text or aria-label")
    for name, src in [(page.path.name, src) for src in page.script_srcs]:
        if not src.startswith(("assets/", "../assets/")):
            fail(f"{name}: script outside assets/: {src}")


def check_page(page: Page) -> None:
    name = page.path.name
    if not page.has_doctype:
        fail(f"{name}: missing <!doctype html>")
    if page.lang != "en":
        fail(f'{name}: <html> needs lang="en"')
    if not page.meta_viewport:
        fail(f"{name}: missing responsive viewport meta")
    if not page.meta_description:
        fail(f"{name}: missing meta description")
    if not page.has_main:
        fail(f"{name}: missing <main> landmark")
    if not page.has_header or not page.has_nav:
        fail(f"{name}: missing header/nav landmarks")
    if page.duplicate_ids:
        fail(f"{name}: duplicate ids: {', '.join(sorted(page.duplicate_ids))}")
    h1s = [text for tag, text in page.headings if tag == "h1"]
    if len(h1s) != 1:
        fail(f"{name}: expected exactly one h1, found {len(h1s)}")
    if not any(text for text in h1s):
        fail(f"{name}: h1 has no text")
    skip = [
        link
        for link in page.links
        if "skip" in link["text"].lower() or "skip" in link["label"].lower()
    ]
    if not skip:
        fail(f"{name}: missing skip link")
    else:
        href = skip[0]["href"]
        if href.startswith("#") and href[1:] not in page.ids:
            fail(f"{name}: skip link target {href} does not exist")
    if page.text.count("<main") > 1:
        fail(f"{name}: more than one <main> element")


def check_site_structure(payload: dict) -> None:
    for name in PAGES:
        if not (DOCS / name).is_file():
            fail(f"missing page docs/{name}")
    for name in REQUIRED_ASSETS:
        if not (DOCS / name).is_file():
            fail(f"missing asset docs/{name}")
    if not (DOCS / ".nojekyll").is_file():
        fail("missing docs/.nojekyll (GitHub Pages must serve files verbatim)")
    css = (DOCS / "assets" / "site.css").read_text(encoding="utf-8")
    for needle in (
        "prefers-reduced-motion",
        "max-width: 1020px",
        "max-width: 820px",
        "max-width: 560px",
    ):
        if needle not in css:
            fail(f"site.css: missing {needle}")
    for needle in (":focus-visible", ".skip-link", ".sr-only"):
        if needle not in css:
            fail(f"site.css: missing {needle}")
    for name in ("site.js", "demo.js", "hero3d.js", "demo-engine.mjs"):
        text = (DOCS / "assets" / name).read_text(encoding="utf-8")
        if "http://" in text or "https://" in text:
            fail(f"assets/{name}: network URL found in frontend JavaScript")
    for name in ("live.js", "live-engine.mjs"):
        text = (DOCS / "assets" / name).read_text(encoding="utf-8")
        if re.search(r"https?://", text):
            fail(
                f"assets/{name}: hard-coded backend URL; use live-config.js "
                "or the in-page field instead"
            )
    config = (DOCS / "assets" / "live-config.js").read_text(encoding="utf-8")
    configured = re.search(r'^[ \t]*apiBase:[ \t]*"([^"]*)"', config, re.MULTILINE)
    if configured is None:
        fail("assets/live-config.js: missing apiBase")
    else:
        api_base = configured.group(1)
        if api_base:
            parts = urlsplit(api_base)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.query
                or parts.fragment
                or parts.path not in {"", "/"}
            ):
                fail(
                    "assets/live-config.js: apiBase must be empty or a clean "
                    "HTTPS origin (no credentials, path, query or fragment)"
                )
    live = (DOCS / "live.html").read_text(encoding="utf-8")
    for needle in (
        "data-connection",
        "data-file-input",
        "data-progress",
        "data-results",
        "No backend connected",
        "never replays",
    ):
        if needle not in live:
            fail(f"live.html: missing live-scan marker {needle!r}")
    if payload["meta"]["version"] != __version__:
        fail(
            "fixtures version "
            f"{payload['meta']['version']} does not match package {__version__}"
        )
    allowed = {"high", "medium", "low"}
    for group in ("code", "datasets", "overlaps"):
        for fixture in payload[group]:
            findings = fixture["findings"]
            counts = fixture["counts"]
            expected = {
                level: sum(f["severity"] == level for f in findings)
                for level in allowed
            }
            if counts != expected:
                fail(f"fixture {fixture['id']}: counts do not match findings")
            for finding in findings:
                if finding["severity"] not in allowed:
                    fail(f"fixture {fixture['id']}: bad severity {finding['severity']}")
                if not finding.get("fix"):
                    fail(
                        f"fixture {fixture['id']}: finding {finding['kind']} has no fix"
                    )
                if not finding.get("detail"):
                    fail(
                        f"fixture {fixture['id']}: finding {finding['kind']} has no detail"
                    )
            expected_score = max(
                0,
                100
                - 35 * expected["high"]
                - 15 * expected["medium"]
                - 5 * expected["low"],
            )
            if fixture.get("trustScore") != expected_score:
                fail(
                    f"fixture {fixture['id']}: trustScore {fixture.get('trustScore')} "
                    f"!= {expected_score}"
                )
            if fixture["clean"] != (not findings):
                fail(f"fixture {fixture['id']}: clean flag does not match findings")


def check_fixture_freshness(expected: str) -> None:
    if not FIXTURES.is_file():
        fail("docs/assets/fixtures.mjs is missing; run python scripts/build_site.py")
        return
    if FIXTURES.read_text(encoding="utf-8") == expected:
        return
    if not build_site_fixtures.serves_shipped_artifact():
        # This scikit-learn minor refits instead of loading the shipped artifact,
        # so its threshold and margins differ; the structural checks still run.
        print(
            "note: AI fixture text was recorded with the shipped artifact and is "
            "not reproducible on this scikit-learn minor"
        )
        return
    fail("docs/assets/fixtures.mjs is stale; run python scripts/build_site.py")


def run(check_only: bool) -> int:
    ERRORS.clear()
    payload = build_site_fixtures.build()
    rendered = build_site_fixtures.render(payload)
    if not check_only:
        FIXTURES.parent.mkdir(parents=True, exist_ok=True)
        FIXTURES.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote docs/assets/fixtures.mjs ({FIXTURES.stat().st_size:,} bytes)")
    check_fixture_freshness(rendered)
    check_site_structure(payload)
    fixture_ids = {
        item["id"]
        for group in ("code", "datasets", "overlaps")
        for item in payload[group]
    }
    pages = {name: parse_page(name) for name in PAGES}
    fragments = {name: page.ids for name, page in pages.items()}
    for page in pages.values():
        check_page(page)
        check_assets(page)
        check_links(page, fragments, fixture_ids)
    if ERRORS:
        print("site validation failed:", file=sys.stderr)
        for message in ERRORS:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(
        f"site ok: {len(PAGES)} pages, {len(REQUIRED_ASSETS)} required assets, "
        f"fixtures v{__version__}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return run(check_only="--check" in argv)


if __name__ == "__main__":
    raise SystemExit(main())
