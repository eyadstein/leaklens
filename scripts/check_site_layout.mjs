/* Responsive and accessibility smoke checks in a real headless browser.
   Uses the Chrome DevTools Protocol directly -- no Playwright/Puppeteer.

   Usage:
     node scripts/check_site_layout.mjs
     node scripts/check_site_layout.mjs --screenshots .site-shots
     node scripts/check_site_layout.mjs --strict   # fail when no browser exists

   The script exits 0 when every check passes, 1 on a failed check, and 2 when
   no supported browser is installed (unless --strict is used). It is a local
   verification tool; CI does not require a browser. */
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";
import { createServer as createNetServer } from "node:net";

const root = resolve(fileURLToPath(new URL("../docs", import.meta.url)));
const tempRoot = process.env.TEMP || process.env.TMPDIR || tmpdir();
const screenshotArg = process.argv.indexOf("--screenshots");
const screenshotDir = screenshotArg > -1 ? process.argv[screenshotArg + 1] : null;
const strict = process.argv.includes("--strict");
const failures = [];

const BROWSERS = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].filter(Boolean);

const browser = BROWSERS.find((candidate) => existsSync(candidate));

const check = (condition, message) => {
  if (!condition) failures.push(message);
};

const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".png": "image/png",
};

const freePort = () =>
  new Promise((resolvePort, reject) => {
    const server = createNetServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolvePort(port));
    });
  });

const startServer = () =>
  new Promise((resolveServer, reject) => {
    const server = createServer(async (request, response) => {
      const url = new URL(request.url, "http://localhost");
      const path = decodeURIComponent(url.pathname);
      const target = resolve(root, `.${path === "/" ? "/index.html" : path}`);
      if (!target.startsWith(root)) {
        response.writeHead(403).end("forbidden");
        return;
      }
      try {
        const body = await readFile(target);
        response.writeHead(200, { "content-type": types[extname(target)] || "application/octet-stream" });
        response.end(body);
      } catch {
        response.writeHead(404).end("not found");
      }
    });
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolveServer(server));
  });

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

class Cdp {
  constructor(url) {
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    this.ws = new WebSocket(url);
    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve: done, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(message.error.message));
        else done(message.result);
      } else if (message.method) {
        this.events.push(message);
      }
    });
  }

  open() {
    return new Promise((done, reject) => {
      this.ws.addEventListener("open", done, { once: true });
      this.ws.addEventListener("error", () => reject(new Error("CDP socket failed")), { once: true });
    });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.id;
    return new Promise((done, reject) => {
      this.pending.set(id, { resolve: done, reject });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
}

const connect = async (port) => {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (response.ok) {
        const data = await response.json();
        return new Cdp(data.webSocketDebuggerUrl);
      }
    } catch {
      /* keep waiting */
    }
    await sleep(125);
  }
  throw new Error("browser did not expose a DevTools endpoint");
};

const evaluate = async (cdp, session, expression) => {
  const result = await cdp.send(
    "Runtime.evaluate",
    { expression, awaitPromise: true, returnByValue: true },
    session
  );
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "evaluation failed");
  }
  return result.result.value;
};

const navigate = async (cdp, session, url) => {
  await cdp.send("Page.navigate", { url }, session);
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const state = await evaluate(cdp, session, "document.readyState").catch(() => "loading");
    if (state === "complete") break;
    await sleep(50);
  }
  await sleep(500);
};

const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "short", width: 1280, height: 600 },
  { name: "tablet", width: 834, height: 1112 },
  { name: "mobile", width: 390, height: 844 },
];

const pages = ["index.html", "demo.html", "live.html", "docs.html"];

const main = async () => {
  if (!browser) {
    console.log("no Chrome/Edge found; skipping browser layout checks (use CHROME_PATH to override)");
    process.exitCode = strict ? 2 : 0;
    return;
  }
  const server = await startServer();
  const base = `http://127.0.0.1:${server.address().port}`;
  const port = await freePort();
  const profile = join(tempRoot, `leaklens-chrome-${Date.now()}`);
  const child = spawn(
    browser,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--hide-scrollbars",
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: "ignore" }
  );

  try {
    const cdp = await connect(port);
    await cdp.open();
    const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);

    if (screenshotDir) mkdirSync(screenshotDir, { recursive: true });

    for (const viewport of viewports) {
      await cdp.send(
        "Emulation.setDeviceMetricsOverride",
        { width: viewport.width, height: viewport.height, deviceScaleFactor: 1, mobile: viewport.name === "mobile" },
        sessionId
      );
      for (const page of pages) {
        cdp.events.length = 0;
        await navigate(cdp, sessionId, `${base}/${page}`);
        const label = `${page} @ ${viewport.name}`;
        const result = await evaluate(
          cdp,
          sessionId,
          `(() => {
            const style = (el) => getComputedStyle(el);
            const visible = (el) => el && style(el).display !== "none" && style(el).visibility !== "hidden";
            return {
              overflow: document.documentElement.scrollWidth - window.innerWidth,
              h1: document.querySelectorAll("h1").length,
              main: Boolean(document.querySelector("main")),
              skip: Boolean(document.querySelector(".skip-link")),
              navToggleVisible: visible(document.querySelector("[data-nav-toggle]")),
              headerVisible: visible(document.querySelector(".site-header")),
              footerVisible: visible(document.querySelector(".site-footer")),
              title: document.title,
              emptyLinks: [...document.querySelectorAll("a")].filter((a) => !a.textContent.trim() && !a.getAttribute("aria-label")).length,
              emptyButtons: [...document.querySelectorAll("button")].filter((b) => !b.textContent.trim() && !b.getAttribute("aria-label")).length,
              wide: document.documentElement.scrollWidth > window.innerWidth + 1
                ? [...document.querySelectorAll("body *")]
                    .filter((el) => {
                      const rect = el.getBoundingClientRect();
                      const style = getComputedStyle(el);
                      return rect.width > 0 && rect.right > window.innerWidth + 1 &&
                        style.position !== "fixed" && style.overflowX !== "auto" && style.overflowX !== "scroll";
                    })
                    .slice(0, 6)
                    .map((el) => el.tagName.toLowerCase() + "." + String(el.className).replace(/\\s+/g, ".") + " right=" + Math.round(el.getBoundingClientRect().right))
                : [],
            };
          })()`
        );
        check(
          result.overflow <= 1,
          `${label}: horizontal overflow of ${result.overflow}px${result.wide?.length ? ` (${result.wide.join("; ")})` : ""}`
        );
        check(result.h1 === 1, `${label}: expected one h1, found ${result.h1}`);
        check(result.main, `${label}: missing main landmark`);
        check(result.skip, `${label}: missing skip link`);
        check(result.headerVisible, `${label}: header hidden`);
        check(result.footerVisible, `${label}: footer hidden`);
        check(result.emptyLinks === 0, `${label}: ${result.emptyLinks} link(s) without an accessible name`);
        check(result.emptyButtons === 0, `${label}: ${result.emptyButtons} button(s) without an accessible name`);
        if (viewport.name === "mobile") {
          check(result.navToggleVisible, `${label}: mobile nav toggle is not visible`);
        } else {
          check(!result.navToggleVisible, `${label}: mobile nav toggle leaked onto a wide viewport`);
        }

        if (page === "index.html") {
          const landing = await evaluate(
            cdp,
            sessionId,
            `(() => {
              const canvas = document.getElementById("leak-canvas");
              const cards = [...document.querySelectorAll("[data-detect-grid] [data-category]")];
              const chip = document.querySelector('[data-filter="tuning"]');
              chip?.click();
              const shown = cards.filter((card) => !card.hidden).length;
              const toggle = document.querySelector("[data-nav-toggle]");
              const nav = document.getElementById("site-nav");
              toggle?.click();
              const opened = nav?.classList.contains("is-open") && toggle?.getAttribute("aria-expanded") === "true";
              document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
              const closed = nav && !nav.classList.contains("is-open");

              const auditToggle = document.querySelector("[data-audit-toggle]");
              const diff = document.querySelector("[data-audit-diff]");
              const score = document.querySelector("[data-audit] .score-ring span")?.textContent;
              auditToggle?.click();
              const diffOpen = diff ? !diff.hidden : false;
              auditToggle?.click();
              const diffClosed = diff ? diff.hidden : false;
              const scoreUnchanged =
                document.querySelector("[data-audit] .score-ring span")?.textContent === score;

              const dup = document.querySelector("[data-dup-toggle]");
              const dupBefore = dup?.textContent;
              dup?.click();
              const dupAfter = dup?.textContent;
              dup?.click();

              return {
                canvas: canvas ? canvas.toDataURL("image/png").length : 0,
                cards: cards.length,
                shown,
                filtersWork: shown > 0 && shown < cards.length,
                audit: Boolean(document.querySelector("[data-audit]")),
                opened,
                closed,
                diffOpen,
                diffClosed,
                scoreUnchanged,
                dupToggles: dupBefore !== dupAfter,
              };
            })()`
          );
          check(landing.canvas > 5000, `${label}: 3-D canvas appears blank`);
          check(landing.cards >= 12, `${label}: detection gallery is incomplete`);
          check(landing.filtersWork, `${label}: detection filter did not narrow the gallery`);
          check(landing.audit, `${label}: landing audit card missing`);
          check(landing.diffOpen, `${label}: proposed patch did not open`);
          check(landing.diffClosed, `${label}: proposed patch did not close`);
          check(landing.scoreUnchanged, `${label}: the recorded card changed its trust score`);
          check(landing.dupToggles, `${label}: duplicate toggle did not change state`);
          if (viewport.name === "mobile") {
            check(landing.opened, `${label}: mobile navigation did not open`);
            check(landing.closed, `${label}: Escape did not close the mobile navigation`);
          }
        }

        if (page === "demo.html") {
          const fixtureCounts = await evaluate(
            cdp,
            sessionId,
            `(() => ({
              total: document.querySelectorAll("[data-fixture-list] button").length,
              visible: document.querySelectorAll("[data-fixture-list] li:not([hidden]) button").length,
            }))()`
          );
          check(fixtureCounts.total === 8, `${label}: expected 8 built fixtures, found ${fixtureCounts.total}`);
          check(fixtureCounts.visible === 3, `${label}: expected 3 visible code fixtures, found ${fixtureCounts.visible}`);
          const demo = await evaluate(
            cdp,
            sessionId,
            `(async () => {
              const wait = (ms) => new Promise((done) => setTimeout(done, ms));
              const frame = () => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
              const root = document.querySelector("[data-demo]");
              const side = root.querySelector(".demo-side");
              const tabs = [...root.querySelectorAll('[data-group-tabs] [role="tab"]')];
              const list = root.querySelector("[data-fixture-list]");
              const visibleButtons = () => [...list.querySelectorAll("li:not([hidden]) button")];
              const selectedTab = () =>
                tabs
                  .find((tab) => tab.getAttribute("aria-selected") === "true")
                  ?.textContent.trim()
                  .split(" · ")[0];
              const scrollable = side.scrollHeight > side.clientHeight + 1;
              if (scrollable) side.scrollTop = 24;
              const start = { top: side.scrollTop, y: scrollY, selected: selectedTab() };
              const timeline = [];
              const picks = ${JSON.stringify(
                viewport.name === "mobile"
                  ? ["Datasets", "Splits", "Code"]
                  : ["Datasets", "Splits", "Code", "Datasets"]
              )};
              for (const name of picks) {
                tabs.find((tab) => tab.textContent.includes(name)).click();
                await wait(140);
                timeline.push({
                  picked: name,
                  selected: selectedTab(),
                  top: side.scrollTop,
                  y: scrollY,
                  pressed: visibleButtons().filter((button) => button.getAttribute("aria-pressed") === "true").length,
                  active: root.dataset.activeFixture,
                });
              }

              tabs.find((tab) => tab.textContent.includes("Code")).click();
              await wait(150);
              const codeLines = root.querySelectorAll(".source-line").length;
              const details = root.querySelector(".finding");
              if (details) details.open = true;
              await wait(100);
              const highlighted = root.querySelectorAll(".source-line.is-target").length;
              tabs.find((tab) => tab.textContent.includes("Datasets")).click();
              await wait(150);
              const dataset = {
                score: root.querySelector(".score-ring span")?.textContent,
                findings: root.querySelectorAll(".finding").length,
                evidence: root.querySelectorAll(".finding-evidence").length,
                preview: root.querySelectorAll(".preview-table tbody tr").length,
              };

              // Focus must survive a selection: the sidebar is updated, not replaced.
              const target = visibleButtons()[visibleButtons().length - 1];              target.focus();
              target.click();
              await wait(140);
              const focusHeld = document.activeElement === target;
              const pressedHeld = target.getAttribute("aria-pressed") === "true";
              const afterFixtureClick = { tab: selectedTab(), group: root.dataset.activeGroup };

              const replayButton = [...root.querySelectorAll("[data-output-head] button")].find((button) =>
                /Replay/.test(button.textContent)
              );
              replayButton.focus();
              replayButton.click();
              await wait(700);
              const replay = {
                scanning: root.querySelector("[data-output]").classList.contains("is-scanning"),
                focus: document.activeElement === replayButton,
                chip: [...root.querySelectorAll(".demo-filters .chip")].find(
                  (chip) => chip.getAttribute("aria-pressed") === "true"
                )?.textContent,
              };

              const copyButton = root.querySelector("[data-output] .copy-button");
              const copyOriginal = copyButton.textContent;
              copyButton.click();
              await wait(200);
              const copyFeedback =
                copyButton.classList.contains("is-copied") || copyButton.classList.contains("is-failed");
              await wait(1700);
              const copyRestored = copyButton.textContent === copyOriginal;

              location.hash = "#overlap_exact";
              await wait(200);
              const deepLink = {
                active: root.dataset.activeFixture,
                tab: selectedTab(),
                pressed: visibleButtons().some((button) => button.getAttribute("aria-pressed") === "true"),
              };

              return {
                start,
                scrollable,
                timeline,
                dataset,
                codeLines,
                highlighted,
                focusHeld,
                pressedHeld,
                afterFixtureClick,
                replay,
                copyFeedback,
                copyRestored,
                deepLink,
              };
            })()`
          );
          check(demo.dataset.score === "50", `${label}: leaky dataset score should be 50, got ${demo.dataset.score}`);
          check(demo.dataset.findings === 2, `${label}: expected 2 dataset findings, got ${demo.dataset.findings}`);
          check(demo.dataset.preview > 0, `${label}: dataset preview table is empty`);
          check(demo.dataset.evidence >= 2, `${label}: dataset findings did not render their evidence records`);
          check(demo.codeLines > 10, `${label}: code fixture did not render its source`);
          check(demo.highlighted === 1, `${label}: opening a finding did not highlight its source line`);
          check(demo.timeline.every((step) => step.selected === step.picked), `${label}: demo selection did not stick`);
          check(demo.timeline.every((step) => step.pressed === 1), `${label}: demo selection lost its highlight`);
          check(demo.timeline.every((step) => step.y === demo.start.y), `${label}: demo selection scrolled the page`);
          if (demo.scrollable) {
            check(
              demo.timeline.every((step) => step.top === demo.start.top),
              `${label}: demo selection reset the sidebar scroll (${demo.timeline.map((step) => step.top).join(",")})`
            );
          }
          check(demo.focusHeld, `${label}: selecting a fixture dropped keyboard focus`);
          check(demo.pressedHeld, `${label}: selecting a fixture did not update its pressed state`);
          check(
            demo.afterFixtureClick.tab === "Datasets" && demo.afterFixtureClick.group === "datasets",
            `${label}: selecting a dataset fixture reset the sidebar to ${demo.afterFixtureClick.tab}`
          );
          check(!demo.replay.scanning, `${label}: replay never finished`);
          check(demo.replay.focus, `${label}: replay stole focus from its button`);
          check(/^All /.test(demo.replay.chip || ""), `${label}: replay did not restore the findings filter`);
          check(demo.copyFeedback, `${label}: copy button gave no visible feedback`);
          check(demo.copyRestored, `${label}: copy button label did not reset`);
          check(demo.deepLink.active === "overlap_exact", `${label}: deep link did not restore the fixture`);
          check(demo.deepLink.tab === "Splits", `${label}: deep link did not restore its group`);
          check(demo.deepLink.pressed, `${label}: deep link did not highlight the fixture`);
        }

        if (page === "live.html") {
          const live = await evaluate(
            cdp,
            sessionId,
            `(() => {
              const connection = document.querySelector("[data-connection]");
              const offline = document.querySelector("[data-offline]");
              const results = document.querySelector("[data-results]");
              const input = document.querySelector("[data-file-input]");
              const scan = document.querySelector("[data-scan]");
              return {
                state: connection?.dataset.state,
                label: document.querySelector("[data-connection-label]")?.textContent,
                offlineVisible: Boolean(offline) && !offline.hidden,
                resultsHidden: Boolean(results?.hidden),
                scanDisabled: Boolean(scan?.disabled),
                accept: input?.getAttribute("accept") || "",
                dropzone: Boolean(document.querySelector("[data-dropzone]")),
                status: Boolean(document.querySelector("[data-live-status]")),
                previewNote: document.querySelector("[data-preview-note]")?.textContent || "",
              };
            })()`
          );
          check(live.state === "offline", `${label}: live page should start disconnected, got ${live.state}`);
          check(/not connected/i.test(live.label || ""), `${label}: live page connection label is not explicit`);
          check(live.offlineVisible, `${label}: offline fallback is not visible`);
          check(live.resultsHidden, `${label}: results panel must stay hidden without a scan`);
          check(live.scanDisabled, `${label}: scan button must be disabled without a backend`);
          check(live.accept.includes(".csv"), `${label}: file input does not accept CSV`);
          check(live.dropzone, `${label}: drag-and-drop zone missing`);
          check(live.status, `${label}: live status region missing`);
          check(/connect a backend/i.test(live.previewNote), `${label}: preview does not explain the offline state`);
        }

        if (page === "docs.html") {
          const docs = await evaluate(
            cdp,
            sessionId,
            `(async () => {
              const wait = (ms) => new Promise((done) => setTimeout(done, ms));
              const frame = () =>
                new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
              const nav = document.querySelector("[data-docs-nav]");
              const toggle = document.querySelector("[data-docs-toggle]");
              const links = [...nav.querySelectorAll('a[href^="#"]')];
              const sections = [...document.querySelectorAll("[data-doc-section]")];
              const ids = sections.map((section) => section.id);
              const order = links.map((link) => link.getAttribute("href").slice(1));
              const activeId = () => nav.querySelector("a.is-active")?.getAttribute("href")?.slice(1) || null;
              const isMobile = ${viewport.name === "mobile"};
              const stickyTop =
                parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-h")) + 20;

              // Walk the whole page: the highlight must only ever move forward,
              // and it must reach every section in document order. The sidebar
              // itself may scroll away at the very end, but never bob upwards.
              const seen = [];
              const navTops = [];
              const max = document.documentElement.scrollHeight - innerHeight;
              const step = Math.max(150, Math.round(innerHeight * 0.2));
              for (let y = 0; y <= max; y += step) {
                window.scrollTo({ top: y, behavior: "instant" });
                await frame();
                const id = activeId();
                if (id && seen[seen.length - 1] !== id) seen.push(id);
                navTops.push(Math.round(nav.getBoundingClientRect().top));
              }
              window.scrollTo({ top: Math.round(max / 2), behavior: "instant" });
              await frame();
              const midTop = Math.round(nav.getBoundingClientRect().top);
              window.scrollTo({ top: max, behavior: "instant" });
              await frame();
              const bottom = activeId();
              const navRect = nav.getBoundingClientRect();
              const activeRect = nav.querySelector("a.is-active")?.getBoundingClientRect();
              const activeVisible = Boolean(
                activeRect &&
                  activeRect.top >= navRect.top - 1 &&
                  activeRect.bottom <= navRect.bottom + 1
              );

              // Every nav link must land its section under the sticky header
              // and highlight that same section.
              const jumps = [];
              for (const [index, link] of links.entries()) {
                const id = link.getAttribute("href").slice(1);
                link.click();
                await wait(60);
                let last = -2;
                let stable = 0;
                for (let attempt = 0; attempt < 180 && stable < 3; attempt += 1) {
                  await frame();
                  const y = Math.round(scrollY);
                  stable = y === last ? stable + 1 : 0;
                  last = y;
                }
                jumps.push({
                  id,
                  last: index === links.length - 1,
                  hash: location.hash.slice(1),
                  active: activeId(),
                  top: Math.round(document.getElementById(id).getBoundingClientRect().top),
                });
              }

              return {
                nav: Boolean(nav),
                toggleVisible: toggle ? getComputedStyle(toggle).display !== "none" : false,
                sections: sections.length,
                order,
                ids,
                seen,
                bottom,
                isMobile,
                stickyTop,
                midTop,
                navTops,
                scrollable: nav.scrollHeight > nav.clientHeight + 1,
                activeVisible,
                jumps,
              };
            })()`
          );
          check(docs.nav, `${label}: docs sidebar missing`);
          check(docs.sections >= 10, `${label}: docs sections are incomplete`);
          check(
            viewport.name === "mobile" ? docs.toggleVisible : !docs.toggleVisible,
            `${label}: docs contents toggle visibility is wrong`
          );
          check(
            docs.order.join() === docs.ids.join(),
            `${label}: sidebar order ${docs.order.join(",")} does not match document order ${docs.ids.join(",")}`
          );
          check(
            docs.seen.join() === docs.ids.join(),
            `${label}: scrolling highlighted ${docs.seen.join(",")} instead of every section in order`
          );
          check(docs.bottom === docs.ids[docs.ids.length - 1], `${label}: the last section is not active at the page end`);
          check(
            docs.navTops.every((top, index) => index === 0 || top <= docs.navTops[index - 1] + 1),
            `${label}: the sidebar bobbed while the page scrolled (${docs.navTops.slice(0, 12).join(",")})`
          );
          if (!docs.isMobile) {
            check(
              Math.abs(docs.midTop - docs.stickyTop) <= 2,
              `${label}: the sidebar is not pinned at ${docs.stickyTop}px mid-page (${docs.midTop}px)`
            );
          }
          if (docs.scrollable) {
            check(docs.activeVisible, `${label}: the active docs link is outside the sidebar's scroll box`);
          }
          for (const jump of docs.jumps) {
            check(jump.hash === jump.id, `${label}: clicking ${jump.id} set the hash to #${jump.hash}`);
            check(jump.active === jump.id, `${label}: clicking ${jump.id} highlighted ${jump.active}`);
            if (!jump.last) {
              check(
                Math.abs(jump.top - 90) <= 24,
                `${label}: clicking ${jump.id} left its heading at ${jump.top}px`
              );
            }
          }
        }

        const pageErrors = cdp.events.filter(
          (event) =>
            event.method === "Runtime.exceptionThrown" ||
            (event.method === "Runtime.consoleAPICalled" && event.params.type === "error")
        );
        check(
          pageErrors.length === 0,
          `${label}: ${pageErrors.length} uncaught/console error(s): ${pageErrors
            .map((event) =>
              String(
                event.params.exceptionDetails?.exception?.description ||
                  event.params.exceptionDetails?.text ||
                  event.params.args?.map((arg) => arg.value ?? arg.description).join(" ") ||
                  "unknown"
              ).replace(/\s+/g, " ")
            )
            .join(" | ")
            .slice(0, 400)}`
        );

        if (screenshotDir) {
          const shot = await cdp.send(
            "Page.captureScreenshot",
            { format: "png", captureBeyondViewport: false },
            sessionId
          );
          writeFileSync(
            join(screenshotDir, `${page.replace(".html", "")}-${viewport.name}.png`),
            Buffer.from(shot.data, "base64")
          );
        }
      }
    }

    /* Reduced motion: reveals must be visible immediately, not animated in. */
    await cdp.send(
      "Emulation.setEmulatedMedia",
      { features: [{ name: "prefers-reduced-motion", value: "reduce" }] },
      sessionId
    );
    await navigate(cdp, sessionId, `${base}/index.html`);
    const reduced = await evaluate(
      cdp,
      sessionId,
      `(() => {
        const item = document.querySelector(".reveal");
        return {
          opacity: item ? getComputedStyle(item).opacity : "0",
          seconds: item ? parseFloat(getComputedStyle(item).animationDuration) : 0,
        };
      })()`
    );
    check(reduced.opacity === "1", "reduced motion: reveal content is still hidden");
    check(reduced.seconds <= 0.001, `reduced motion: animations still run (${reduced.seconds}s)`);

    cdp.ws.close();
  } finally {
    child.kill();
    server.close();
  }

  if (failures.length) {
    console.error("layout checks failed:");
    for (const failure of failures) console.error(`  - ${failure}`);
    process.exitCode = 1;
    return;
  }
  console.log(`layout ok: ${pages.length} pages × ${viewports.length} viewports${screenshotDir ? ` (screenshots in ${screenshotDir})` : ""}`);
  process.exitCode = 0;
};

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
