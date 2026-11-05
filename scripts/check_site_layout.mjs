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
    this.ws = new WebSocket(url);
    this.ws.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && this.pending.has(message.id)) {
        const { resolve: done, reject } = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) reject(new Error(message.error.message));
        else done(message.result);
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
              auditToggle?.click();
              const fixedScore = document.querySelector("[data-audit-score]")?.textContent;
              const fixed = document.querySelector("[data-audit]")?.classList.contains("is-fixed");
              auditToggle?.click();
              const restoredScore = document.querySelector("[data-audit-score]")?.textContent;

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
                fixedScore,
                fixed,
                restoredScore,
                dupToggles: dupBefore !== dupAfter,
              };
            })()`
          );
          check(landing.canvas > 5000, `${label}: 3-D canvas appears blank`);
          check(landing.cards >= 12, `${label}: detection gallery is incomplete`);
          check(landing.filtersWork, `${label}: detection filter did not narrow the gallery`);
          check(landing.audit, `${label}: landing audit card missing`);
          check(landing.fixed && landing.fixedScore === "100", `${label}: recorded fix did not reach 100`);
          check(landing.restoredScore === "50", `${label}: recorded fix did not restore to 50`);
          check(landing.dupToggles, `${label}: duplicate toggle did not change state`);
          if (viewport.name === "mobile") {
            check(landing.opened, `${label}: mobile navigation did not open`);
            check(landing.closed, `${label}: Escape did not close the mobile navigation`);
          }
        }

        if (page === "demo.html") {
          const fixtureCount = await evaluate(
            cdp,
            sessionId,
            `document.querySelectorAll("[data-fixture-list] button").length`
          );
          check(fixtureCount === 3, `${label}: expected 3 code fixtures, found ${fixtureCount}`);
          const demo = await evaluate(
            cdp,
            sessionId,
            `(async () => {
              const wait = (ms) => new Promise((done) => setTimeout(done, ms));
              const tabs = [...document.querySelectorAll('[data-group-tabs] [role="tab"]')];
              tabs.find((tab) => tab.textContent.includes("Datasets"))?.click();
              await wait(120);
              const score = document.querySelector("[data-output] .score-ring span")?.textContent;
              const findings = document.querySelectorAll("[data-output] .finding").length;
              const high = [...document.querySelectorAll("[data-output] .demo-filters .chip")][1];
              high?.click();
              await wait(80);
              const pressed = [...document.querySelectorAll("[data-output] .demo-filters .chip")].findIndex((chip) => chip.getAttribute("aria-pressed") === "true");
              const preview = document.querySelectorAll("[data-output] .preview-table tbody tr").length;
              const active = document.querySelector("[data-demo]")?.dataset.activeFixture;
              tabs.find((tab) => tab.textContent.includes("Code"))?.click();
              await wait(120);
              const details = document.querySelector("[data-output] .finding");
              if (details) details.open = true;
              await wait(80);
              return {
                score,
                findings,
                pressed,
                preview,
                active,
                sourceLines: document.querySelectorAll("[data-output] .source-line").length,
                highlighted: document.querySelectorAll("[data-output] .source-line.is-target").length,
              };
            })()`
          );
          check(demo.score === "50", `${label}: leaky dataset score should be 50, got ${demo.score}`);
          check(demo.findings === 2, `${label}: expected 2 dataset findings, got ${demo.findings}`);
          check(demo.pressed === 1, `${label}: severity filter did not activate`);
          check(demo.preview > 0, `${label}: dataset preview table is empty`);
          check(demo.active === "leaky_transactions", `${label}: dataset tab did not select a fixture`);
          check(demo.sourceLines > 10, `${label}: code fixture did not render its source`);
          check(demo.highlighted === 1, `${label}: opening a finding did not highlight its source line`);
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
            `(() => {
              const nav = document.querySelector("[data-docs-nav]");
              const toggle = document.querySelector("[data-docs-toggle]");
              return {
                nav: Boolean(nav),
                toggleVisible: toggle ? getComputedStyle(toggle).display !== "none" : false,
                sections: document.querySelectorAll("[data-doc-section]").length,
              };
            })()`
          );
          check(docs.nav, `${label}: docs sidebar missing`);
          check(docs.sections >= 10, `${label}: docs sections are incomplete`);
          check(
            viewport.name === "mobile" ? docs.toggleVisible : !docs.toggleVisible,
            `${label}: docs contents toggle visibility is wrong`
          );
        }

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
