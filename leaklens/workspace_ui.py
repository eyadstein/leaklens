"""The single-page workspace UI, served by ``leaklens.workspace``.

Kept as a Python string so the package needs no changes to its data-file
configuration; the page has no build step and no external requests.
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LeakLens workspace</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--muted:#8b949e;
--high:#ff7b72;--medium:#e3b341;--low:#7ee787;--accent:#58a6ff;--ok:#2ea043}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg)}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;padding:16px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0}
header .muted{font-size:12px}
main{padding:16px 20px 40px;max-width:1400px;margin:0 auto}
.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
input[type=text],input[type=number],select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit;min-width:0}
input[type=text],input[type=number]{width:100%}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}
.row>.grow{flex:1 1 260px}
button{font:inherit;border-radius:6px;border:1px solid var(--line);padding:8px 14px;cursor:pointer;background:#21262d;color:var(--fg)}
button.primary{background:#238636;border-color:#2ea043}
button:disabled{opacity:.5;cursor:not-allowed}
button.finding{display:block;width:100%;text-align:left;background:none;border:1px solid transparent;border-radius:6px;padding:8px;color:var(--fg)}
button.finding:hover,button.finding:focus-visible{border-color:var(--line);background:#1c2128;outline:none}
button.finding[aria-current=true]{border-color:var(--accent);background:#1c2128}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid currentColor;margin-right:6px}
.high{color:var(--high)}.medium{color:var(--medium)}.low{color:var(--low)}
.status{color:var(--accent)}
.status.new{color:var(--high)}.status.changed{color:var(--medium)}
.status.unchanged,.status.suppressed{color:var(--muted)}
.status.suppressed{text-decoration:line-through}
.status.resolved{color:var(--ok)}
.badge{display:inline-block;font-size:12px;padding:2px 9px;border-radius:999px;background:#21262d;border:1px solid var(--line)}
.badge.running{color:var(--accent);border-color:var(--accent)}
.badge.done{color:var(--ok);border-color:var(--ok)}
.badge.cancelled,.badge.failed{color:var(--high);border-color:var(--high)}
progress{width:100%;height:10px;accent-color:var(--ok)}
.stats{display:flex;flex-wrap:wrap;gap:18px;font-size:14px;margin-top:8px}
.stats b{font-size:18px;display:block}
.layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(340px,440px);gap:16px;align-items:start}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
details.cluster{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;background:#0f141a}
details.cluster>summary{cursor:pointer;padding:9px 12px;font-size:13px}
.files{list-style:none;margin:0;padding:0 8px 8px}
.meta{font-size:12px;color:var(--muted);margin-top:4px}
pre{background:#010409;border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto;font-size:12.5px;line-height:1.5;margin:8px 0}
pre .hl{background:#1f6feb33;display:block}
pre .ln{color:var(--muted);display:inline-block;width:3.2em;user-select:none}
.evidence{margin:6px 0 0;padding-left:18px;font-size:13px}
.evidence li{margin-bottom:4px}
.kv{font-size:12px;color:var(--muted);word-break:break-all}
a{color:var(--accent)}
.empty{color:var(--muted);font-size:14px;padding:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px 12px;margin-top:12px}
h3{font-size:13px;margin:16px 0 6px}
.table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}
.table th,.table td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.table th{color:var(--muted);font-weight:600}
.recs{margin:0;padding-left:20px;font-size:13px}
.recs>li{margin-bottom:12px}
.recs ul{margin:4px 0 0;padding-left:18px}
.graph-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(340px,440px);gap:16px;align-items:start}
@media(max-width:1000px){.graph-layout{grid-template-columns:1fr}}
.node-list{list-style:none;margin:0;padding:0}
button.node{display:block;width:100%;text-align:left;background:none;border:1px solid transparent;border-radius:6px;padding:8px;color:var(--fg)}
button.node:hover,button.node:focus-visible{border-color:var(--line);background:#1c2128;outline:none}
button.node[aria-current=true]{border-color:var(--accent);background:#1c2128}
.node-arrow{color:var(--muted);margin:0 6px}
.node-relation{font-size:12px;color:var(--accent)}
.pager{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
.finding-highlight{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}
footer{padding:0 20px 40px;font-size:12px;color:var(--muted)}
</style></head><body>
<header>
  <h1>LeakLens workspace</h1>
  <span class="badge">v@@VERSION@@</span>
  <span class="muted">root: <code id="root">@@ROOT@@</code></span>
</header>
<main>
  <section class="card" aria-labelledby="policy-title">
    <h2 id="policy-title" style="font-size:14px;margin:0 0 10px">Baseline and suppression policy</h2>
    <div class="row">
      <div class="grow">
        <label for="baseline">Baseline file, relative to the workspace root (blank = <code>[baseline].path</code> from <code>.leaklens.toml</code>)</label>
        <input id="baseline" type="text" placeholder="leaklens-baseline.json" spellcheck="false" autocomplete="off">
      </div>
    </div>
    <p id="policy-info" class="meta" role="status" aria-live="polite"></p>
    <p class="meta">Suppression rules in <code>.leaklens.toml</code> apply to both scan types.
      Suppressed findings stay in the results with their reason and fingerprint and do not count
      toward the trust score. The workspace never writes or updates a baseline: use
      <code>leaklens --scan . --write-baseline</code> or <code>--update-baseline</code>.</p>
  </section>

  <section class="card" aria-labelledby="scan-title">
    <h2 id="scan-title" style="font-size:14px;margin:0 0 10px">Start a scan</h2>
    <form id="scan-form" class="row">
      <div class="grow">
        <label for="path">File or folder, relative to the workspace root</label>
        <input id="path" name="path" type="text" value="." spellcheck="false" autocomplete="off">
      </div>
      <div>
        <label for="sql">Include .sql files</label>
        <input id="sql" type="checkbox">
      </div>
      <div>
        <label for="ai">Include AI hypotheses</label>
        <input id="ai" type="checkbox">
      </div>
      <div>
        <button id="start" class="primary" type="submit">Scan</button>
        <button id="cancel" type="button" disabled>Cancel</button>
      </div>
    </form>
    <p id="notice" class="meta" role="status" aria-live="polite"></p>
    <progress id="bar" max="1" value="0" aria-label="Scan progress"></progress>
    <div class="stats">
      <span><b id="state-badge" class="badge">idle</b></span>
      <span class="muted">files <b class="muted" id="done">0</b></span>
      <span class="muted">current <span id="current">-</span></span>
      <span class="muted">elapsed <span id="elapsed">0</span>s</span>
      <span class="high">high <b id="c-high">0</b></span>
      <span class="medium">medium <b id="c-medium">0</b></span>
      <span class="low">low <b id="c-low">0</b></span>
    </div>
  </section>

  <section id="results" hidden>
    <div class="card">
      <div class="row">
        <span><label>Trust score</label><b id="trust" style="font-size:22px">-</b></span>
        <span class="grow"></span>
        <span><label for="f-sev">Severity</label>
          <select id="f-sev"><option value="">all</option><option value="high">high</option>
          <option value="medium">medium</option><option value="low">low</option></select></span>
        <span><label for="f-kind">Kind</label><select id="f-kind"><option value="">all</option></select></span>
        <span><label for="f-status">Status</label><select id="f-status"><option value="">all</option></select></span>
        <span><label for="f-q">Search</label><input id="f-q" type="text" placeholder="kind, file, detail" style="min-width:200px"></span>
        <span><button id="g-open-scan" type="button">Evidence graph</button></span>
      </div>
      <p class="meta" id="summary"></p>
      <p class="meta" id="policy-notes"></p>
      <p class="meta" id="ai-summary"></p>
    </div>
    <div class="layout">
      <div>
        <h2 style="font-size:14px">Findings by evidence cluster</h2>
        <div id="clusters"><p class="empty">No findings.</p></div>
        <div id="resolved"></div>
      </div>
      <aside class="card" id="detail" aria-live="polite">
        <p class="empty">Select a finding to see its evidence and source.</p>
      </aside>
    </div>
  </section>

  <section class="card" aria-labelledby="ds-title">
    <h2 id="ds-title" style="font-size:14px;margin:0 0 10px">Audit a dataset</h2>
    <p class="meta">Reads a dataset from the workspace root and runs the same audit engine as
      <code>leaklens FILE --target COL</code>. Nothing is uploaded or written to disk.</p>
    <p class="meta" id="ds-formats"></p>
    <form id="ds-form" class="row">
      <div class="grow">
        <label for="ds-path">Dataset file, relative to the workspace root</label>
        <input id="ds-path" type="text" placeholder="train.csv" spellcheck="false" autocomplete="off">
      </div>
      <div>
        <button id="ds-inspect" type="submit">Inspect</button>
      </div>
    </form>
    <p id="ds-notice" class="meta" role="status" aria-live="polite"></p>
    <div id="ds-config" hidden>
      <div id="ds-preview"></div>
      <div class="grid">
        <div>
          <label for="ds-target">Target column (required)</label>
          <select id="ds-target"></select>
        </div>
        <div>
          <label for="ds-threshold">Threshold (0.5 to 1.0)</label>
          <input id="ds-threshold" type="number" min="0.5" max="1" step="0.01" value="0.95">
        </div>
        <div>
          <label for="ds-time">Time column (optional)</label>
          <select id="ds-time"></select>
        </div>
        <div>
          <label for="ds-group">Group column (optional)</label>
          <select id="ds-group"></select>
        </div>
        <div>
          <label for="ds-test">Held-out dataset (optional)</label>
          <input id="ds-test" type="text" placeholder="test.csv" spellcheck="false" autocomplete="off">
        </div>
        <div>
          <label for="ds-ai">Include AI hypotheses</label>
          <input id="ds-ai" type="checkbox">
        </div>
      </div>
      <div class="row" style="margin-top:12px">
        <button id="ds-run" class="primary" type="button">Audit</button>
        <button id="ds-cancel" type="button" disabled>Cancel</button>
        <span id="ds-state" class="badge">idle</span>
        <span class="muted" id="ds-elapsed"></span>
      </div>
    </div>
  </section>

  <section id="ds-results" hidden>
    <div class="card">
      <div class="row">
        <span><label>Trust score</label><b id="ds-trust" style="font-size:22px">-</b></span>
        <span class="grow"></span>
        <span><label for="ds-status">Status</label>
          <select id="ds-status"><option value="">all</option></select></span>
        <span><button id="g-open-dataset" type="button">Evidence graph</button></span>
        <span id="ds-links"></span>
      </div>
      <p class="meta" id="ds-summary"></p>
      <p class="meta" id="ds-policy-notes"></p>
      <p class="meta" id="ds-ai-summary"></p>
      <p class="meta high" id="ds-error"></p>
    </div>
    <div class="layout">
      <div>
        <h2 style="font-size:14px">Findings by evidence cluster</h2>
        <div id="ds-clusters"></div>
        <h2 style="font-size:14px">Recommended next actions</h2>
        <div id="ds-recs"></div>
        <div id="ds-resolved"></div>
      </div>
      <aside class="card" aria-live="polite">
        <h3 style="margin-top:0">Dataset</h3>
        <div id="ds-audit" class="kv"></div>
        <h3>Held-out impact</h3>
        <div id="ds-impact" class="kv"></div>
        <h3>Run</h3>
        <div id="ds-request" class="kv"></div>
      </aside>
    </div>
  </section>

  <section class="card" aria-labelledby="history-title">
    <div class="row">
      <h2 id="history-title" style="font-size:14px;margin:0">Session history</h2>
      <span class="grow"></span>
      <button id="history-refresh" type="button">Refresh</button>
    </div>
    <p class="meta" id="history-note" role="status" aria-live="polite">Loading retained sessions…</p>
    <div id="history-list"></div>
  </section>

  <section class="card" aria-labelledby="compare-title">
    <h2 id="compare-title" style="font-size:14px;margin:0 0 10px">Compare retained sessions</h2>
    <div class="row">
      <span class="grow"><label for="cmp-previous">Previous</label>
        <select id="cmp-previous" style="width:100%"></select></span>
      <span class="grow"><label for="cmp-current">Current</label>
        <select id="cmp-current" style="width:100%"></select></span>
      <button id="cmp-run" type="button">Compare</button>
    </div>
    <p class="meta">Comparison uses stable finding fingerprints and the existing baseline
      classification. A finding missing from a cancelled or partial scan is inconclusive,
      never resolved.</p>
    <p id="cmp-notice" class="meta" role="status" aria-live="polite"></p>
    <div id="cmp-summary"></div>
    <div id="cmp-list"></div>
  </section>

  <section id="graph-panel" hidden aria-labelledby="graph-title">
    <div class="card">
      <div class="row">
        <h2 id="graph-title" style="font-size:14px;margin:0">Evidence graph</h2>
        <span class="grow"></span>
        <span id="graph-source" class="badge">result</span>
        <button id="graph-close" type="button">Hide</button>
      </div>
      <p class="meta">Nodes are evidence records the engine already produced; relationships are the
        engine's recorded links. Nothing on this page is inferred or added.</p>
      <div class="row">
        <span><label for="g-kind">Node type</label><select id="g-kind"><option value="">all</option></select></span>
        <span><label for="g-relation">Relationship</label><select id="g-relation"><option value="">all</option></select></span>
        <span><label for="g-sev">Severity</label><select id="g-sev">
          <option value="">all</option><option value="high">high</option>
          <option value="medium">medium</option><option value="low">low</option></select></span>
        <span><label for="g-status">Status</label><select id="g-status"><option value="">all</option></select></span>
        <span class="grow"><label for="g-q">Search</label><input id="g-q" type="text" placeholder="evidence id, kind, description, finding" spellcheck="false" autocomplete="off"></span>
      </div>
      <p class="meta" id="g-summary" role="status" aria-live="polite"></p>
      <p class="meta high" id="g-error"></p>
    </div>
    <div class="graph-layout">
      <div class="card">
        <h3 style="margin-top:0">Nodes</h3>
        <div id="g-nodes"><p class="empty">Open a scan or dataset result to browse its evidence graph.</p></div>
        <div class="pager">
          <button id="g-prev" type="button" disabled>Previous</button>
          <span class="meta" id="g-page"></span>
          <button id="g-next" type="button" disabled>Next</button>
        </div>
      </div>
      <aside class="card" id="g-detail" aria-live="polite">
        <p class="empty">Select a node to inspect its evidence and relationships.</p>
      </aside>
    </div>
  </section>
</main>
<footer>
  <p><a id="report-link" href="#" hidden>Download HTML report</a>
     <a id="json-link" href="#" hidden style="margin-left:16px">Raw results (JSON)</a></p>
  <p>Read-only: scans run in memory, nothing is written and no code is executed.</p>
</footer>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const state = {id: null, info: null, results: null, selected: null, source: null};
const history = {data: null};
const comparison = {data: null, type: null, current: null};
const graph = {context: null, id: null, data: null, selected: null,
  offset: 0, limit: 50, focus: null, finding: null, request: 0, timer: null};

function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, value);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
}

async function api(path, opts) {
  const response = await fetch(path, opts);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = body && body.detail;
    const message = typeof detail === "string" ? detail
      : detail ? JSON.stringify(detail) : String(body || response.status);
    throw new Error(message);
  }
  return body;
}

function setState(name) {
  const badge = $("state-badge");
  badge.textContent = name;
  badge.className = "badge " + name;
}

function counts(c) {
  $("c-high").textContent = c.high;
  $("c-medium").textContent = c.medium;
  $("c-low").textContent = c.low;
}

function statusChip(finding) {
  return finding.status ? el("span", {class: "chip status " + finding.status,
    text: finding.status}) : null;
}

function graphButton(context, index) {
  const button = el("button", {type: "button", text: "Explore in evidence graph"});
  button.addEventListener("click", () => openGraph(context,
    context === "scan" ? state.id : ds.id, index));
  return el("div", {}, button);
}

function policyText(policy) {
  if (!policy) return "";
  const parts = [];
  if (policy.baseline) {
    parts.push("baseline " + policy.baseline.new + " new, " +
      policy.baseline.unchanged + " unchanged, " +
      policy.baseline.changed + " changed, " +
      policy.baseline.resolved + " resolved" +
      (policy.baseline.path ? " (" + policy.baseline.path + ")" : ""));
  } else if (!policy.baseline_selected) {
    parts.push("no baseline selected: findings are not compared");
  }
  const statuses = policy.statuses || {};
  for (const name of Object.keys(statuses).sort()) {
    parts.push(statuses[name] + " " + name);
  }
  if (policy.unclassified) parts.push(policy.unclassified + " not compared");
  if (policy.suppressions_configured) {
    parts.push(policy.suppressions_configured + " suppression rule(s) configured");
  }
  if (policy.baseline_skipped) parts.push(policy.baseline_skipped);
  return parts.join(" · ");
}

function policyNotes(policy) {
  const diagnostics = (policy && policy.diagnostics) || [];
  return diagnostics.length ? "Policy diagnostics: " + diagnostics.join("; ") : "";
}

function aiSummary(data) {
  if (!("ai" in data)) return "";
  if (!data.ai) {
    return "AI analysis unavailable; deterministic results are unchanged.";
  }
  const ai = data.ai;
  const hypotheses = ai.hypotheses || [];
  const counts = {supported: 0, weak: 0, insufficient: 0};
  for (const item of hypotheses) {
    const key = item.sufficiency in counts ? item.sufficiency : "weak";
    counts[key] += 1;
  }
  const parts = [];
  if (hypotheses.length) {
    parts.push(hypotheses.length + " hypotheses");
    if (counts.supported) parts.push(counts.supported + " supported");
    if (counts.weak) parts.push(counts.weak + " weak");
    if (counts.insufficient) parts.push(counts.insufficient + " ranking only");
  }
  if (ai.selection && ai.selection.applied) {
    parts.push("selective reporting at p >= " +
      Number(ai.selection.threshold).toFixed(2));
  }
  return "AI (experimental, never changes the trust score): " +
    (parts.length ? parts.join(" · ") : "no hypotheses") +
    ". Open the HTML report for evidence and limitations.";
}

function renderResolved(box, items) {
  box.replaceChildren();
  if (!items || !items.length) return;
  box.append(el("details", {class: "cluster"},
    el("summary", {text: "Resolved since baseline (" + items.length + ")"}),
    el("ul", {class: "files"}, ...items.map((entry) => el("li", {},
      el("span", {class: "chip status resolved", text: "resolved"}),
      el("strong", {text: entry.kind}),
      el("span", {class: "meta", text: " · " + entry.column + " · " +
        entry.fingerprint.slice(0, 12)}))))));
}

function fillStatusSelect(select, policy) {
  const statuses = Object.keys((policy && policy.statuses) || {}).sort();
  const options = [el("option", {value: "", text: "all"}),
    ...statuses.map((name) => el("option", {value: name, text: name}))];
  if (policy && policy.unclassified) {
    options.push(el("option", {value: "__none__", text: "not compared"}));
  }
  select.replaceChildren(...options);
}

function renderStatus(status) {
  setState(status.state);
  $("done").textContent = status.done + "/" + status.total;
  $("current").textContent = status.current || "-";
  $("elapsed").textContent = (status.elapsed_ms / 1000).toFixed(1);
  $("bar").max = Math.max(status.total, 1);
  $("bar").value = status.done;
  counts(status.counts);
  $("cancel").disabled = status.state !== "running";
  if (status.errors.length) {
    $("notice").textContent = status.errors.length + " file(s) could not be scanned: " +
      status.errors.map((e) => e.path + " (" + e.message + ")").join("; ");
  }
}

async function poll() {
  if (!state.id) return;
  let status;
  try {
    status = await api("/api/scans/" + state.id);
  } catch (error) {
    $("notice").textContent = error.message;
    return;
  }
  renderStatus(status);
  if (status.state === "running") {
    setTimeout(poll, 400);
    return;
  }
  await loadResults();
  loadHistory();
}

function stopScanWatch() {
  if (state.source) {
    state.source.close();
    state.source = null;
  }
}

function watchScan() {
  stopScanWatch();
  if (!state.id) return;
  if (!window.EventSource) {
    poll();
    return;
  }
  const id = state.id;
  const source = new EventSource("/api/scans/" + id + "/events");
  state.source = source;
  let finished = false;
  const stop = () => {
    finished = true;
    source.close();
    if (state.source === source) state.source = null;
  };
  source.addEventListener("snapshot", (event) => {
    if (finished || id !== state.id) return;
    const status = JSON.parse(event.data);
    renderStatus(status);
  });
  source.addEventListener("failure", (event) => {
    if (finished || id !== state.id) return;
    const status = JSON.parse(event.data);
    renderStatus(status);
    const failure = status.failure || {};
    const details = (failure.errors || [])
      .map((item) => item.path + " (" + item.message + ")").join("; ");
    $("notice").textContent = (failure.message || "the scan failed") +
      (details ? ": " + details : "");
  });
  source.addEventListener("done", async (event) => {
    if (finished || id !== state.id) return;
    stop();
    const status = JSON.parse(event.data);
    renderStatus(status);
    $("notice").textContent = "Live updates complete.";
    try {
      await loadResults();
    } catch (error) {
      $("notice").textContent = error.message;
    }
    loadHistory();
  });
  source.onerror = () => {
    if (finished || id !== state.id) return;
    stop();
    $("notice").textContent =
      "Live updates disconnected; reconnecting with polling.";
    poll();
  };
}

async function loadResults() {
  const data = await api("/api/scans/" + state.id + "/results");
  state.results = data;
  $("results").hidden = false;
  $("trust").textContent = data.trust_score + "/100";
  const c = data.counts;
  $("summary").textContent =
    data.findings.length + " findings from " + data.files + " files in " +
    (data.duration_ms / 1000).toFixed(1) + "s · high " + c.high +
    " · medium " + c.medium + " · low " + c.low +
    (policyText(data.policy) ? " · " + policyText(data.policy) : "");
  $("policy-notes").textContent = policyNotes(data.policy);
  $("ai-summary").textContent = aiSummary(data);
  $("report-link").href = "/api/scans/" + state.id + "/report";
  $("report-link").hidden = false;
  $("json-link").href = "/api/scans/" + state.id + "/results";
  $("json-link").hidden = false;
  const kinds = [...new Set(data.findings.map((f) => f.kind))].sort();
  const kindSelect = $("f-kind");
  kindSelect.replaceChildren(el("option", {value: "", text: "all"}),
    ...kinds.map((kind) => el("option", {value: kind, text: kind})));
  fillStatusSelect($("f-status"), data.policy);
  renderResolved($("resolved"), data.resolved);
  renderFindings();
}

function visibleFindings() {
  const data = state.results;
  if (!data) return [];
  const severity = $("f-sev").value;
  const kind = $("f-kind").value;
  const status = $("f-status").value;
  const query = $("f-q").value.trim().toLowerCase();
  return data.findings.filter((f) => {
    if (severity && f.severity !== severity) return false;
    if (kind && f.kind !== kind) return false;
    if (status === "__none__") {
      if (f.status) return false;
    } else if (status && f.status !== status) return false;
    if (!query) return true;
    return (f.kind + " " + f.column + " " + f.detail + " " +
      (f.advice || "")).toLowerCase().includes(query);
  });
}

function renderFindings() {
  const data = state.results;
  const box = $("clusters");
  box.replaceChildren();
  if (!data || !data.findings.length) {
    box.append(el("p", {class: "empty", text: "No findings."}));
    return;
  }
  const visible = visibleFindings();
  if (!visible.length) {
    box.append(el("p", {class: "empty", text: "No findings match these filters."}));
    return;
  }
  const allowed = new Set(visible.map((f) => f.index));
  const byIndex = new Map(data.findings.map((f) => [f.index, f]));
  const grouped = new Set();
  for (const cluster of data.clusters) {
    const members = cluster.finding_indices.filter((i) => allowed.has(i));
    if (!members.length) continue;
    members.forEach((i) => grouped.add(i));
    box.append(clusterNode(cluster, members.map((i) => byIndex.get(i))));
  }
  const rest = visible.filter((f) => !grouped.has(f.index));
  if (rest.length) {
    box.append(el("details", {class: "cluster", open: ""},
      el("summary", {text: "Other findings (" + rest.length + ")"}),
      el("ul", {class: "files"}, ...rest.map(findingNode))));
  }
}

function clusterNode(cluster, findings) {
  const confidence = Math.round((cluster.confidence.score || 0) * 100);
  const head = el("summary", {},
    el("span", {class: "chip " + cluster.priority, text: cluster.priority}),
    el("strong", {text: cluster.anchor}),
    el("span", {class: "muted", text: " · " + findings.length + " finding(s) · " +
      cluster.anchor_kind + " · confidence " + confidence + "%"}));
  return el("details", {class: "cluster", open: ""}, head,
    el("ul", {class: "files"}, ...findings.map(findingNode)));
}

function findingNode(finding) {
  const where = finding.file ? finding.file + ":" + finding.line : finding.column;
  const row = el("button", {class: "finding", type: "button"},
    el("span", {class: "chip " + finding.severity, text: finding.severity}),
    statusChip(finding),
    el("strong", {text: finding.kind}),
    el("span", {class: "meta", text: " · " + where}),
    el("div", {class: "meta", text: finding.detail}));
  row.addEventListener("click", () => select(finding, row));
  row.setAttribute("data-index", String(finding.index));
  if (state.selected === finding.index) row.setAttribute("aria-current", "true");
  return el("li", {}, row);
}

async function select(finding, row) {
  state.selected = finding.index;
  document.querySelectorAll("button.finding[aria-current]").forEach((node) => {
    node.removeAttribute("aria-current");
  });
  if (row) row.setAttribute("aria-current", "true");
  const panel = $("detail");
  panel.replaceChildren();
  const blocks = [
    el("div", {},
      el("span", {class: "chip " + finding.severity, text: finding.severity}),
      statusChip(finding),
      el("strong", {text: finding.kind})),
    el("p", {text: finding.detail}),
    el("p", {class: "kv", text: (finding.file || finding.column) +
      (finding.line ? " · line " + finding.line : "")}),
    finding.status ? el("p", {class: "meta", text:
      "Policy status: " + finding.status + " · " +
      (finding.requires_attention
        ? "requires attention and counts toward the trust score"
        : "suppressed, so it does not count toward the trust score; the underlying finding and its evidence remain")}) : null,
    finding.previous ? el("div", {},
      el("label", {text: "Previous baseline entry"}),
      el("p", {class: "kv", text: "severity " + finding.previous.severity +
        " · kind " + finding.previous.kind + " · " + finding.previous.column})) : null,
    finding.suppression ? el("div", {},
      el("label", {text: "Suppression rule"}),
      el("p", {class: "kv", text: "reason: " + finding.suppression.reason +
        " · fingerprint: " + finding.suppression.fingerprint +
        (finding.suppression.expires ? " · expires " + finding.suppression.expires : "")})) : null,
    finding.advice ? el("div", {},
      el("label", {text: "Suggested fix"}),
      el("p", {text: finding.advice})) : null,
    graphButton("scan", finding.index),
    el("div", {},
      el("label", {text: "Fingerprint"}),
      el("p", {class: "kv", text: finding.fingerprint})),
  ];
  panel.append(...blocks.filter(Boolean));
  if (finding.evidence && finding.evidence.length) {
    panel.append(el("label", {text: "Evidence (" + finding.evidence.length + ")"}),
      el("ul", {class: "evidence"}, ...finding.evidence.map((item) =>
        el("li", {}, el("strong", {text: item.kind + ": "}), el("span", {text: item.description})))));
  }
  const previewable = finding.file && /\\.(py|sql)$/i.test(finding.file) && finding.line;
  if (previewable) {
    const source = el("div", {}, el("label", {text: "Source"}), el("p", {class: "meta", text: "loading..."}));
    panel.append(source);
    try {
      const data = await api("/api/source?path=" + encodeURIComponent(finding.file) +
        "&line=" + finding.line);
      const pre = el("pre", {});
      for (const line of data.lines) {
        const span = el("span", {class: line.number === data.line ? "hl" : ""},
          el("span", {class: "ln", text: String(line.number)}),
          document.createTextNode(line.text));
        pre.append(span, document.createTextNode("\\n"));
      }
      source.replaceChildren(el("label", {text: "Source · " + data.path}), pre);
    } catch (error) {
      source.replaceChildren(el("label", {text: "Source"}),
        el("p", {class: "empty", text: error.message}));
    }
  } else if (finding.file) {
    panel.append(el("label", {text: "Source"}),
      el("p", {class: "meta", text: "Open " + finding.file +
        (finding.cell != null ? " cell " + finding.cell : "") + " in your editor."}));
  }
}

$("scan-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("start").disabled = true;
  $("notice").textContent = "";
  $("results").hidden = true;
  try {
    const started = await api("/api/scans", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: $("path").value, sql: $("sql").checked,
        include_ai: $("ai").checked,
        baseline: $("baseline").value.trim() || null}),
    });
    state.id = started.scan_id;
    state.selected = null;
    closeGraph();
    setState("running");
    $("cancel").disabled = false;
    watchScan();
    loadHistory();
  } catch (error) {
    setState("failed");
    $("notice").textContent = error.message;
  } finally {
    $("start").disabled = false;
  }
});

$("cancel").addEventListener("click", async () => {
  if (!state.id) return;
  try {
    const status = await api("/api/scans/" + state.id + "/cancel", {method: "POST"});
    renderStatus(status);
  } catch (error) {
    $("notice").textContent = error.message;
  }
});

for (const id of ["f-sev", "f-kind", "f-status", "f-q"]) {
  $(id).addEventListener("input", renderFindings);
}

// --- dataset audits -------------------------------------------------------

const ds = {id: null, columns: [], results: null, source: null};

function dsState(name) {
  const badge = $("ds-state");
  badge.textContent = name;
  badge.className = "badge " + name;
}

function dsBusy(busy) {
  $("ds-run").disabled = busy;
  $("ds-inspect").disabled = busy;
  $("ds-cancel").disabled = !busy;
}

function selectOptions(names, withEmpty) {
  const items = withEmpty ? [el("option", {value: "", text: "none"})] : [];
  return items.concat(names.map((name) => el("option", {value: name, text: name})));
}

function previewTable(data) {
  const names = data.columns.map((column) => column.name);
  return el("table", {class: "table"},
    el("thead", {}, el("tr", {}, ...names.map((name) => el("th", {text: name})))),
    el("tbody", {}, ...data.preview.map((row) => el("tr", {},
      ...names.map((name) => el("td", {text: row[name] == null ? "-" : String(row[name])}))))));
}

async function inspectDataset() {
  const path = $("ds-path").value.trim();
  $("ds-notice").textContent = "";
  $("ds-config").hidden = true;
  $("ds-results").hidden = true;
  if (!path) {
    $("ds-notice").textContent = "Enter a dataset path first.";
    return;
  }
  dsBusy(true);
  $("ds-notice").textContent = "Reading " + path + "...";
  try {
    const data = await api("/api/datasets/inspect", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: path}),
    });
    ds.columns = data.columns.map((column) => column.name);
    $("ds-target").replaceChildren(...selectOptions(ds.columns, false));
    $("ds-time").replaceChildren(...selectOptions(ds.columns, true));
    $("ds-group").replaceChildren(...selectOptions(ds.columns, true));
    if (data.target_candidates.length) $("ds-target").value = data.target_candidates[0];
    $("ds-preview").replaceChildren(previewTable(data));
    $("ds-notice").textContent = data.rows + " rows x " + ds.columns.length +
      " columns read from " + data.file + ". Pick a target and audit.";
    $("ds-config").hidden = false;
    dsState("ready");
  } catch (error) {
    $("ds-notice").textContent = error.message;
  } finally {
    dsBusy(false);
  }
}

async function pollDataset() {
  if (!ds.id) return;
  let status;
  try {
    status = await api("/api/datasets/" + ds.id);
  } catch (error) {
    $("ds-notice").textContent = error.message;
    dsBusy(false);
    return;
  }
  dsState(status.state);
  $("ds-elapsed").textContent = (status.elapsed_ms / 1000).toFixed(1) + "s";
  if (status.state === "queued" || status.state === "auditing") {
    $("ds-notice").textContent = "Auditing " + status.request.file + " (target " +
      status.request.target + ")...";
    setTimeout(pollDataset, 400);
    return;
  }
  dsBusy(false);
  if (status.state === "done") {
    await loadDatasetResults();
    loadHistory();
    return;
  }
  if (status.state === "failed" && status.error) {
    $("ds-notice").textContent = status.error.message + " (" + status.error.code + ")";
    renderDatasetError(status.error);
    loadHistory();
    return;
  }
  $("ds-notice").textContent = "Audit " + status.state + ".";
  loadHistory();
}

function stopDatasetWatch() {
  if (ds.source) {
    ds.source.close();
    ds.source = null;
  }
}

function watchDataset() {
  stopDatasetWatch();
  if (!ds.id) return;
  if (!window.EventSource) {
    pollDataset();
    return;
  }
  const id = ds.id;
  const source = new EventSource("/api/datasets/" + id + "/events");
  ds.source = source;
  let finished = false;
  const stop = () => {
    finished = true;
    source.close();
    if (ds.source === source) ds.source = null;
  };
  source.addEventListener("snapshot", (event) => {
    if (finished || id !== ds.id) return;
    const status = JSON.parse(event.data);
    dsState(status.state);
    $("ds-elapsed").textContent = (status.elapsed_ms / 1000).toFixed(1) + "s";
    if (status.state === "queued" || status.state === "auditing") {
      $("ds-notice").textContent = "Auditing " + status.request.file +
        " (target " + status.request.target + ")..." +
        (status.state === "queued"
          ? " queued: cancellation is still available."
          : " the engine is running; it cannot be interrupted.");
    }
  });
  source.addEventListener("failure", (event) => {
    if (finished || id !== ds.id) return;
    const status = JSON.parse(event.data);
    dsState(status.state);
    const failure = status.failure || {};
    $("ds-notice").textContent = (failure.message || "the audit failed") +
      (failure.code ? " (" + failure.code + ")" : "");
    renderDatasetError({message: failure.message || "the audit failed",
      code: failure.code || "audit_failed"});
  });
  source.addEventListener("done", async (event) => {
    if (finished || id !== ds.id) return;
    stop();
    const status = JSON.parse(event.data);
    dsState(status.state);
    $("ds-elapsed").textContent = (status.elapsed_ms / 1000).toFixed(1) + "s";
    dsBusy(false);
    if (status.state === "done") {
      await loadDatasetResults();
      $("ds-notice").textContent = "Audit complete.";
    } else if (status.state === "failed" && status.error) {
      $("ds-notice").textContent = status.error.message + " (" + status.error.code + ")";
      renderDatasetError(status.error);
    } else {
      $("ds-notice").textContent = "Audit " + status.state + ".";
    }
    loadHistory();
  });
  source.onerror = () => {
    if (finished || id !== ds.id) return;
    stop();
    $("ds-notice").textContent =
      "Live updates disconnected; reconnecting with polling.";
    pollDataset();
  };
}

async function loadDatasetResults() {
  try {
    const data = await api("/api/datasets/" + ds.id + "/results");
    ds.results = data;
    renderDataset(data);
  } catch (error) {
    $("ds-notice").textContent = error.message;
  }
}

function kvRows(box, pairs) {
  box.replaceChildren(...pairs
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) => el("div", {},
      el("b", {text: label + ": "}), el("span", {text: String(value)}))));
}

function datasetFinding(finding) {
  return el("li", {"data-index": String(finding.index)},
    el("span", {class: "chip " + finding.severity, text: finding.severity}),
    statusChip(finding),
    el("strong", {text: finding.kind}),
    el("span", {class: "meta", text: " · " + finding.column}),
    el("div", {class: "meta", text: finding.detail}),
    finding.previous ? el("div", {class: "meta", text: "previous baseline severity: " +
      finding.previous.severity}) : null,
    finding.suppression ? el("div", {class: "meta", text: "suppressed: " +
      finding.suppression.reason + " (" + finding.suppression.fingerprint.slice(0, 12) + ")"}) : null,
    finding.advice ? el("div", {class: "meta", text: "fix: " + finding.advice}) : null,
    graphButton("dataset", finding.index));
}

function dsVisibleFindings(data) {
  const found = data.findings || [];
  const status = $("ds-status").value;
  if (!status) return found;
  if (status === "__none__") return found.filter((finding) => !finding.status);
  return found.filter((finding) => finding.status === status);
}

function datasetClusters(data) {
  const box = $("ds-clusters");
  box.replaceChildren();
  const findings = data.findings || [];
  if (!findings.length) {
    box.append(el("p", {class: "empty", text: "No leakage found."}));
    return;
  }
  const visible = dsVisibleFindings(data);
  if (!visible.length) {
    box.append(el("p", {class: "empty", text: "No findings match these filters."}));
    return;
  }
  const allowed = new Set(visible.map((finding) => finding.index));
  const byIndex = new Map(findings.map((finding) => [finding.index, finding]));
  const grouped = new Set();
  for (const cluster of data.intelligence.clusters || []) {
    const members = cluster.finding_indices
      .filter((index) => allowed.has(index))
      .map((index) => byIndex.get(index)).filter(Boolean);
    if (!members.length) continue;
    members.forEach((finding) => grouped.add(finding.index));
    const confidence = Math.round((cluster.confidence.score || 0) * 100);
    box.append(el("details", {class: "cluster", open: ""},
      el("summary", {},
        el("span", {class: "chip " + cluster.priority, text: cluster.priority}),
        el("strong", {text: cluster.anchor}),
        el("span", {class: "muted", text: " · " + members.length + " finding(s) · " +
          cluster.anchor_kind + " · confidence " + confidence + "%"})),
      el("ul", {class: "files"}, ...members.map(datasetFinding))));
  }
  const rest = visible.filter((finding) => !grouped.has(finding.index));
  if (rest.length) {
    box.append(el("details", {class: "cluster", open: ""},
      el("summary", {text: "Other findings (" + rest.length + ")"}),
      el("ul", {class: "files"}, ...rest.map(datasetFinding))));
  }
}

function datasetRecommendations(data) {
  const box = $("ds-recs");
  box.replaceChildren();
  const items = data.intelligence.recommendations || [];
  if (!items.length) {
    box.append(el("p", {class: "empty", text: "No recommendations."}));
    return;
  }
  box.append(el("ol", {class: "recs"}, ...items.map((item) => el("li", {},
    el("span", {class: "chip " + item.priority, text: item.priority}),
    el("strong", {text: item.title}),
    el("div", {class: "meta", text: item.description}),
    el("ul", {}, ...(item.actions || []).map((action) => el("li", {text: action})))))));
}

function datasetSidebar(data) {
  const audit = data.audit || {};
  kvRows($("ds-audit"), [
    ["rows", audit.rows], ["columns", audit.cols],
    ["duplicate rows", audit.duplicate_rows], ["minority rate", audit.minority_rate],
  ]);
  const impact = data.impact;
  if (!impact) {
    $("ds-impact").replaceChildren(el("p", {class: "empty",
      text: "Add a held-out dataset to measure train/test impact."}));
  } else {
    kvRows($("ds-impact"), [
      ["train rows", impact.train_rows], ["held-out rows", impact.test_rows],
      ["duplicate held-out rows", impact.duplicate_test_rows],
      ["duplicate held-out %", impact.duplicate_test_pct],
      ["shared groups", impact.group_column ? impact.shared_group_count : null],
      ["shared group held-out %", impact.group_column ? impact.shared_group_test_pct : null],
      ["temporal overlap", impact.temporal_overlap === null ? null : String(impact.temporal_overlap)],
      ["train max time", impact.train_max_time], ["held-out min time", impact.test_min_time],
    ]);
  }
  const request = data.request || {};
  kvRows($("ds-request"), [
    ["file", request.file], ["held-out", request.test], ["target", request.target],
    ["threshold", request.threshold], ["time", request.time_column],
    ["group", request.group_column],
  ]);
}

function renderDataset(data) {
  $("ds-results").hidden = false;
  $("ds-error").textContent = "";
  $("ds-trust").textContent = data.trust_score + "/100";
  const counts = data.counts;
  const summary = data.summary;
  $("ds-summary").textContent = summary.file + " · target " + summary.target +
    " · threshold " + summary.threshold + " · " + summary.rows + " rows x " +
    summary.columns + " columns" +
    (summary.test_rows != null ? " · held-out " + summary.test_rows + " rows" : "") +
    " · " + (data.elapsed_ms / 1000).toFixed(1) + "s" +
    " · high " + (counts.high || 0) + " · medium " + (counts.medium || 0) +
    " · low " + (counts.low || 0) +
    (policyText(data.policy) ? " · " + policyText(data.policy) : "");
  $("ds-policy-notes").textContent = policyNotes(data.policy);
  $("ds-ai-summary").textContent = aiSummary(data);
  $("ds-links").replaceChildren(
    el("a", {href: "/api/datasets/" + ds.id + "/report", text: "HTML report"}),
    el("a", {href: "/api/datasets/" + ds.id + "/results", text: "Raw results (JSON)",
      style: "margin-left:16px"}));
  fillStatusSelect($("ds-status"), data.policy);
  datasetClusters(data);
  datasetRecommendations(data);
  datasetSidebar(data);
  renderResolved($("ds-resolved"), data.resolved);
}

function renderDatasetError(error) {
  $("ds-results").hidden = false;
  $("ds-trust").textContent = "-";
  $("ds-summary").textContent = "";
  $("ds-policy-notes").textContent = "";
  $("ds-ai-summary").textContent = "";
  $("ds-links").replaceChildren();
  $("ds-error").textContent = error.message + " [" + error.code + "]";
  $("ds-status").replaceChildren(el("option", {value: "", text: "all"}));
  datasetClusters({findings: [], intelligence: {clusters: []}});
  datasetRecommendations({intelligence: {recommendations: []}});
  kvRows($("ds-audit"), []);
  kvRows($("ds-impact"), []);
  kvRows($("ds-request"), []);
  $("ds-resolved").replaceChildren();
}

$("ds-form").addEventListener("submit", (event) => {
  event.preventDefault();
  inspectDataset();
});

$("ds-run").addEventListener("click", async () => {
  const body = {
    path: $("ds-path").value.trim(),
    target: $("ds-target").value,
    threshold: Number($("ds-threshold").value),
    test_path: $("ds-test").value.trim() || null,
    time_col: $("ds-time").value || null,
    group_col: $("ds-group").value || null,
    baseline: $("baseline").value.trim() || null,
    include_ai: $("ds-ai").checked,
  };
  $("ds-notice").textContent = "";
  $("ds-results").hidden = true;
  dsBusy(true);
  try {
    const started = await api("/api/datasets", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    ds.id = started.dataset_id;
    ds.results = null;
    closeGraph();
    dsState("queued");
    watchDataset();
    loadHistory();
  } catch (error) {
    dsState("failed");
    $("ds-notice").textContent = error.message;
    dsBusy(false);
  }
});

$("ds-cancel").addEventListener("click", async () => {
  if (!ds.id) return;
  try {
    const status = await api("/api/datasets/" + ds.id + "/cancel", {method: "POST"});
    dsState(status.state);
    $("ds-notice").textContent = "Audit cancelled before the engine started.";
    dsBusy(false);
  } catch (error) {
    $("ds-notice").textContent = error.message;
  }
});

$("ds-status").addEventListener("input", () => {
  if (ds.results) datasetClusters(ds.results);
});

// --- session history and comparisons --------------------------------------

function compareValue(entry) {
  return entry.type + ":" + entry.id;
}

function describeEntry(entry) {
  if (!entry) return "-";
  const when = entry.started_at ? new Date(entry.started_at).toLocaleString() : "unknown start";
  return (entry.type === "scan" ? "source scan" : "dataset audit") + " · " +
    entry.target + " · " + entry.state + " · " + when;
}

function historyRow(entry) {
  const summary = entry.summary || {};
  const pieces = [];
  if (summary.findings != null) pieces.push(summary.findings + " findings");
  if (summary.trust_score != null) pieces.push("trust " + summary.trust_score + "/100");
  if (entry.type === "dataset") {
    if (summary.rows != null) pieces.push(summary.rows + " rows");
    if (summary.target) pieces.push("target " + summary.target);
  } else if (summary.files != null) {
    pieces.push(summary.files + " files");
  }
  if (summary.errors) pieces.push(summary.errors + " error(s)");
  const actions = [];
  if (entry.results_available) {
    const open = el("button", {type: "button", text: "Open results"});
    open.addEventListener("click", async () => {
      try {
        await openHistoryEntry(entry);
      } catch (error) {
        $("notice").textContent = error.message;
      }
    });
    actions.push(open);
    const graphButton = el("button", {type: "button", text: "Graph"});
    graphButton.addEventListener("click", async () => {
      try {
        await openHistoryEntry(entry);
        openGraph(entry.type === "scan" ? "scan" : "dataset", entry.id);
      } catch (error) {
        $("notice").textContent = error.message;
      }
    });
    actions.push(graphButton);
  }
  if (entry.report_available) {
    actions.push(el("a", {href: "/api/" + (entry.type === "scan" ? "scans" : "datasets") +
      "/" + entry.id + "/report", text: "Report"}));
  }
  if (entry.results_available) {
    const compare = el("button", {type: "button", text: "Compare"});
    compare.addEventListener("click", () => compareSelection(entry));
    actions.push(compare);
  }
  const when = entry.started_at ? new Date(entry.started_at).toLocaleString() : "-";
  const duration = entry.duration_ms != null
    ? (entry.duration_ms / 1000).toFixed(1) + "s" : "-";
  return el("tr", {},
    el("td", {}, el("span", {class: "badge " + entry.state, text: entry.state})),
    el("td", {text: entry.type === "scan" ? "source scan" : "dataset audit"}),
    el("td", {class: "kv", text: entry.target}),
    el("td", {class: "kv", text: when}),
    el("td", {text: duration}),
    el("td", {text: pieces.join(" · ")}),
    el("td", {}, ...actions));
}

function compareOption(entry) {
  const summary = entry.summary || {};
  const bits = [entry.type === "scan" ? "scan" : "dataset", entry.target, entry.state];
  if (summary.findings != null) bits.push(summary.findings + " findings");
  if (entry.started_at) bits.push(new Date(entry.started_at).toLocaleString());
  return el("option", {value: compareValue(entry), text: bits.join(" · ")});
}

function fillCompareSelect(select, entries) {
  const previous = select.value;
  const usable = (entries || []).filter((entry) => entry.results_available);
  const scans = usable.filter((entry) => entry.type === "scan");
  const datasets = usable.filter((entry) => entry.type === "dataset");
  const children = [];
  if (scans.length) {
    children.push(el("optgroup", {label: "Source scans"}, ...scans.map(compareOption)));
  }
  if (datasets.length) {
    children.push(el("optgroup", {label: "Dataset audits"}, ...datasets.map(compareOption)));
  }
  if (!children.length) children.push(el("option", {value: "", text: "No retained results yet"}));
  select.replaceChildren(...children);
  if (previous && Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  }
}

async function loadHistory() {
  const note = $("history-note");
  try {
    const data = await api("/api/history");
    history.data = data;
    note.textContent = data.note + " Retained now: " + data.scans.length +
      " source scan(s) and " + data.datasets.length + " dataset audit(s); limits " +
      data.limits.scans + " and " + data.limits.datasets + ".";
    const list = $("history-list");
    list.replaceChildren();
    if (!data.entries.length) {
      list.append(el("p", {class: "empty",
        text: "No sessions yet. Start a scan or audit above."}));
    } else {
      list.append(el("table", {class: "table"},
        el("thead", {}, el("tr", {},
          ...["Status", "Type", "Target", "Started", "Duration", "Summary", "Actions"]
            .map((name) => el("th", {text: name})))),
        el("tbody", {}, ...data.entries.map(historyRow))));
    }
    fillCompareSelect($("cmp-previous"), data.entries);
    fillCompareSelect($("cmp-current"), data.entries);
  } catch (error) {
    note.textContent = "History unavailable: " + error.message;
  }
}

async function openHistoryEntry(entry) {
  closeGraph();
  if (entry.type === "scan") {
    stopScanWatch();
    state.id = entry.id;
    state.selected = null;
    state.results = null;
    $("results").hidden = true;
    $("notice").textContent = "";
    setState(entry.state);
    await loadResults();
    $("results").scrollIntoView({behavior: "smooth", block: "start"});
    return;
  }
  stopDatasetWatch();
  ds.id = entry.id;
  ds.results = null;
  $("ds-results").hidden = true;
  $("ds-notice").textContent = "";
  dsState(entry.state);
  await loadDatasetResults();
  $("ds-results").scrollIntoView({behavior: "smooth", block: "start"});
}

function changeClass(change) {
  if (change === "added") return "new";
  if (change === "no_longer_present") return "resolved";
  if (change === "inconclusive") return "unchanged";
  return change;
}

function comparisonStat(label, value, cls) {
  return el("span", {class: cls || "muted"},
    el("b", {text: String(value == null ? 0 : value)}),
    el("span", {class: "muted", text: label}));
}

function comparisonRow(entry) {
  const subject = entry.current || entry.previous || {};
  const where = subject.file
    ? subject.file + (subject.line ? ":" + subject.line : "")
    : (subject.column || "");
  const details = [];
  if (entry.change === "changed") {
    details.push("severity " + ((entry.previous || {}).severity || "?") + " → " +
      ((entry.current || {}).severity || "?"));
  }
  if (entry.current && entry.current.status) {
    details.push("current status " + entry.current.status);
  }
  if (entry.current && entry.current.suppression) {
    details.push("suppressed: " + entry.current.suppression.reason);
  }
  if (entry.current && entry.current.requires_attention === false) {
    details.push("does not count toward the trust score");
  }
  if (entry.reason) details.push(entry.reason);
  const actions = [];
  if (entry.current && entry.current.index != null) {
    const open = el("button", {type: "button", text: "Open finding"});
    open.addEventListener("click", () => openComparisonFinding(entry.current.index));
    actions.push(open);
    const graphButton = el("button", {type: "button", text: "Graph"});
    graphButton.addEventListener("click", () => openGraph(
      comparison.type === "scan" ? "scan" : "dataset",
      comparison.current, entry.current.index));
    actions.push(graphButton);
  }
  return el("tr", {},
    el("td", {}, el("span", {class: "chip status " + changeClass(entry.change),
      text: entry.change})),
    el("td", {}, subject.severity
      ? el("span", {class: "chip " + subject.severity, text: subject.severity}) : null),
    el("td", {text: subject.kind || ""}),
    el("td", {class: "kv", text: where}),
    el("td", {class: "kv", text: (entry.fingerprint || "").slice(0, 16)}),
    el("td", {class: "meta", text: details.join(" · ")}),
    el("td", {}, ...actions));
}

function renderComparison(data) {
  const notice = $("cmp-notice");
  const summary = $("cmp-summary");
  const list = $("cmp-list");
  summary.replaceChildren();
  list.replaceChildren();
  if (!data.available) {
    notice.textContent = data.reason || "Comparison is unavailable.";
    if (data.checks && data.checks.length) {
      summary.append(el("table", {class: "table"},
        el("thead", {}, el("tr", {},
          ...["Setting", "Compatible", "Previous", "Current"]
            .map((name) => el("th", {text: name})))),
        el("tbody", {}, ...data.checks.map((check) => el("tr", {},
          el("td", {text: check.name}),
          el("td", {text: check.compatible ? "yes" : "no"}),
          el("td", {class: "kv", text: String(check.previous)}),
          el("td", {class: "kv", text: String(check.current)}))))));
    }
    return;
  }
  notice.textContent = "";
  const s = data.summary;
  summary.append(
    el("p", {class: "meta", text: "Previous: " + describeEntry(data.previous)}),
    el("p", {class: "meta", text: "Current: " + describeEntry(data.current)}),
    el("div", {class: "stats"},
      comparisonStat("added", s.added, "high"),
      comparisonStat("unchanged", s.unchanged, "muted"),
      comparisonStat("changed", s.changed, "medium"),
      comparisonStat("no longer present", s.no_longer_present, "low"),
      comparisonStat("inconclusive", s.inconclusive, "muted")),
    el("p", {class: "meta", text: s.suppressed + " suppressed · " +
      s.unclassified + " unclassified · " +
      (data.counts_consistent ? "counts consistent" : "counts inconsistent") +
      (data.truncated ? " · list truncated, " + s.total + " change(s) total" : "")}),
    ...(data.limitations || []).map((text) => el("p", {class: "meta high", text: text})),
    ...(data.notes || []).map((text) => el("p", {class: "meta", text: text})));
  if (!data.changes.length) {
    list.append(el("p", {class: "empty", text: "No findings to compare."}));
    return;
  }
  list.append(el("table", {class: "table"},
    el("thead", {}, el("tr", {},
      ...["Change", "Severity", "Kind", "Column / file", "Fingerprint", "Details", "Actions"]
        .map((name) => el("th", {text: name})))),
    el("tbody", {}, ...data.changes.map(comparisonRow))));
}

async function runComparison() {
  const before = $("cmp-previous").value;
  const after = $("cmp-current").value;
  const notice = $("cmp-notice");
  $("cmp-summary").replaceChildren();
  $("cmp-list").replaceChildren();
  if (!before || !after) {
    notice.textContent = "Choose two retained sessions with results.";
    return;
  }
  if (before === after) {
    notice.textContent = "Choose two different sessions.";
    return;
  }
  const beforeType = before.split(":")[0];
  const afterType = after.split(":")[0];
  if (beforeType !== afterType) {
    notice.textContent = "Compare a source scan with a source scan, or a dataset " +
      "audit with a dataset audit.";
    return;
  }
  const beforeId = before.split(":")[1];
  const afterId = after.split(":")[1];
  notice.textContent = "Comparing…";
  try {
    const data = await api("/api/compare/" + (afterType === "scan" ? "scans" : "datasets") +
      "?previous=" + encodeURIComponent(beforeId) +
      "&current=" + encodeURIComponent(afterId));
    comparison.data = data;
    comparison.type = afterType;
    comparison.current = afterId;
    renderComparison(data);
  } catch (error) {
    notice.textContent = error.message;
  }
}

function compareSelection(entry) {
  if (!history.data) return;
  $("cmp-current").value = compareValue(entry);
  const peer = (history.data.entries || []).find((item) =>
    item.type === entry.type && item.id !== entry.id && item.results_available);
  if (peer) $("cmp-previous").value = compareValue(peer);
  runComparison();
}

async function openComparisonFinding(index) {
  const type = comparison.type;
  const id = comparison.current;
  if (type === "scan") {
    if (state.id !== id) {
      const entry = (history.data ? history.data.entries : []).find((item) =>
        item.type === "scan" && item.id === id);
      if (entry) await openHistoryEntry(entry);
    }
    if (state.results) await focusFinding("scan", index);
    return;
  }
  if (ds.id !== id) {
    const entry = (history.data ? history.data.entries : []).find((item) =>
      item.type === "dataset" && item.id === id);
    if (entry) await openHistoryEntry(entry);
  }
  if (ds.results) await focusFinding("dataset", index);
}

$("history-refresh").addEventListener("click", loadHistory);
$("cmp-run").addEventListener("click", runComparison);
for (const id of ["cmp-previous", "cmp-current"]) {
  $(id).addEventListener("change", () => {
    $("cmp-notice").textContent = "";
    $("cmp-summary").replaceChildren();
    $("cmp-list").replaceChildren();
  });
}

// --- evidence graph -------------------------------------------------------

function closeGraph() {
  graph.context = null;
  graph.id = null;
  graph.data = null;
  graph.selected = null;
  graph.offset = 0;
  graph.focus = null;
  graph.finding = null;
  $("graph-panel").hidden = true;
}

async function openGraph(context, id, finding) {
  if (!id) return;
  graph.context = context;
  graph.id = id;
  graph.data = null;
  graph.selected = null;
  graph.offset = 0;
  graph.focus = null;
  graph.finding = typeof finding === "number" ? finding : null;
  graph.limit = 50;
  $("graph-source").textContent = context === "scan" ? "source scan" : "dataset audit";
  $("g-kind").value = "";
  $("g-relation").value = "";
  $("g-sev").value = "";
  $("g-status").value = "";
  $("g-q").value = "";
  $("graph-panel").hidden = false;
  $("graph-panel").scrollIntoView({behavior: "smooth", block: "start"});
  await loadGraph();
}

function fillGraphSelect(select, values, allLabel, labels) {
  const current = select.value;
  select.replaceChildren(el("option", {value: "", text: allLabel}),
    ...values.map((value) => el("option", {value: value,
      text: labels && labels[value] ? labels[value] : value})));
  if (values.includes(current)) select.value = current;
}

function renderGraphFilters(filters) {
  fillGraphSelect($("g-kind"), filters.kinds || [], "all");
  fillGraphSelect($("g-relation"), filters.relations || [], "all");
  const statuses = filters.statuses || [];
  fillGraphSelect($("g-status"), statuses, "all",
    {unclassified: "not compared"});
  const severity = $("g-sev").value;
  if (severity && !(filters.severities || []).includes(severity)) $("g-sev").value = "";
}

function renderGraphSummary(data) {
  const s = data.summary;
  const p = data.page;
  const parts = [
    s.nodes + " evidence node(s)",
    s.links + " relationship(s)",
    s.clusters + " cluster(s)",
    s.components + " component(s)",
    s.isolated + " isolated",
  ];
  if (p.total !== s.nodes) parts.push(p.total + " node(s) match these filters");
  if (s.relationships_truncated === true) {
    parts.push("the engine truncated relationships at its limit");
  } else if (s.relationships_truncated === null) {
    parts.push("relationship completeness is not reported for this result");
  }
  if (s.dangling_links) {
    parts.push(s.dangling_links + " relationship(s) referenced missing evidence and were ignored");
  }
  if (s.invalid_nodes || s.invalid_links) {
    parts.push("some malformed graph records were skipped");
  }
  $("g-summary").textContent = parts.join(" · ");
  $("g-page").textContent = p.total
    ? "Showing " + (p.offset + 1) + "-" + (p.offset + p.returned) + " of " + p.total
    : "No matching nodes";
  $("g-prev").disabled = p.offset <= 0;
  $("g-next").disabled = p.next_offset == null;
}

function graphNodeButton(node) {
  const severity = (node.finding && node.finding.severity) || node.severity;
  const row = el("button", {class: "node", type: "button"},
    el("span", {class: "chip " + severity, text: severity}),
    statusChip(node.finding || {}),
    el("strong", {text: node.kind}),
    el("span", {class: "meta", text: " · " + node.source +
      (node.location ? " · " + node.location : "")}),
    el("div", {class: "meta", text: node.description}),
    el("div", {class: "meta", text: node.degree + " relationship(s) · component " +
      node.component + (node.isolated ? " · isolated" : "")}));
  row.addEventListener("click", () => selectGraphNode(node.evidence_id));
  if (graph.selected === node.evidence_id) row.setAttribute("aria-current", "true");
  return el("li", {}, row);
}

function renderGraphNodes(data) {
  const box = $("g-nodes");
  box.replaceChildren();
  const nodes = data.nodes || [];
  if (!nodes.length) {
    box.append(el("p", {class: "empty", text: data.summary.nodes
      ? "No evidence nodes match these filters."
      : "This result has no evidence records."}));
    return;
  }
  box.append(el("ul", {class: "node-list"}, ...nodes.map(graphNodeButton)));
}

function renderGraphDetail(node, data) {
  const panel = $("g-detail");
  panel.replaceChildren();
  if (!node) {
    panel.append(el("p", {class: "empty",
      text: "Select a node to inspect its evidence and relationships."}));
    return;
  }
  const severity = (node.finding && node.finding.severity) || node.severity;
  const blocks = [
    el("div", {},
      el("span", {class: "chip " + severity, text: severity}),
      statusChip(node.finding || {}),
      el("strong", {text: node.kind}),
      el("span", {class: "meta", text: " · " + node.source})),
    el("p", {text: node.description}),
    el("p", {class: "kv", text: node.evidence_id +
      (node.location ? " · " + node.location : "")}),
    el("p", {class: "meta", text: "component " + node.component +
      (node.isolated ? " · isolated: no recorded relationship"
        : " · " + node.degree + " relationship(s)")}),
    node.cluster_id ? el("p", {class: "meta", text: "cluster " +
      node.cluster_id + (node.anchor ? " · " + node.anchor : "")}) : null,
  ];
  if (node.finding) {
    const finding = node.finding;
    const show = el("button", {type: "button", text: "Show finding"});
    show.addEventListener("click", () => focusFinding(graph.context, finding.index));
    blocks.push(el("div", {},
      el("label", {text: "Affected finding"}),
      el("p", {}, el("span", {class: "chip " + finding.severity,
        text: finding.severity}), statusChip(finding),
        el("strong", {text: finding.kind})),
      el("p", {class: "meta", text: (finding.column || "") +
        (finding.detail ? " · " + finding.detail : "")}),
      finding.previous ? el("p", {class: "meta", text: "previous baseline severity: " +
        finding.previous.severity}) : null,
      finding.suppression ? el("p", {class: "meta", text: "suppressed: " +
        finding.suppression.reason + " (" +
        finding.suppression.fingerprint.slice(0, 12) + ")"}) : null,
      finding.fingerprint ? el("p", {class: "kv", text: finding.fingerprint}) : null,
      show));
  }
  const metadata = Object.entries(node.metadata || {});
  if (metadata.length) {
    blocks.push(el("details", {},
      el("summary", {class: "meta", text: "Evidence metadata"}),
      el("div", {class: "kv"}, ...metadata.map(([key, value]) => el("div", {},
        el("b", {text: key + ": "}), el("span", {text: String(value)}))))));
  }
  const neighbors = node.neighbors || [];
  if (!neighbors.length) {
    blocks.push(el("div", {},
      el("label", {text: "Relationships"}),
      el("p", {class: "empty",
        text: "No relationship is recorded for this evidence node; it is isolated in this graph."})));
  } else {
    const items = neighbors.map((item) => {
      const inspect = el("button", {type: "button", text: "Inspect"});
      inspect.addEventListener("click", () => selectGraphNode(item.evidence_id));
      return el("li", {},
        el("span", {class: "node-relation", text: item.relation}),
        el("span", {class: "node-arrow", text: item.direction === "out" ? "→" : "←"}),
        el("strong", {text: item.evidence_id}),
        el("span", {class: "meta", text: " · " + item.kind}),
        el("div", {class: "meta", text: item.rationale}),
        inspect);
    });
    blocks.push(el("div", {},
      el("label", {text: "Relationships (" + neighbors.length + ")"}),
      el("ul", {class: "evidence"}, ...items)));
    if (data && data.summary.relationships_truncated === true) {
      blocks.push(el("p", {class: "meta",
        text: "The engine truncated this result's relationship list; some links may be absent."}));
    }
  }
  panel.append(...blocks.filter(Boolean));
}

async function selectGraphNode(evidenceId) {
  const data = graph.data;
  if (!data) return;
  let node = (data.nodes || []).find((item) => item.evidence_id === evidenceId);
  if (!node && data.focus && data.focus.evidence_id === evidenceId) node = data.focus;
  if (!node) {
    graph.focus = evidenceId;
    graph.selected = evidenceId;
    await loadGraph();
    return;
  }
  graph.selected = evidenceId;
  renderGraphDetail(node, data);
  renderGraphNodes(data);
}

async function loadGraph() {
  if (!graph.id) return;
  const token = ++graph.request;
  $("g-error").textContent = "";
  $("g-nodes").replaceChildren(el("p", {class: "empty", text: "Loading evidence graph…"}));
  $("g-detail").replaceChildren(el("p", {class: "empty", text: "Loading evidence…"}));
  const params = new URLSearchParams();
  for (const [key, id] of [["kind", "g-kind"], ["relation", "g-relation"],
    ["severity", "g-sev"], ["status", "g-status"]]) {
    const value = $(id).value;
    if (value) params.set(key, value);
  }
  const query = $("g-q").value.trim();
  if (query) params.set("q", query);
  params.set("offset", String(graph.offset));
  params.set("limit", String(graph.limit));
  if (graph.focus) params.set("focus", graph.focus);
  if (graph.finding != null) params.set("finding", String(graph.finding));
  const base = graph.context === "scan" ? "scans" : "datasets";
  try {
    const data = await api("/api/" + base + "/" + graph.id + "/graph?" + params.toString());
    if (token !== graph.request) return;
    graph.focus = null;
    graph.finding = null;
    graph.data = data;
    if (!data.available) {
      $("g-summary").textContent = "";
      $("g-page").textContent = "";
      $("g-prev").disabled = true;
      $("g-next").disabled = true;
      $("g-nodes").replaceChildren(el("p", {class: "empty",
        text: "No evidence graph is available for this result: " +
          (data.reason || "unknown reason") + "."}));
      $("g-detail").replaceChildren(el("p", {class: "empty",
        text: "Nothing to inspect."}));
      return;
    }
    renderGraphFilters(data.filters);
    renderGraphSummary(data);
    renderGraphNodes(data);
    const selected = data.focus || (data.nodes || [])
      .find((item) => item.evidence_id === graph.selected);
    if (selected) {
      graph.selected = selected.evidence_id;
      renderGraphDetail(selected, data);
      renderGraphNodes(data);
    } else {
      graph.selected = null;
      renderGraphDetail(null, data);
    }
  } catch (error) {
    if (token !== graph.request) return;
    $("g-error").textContent = error.message;
    $("g-nodes").replaceChildren(el("p", {class: "empty",
      text: "The evidence graph could not be loaded."}));
    $("g-detail").replaceChildren(el("p", {class: "empty", text: error.message}));
  }
}

async function focusFinding(context, index) {
  if (context === "scan") {
    if (!state.results) return;
    const finding = state.results.findings.find((item) => item.index === index);
    if (!finding) return;
    let row = document.querySelector(
      '#clusters button.finding[data-index="' + index + '"]');
    if (!row) {
      $("f-sev").value = "";
      $("f-kind").value = "";
      $("f-status").value = "";
      $("f-q").value = "";
      renderFindings();
      row = document.querySelector(
        '#clusters button.finding[data-index="' + index + '"]');
    }
    if (row) {
      row.scrollIntoView({behavior: "smooth", block: "center"});
      await select(finding, row);
    }
    return;
  }
  if (!ds.results) return;
  const finding = ds.results.findings.find((item) => item.index === index);
  if (!finding) return;
  $("ds-status").value = "";
  datasetClusters(ds.results);
  const row = document.querySelector('#ds-clusters li[data-index="' + index + '"]');
  if (row) {
    row.scrollIntoView({behavior: "smooth", block: "center"});
    row.classList.add("finding-highlight");
    setTimeout(() => row.classList.remove("finding-highlight"), 2000);
  }
}

$("g-open-scan").addEventListener("click", () => {
  if (state.id) openGraph("scan", state.id);
});
$("g-open-dataset").addEventListener("click", () => {
  if (ds.id) openGraph("dataset", ds.id);
});
$("graph-close").addEventListener("click", closeGraph);

for (const id of ["g-kind", "g-relation", "g-sev", "g-status"]) {
  $(id).addEventListener("input", () => {
    graph.offset = 0;
    graph.focus = null;
    graph.finding = null;
    loadGraph();
  });
}
$("g-q").addEventListener("input", () => {
  clearTimeout(graph.timer);
  graph.offset = 0;
  graph.focus = null;
  graph.finding = null;
  graph.timer = setTimeout(loadGraph, 250);
});
$("g-prev").addEventListener("click", () => {
  graph.offset = Math.max(0, graph.offset - graph.limit);
  graph.focus = null;
  graph.finding = null;
  loadGraph();
});
$("g-next").addEventListener("click", () => {
  graph.offset += graph.limit;
  graph.focus = null;
  graph.finding = null;
  loadGraph();
});

api("/api/info").then((info) => {
  state.info = info;
  $("sql").disabled = !info.sql_available;
  if (!info.sql_available) $("sql").parentElement.title = "requires the sql extra";
  $("root").textContent = info.root;
  const config = info.config || {};
  if (config.error) {
    $("policy-info").textContent = "Configuration error: " + config.error;
    $("policy-info").className = "meta high";
  } else if (config.baseline) {
    $("baseline").value = config.baseline;
    $("policy-info").textContent = "Using [baseline].path = " + config.baseline +
      " from " + config.path + " · " + config.suppressions +
      " suppression rule(s) configured.";
  } else {
    $("policy-info").textContent = "No [baseline].path in " + (config.path ||
      ".leaklens.toml") + " · " + config.suppressions +
      " suppression rule(s) configured. Enter a baseline file above to compare.";
  }
  const formats = info.dataset_formats.filter((item) => item.enabled)
    .map((item) => item.extension);
  const missing = info.dataset_formats
    .filter((item) => !item.enabled && item.configured)
    .map((item) => item.extension + " (needs " + item.extra + ")");
  const limits = info.dataset_limits;
  $("ds-formats").textContent = "Accepts " + formats.join(", ") +
    (missing.length ? "; unavailable here: " + missing.join(", ") : "") +
    ". Limits: " + (limits.max_bytes / (1024 * 1024)) + " MiB, " +
    limits.max_rows + " rows, " + limits.max_columns + " columns, " +
    limits.max_cells + " cells.";
  loadHistory();
});
</script>
</body></html>
"""
