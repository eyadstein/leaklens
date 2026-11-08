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
input[type=text],select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit;min-width:0}
input[type=text]{width:100%}
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
footer{padding:0 20px 40px;font-size:12px;color:var(--muted)}
</style></head><body>
<header>
  <h1>LeakLens workspace</h1>
  <span class="badge">v@@VERSION@@</span>
  <span class="muted">root: <code id="root">@@ROOT@@</code></span>
</header>
<main>
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
        <span><label for="f-q">Search</label><input id="f-q" type="text" placeholder="kind, file, detail" style="min-width:200px"></span>
      </div>
      <p class="meta" id="summary"></p>
    </div>
    <div class="layout">
      <div>
        <h2 style="font-size:14px">Findings by evidence cluster</h2>
        <div id="clusters"><p class="empty">No findings.</p></div>
      </div>
      <aside class="card" id="detail" aria-live="polite">
        <p class="empty">Select a finding to see its evidence and source.</p>
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
const state = {id: null, info: null, results: null, selected: null};

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
    const message = body && body.detail ? body.detail : String(body || response.status);
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
    " · medium " + c.medium + " · low " + c.low;
  $("report-link").href = "/api/scans/" + state.id + "/report";
  $("report-link").hidden = false;
  $("json-link").href = "/api/scans/" + state.id + "/results";
  $("json-link").hidden = false;
  const kinds = [...new Set(data.findings.map((f) => f.kind))].sort();
  const kindSelect = $("f-kind");
  kindSelect.replaceChildren(el("option", {value: "", text: "all"}),
    ...kinds.map((kind) => el("option", {value: kind, text: kind})));
  renderFindings();
}

function visibleFindings() {
  const data = state.results;
  if (!data) return [];
  const severity = $("f-sev").value;
  const kind = $("f-kind").value;
  const query = $("f-q").value.trim().toLowerCase();
  return data.findings.filter((f) => {
    if (severity && f.severity !== severity) return false;
    if (kind && f.kind !== kind) return false;
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
    el("strong", {text: finding.kind}),
    el("span", {class: "meta", text: " · " + where}),
    el("div", {class: "meta", text: finding.detail}));
  row.addEventListener("click", () => select(finding, row));
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
      el("strong", {text: finding.kind})),
    el("p", {text: finding.detail}),
    el("p", {class: "kv", text: (finding.file || finding.column) +
      (finding.line ? " · line " + finding.line : "")}),
    finding.advice ? el("div", {},
      el("label", {text: "Suggested fix"}),
      el("p", {text: finding.advice})) : null,
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
      body: JSON.stringify({path: $("path").value, sql: $("sql").checked}),
    });
    state.id = started.scan_id;
    state.selected = null;
    setState("running");
    $("cancel").disabled = false;
    poll();
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

for (const id of ["f-sev", "f-kind", "f-q"]) {
  $(id).addEventListener("input", renderFindings);
}

api("/api/info").then((info) => {
  state.info = info;
  $("sql").disabled = !info.sql_available;
  if (!info.sql_available) $("sql").parentElement.title = "requires the sql extra";
  $("root").textContent = info.root;
});
</script>
</body></html>
"""
