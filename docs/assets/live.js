// Live scan controller. All findings rendered here come from the configured
// LeakLens Python backend; the page has no recorded fallback results.
import {
  DEFAULT_FORMATS,
  DEFAULT_LIMITS,
  capabilitySummary,
  componentLabel,
  errorMessage,
  filterFindings,
  findingKinds,
  formatBytes,
  formatDuration,
  limitsFrom,
  makeReportFilename,
  normalizeApiBase,
  previewColumns,
  scoreTone,
  scoreWord,
  severityCounts,
  supportedExtensions,
  targetOptions,
  validateFile,
} from "./live-engine.mjs";

const doc = document;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const CONFIG = window.LEAKLENS_LIVE_CONFIG || {};
const STORAGE_KEY = "leaklens.live.api";

const $ = (selector, root = doc) => root.querySelector(selector);
const $$ = (selector, root = doc) => [...root.querySelectorAll(selector)];

const state = {
  apiBase: "",
  connected: false,
  checking: false,
  capabilities: null,
  file: null,
  testFile: null,
  inspect: null,
  testInspect: null,
  result: null,
  filter: { severity: "all", kind: "all", query: "" },
  busy: false,
  cancelled: false,
  controller: null,
  inspectController: null,
  elapsedTimer: null,
  stepTimer: null,
  startedAt: 0,
};

const ui = {
  apiForm: $("[data-api-form]"),
  apiInput: $("[data-api-input]"),
  connect: $("[data-connect]"),
  connection: $("[data-connection]"),
  connectionLabel: $("[data-connection-label]"),
  connectionDetail: $("[data-connection-detail]"),
  connectionBadge: $("[data-connection-badge]"),
  offline: $("[data-offline]"),
  limits: $("[data-limits]"),
  formats: $("[data-formats]"),
  upload: $("[data-upload-panel]"),
  config: $("[data-config-panel]"),
  configFields: $("[data-config-fields]"),
  dropzone: $("[data-dropzone]"),
  fileInput: $("[data-file-input]"),
  fileInfo: $("[data-file-info]"),
  fileName: $("[data-file-name]"),
  fileMeta: $("[data-file-meta]"),
  fileErrors: $("[data-file-errors]"),
  fileWarnings: $("[data-file-warnings]"),
  clearFile: $("[data-clear-file]"),
  testDropzone: $("[data-test-dropzone]"),
  testInput: $("[data-test-input]"),
  testInfo: $("[data-test-info]"),
  testName: $("[data-test-name]"),
  testMeta: $("[data-test-meta]"),
  testErrors: $("[data-test-errors]"),
  testWarnings: $("[data-test-warnings]"),
  clearTest: $("[data-clear-test]"),
  target: $("[data-target]"),
  threshold: $("[data-threshold]"),
  thresholdValue: $("[data-threshold-value]"),
  group: $("[data-group]"),
  time: $("[data-time]"),
  aiToggle: $("[data-ai-toggle]"),
  aiResult: $("[data-ai-result]"),
  aiBody: $("[data-ai]"),
  scan: $("[data-scan]"),
  reset: $("[data-reset]"),
  progress: $("[data-progress]"),
  progressLabel: $("[data-progress-label]"),
  progressElapsed: $("[data-progress-elapsed]"),
  progressSteps: $$("[data-progress-step]"),
  cancel: $("[data-cancel]"),
  error: $("[data-error]"),
  errorTitle: $("[data-error-title]"),
  errorMessage: $("[data-error-message]"),
  retry: $("[data-retry]"),
  results: $("[data-results]"),
  score: $("[data-score]"),
  scoreWord: $("[data-score-word]"),
  scoreRing: $("[data-score-ring]"),
  counts: $("[data-counts]"),
  summary: $("[data-summary]"),
  audit: $("[data-audit]"),
  impact: $("[data-impact]"),
  findings: $("[data-findings]"),
  findingsCount: $("[data-findings-count]"),
  filterSeverity: $("[data-filter-severity]"),
  filterKind: $("[data-filter-kind]"),
  filterQuery: $("[data-filter-query]"),
  preview: $("[data-preview]"),
  previewNote: $("[data-preview-note]"),
  downloadReport: $("[data-download-report]"),
  downloadJson: $("[data-download-json]"),
  newScan: $("[data-new-scan]"),
  status: $("[data-live-status]"),
};

const el = (tag, props = {}, children = []) => {
  const node = doc.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "className") node.className = value;
    else if (key === "textContent") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "style") Object.assign(node.style, value);
    else if (key === "onClick") node.addEventListener("click", value);
    else if (key === "onChange") node.addEventListener("change", value);
    else if (key === "onInput") node.addEventListener("input", value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : doc.createTextNode(String(child)));
  }
  return node;
};

const setText = (node, value) => {
  if (node) node.textContent = value;
};

const announce = (message) => {
  if (!ui.status) return;
  ui.status.textContent = "";
  window.setTimeout(() => {
    ui.status.textContent = message;
  }, 20);
};

const setConnection = (kind, label, detail) => {
  if (ui.connection) ui.connection.dataset.state = kind;
  if (ui.connectionBadge) ui.connectionBadge.dataset.state = kind;
  setText(ui.connectionLabel, label);
  setText(ui.connectionDetail, detail);
};

const setBusy = (busy) => {
  state.busy = busy;
  if (ui.configFields) ui.configFields.disabled = busy;
  if (ui.aiToggle) ui.aiToggle.disabled = busy;
  if (ui.fileInput) ui.fileInput.disabled = busy;
  if (ui.testInput) ui.testInput.disabled = busy;
  if (ui.scan) ui.scan.disabled = busy || !state.connected || !state.file;
  if (ui.cancel) ui.cancel.hidden = !busy;
  if (ui.results) ui.results.setAttribute("aria-busy", String(busy));
};

const clear = (node) => {
  if (node) node.replaceChildren();
};

const listBlock = (items, className) => {
  if (!items.length) return null;
  return el(
    "ul",
    { className },
    items.map((item) => el("li", { textContent: item }))
  );
};

const showFileMessages = (validation, target) => {
  const errors = validation.errors || [];
  const warnings = validation.warnings || [];
  const errorBox = target === "test" ? ui.testErrors : ui.fileErrors;
  const warningBox = target === "test" ? ui.testWarnings : ui.fileWarnings;
  if (errorBox) {
    clear(errorBox);
    if (errors.length) errorBox.append(listBlock(errors, "file-errors"));
  }
  if (warningBox) {
    clear(warningBox);
    if (warnings.length) warningBox.append(listBlock(warnings, "file-warnings"));
  }
  if (errors.length) {
    if (target === "test" && ui.testInfo) ui.testInfo.hidden = false;
    else if (ui.fileInfo) ui.fileInfo.hidden = false;
    announce(errors[0]);
  }
};

const clearResult = () => {
  state.result = null;
  state.filter = { severity: "all", kind: "all", query: "" };
  if (ui.results) ui.results.hidden = true;
  if (ui.aiResult) ui.aiResult.hidden = true;
  if (ui.aiBody) clear(ui.aiBody);
  hideError();
  hideProgress();
  if (ui.findings) clear(ui.findings);
};

const showError = (info, retry) => {
  if (!ui.error) return;
  ui.error.hidden = false;
  setText(ui.errorTitle, info.title);
  setText(ui.errorMessage, info.message);
  if (ui.retry) {
    ui.retry.hidden = !info.retryable;
    ui.retry.onclick = info.retryable && retry ? retry : null;
  }
  announce(`${info.title}. ${info.message}`);
};

const hideError = () => {
  if (ui.error) ui.error.hidden = true;
};

const renderLimits = (capabilities = null) => {
  const summary = capabilitySummary(capabilities);
  if (ui.formats) {
    setText(ui.formats, (summary.formats.length ? summary.formats : DEFAULT_FORMATS).join("  "));
  }
  if (ui.limits) {
    const rows = Number(summary.maxRows || DEFAULT_LIMITS.max_rows);
    setText(
      ui.limits,
      `${formatBytes(summary.maxBytes)} per file · ${rows.toLocaleString()} rows · ${summary.timeoutSeconds}s scan timeout`
    );
  }
};

const setOffline = (message) => {
  state.connected = false;
  state.capabilities = null;
  if (ui.offline) ui.offline.hidden = false;
  if (ui.offline) {
    const detail = $("[data-offline-detail]", ui.offline);
    if (detail) setText(detail, message);
  }
  setConnection(
    "offline",
    "Not connected",
    "No live LeakLens backend is configured. The static site never fabricates a scan result."
  );
  if (ui.scan) ui.scan.disabled = true;
  renderLimits(null);
  if (ui.upload) ui.upload.dataset.ready = "false";
};

const persistBase = (base) => {
  try {
    window.localStorage.setItem(STORAGE_KEY, base);
  } catch {
    /* storage can be disabled; the page still works for this session */
  }
};

const storedBase = () => {
  try {
    return window.localStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
};

class RequestTimeout extends Error {
  constructor() {
    super("request timed out");
    this.name = "RequestTimeout";
  }
}

class ApiFailure extends Error {
  constructor(status, payload) {
    super("api failure");
    this.name = "ApiFailure";
    this.status = status;
    this.payload = payload;
  }
}

const parseJson = async (response) => {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
};

const send = async (path, options, timeoutMs, controller) => {
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(`${state.apiBase}${path}`, {
      ...options,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
    const payload = await parseJson(response);
    if (!response.ok) throw new ApiFailure(response.status, payload);
    return payload;
  } catch (error) {
    if (timedOut) throw new RequestTimeout();
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
};

const connect = async (raw, { auto = false } = {}) => {
  if (state.checking) return;
  const result = normalizeApiBase(raw, { allowHttpLocalhost: true });
  if (!result.ok) {
    setConnection("error", "Not connected", result.error);
    showError(
      { title: "Check the backend URL", message: result.error, retryable: false },
      null
    );
    return;
  }
  state.checking = true;
  state.apiBase = result.base;
  if (ui.connect) ui.connect.disabled = true;
  setConnection(
    "checking",
    auto ? "Reconnecting…" : "Checking backend…",
    `Contacting ${result.host}. A free backend can take about a minute to wake up.`
  );
  const controller = new AbortController();
  try {
    const health = await send("/health", { method: "GET" }, 120_000, controller);
    if (!health || health.live !== true) {
      throw new Error("not a LeakLens live API");
    }
    const capabilities = await send("/capabilities", { method: "GET" }, 20_000, controller);
    if (!capabilities || capabilities.live !== true) {
      throw new Error("missing capabilities");
    }
    state.apiBase = result.base;
    state.capabilities = capabilities;
    state.connected = true;
    if (ui.apiInput) ui.apiInput.value = result.base;
    persistBase(result.base);
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("api", result.base);
      window.history.replaceState({}, "", url);
    } catch {
      /* ignore history failures */
    }
    if (ui.offline) ui.offline.hidden = true;
    if (ui.upload) ui.upload.dataset.ready = "true";
    setConnection(
      "live",
      "Live backend",
      `Connected to LeakLens ${capabilities.version || ""} at ${result.host}. Scans run on that backend, not in your browser.`
    );
    renderLimits(capabilities);
    hideError();
    announce("Connected to the live LeakLens backend.");
    if (state.file && !state.inspect) await inspect("primary");
    setBusy(false);
  } catch (error) {
    state.connected = false;
    state.capabilities = null;
    if (ui.offline) ui.offline.hidden = false;
    if (ui.scan) ui.scan.disabled = true;
    const timedOut = error instanceof RequestTimeout;
    const message = timedOut
      ? `The backend at ${result.host} did not answer in time. It may be waking up; retry in a minute.`
      : error instanceof ApiFailure
        ? errorMessage(error.status, error.payload).message
        : `Could not reach ${result.host}. Check the URL and whether the backend is running.`;
    setConnection("error", "Backend unavailable", message);
    showError(
      {
        title: timedOut ? "Backend waking up" : "Backend unavailable",
        message,
        retryable: true,
      },
      () => connect(state.apiBase || result.base)
    );
    announce(message);
  } finally {
    state.checking = false;
    if (ui.connect) ui.connect.disabled = false;
  }
};

const setFile = async (file, kind = "primary") => {
  clearResult();
  const validation = validateFile(file, state.capabilities);
  showFileMessages(validation, kind);
  if (!validation.ok) {
    if (kind === "test") {
      state.testFile = null;
      setText(ui.testName, "Held-out file not accepted");
      setText(ui.testMeta, "Choose a supported file or continue without one.");
      if (ui.clearTest) ui.clearTest.hidden = true;
    } else {
      state.file = null;
      setText(ui.fileName, "File not accepted");
      setText(ui.fileMeta, "Choose another file. The reason is listed below.");
      if (ui.clearFile) ui.clearFile.hidden = true;
      if (ui.scan) ui.scan.disabled = true;
    }
    return;
  }
  if (kind === "test") {
    state.testFile = file;
    state.testInspect = null;
    if (ui.testInfo) ui.testInfo.hidden = false;
    setText(ui.testName, file.name);
    setText(
      ui.testMeta,
      `${formatBytes(file.size)} · ${validation.extension} · validating on the backend…`
    );
    if (ui.clearTest) ui.clearTest.hidden = false;
    if (state.connected) await inspect("test");
    else {
      setText(
        ui.testMeta,
        `${formatBytes(file.size)} · ${validation.extension} · connect a backend to inspect it.`
      );
    }
    announce(`Held-out file ${file.name} selected.`);
    return;
  }
  state.file = file;
  state.inspect = null;
  if (ui.fileInfo) ui.fileInfo.hidden = false;
  setText(ui.fileName, file.name);
  setText(
    ui.fileMeta,
    `${formatBytes(file.size)} · ${validation.extension} · validating on the backend…`
  );
  if (ui.clearFile) ui.clearFile.hidden = false;
  if (state.connected) await inspect("primary");
  else {
    setText(
      ui.fileMeta,
      `${formatBytes(file.size)} · ${validation.extension} · connect a backend to inspect columns.`
    );
    if (ui.scan) ui.scan.disabled = true;
  }
  announce(`File ${file.name} selected.`);
};

const option = (value, label = value) => el("option", { value, textContent: label });

const populateSelect = (select, names, { includeNone = false, preferred = "" } = {}) => {
  if (!select) return;
  const previous = preferred || select.value;
  const options = [];
  if (includeNone) options.push(option("", "None"));
  for (const name of names) options.push(option(name, name));
  select.replaceChildren(...options);
  if (previous && names.includes(previous)) select.value = previous;
  else if (!includeNone && names.length) select.value = names[0];
  else if (includeNone) select.value = "";
};

const renderPreview = (inspection) => {
  if (!ui.preview) return;
  clear(ui.preview);
  const columns = Array.isArray(inspection?.columns) ? inspection.columns : [];
  if (!columns.length) {
    setText(ui.previewNote, "No preview is available.");
    return;
  }
  const { shown, hidden } = previewColumns(columns, 8);
  const rows = Array.isArray(inspection?.preview) ? inspection.preview : [];
  const head = el("tr", {}, shown.map((column) => el("th", { scope: "col", textContent: column.name })));
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      shown.map((column) => {
        const value = row[column.name];
        return el("td", {
          textContent: value === null || value === undefined ? "—" : String(value),
        });
      })
    )
  );
  const table = el("table", { className: "live-preview-table" }, [
    el("caption", {
      textContent: `First ${rows.length || 0} rows · showing ${shown.length} of ${columns.length} columns`,
    }),
    el("thead", {}, [head]),
    el("tbody", {}, body),
  ]);
  ui.preview.append(table);
  setText(
    ui.previewNote,
    hidden
      ? `${hidden} more column${hidden === 1 ? "" : "s"} hidden here; every column is available in the target, group and time selectors.`
      : "All columns are shown. Values are rendered as text, never as HTML."
  );
};

const inspect = async (kind) => {
  const file = kind === "test" ? state.testFile : state.file;
  if (!file || !state.connected || state.busy) return;
  if (state.inspectController) state.inspectController.abort();
  const controller = new AbortController();
  state.inspectController = controller;
  const meta = kind === "test" ? ui.testMeta : ui.fileMeta;
  if (meta) setText(meta, "Inspecting columns on the backend…");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const payload = await send("/inspect", { method: "POST", body: form }, 60_000, controller);
    if (kind === "test") {
      state.testInspect = payload;
      setText(
        ui.testMeta,
        `${formatBytes(file.size)} · ${payload.rows.toLocaleString()} rows · ${payload.columns.length} columns`
      );
      announce("Held-out file validated.");
      return;
    }
    state.inspect = payload;
    const { names, candidates } = targetOptions(payload);
    populateSelect(ui.target, names, {
      preferred: ui.target?.value || candidates[0] || names[0] || "",
    });
    populateSelect(ui.group, names, { includeNone: true });
    populateSelect(ui.time, names, { includeNone: true });
    setText(
      ui.fileMeta,
      `${formatBytes(file.size)} · ${payload.rows.toLocaleString()} rows · ${payload.columns.length} columns · inspected`
    );
    renderPreview(payload);
    if (ui.config) {
      ui.config.dataset.ready = "true";
      const note = $("[data-config-note]", ui.config);
      if (note) {
        setText(
          note,
          candidates.length
            ? `Suggested targets: ${candidates.slice(0, 6).join(", ")}. Choose the column you predict.`
            : "Choose the column your model predicts."
        );
      }
    }
    if (ui.scan) ui.scan.disabled = !ui.target?.value;
    announce(`Inspected ${payload.rows.toLocaleString()} rows and ${payload.columns.length} columns.`);
  } catch (error) {
    if (error instanceof ApiFailure) {
      showError(errorMessage(error.status, error.payload), () => inspect(kind));
    } else if (error.name !== "AbortError") {
      showError(
        {
          title: "Could not inspect the file",
          message: error instanceof RequestTimeout
            ? "The backend did not answer in time. Try a smaller file."
            : "The browser could not reach the backend.",
          retryable: true,
        },
        () => inspect(kind)
      );
    }
    if (meta) setText(meta, "Inspection failed; see the message below.");
    if (ui.scan) ui.scan.disabled = true;
  } finally {
    if (state.inspectController === controller) state.inspectController = null;
  }
};

const progressLabel = (index) => {
  const labels = [
    "Uploading and validating the file",
    "Parsing with bounded row and column limits",
    "Running LeakLens detectors on the backend",
    "Generating the HTML report",
  ];
  return labels[Math.min(index, labels.length - 1)];
};

const setProgressStep = (index) => {
  ui.progressSteps.forEach((step, position) => {
    step.dataset.state = position < index ? "done" : position === index ? "active" : "pending";
  });
  setText(ui.progressLabel, progressLabel(index));
};

const showProgress = () => {
  hideError();
  if (!ui.progress) return;
  ui.progress.hidden = false;
  setProgressStep(0);
  state.startedAt = performance.now();
  setText(ui.progressElapsed, "0.0 s");
  window.clearInterval(state.elapsedTimer);
  window.clearInterval(state.stepTimer);
  state.elapsedTimer = window.setInterval(() => {
    const seconds = (performance.now() - state.startedAt) / 1000;
    setText(ui.progressElapsed, `${seconds.toFixed(1)} s`);
  }, 200);
  let step = 0;
  state.stepTimer = window.setInterval(() => {
    step = Math.min(step + 1, 3);
    setProgressStep(step);
  }, 4_000);
  announce("Scan started on the backend. Exact server progress is not streamed.");
};

const hideProgress = () => {
  window.clearInterval(state.elapsedTimer);
  window.clearInterval(state.stepTimer);
  state.elapsedTimer = null;
  state.stepTimer = null;
  if (ui.progress) ui.progress.hidden = true;
};

const stat = (label, value) =>
  el("div", { className: "live-stat" }, [
    el("span", { textContent: label }),
    el("strong", { textContent: value }),
  ]);

const renderCounts = (counts) => {
  if (!ui.counts) return;
  clear(ui.counts);
  for (const severity of ["high", "medium", "low"]) {
    ui.counts.append(
      el("span", { className: `badge badge-${severity}`, textContent: `${severity} ${counts[severity] || 0}` })
    );
  }
  ui.counts.append(
    el("span", { className: "badge", textContent: `${counts.total || 0} total` })
  );
};

const renderSummary = (data) => {
  if (!ui.summary) return;
  clear(ui.summary);
  const summary = data.summary || {};
  const counts = data.counts || severityCounts(data.findings);
  ui.summary.append(
    stat("Rows", Number(summary.rows || 0).toLocaleString()),
    stat("Columns", Number(summary.columns || 0).toLocaleString()),
    stat("Findings", String(counts.total || 0)),
    stat("Duration", formatDuration(summary.duration_ms)),
    stat("Target", String(summary.target || "—")),
    stat("Threshold", String(summary.threshold ?? "—"))
  );
  if (summary.test_rows !== null && summary.test_rows !== undefined) {
    ui.summary.append(stat("Held-out rows", Number(summary.test_rows).toLocaleString()));
  }
  if (summary.group_column) ui.summary.append(stat("Group", summary.group_column));
  if (summary.time_column) ui.summary.append(stat("Time", summary.time_column));
};

const keyValueTable = (rows, caption) => {
  const body = rows
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) =>
      el("tr", {}, [
        el("th", { scope: "row", textContent: label }),
        el("td", { textContent: String(value) }),
      ])
    );
  if (!body.length) return null;
  return el("table", { className: "live-impact-table" }, [
    el("caption", { textContent: caption }),
    el("tbody", {}, body),
  ]);
};

const renderAudit = (data) => {
  if (!ui.audit) return;
  clear(ui.audit);
  const audit = data.audit || {};
  const rows = [
    ["Rows", audit.rows],
    ["Columns", audit.cols],
    ["Duplicate rows", audit.duplicate_rows],
    ["Minority rate", audit.minority_rate === undefined ? null : `${(Number(audit.minority_rate) * 100).toFixed(2)}%`],
  ];
  const table = keyValueTable(rows, "Dataset audit");
  if (table) ui.audit.append(table);
  if (audit.target_distribution && typeof audit.target_distribution === "object") {
    const distribution = Object.entries(audit.target_distribution)
      .slice(0, 8)
      .map(([label, share]) => `${label}: ${(Number(share) * 100).toFixed(2)}%`)
      .join(" · ");
    ui.audit.append(el("p", { className: "live-note", textContent: `Target distribution — ${distribution}` }));
  }
};

const renderImpact = (data) => {
  if (!ui.impact) return;
  clear(ui.impact);
  const impact = data.impact;
  if (!impact) {
    ui.impact.append(
      el("p", {
        className: "live-note",
        textContent:
          "Add a held-out test file to measure duplicate rows, group overlap and temporal boundary overlap.",
      })
    );
    return;
  }
  const rows = [
    ["Training rows", Number(impact.train_rows || 0).toLocaleString()],
    ["Test rows", Number(impact.test_rows || 0).toLocaleString()],
    [
      "Duplicate test rows",
      `${Number(impact.duplicate_test_rows || 0).toLocaleString()} (${Number(impact.duplicate_test_pct || 0).toFixed(2)}%)`,
    ],
  ];
  if (impact.group_column) {
    rows.push(
      ["Group column", impact.group_column],
      ["Shared groups", impact.shared_group_count],
      [
        "Test rows in shared groups",
        `${impact.shared_group_test_rows || 0} (${Number(impact.shared_group_test_pct || 0).toFixed(2)}%)`,
      ]
    );
  }
  if (impact.time_column) {
    rows.push(
      ["Time column", impact.time_column],
      [
        "Temporal overlap",
        impact.temporal_overlap === null || impact.temporal_overlap === undefined
          ? "Not measurable"
          : impact.temporal_overlap
            ? "Yes"
            : "No",
      ],
      ["Latest training time", impact.train_max_time || "Unavailable"],
      ["Earliest test time", impact.test_min_time || "Unavailable"]
    );
  }
  const table = keyValueTable(rows, "Measured split impact");
  if (table) ui.impact.append(table);
  ui.impact.append(
    el("p", {
      className: "live-note",
      textContent:
        "These are measured split statistics, not estimates of model-performance loss.",
    })
  );
};

const findingNode = (finding, index) => {
  const severity = ["high", "medium", "low"].includes(finding.severity)
    ? finding.severity
    : "low";
  const summary = el("summary", {}, [
    el("span", { className: `badge badge-${severity}`, textContent: severity }),
    el("code", { className: "f-kind", textContent: finding.kind || "finding" }),
    el("code", { className: "f-col", textContent: finding.column || "—" }),
    el("span", { className: "f-detail", textContent: finding.detail || "" }),
  ]);
  const body = el("div", { className: "finding-body" }, [
    el("p", {}, [
      el("strong", { textContent: "Evidence: " }),
      finding.detail || "No additional evidence was returned.",
    ]),
    finding.fix
      ? el("p", { className: "fix" }, [
          el("strong", { textContent: "Fix: " }),
          finding.fix,
        ])
      : el("p", { className: "fix", textContent: "No remediation hint is available for this finding." }),
  ]);
  const details = el("details", { className: `finding ${severity}` }, [summary, body]);
  if (index === 0 && severity === "high") details.open = true;
  return details;
};

const renderFindings = () => {
  if (!ui.findings || !state.result) return;
  const all = Array.isArray(state.result.findings) ? state.result.findings : [];
  const shown = filterFindings(all, state.filter);
  clear(ui.findings);
  if (!shown.length) {
    const message = all.length
      ? "No findings match these filters."
      : "No leakage findings were returned for this target. Check the audit and impact panels for context.";
    ui.findings.append(el("p", { className: "live-empty", textContent: message }));
  } else {
    ui.findings.append(...shown.map(findingNode));
  }
  setText(
    ui.findingsCount,
    all.length
      ? `Showing ${shown.length} of ${componentLabel(all.length, "finding")}`
      : "No findings"
  );
};

const renderFilters = () => {
  const all = Array.isArray(state.result?.findings) ? state.result.findings : [];
  const counts = severityCounts(all);
  if (ui.filterSeverity) {
    clear(ui.filterSeverity);
    const options = [
      ["all", `All ${counts.total}`],
      ["high", `High ${counts.high}`],
      ["medium", `Medium ${counts.medium}`],
      ["low", `Low ${counts.low}`],
    ];
    for (const [value, label] of options) {
      const button = el("button", {
        type: "button",
        className: "chip",
        textContent: label,
        "aria-pressed": String(state.filter.severity === value),
      });
      button.addEventListener("click", () => {
        state.filter.severity = value;
        renderFilters();
        renderFindings();
      });
      ui.filterSeverity.append(button);
    }
  }
  if (ui.filterKind) {
    const current = state.filter.kind;
    const kinds = findingKinds(all);
    ui.filterKind.replaceChildren(
      option("all", "All kinds"),
      ...kinds.map((kind) => option(kind, kind))
    );
    state.filter.kind = kinds.includes(current) ? current : "all";
    ui.filterKind.value = state.filter.kind;
  }
  if (ui.filterQuery) ui.filterQuery.value = state.filter.query;
};

const aiList = (title, items) => {
  if (!Array.isArray(items) || !items.length) return null;
  return el("div", { className: "live-ai-list" }, [
    el("strong", { textContent: title }),
    listBlock(items.map((item) => String(item)), "live-list"),
  ]);
};

const aiText = (value) =>
  value === null || value === undefined || value === "" ? null : String(value);

const renderAI = (data) => {
  if (!ui.aiResult || !ui.aiBody) return;
  clear(ui.aiBody);
  const ai = data?.ai;
  if (!ai || typeof ai !== "object") {
    if (data?.ai_unavailable) {
      ui.aiResult.hidden = false;
      ui.aiBody.append(
        el("p", {
          className: "live-note",
          textContent:
            "AI analysis was requested but the backend could not produce it. Deterministic results are unchanged.",
        })
      );
    } else {
      ui.aiResult.hidden = true;
    }
    return;
  }
  ui.aiResult.hidden = false;

  const plan = ai.plan && typeof ai.plan === "object" ? ai.plan : {};
  const groups = Array.isArray(plan.root_causes) ? plan.root_causes : [];
  const hypotheses = Array.isArray(ai.hypotheses) ? ai.hypotheses : [];
  const models = Array.isArray(ai.models) ? ai.models : [];
  const notes = Array.isArray(plan.llm_notes) ? plan.llm_notes : [];
  const score = ai.deterministic_trust_score ?? data.trust_score;

  ui.aiBody.append(
    el("p", {
      className: "live-note",
      textContent: `Scope: ${String(ai.scope || "unknown")} · ${hypotheses.length} hypothesis(es) · ${groups.length} root-cause group(s) · deterministic trust score ${String(score)} (unchanged)`,
    })
  );
  if (aiText(plan.summary)) {
    ui.aiBody.append(el("p", { textContent: aiText(plan.summary) }));
  }
  if (aiText(plan.llm_summary)) {
    ui.aiBody.append(
      el("p", { className: "live-ai-llm", textContent: `LLM rephrasing: ${aiText(plan.llm_summary)}` })
    );
  }
  if (plan.enrichment === "deterministic_fallback" && aiText(plan.fallback_reason)) {
    ui.aiBody.append(
      el("p", {
        className: "live-note",
        textContent: `LLM enrichment was not used: ${aiText(plan.fallback_reason)}`,
      })
    );
  }

  if (!groups.length && !hypotheses.length) {
    ui.aiBody.append(
      el("p", {
        className: "live-note",
        textContent:
          "No AI hypotheses were produced. A clean AI pass does not prove the pipeline is risk-free.",
      })
    );
  }

  for (const group of groups) {
    const card = el("section", { className: "live-ai-card" });
    card.append(el("h4", { textContent: aiText(group.title) || "Root-cause group" }));
    card.append(
      el("p", {
        className: "live-note",
        textContent: `Priority: ${String(group.priority || "unknown")} · Confidence: ${String(group.confidence ?? "—")} · Source: ${String(group.source || "unknown")}`,
      })
    );
    if (aiText(group.explanation)) card.append(el("p", { textContent: aiText(group.explanation) }));
    const remediation = group.remediation && typeof group.remediation === "object" ? group.remediation : {};
    const actions = aiList("Remediation", remediation.actions);
    const verification = aiList("Verification", remediation.verification);
    if (actions) card.append(actions);
    if (verification) card.append(verification);
    if (aiText(remediation.code_hint)) {
      card.append(el("pre", {}, [el("code", { textContent: aiText(remediation.code_hint) })]));
    }
    const note = notes.find((item) => item?.group_id === group.group_id);
    if (note) {
      const noteBlock = el("div", { className: "live-ai-note" }, [
        el("strong", { textContent: "LLM rephrasing (validated, no new evidence)" }),
      ]);
      if (aiText(note.explanation)) noteBlock.append(el("p", { textContent: aiText(note.explanation) }));
      const noteActions = aiList("Suggested actions", note.actions);
      const noteVerification = aiList("Suggested verification", note.verification);
      if (noteActions) noteBlock.append(noteActions);
      if (noteVerification) noteBlock.append(noteVerification);
      card.append(noteBlock);
    }
    ui.aiBody.append(card);
  }

  for (const hypothesis of hypotheses) {
    const card = el("section", { className: "live-ai-card" });
    card.append(el("h4", { textContent: aiText(hypothesis.title) || "AI hypothesis" }));
    card.append(
      el("p", {
        className: "live-note",
        textContent: `Concept: ${String(hypothesis.concept || "unknown")} · Confidence: ${String(hypothesis.confidence ?? "—")} · Severity hint: ${String(hypothesis.severity_hint || "info")} · Location: ${String(hypothesis.location || "—")}`,
      })
    );
    if (aiText(hypothesis.explanation)) {
      card.append(el("p", { textContent: aiText(hypothesis.explanation) }));
    }
    if (hypothesis.corroborates) {
      card.append(
        el("p", {
          className: "live-note",
          textContent: `Corroborates deterministic finding: ${String(hypothesis.corroborates)}`,
        })
      );
    }
    const evidence = Array.isArray(hypothesis.evidence) ? hypothesis.evidence : [];
    if (evidence.length) {
      card.append(
        aiList(
          "Evidence",
          evidence.map(
            (span) =>
              `${String(span.kind || "evidence")} at ${String(span.location || "—")}: ${String(span.excerpt || "")}`
          )
        )
      );
    }
    const limitations = aiList("Limitations", hypothesis.limitations);
    if (limitations) card.append(limitations);
    ui.aiBody.append(card);
  }

  if (models.length) {
    const details = el("details", { className: "live-ai-model" }, [
      el("summary", { textContent: "Model card and measured performance" }),
    ]);
    for (const model of models) {
      const external = Array.isArray(model.external)
        ? model.external
            .map((item) => (Array.isArray(item) ? `${String(item[0])}: ${String(item[1])}` : ""))
            .filter(Boolean)
            .join(" · ")
        : "";
      details.append(
        el("p", {
          textContent: `${String(model.name || "model")} · ${String(model.kind || "")} · threshold ${String(model.threshold ?? "—")} · precision ${String(model.precision ?? "—")} · recall ${String(model.recall ?? "—")} · F1 ${String(model.f1 ?? "—")} · FPR ${String(model.false_positive_rate ?? "—")}`,
        })
      );
      if (aiText(model.evaluation)) {
        details.append(el("p", { className: "live-note", textContent: aiText(model.evaluation) }));
      }
      if (external) {
        details.append(
          el("p", { className: "live-note", textContent: `External transfer: ${external}` })
        );
      }
      const modelLimitations = aiList("Model limitations", model.limitations);
      if (modelLimitations) details.append(modelLimitations);
    }
    ui.aiBody.append(details);
  }

  const globalLimitations = aiList("AI limitations", ai.limitations);
  if (globalLimitations) ui.aiBody.append(globalLimitations);
};

const renderResult = (data) => {
  state.result = data;
  state.filter = { severity: "all", kind: "all", query: "" };
  hideError();
  hideProgress();
  if (!ui.results) return;
  ui.results.hidden = false;
  const score = Number(data.trust_score);
  const tone = scoreTone(score);
  setText(ui.score, Number.isFinite(score) ? String(score) : "—");
  setText(ui.scoreWord, scoreWord(score));
  if (ui.scoreRing) {
    ui.scoreRing.style.setProperty("--score", String(Number.isFinite(score) ? score : 0));
    ui.scoreRing.dataset.tone = tone;
    ui.scoreRing.setAttribute("aria-label", `Trust score ${score} out of 100`);
  }
  renderCounts(data.counts || severityCounts(data.findings));
  renderSummary(data);
  renderAudit(data);
  renderImpact(data);
  renderAI(data);
  renderFilters();
  renderFindings();
  const heading = $("#live-results-title");
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
  }
  ui.results.scrollIntoView({
    behavior: reduceMotion.matches ? "auto" : "smooth",
    block: "start",
  });
  announce(
    `Scan complete. Trust score ${score} out of 100. ${componentLabel(
      Array.isArray(data.findings) ? data.findings.length : 0,
      "finding"
    )}.`
  );
};

const runScan = async (event) => {
  event.preventDefault();
  if (!state.connected || !state.file || state.busy) return;
  const target = ui.target?.value;
  if (!target) {
    showError(
      {
        title: "Choose a target",
        message: "Inspect the file, then choose the column your model predicts.",
        retryable: false,
      },
      null
    );
    return;
  }
  const limits = limitsFrom(state.capabilities);
  const form = new FormData();
  form.append("file", state.file, state.file.name);
  form.append("target", target);
  form.append("threshold", String(Number(ui.threshold?.value || 0.95)));
  if (ui.group?.value) form.append("group", ui.group.value);
  if (ui.time?.value) form.append("time", ui.time.value);
  if (ui.aiToggle?.checked) form.append("include_ai", "true");
  if (state.testFile) form.append("test", state.testFile, state.testFile.name);
  state.cancelled = false;
  state.controller = new AbortController();
  setBusy(true);
  showProgress();
  const timeoutMs = Math.max(20_000, (Number(limits.timeout_seconds) + 20) * 1000);
  try {
    const payload = await send(
      "/audit",
      { method: "POST", body: form },
      timeoutMs,
      state.controller
    );
    renderResult(payload);
  } catch (error) {
    if (error.name === "AbortError" && state.cancelled) {
      hideProgress();
      announce("Scan cancelled. No result was returned.");
      showError(
        {
          title: "Scan cancelled",
          message: "The request was cancelled in the browser. The backend may still finish its work.",
          retryable: false,
        },
        null
      );
    } else if (error instanceof RequestTimeout) {
      showError(
        {
          title: "Scan timed out",
          message: `The backend did not respond within ${Math.round(timeoutMs / 1000)} seconds. It may still be working; retry with a smaller file.`,
          retryable: true,
        },
        () => runScan(new Event("submit"))
      );
    } else if (error instanceof ApiFailure) {
      showError(errorMessage(error.status, error.payload), () => runScan(new Event("submit")));
      if (error.status === 503 || error.status === 504) {
        setConnection(
          "live",
          "Live backend",
          "The backend is reachable but busy or out of time. Retry shortly."
        );
      }
    } else {
      showError(errorMessage(0, null), () => runScan(new Event("submit")));
      setConnection("error", "Connection lost", "The request did not reach the backend.");
    }
  } finally {
    state.controller = null;
    setBusy(false);
    hideProgress();
  }
};

const download = (filename, content, type) => {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = el("a", { href: url, download: filename });
  doc.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 5_000);
};

const resetScan = () => {
  state.file = null;
  state.testFile = null;
  state.inspect = null;
  state.testInspect = null;
  clearResult();
  if (ui.fileInput) ui.fileInput.value = "";
  if (ui.testInput) ui.testInput.value = "";
  if (ui.fileInfo) ui.fileInfo.hidden = true;
  if (ui.testInfo) ui.testInfo.hidden = true;
  if (ui.fileErrors) clear(ui.fileErrors);
  if (ui.fileWarnings) clear(ui.fileWarnings);
  if (ui.testErrors) clear(ui.testErrors);
  if (ui.testWarnings) clear(ui.testWarnings);
  if (ui.clearFile) ui.clearFile.hidden = true;
  if (ui.clearTest) ui.clearTest.hidden = true;
  if (ui.preview) clear(ui.preview);
  if (ui.target) populateSelect(ui.target, []);
  if (ui.group) populateSelect(ui.group, [], { includeNone: true });
  if (ui.time) populateSelect(ui.time, [], { includeNone: true });
  if (ui.config) ui.config.dataset.ready = "false";
  setText(ui.fileMeta, "Drop a file here or use the file picker.");
  setText(ui.testMeta, "Optional: compare a separate test split.");
  setBusy(false);
  announce("Ready for another dataset.");
};

const bindDropzone = (zone, input, kind) => {
  if (!zone || !input) return;
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) setFile(file, kind);
  });
  for (const eventName of ["dragenter", "dragover"]) {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "dragend"]) {
    zone.addEventListener(eventName, () => zone.classList.remove("is-dragging"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("is-dragging");
    const file = event.dataTransfer?.files?.[0];
    if (file) setFile(file, kind);
  });
};

const bind = () => {
  ui.apiForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    connect(ui.apiInput?.value || "");
  });
  ui.threshold?.addEventListener("input", () => {
    setText(ui.thresholdValue, Number(ui.threshold.value).toFixed(2));
  });
  ui.filterKind?.addEventListener("change", () => {
    state.filter.kind = ui.filterKind.value;
    renderFindings();
  });
  ui.filterQuery?.addEventListener("input", () => {
    state.filter.query = ui.filterQuery.value;
    renderFindings();
  });
  ui.config?.addEventListener("submit", runScan);
  ui.cancel?.addEventListener("click", () => {
    state.cancelled = true;
    state.controller?.abort();
  });
  ui.clearFile?.addEventListener("click", () => resetScan());
  ui.clearTest?.addEventListener("click", () => {
    state.testFile = null;
    state.testInspect = null;
    if (ui.testInput) ui.testInput.value = "";
    if (ui.testInfo) ui.testInfo.hidden = true;
    if (ui.testErrors) clear(ui.testErrors);
    if (ui.testWarnings) clear(ui.testWarnings);
    if (ui.clearTest) ui.clearTest.hidden = true;
    setText(ui.testMeta, "Optional: compare a separate test split.");
    announce("Held-out file removed.");
  });
  ui.downloadReport?.addEventListener("click", () => {
    if (!state.result?.report_html) return;
    download(
      makeReportFilename(state.result.summary?.file || "leaklens-report", "html"),
      state.result.report_html,
      "text/html;charset=utf-8"
    );
  });
  ui.downloadJson?.addEventListener("click", () => {
    if (!state.result) return;
    const copy = { ...state.result };
    delete copy.report_html;
    download(
      makeReportFilename(state.result.summary?.file || "leaklens-report", "json"),
      JSON.stringify(copy, null, 2),
      "application/json;charset=utf-8"
    );
  });
  ui.newScan?.addEventListener("click", resetScan);
  bindDropzone(ui.dropzone, ui.fileInput, "primary");
  bindDropzone(ui.testDropzone, ui.testInput, "test");
};

const start = () => {
  bind();
  renderLimits(null);
  setBusy(false);
  const query = new URLSearchParams(window.location.search).get("api") || "";
  const configured = query || storedBase() || CONFIG.apiBase || "";
  if (configured) {
    if (ui.apiInput) ui.apiInput.value = configured;
    connect(configured, { auto: true });
  } else {
    setOffline(
      "This GitHub Pages build is static. Run the API locally or deploy the documented Render service, then connect its URL above."
    );
  }
  if (!("fetch" in window) || !window.AbortController) {
    setConnection("error", "Browser unsupported", "This page needs fetch and AbortController.");
    showError(
      {
        title: "Browser not supported",
        message: "Use a current browser to run a live scan, or use the prerecorded demo.",
        retryable: false,
      },
      null
    );
  }
};

if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", start);
else start();
