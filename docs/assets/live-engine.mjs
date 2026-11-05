// Pure helpers for the live scan page. Kept free of DOM and network access so
// the same functions can run under Node's built-in test runner.

export const DEFAULT_LIMITS = Object.freeze({
  max_bytes: 4 * 1024 * 1024,
  max_rows: 5000,
  max_columns: 200,
  max_cells: 750000,
  timeout_seconds: 30,
  max_source_bytes: 262144,
});

export const DEFAULT_FORMATS = Object.freeze([".csv", ".json"]);

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);
const EXTENSION_BY_MIME = Object.freeze({
  "text/csv": ".csv",
  "application/csv": ".csv",
  "application/json": ".json",
  "text/json": ".json",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
  "application/vnd.apache.parquet": ".parquet",
});

export function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "unknown size";
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(value < 10240 ? 1 : 0)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatDuration(ms) {
  const value = Number(ms);
  if (!Number.isFinite(value) || value < 0) return "unknown";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 2 : 1)} s`;
}

export function normalizeApiBase(raw, { allowHttpLocalhost = true } = {}) {
  const text = String(raw ?? "").trim();
  if (!text) {
    return { ok: false, base: "", host: "", error: "Enter the URL of a LeakLens live-scan API." };
  }
  let candidate = text;
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(candidate)) {
    const hostPart = candidate.split("/")[0].toLowerCase();
    const hostname = hostPart.replace(/^\[/, "").replace(/\].*$/, "").split(":")[0];
    candidate = `${LOCAL_HOSTS.has(hostname) ? "http" : "https"}://${candidate}`;
  }
  let url;
  try {
    url = new URL(candidate);
  } catch {
    return { ok: false, base: "", host: "", error: "That is not a valid URL." };
  }
  if (url.username || url.password) {
    return { ok: false, base: "", host: "", error: "Do not put credentials in the API URL." };
  }
  if (url.search || url.hash) {
    return { ok: false, base: "", host: "", error: "Remove the query string and fragment from the API URL." };
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    return { ok: false, base: "", host: "", error: "The API URL must use HTTPS (or HTTP on localhost)." };
  }
  const local = LOCAL_HOSTS.has(url.hostname.toLowerCase());
  if (url.protocol === "http:" && !(allowHttpLocalhost && local)) {
    return {
      ok: false,
      base: "",
      host: url.host,
      error: "Public scan backends must use HTTPS. Plain HTTP is allowed only for localhost.",
    };
  }
  let path = url.pathname.replace(/\/+$/, "");
  if (path === "/") path = "";
  const base = `${url.origin}${path}`;
  return { ok: true, base, host: url.host, secure: url.protocol === "https:" };
}

export function supportedExtensions(capabilities) {
  const formats = capabilities?.formats;
  if (!Array.isArray(formats)) return [...DEFAULT_FORMATS];
  const enabled = formats
    .filter((item) => item && item.enabled && typeof item.extension === "string")
    .map((item) => item.extension.toLowerCase());
  return enabled.length ? enabled : [...DEFAULT_FORMATS];
}

export function limitsFrom(capabilities) {
  return { ...DEFAULT_LIMITS, ...(capabilities?.limits || {}) };
}

export function validateFile(file, capabilities) {
  const errors = [];
  const warnings = [];
  if (!file || typeof file.name !== "string") {
    return { ok: false, errors: ["Choose a dataset file."], warnings, extension: "" };
  }
  const name = file.name.trim();
  const dot = name.lastIndexOf(".");
  const extension = dot > 0 ? name.slice(dot).toLowerCase() : "";
  const limits = limitsFrom(capabilities);
  const allowed = supportedExtensions(capabilities);
  if (!extension) {
    errors.push("The file has no extension. Use CSV or another supported format.");
  } else if (!allowed.includes(extension)) {
    errors.push(`This backend accepts ${allowed.join(", ")}; received ${extension}.`);
  }
  if (!Number.isFinite(Number(file.size)) || Number(file.size) <= 0) {
    errors.push("The file is empty.");
  } else if (Number(file.size) > Number(limits.max_bytes)) {
    errors.push(`The file is ${formatBytes(file.size)}; the limit is ${formatBytes(limits.max_bytes)}.`);
  }
  const mime = String(file.type || "").toLowerCase();
  if (mime && EXTENSION_BY_MIME[mime] && EXTENSION_BY_MIME[mime] !== extension) {
    warnings.push(
      `The browser reports ${mime} but the extension is ${extension}. LeakLens validates the content, not the MIME type.`
    );
  }
  return { ok: errors.length === 0, errors, warnings, extension, limits };
}

export function severityCounts(findings) {
  const counts = { high: 0, medium: 0, low: 0, total: 0 };
  for (const finding of Array.isArray(findings) ? findings : []) {
    if (finding && finding.severity in counts && finding.severity !== "total") {
      counts[finding.severity] += 1;
      counts.total += 1;
    }
  }
  return counts;
}

export function findingKinds(findings) {
  const kinds = new Set();
  for (const finding of Array.isArray(findings) ? findings : []) {
    if (finding?.kind) kinds.add(String(finding.kind));
  }
  return [...kinds].sort();
}

export function filterFindings(findings, { severity = "all", kind = "all", query = "" } = {}) {
  const needle = String(query || "").trim().toLowerCase();
  return (Array.isArray(findings) ? findings : []).filter((finding) => {
    if (!finding) return false;
    if (severity !== "all" && finding.severity !== severity) return false;
    if (kind !== "all" && finding.kind !== kind) return false;
    if (!needle) return true;
    return [finding.kind, finding.column, finding.detail, finding.fix]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(needle));
  });
}

export function scoreTone(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "unknown";
  if (value >= 85) return "safe";
  if (value >= 60) return "warn";
  return "danger";
}

export function scoreWord(score) {
  const tone = scoreTone(score);
  if (tone === "safe") return "Trusted";
  if (tone === "warn") return "Review";
  if (tone === "danger") return "Leaky";
  return "Unknown";
}

export function componentLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function errorMessage(status, payload) {
  const code = Number(status);
  const server = payload?.error;
  const safeServerMessage =
    typeof server?.message === "string" && server.message.length <= 320
      ? server.message
      : "";
  if (code === 0) {
    return {
      title: "Backend unreachable",
      message:
        "The browser could not reach the scan API. Check the URL, the backend status, and your network connection.",
      retryable: true,
    };
  }
  if (code === 413) {
    return {
      title: "Upload too large",
      message: safeServerMessage || "The file or source exceeds this backend's limit. Try a smaller extract.",
      retryable: false,
    };
  }
  if (code === 415) {
    return {
      title: "Unsupported file",
      message: safeServerMessage || "This backend does not accept that format. Supported formats are listed beside the upload.",
      retryable: false,
    };
  }
  if (code === 422) {
    return {
      title: "Check the dataset",
      message: safeServerMessage || "The file parsed, but the scan options were rejected. Inspect the columns and target.",
      retryable: false,
    };
  }
  if (code === 429) {
    return {
      title: "Rate limited",
      message: safeServerMessage || "Too many scans were sent from this client. Wait a moment and try again.",
      retryable: true,
    };
  }
  if (code === 503) {
    return {
      title: "Server busy",
      message: safeServerMessage || "Another scan is running on this small backend. Retry in a few seconds.",
      retryable: true,
    };
  }
  if (code === 504) {
    return {
      title: "Scan timed out",
      message: safeServerMessage || "The backend exceeded its time limit. Try fewer rows or fewer columns.",
      retryable: true,
    };
  }
  if (code === 400 || code === 404 || code === 405) {
    return {
      title: "Request rejected",
      message: safeServerMessage || "The backend rejected the request. Check the API URL and try again.",
      retryable: false,
    };
  }
  return {
    title: "Scan failed",
    message: safeServerMessage || "The backend returned an unexpected response. Try again, or run LeakLens locally.",
    retryable: true,
  };
}

export function capabilitySummary(capabilities) {
  const limits = limitsFrom(capabilities);
  return {
    formats: supportedExtensions(capabilities),
    maxBytes: limits.max_bytes,
    maxRows: limits.max_rows,
    maxColumns: limits.max_columns,
    maxCells: limits.max_cells,
    timeoutSeconds: limits.timeout_seconds,
    maxSourceBytes: limits.max_source_bytes,
    checks: Array.isArray(capabilities?.checks) ? capabilities.checks : [],
    privacy: typeof capabilities?.privacy === "string" ? capabilities.privacy : "",
    storage: capabilities?.storage || "unknown",
    version: capabilities?.version || "",
  };
}

export function previewColumns(columns, max = 8) {
  const list = Array.isArray(columns) ? columns : [];
  return { shown: list.slice(0, max), hidden: Math.max(0, list.length - max) };
}

export function targetOptions(inspection) {
  const columns = Array.isArray(inspection?.columns) ? inspection.columns : [];
  const names = columns.map((column) => String(column.name)).filter(Boolean);
  const candidates = Array.isArray(inspection?.target_candidates)
    ? inspection.target_candidates.filter((name) => names.includes(name))
    : [];
  return { names, candidates };
}

export function makeReportFilename(name, extension = "html") {
  const stem = String(name || "report")
    .replace(/\.[^.]+$/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "")
    .slice(0, 80);
  return `leaklens-${stem || "report"}.${extension}`;
}
