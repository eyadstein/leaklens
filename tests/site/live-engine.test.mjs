import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_FORMATS,
  DEFAULT_LIMITS,
  capabilitySummary,
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
} from "../../docs/assets/live-engine.mjs";

const caps = {
  live: true,
  version: "0.13.0",
  limits: {
    max_bytes: 1024,
    max_rows: 100,
    max_columns: 20,
    max_cells: 2000,
    timeout_seconds: 12,
    max_source_bytes: 4096,
  },
  formats: [
    { extension: ".csv", enabled: true },
    { extension: ".json", enabled: true },
    { extension: ".parquet", enabled: false },
  ],
  checks: ["target leakage"],
  privacy: "in memory",
  storage: "none",
};

const findings = [
  { severity: "high", kind: "target_leakage", column: "chargeback_flag", detail: "perfect", fix: "drop" },
  { severity: "medium", kind: "id_like", column: "txn_id", detail: "unique", fix: "drop id" },
  { severity: "low", kind: "parse_error", column: "file.py", detail: "syntax", fix: "" },
];

test("normalizeApiBase accepts HTTPS and strips a trailing slash", () => {
  const result = normalizeApiBase("https://leaklens.onrender.com/");
  assert.equal(result.ok, true);
  assert.equal(result.base, "https://leaklens.onrender.com");
  assert.equal(result.secure, true);
});

test("normalizeApiBase allows HTTP only for localhost", () => {
  assert.equal(normalizeApiBase("http://localhost:8000").ok, true);
  assert.equal(normalizeApiBase("http://127.0.0.1:8000").ok, true);
  assert.equal(normalizeApiBase("http://leaklens.example.com").ok, false);
  assert.equal(normalizeApiBase("localhost:8000").ok, true);
  assert.equal(normalizeApiBase("leaklens.example.com").base, "https://leaklens.example.com");
});

test("normalizeApiBase rejects credentials, queries and non-HTTP schemes", () => {
  assert.equal(normalizeApiBase("https://user:pass@example.com").ok, false);
  assert.equal(normalizeApiBase("https://example.com?token=1").ok, false);
  assert.equal(normalizeApiBase("javascript:alert(1)").ok, false);
  assert.equal(normalizeApiBase("").ok, false);
});

test("normalizeApiBase preserves a path prefix but not a trailing slash", () => {
  const result = normalizeApiBase("https://example.com/leaklens/");
  assert.equal(result.base, "https://example.com/leaklens");
});

test("supportedExtensions honours backend capabilities", () => {
  assert.deepEqual(supportedExtensions(caps), [".csv", ".json"]);
  assert.deepEqual(supportedExtensions(null), DEFAULT_FORMATS);
  assert.deepEqual(supportedExtensions({ formats: [] }), DEFAULT_FORMATS);
});

test("limitsFrom merges backend limits over defaults", () => {
  const limits = limitsFrom(caps);
  assert.equal(limits.max_bytes, 1024);
  assert.equal(limits.max_rows, 100);
  assert.equal(limits.timeout_seconds, 12);
  assert.equal(limitsFrom(null).max_bytes, DEFAULT_LIMITS.max_bytes);
});

test("validateFile accepts a bounded CSV and reports its extension", () => {
  const result = validateFile({ name: "data.csv", size: 200, type: "text/csv" }, caps);
  assert.equal(result.ok, true);
  assert.equal(result.extension, ".csv");
  assert.deepEqual(result.errors, []);
});

test("validateFile rejects empty, unsupported and oversized files", () => {
  assert.equal(validateFile({ name: "data.csv", size: 0 }, caps).ok, false);
  assert.equal(validateFile({ name: "data.txt", size: 20 }, caps).ok, false);
  assert.equal(validateFile({ name: "data.parquet", size: 20 }, caps).ok, false);
  const big = validateFile({ name: "data.csv", size: 2048 }, caps);
  assert.equal(big.ok, false);
  assert.match(big.errors.join(" "), /limit/i);
});

test("validateFile warns when the MIME type contradicts the extension", () => {
  const result = validateFile({ name: "data.csv", size: 20, type: "application/json" }, caps);
  assert.equal(result.ok, true);
  assert.equal(result.warnings.length, 1);
  assert.match(result.warnings[0], /content, not the MIME type/);
});

test("severityCounts and findingKinds are total and deterministic", () => {
  assert.deepEqual(severityCounts(findings), { high: 1, medium: 1, low: 1, total: 3 });
  assert.deepEqual(findingKinds(findings), ["id_like", "parse_error", "target_leakage"]);
  assert.deepEqual(severityCounts([]), { high: 0, medium: 0, low: 0, total: 0 });
});

test("filterFindings combines severity, kind and text without mutating", () => {
  const before = findings.length;
  assert.equal(filterFindings(findings, { severity: "high" }).length, 1);
  assert.equal(filterFindings(findings, { kind: "id_like" }).length, 1);
  assert.equal(filterFindings(findings, { query: "CHARGEBACK" }).length, 1);
  assert.equal(filterFindings(findings, { severity: "high", query: "txn" }).length, 0);
  assert.equal(filterFindings(findings, { severity: "all" }).length, 3);
  assert.equal(findings.length, before);
});

test("scoreTone and scoreWord map the central trust score bands", () => {
  assert.equal(scoreTone(100), "safe");
  assert.equal(scoreTone(85), "safe");
  assert.equal(scoreTone(84), "warn");
  assert.equal(scoreTone(60), "warn");
  assert.equal(scoreTone(59), "danger");
  assert.equal(scoreWord(100), "Trusted");
  assert.equal(scoreWord(70), "Review");
  assert.equal(scoreWord(0), "Leaky");
  assert.equal(scoreTone(Number.NaN), "unknown");
});

test("errorMessage keeps server text bounded and marks retryability", () => {
  assert.equal(errorMessage(413, null).retryable, false);
  assert.equal(errorMessage(429, null).retryable, true);
  assert.equal(errorMessage(504, null).retryable, true);
  assert.equal(errorMessage(0, null).title, "Backend unreachable");
  const safe = errorMessage(422, { error: { message: "Target column missing." } });
  assert.match(safe.message, /Target column missing/);
  const long = errorMessage(500, { error: { message: "x".repeat(1000) } });
  assert.doesNotMatch(long.message, /x{400}/);
});

test("capabilitySummary supplies honest defaults", () => {
  const summary = capabilitySummary(caps);
  assert.equal(summary.maxBytes, 1024);
  assert.equal(summary.maxRows, 100);
  assert.deepEqual(summary.formats, [".csv", ".json"]);
  assert.equal(summary.storage, "none");
  assert.equal(capabilitySummary(null).maxBytes, DEFAULT_LIMITS.max_bytes);
});

test("targetOptions prefers real candidates that exist in the columns", () => {
  const result = targetOptions({
    columns: [{ name: "x" }, { name: "y" }],
    target_candidates: ["y", "missing"],
  });
  assert.deepEqual(result.names, ["x", "y"]);
  assert.deepEqual(result.candidates, ["y"]);
  assert.deepEqual(targetOptions(null), { names: [], candidates: [] });
});

test("previewColumns caps the preview and counts hidden columns", () => {
  const columns = Array.from({ length: 11 }, (_, index) => ({ name: `c${index}` }));
  const result = previewColumns(columns, 8);
  assert.equal(result.shown.length, 8);
  assert.equal(result.hidden, 3);
});

test("makeReportFilename strips paths and hostile characters", () => {
  assert.equal(makeReportFilename("../../my data.csv"), "leaklens-my-data.html");
  assert.equal(makeReportFilename(""), "leaklens-report.html");
  assert.equal(makeReportFilename("a.csv", "json"), "leaklens-a.json");
});

test("formatBytes and formatDuration are deterministic", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(2048), "2.0 KiB");
  assert.equal(formatBytes(4 * 1024 * 1024), "4.0 MiB");
  assert.equal(formatDuration(12), "12 ms");
  assert.equal(formatDuration(1500), "1.50 s");
  assert.equal(formatDuration(12000), "12.0 s");
});
