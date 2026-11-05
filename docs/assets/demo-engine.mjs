/* Pure helpers for the prerecorded demo. No DOM, no network: tests import
   these directly. */

export const GROUPS = [
  { id: "code", label: "Code", blurb: "Python and notebooks" },
  { id: "datasets", label: "Datasets", blurb: "Tables and targets" },
  { id: "overlaps", label: "Splits", blurb: "Train/test checks" },
];

export const SEVERITIES = ["high", "medium", "low"];
export const SEVERITY_RANK = { high: 0, medium: 1, low: 2 };

export function groupFixtures(data) {
  return GROUPS.map((group) => ({
    ...group,
    fixtures: Array.isArray(data?.[group.id]) ? data[group.id] : [],
  }));
}

export function allFixtures(data) {
  return GROUPS.flatMap((group) => data?.[group.id] || []);
}

export function findFixture(data, id) {
  return allFixtures(data).find((fixture) => fixture.id === id) || null;
}

export function sortFindings(findings) {
  return [...(findings || [])].sort((a, b) => {
    const bySeverity = (SEVERITY_RANK[a.severity] ?? 9) - (SEVERITY_RANK[b.severity] ?? 9);
    if (bySeverity !== 0) return bySeverity;
    const lineA = Number.isInteger(a.line) ? a.line : Number.MAX_SAFE_INTEGER;
    const lineB = Number.isInteger(b.line) ? b.line : Number.MAX_SAFE_INTEGER;
    if (lineA !== lineB) return lineA - lineB;
    return String(a.kind).localeCompare(String(b.kind));
  });
}

export function filterFindings(findings, severity = "all") {
  const sorted = sortFindings(findings);
  if (!SEVERITIES.includes(severity)) return sorted;
  return sorted.filter((finding) => finding.severity === severity);
}

export function severityCounts(findings) {
  const counts = { high: 0, medium: 0, low: 0, total: 0 };
  for (const finding of findings || []) {
    if (finding.severity in counts) counts[finding.severity] += 1;
    counts.total += 1;
  }
  return counts;
}

/** Mirrors the CLI gate: any high finding exits 1; a trust score below an
 *  explicit --fail-under also exits 1. Code scans have no trust score. */
export function cliGate(fixture, failUnder = null) {
  const findings = fixture?.findings || [];
  const counts = severityCounts(findings);
  const hasScore = Number.isFinite(fixture?.trustScore);
  const belowFloor = hasScore && Number.isFinite(failUnder) && fixture.trustScore < failUnder;
  const failing = counts.high > 0 || belowFloor;
  const reasons = [];
  if (counts.high > 0) reasons.push(`${counts.high} high-severity finding${counts.high === 1 ? "" : "s"}`);
  if (belowFloor) reasons.push(`trust score ${fixture.trustScore} is below --fail-under ${failUnder}`);
  let detail;
  if (!hasScore) {
    detail = failing
      ? "A high finding fails the default CLI gate."
      : "No high findings, so the default CLI gate passes.";
  } else if (belowFloor) {
    detail = `Trust score ${fixture.trustScore}/100 is below the configured floor of ${failUnder}.`;
  } else {
    detail = `Trust score ${fixture.trustScore}/100.${counts.high ? " A high finding still fails the gate." : ""}`;
  }
  return {
    exitCode: failing ? 1 : 0,
    label: failing ? "Exit 1" : "Exit 0",
    detail,
    reasons,
    failing,
  };
}

export function scoreTone(score) {
  if (!Number.isFinite(score)) return "neutral";
  if (score >= 90) return "safe";
  if (score >= 70) return "warn";
  return "danger";
}

export function scoreWord(score) {
  if (!Number.isFinite(score)) return "No score";
  if (score >= 90) return "Clean";
  if (score >= 70) return "Review";
  return "Leaky";
}

/** Split source into numbered lines and mark lines referenced by findings. */
export function sourceLines(fixture) {
  if (!fixture?.source) return [];
  const hits = new Map();
  for (const finding of fixture.findings || []) {
    if (!Number.isInteger(finding.line)) continue;
    const current = hits.get(finding.line);
    if (!current || SEVERITY_RANK[finding.severity] < SEVERITY_RANK[current]) {
      hits.set(finding.line, finding.severity);
    }
  }
  return fixture.source.split(/\r?\n/).map((text, index) => {
    const number = index + 1;
    return {
      number,
      text,
      severity: hits.get(number) || null,
      marked: hits.has(number),
    };
  });
}

export function formatCell(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

export function fixtureMeta(fixture) {
  const counts = severityCounts(fixture?.findings);
  const gate = cliGate(fixture);
  return {
    counts,
    gate,
    hasScore: Number.isFinite(fixture?.trustScore),
    tone: scoreTone(fixture?.trustScore),
    word: scoreWord(fixture?.trustScore),
    columns: fixture?.columns || [],
    rows: fixture?.rows ?? null,
    target: fixture?.target || null,
    check: fixture?.check || null,
    split: fixture?.split || null,
    duplicates: fixture?.plot?.counts?.duplicates ?? null,
  };
}

/** High findings first, then dataset score, for "worst first" listings. */
export function rankFixtures(fixtures) {
  return [...(fixtures || [])].sort((a, b) => {
    const high = severityCounts(b.findings).high - severityCounts(a.findings).high;
    if (high !== 0) return high;
    const scoreA = Number.isFinite(a.trustScore) ? a.trustScore : 101;
    const scoreB = Number.isFinite(b.trustScore) ? b.trustScore : 101;
    return scoreA - scoreB;
  });
}
