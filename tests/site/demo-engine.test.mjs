import assert from "node:assert/strict";
import { test } from "node:test";

import { fixtures } from "../../docs/assets/fixtures.mjs";
import {
  GROUPS,
  SEVERITIES,
  allFixtures,
  cliGate,
  filterFindings,
  findFixture,
  fixtureMeta,
  formatCell,
  groupFixtures,
  rankFixtures,
  scoreTone,
  scoreWord,
  severityCounts,
  sortFindings,
  sourceLines,
} from "../../docs/assets/demo-engine.mjs";

const advanced = findFixture(fixtures, "advanced");
const safe = findFixture(fixtures, "safe");
const leaky = findFixture(fixtures, "leaky_transactions");
const clean = findFixture(fixtures, "clean_transactions");
const near = findFixture(fixtures, "overlap_near");
const exact = findFixture(fixtures, "overlap_exact");

test("fixtures expose the three demo groups", () => {
  const groups = groupFixtures(fixtures);
  assert.deepEqual(
    groups.map((group) => group.id),
    GROUPS.map((group) => group.id)
  );
  assert.equal(groups[0].fixtures.length, 3);
  assert.equal(groups[1].fixtures.length, 2);
  assert.equal(groups[2].fixtures.length, 2);
  assert.equal(allFixtures(fixtures).length, 7);
});

test("findFixture returns a fixture or null", () => {
  assert.equal(findFixture(fixtures, "advanced")?.group, "code");
  assert.equal(findFixture(fixtures, "missing"), null);
});

test("findings sort by severity and then line", () => {
  const sorted = sortFindings(advanced.findings);
  assert.deepEqual(
    sorted.map((finding) => finding.severity),
    ["high", "high", "high", "medium", "medium", "medium", "low"]
  );
  const high = sorted.filter((finding) => finding.severity === "high");
  assert.ok(high[0].line <= high[1].line);
});

test("severity filtering never mutates the fixture", () => {
  const before = advanced.findings.length;
  const high = filterFindings(advanced.findings, "high");
  const all = filterFindings(advanced.findings, "all");
  assert.equal(high.length, 3);
  assert.equal(all.length, before);
  assert.equal(advanced.findings.length, before);
  assert.ok(high.every((finding) => finding.severity === "high"));
});

test("severityCounts totals every finding", () => {
  assert.deepEqual(severityCounts(advanced.findings), {
    high: 3,
    medium: 3,
    low: 1,
    total: 7,
  });
  assert.deepEqual(severityCounts([]), { high: 0, medium: 0, low: 0, total: 0 });
});

test("the CLI gate mirrors high findings and the score floor", () => {
  assert.equal(cliGate(advanced).exitCode, 1);
  assert.match(cliGate(advanced).detail, /high finding/i);
  assert.equal(cliGate(safe).exitCode, 0);
  assert.equal(cliGate(safe).label, "Exit 0");
  assert.equal(cliGate(leaky).exitCode, 1);
  assert.equal(cliGate(clean).exitCode, 0);
  assert.equal(cliGate(clean, 101).exitCode, 1);
  assert.match(cliGate(clean, 101).detail, /below/);
});

test("medium-only overlap passes the default gate but fails a strict floor", () => {
  assert.equal(severityCounts(near.findings).high, 0);
  assert.equal(near.trustScore, 85);
  assert.equal(cliGate(near).exitCode, 0);
  assert.equal(cliGate(near, 90).exitCode, 1);
  assert.equal(cliGate(exact).exitCode, 1);
});

test("sourceRows highlight only the real finding lines", () => {
  const lines = sourceLines(advanced);
  assert.equal(lines.length, advanced.source.split(/\r?\n/).length);
  const marked = lines.filter((line) => line.marked);
  assert.equal(marked.length, 6);
  assert.deepEqual(
    marked.map((line) => line.number).sort((a, b) => a - b),
    [16, 18, 22, 26, 29, 33]
  );
  assert.equal(lines[15].severity, "high");
  assert.equal(lines[0].marked, false);
  assert.deepEqual(sourceLines({ source: "" }), []);
});

test("formatCell keeps table values readable", () => {
  assert.equal(formatCell(null), "—");
  assert.equal(formatCell(3), "3");
  assert.equal(formatCell(3.14159), "3.142");
  assert.equal(formatCell(true), "true");
  assert.equal(formatCell("web"), "web");
});

test("scoreTone and scoreWord use the documented bands", () => {
  assert.equal(scoreTone(100), "safe");
  assert.equal(scoreTone(90), "safe");
  assert.equal(scoreTone(89), "warn");
  assert.equal(scoreTone(70), "warn");
  assert.equal(scoreTone(69), "danger");
  assert.equal(scoreTone(null), "neutral");
  assert.equal(scoreWord(100), "Clean");
  assert.equal(scoreWord(85), "Review");
  assert.equal(scoreWord(50), "Leaky");
  assert.equal(scoreWord(NaN), "No score");
});

test("fixtureMeta exposes the audit inputs without inventing values", () => {
  const meta = fixtureMeta(leaky);
  assert.equal(meta.rows, 600);
  assert.equal(meta.target, "is_fraud");
  assert.equal(meta.gate.exitCode, 1);
  assert.equal(meta.tone, "danger");
  assert.ok(meta.columns.includes("chargeback_flag"));
  const overlap = fixtureMeta(exact);
  assert.equal(overlap.check, "exact row overlap");
  assert.equal(overlap.duplicates, 20);
  assert.equal(overlap.split.train, 420);
});

test("rankFixtures puts the most severe fixture first", () => {
  const ranked = rankFixtures(fixtures.datasets);
  assert.equal(ranked[0].id, "leaky_transactions");
  assert.equal(ranked.at(-1).id, "clean_transactions");
});

test("every severity is a documented value", () => {
  for (const fixture of allFixtures(fixtures)) {
    for (const finding of fixture.findings) {
      assert.ok(SEVERITIES.includes(finding.severity), finding.kind);
    }
  }
});
