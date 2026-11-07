# Deterministic intelligence

`--intelligence` turns the existing scanner findings into a deterministic
evidence graph: confidence, root-cause explanations, remediation steps and -
since Wave 5 - **inspectable traversal and portable Mermaid/DOT/JSON exports**
of the clusters and links. Wave 6 adds **baseline-aware snapshots** that
compare two scans by stable fingerprint and report new, unchanged, changed,
resolved or uncertain evidence. It is opt-in, offline and CPU-only, and it is
available in the base install with no new dependency.

```sh
leaklens train.csv --target label --intelligence
leaklens train.csv --target label --test test.csv --group patient_id --intelligence
leaklens --scan src/ --intelligence --json
leaklens --scan src/ --graph-out evidence.mmd
```

Intelligence is **additive**. It never creates a finding, never edits a
severity or evidence record, never changes `leakage.score()` and never changes
the exit code. The deterministic detectors stay authoritative.

## What it produces

| Section | Meaning |
|---|---|
| `confidence` | Evidence-strength heuristic for the whole report. |
| `clusters` | Existing findings grouped by one inspectable anchor (feature, split or file), each with its own confidence and priority. |
| `links` | Directed relations between existing evidence records: `supports`, `corroborates` or `same_anchor`. |
| `impact` | Optional measured duplicate/group/temporal overlap (`--test` only). |
| `root_causes` | Recognised mechanism per finding kind, unchanged from earlier releases. |
| `recommendations` | Actions plus explicit validation steps, unchanged from earlier releases. |

The CLI prints a short cluster summary, the measured impact attached to the
top cluster, the first remediation action and verification step, and an
evidence-link count by relation. `--json` exposes the full graph under
`intelligence`, and `--html` renders *Evidence clusters* and *Evidence links*
tables. `--graph-out` writes the standalone graph instead (see below).

## Exploring and exporting the graph

Every report already carries the graph; `IntelligenceReport.to_graph()` returns
an `EvidenceGraph` with the same evidence, clusters and links without recomputing
anything. The graph is a read-only view:

```python
from leaklens.intelligence import analyze_findings

graph = analyze_findings(findings).to_graph()
graph.summary()                     # counts and relation totals
graph.evidence_by_id["scanner-000000"]
graph.cluster_for(0)                # cluster containing finding index 0
graph.neighbors("scanner-000000")   # (relation, "in"|"out", Evidence) tuples
```

`neighbors` returns an empty tuple for an unknown id, never raises, and orders
results by neighbouring evidence id, then relation, then direction. Traversal
never mutates the graph.

Export the same graph as Mermaid, Graphviz DOT or JSON:

```sh
leaklens --scan src/ --graph-out evidence.mmd     # Mermaid (also .mermaid, .md)
leaklens --scan src/ --graph-out evidence.dot     # DOT (also .gv)
leaklens --scan src/ --graph-out evidence.json    # JSON
leaklens --scan src/ --graph-out evidence.txt --graph-format mermaid
```

`--graph-out` implies `--intelligence`. Extension inference covers `.mmd`,
`.mermaid`, `.md`, `.dot`, `.gv` and `.json`; `--graph-format` overrides it.
An unknown extension or an unwritable path exits 2 instead of reporting success.
The export does not change findings, the trust score, the exit code or the input
file.

Mermaid and DOT use positional node ids (`n0`, `n1`, ... in evidence order) and
cluster subgraphs (`c0`, `c1`, ... in cluster order). Labels are collapsed and
escaped; brackets, pipes, quotes, backslashes and newlines cannot break the
document. Links that reference a missing record are skipped, and an empty graph
renders a valid empty `flowchart LR` / `digraph leaklens`. JSON is the existing
`to_dict()` payload (`evidence`, `clusters`, `links`).

Both API surfaces expose the same graph only on request: legacy `POST /scan`
and `/audit` accept `intelligence=true`; the hardened public API accepts
`include_intelligence=true` on `/audit` and `/scan`. The default payloads are
unchanged, the response adds an `intelligence` object, and `GET /capabilities`
advertises the opt-in. The public HTML report gains the cluster and link tables
only when requested.

## Baseline-aware graph comparison

A single graph shows one scan. **Snapshots** let you compare two scans
deterministically and review what changed. A snapshot is a separate, validated
JSON file (`schema: "leaklens-evidence-graph"`, version 1); it does not change
the `--graph-out` JSON payload.

```sh
# Save a snapshot from the current graph
leaklens --scan src/ --graph-snapshot evidence-baseline.json

# Compare a later scan against it
leaklens --scan src/ --graph-baseline evidence-baseline.json
leaklens --scan src/ --graph-baseline evidence-baseline.json --json
leaklens --scan src/ --graph-baseline evidence-baseline.json \
  --graph-snapshot evidence-baseline.json --update-graph-snapshot
```

`--graph-snapshot` and `--graph-baseline` imply `--intelligence`. They are
opt-in: without them the findings, trust score, JSON/SARIF payload and exit code
are exactly as before.

### Change states

Evidence is matched by **stable finding fingerprint**, never by list position,
display label or evidence id. Reordering findings, rebuilding evidence ids or
re-running the same scan therefore compares as unchanged.

| State | Meaning |
|---|---|
| `new` | Current evidence has no baseline counterpart. |
| `unchanged` | Same identity and same relevant content and relationships. |
| `changed` | Same identity, but evidence content or an incident relationship changed. |
| `resolved` | Baseline evidence has no current counterpart, and the comparison is complete enough to say so. |
| `uncertain` | Baseline evidence is absent, but the scan was incomplete, the analysis context differs, or dangling links were found. |

`changed` records name the fields that moved (`description`, `location`,
`metadata`, `relationships`, ...). Relationship changes are also listed
separately as `added` / `removed`, and relationship-only movement marks the
incident evidence `changed`. Cluster changes use the same states with their
own anchor and membership details. Duplicate evidence is kept with a
deterministic `#0`, `#1`, ... ordinal and counted in `duplicates`.

### Safety and interpretation

- Snapshots are validated strictly before use: unknown keys, wrong types,
  malformed JSON, non-UTF-8 input, oversized files, unknown evidence references
  and unsupported schemas/versions are rejected with a clear message (exit 2).
- Writes are atomic and refuse directories, symlinks and missing parents. An
  existing snapshot is never replaced unless `--update-graph-snapshot` is given.
- Missing baseline evidence becomes `uncertain`, not `resolved`, when either
  scan reported a parse error, the tool context changed (target, split, group,
  time, dialect, threshold, drift mode) or a dangling link was skipped.
- `tool_version` differences are a warning, not a hard failure; fingerprints
  still come from the schema recorded in the snapshot.
- `--fail-on-graph-new` is the only new gate. It exits 1 when new evidence is
  found and never fails on `changed`, `resolved` or `uncertain`. It requires
  `--graph-baseline`.

A graph comparison describes how evidence moved between two runs. It is **not**
proof that leakage was introduced or fixed, and a `resolved` record is not a
confirmation that the underlying issue was remediated.

### SARIF

With `--graph-snapshot` / `--graph-baseline`, SARIF gains compact per-result
`properties.leaklens.graph` metadata (`evidence_ids`, `evidence_count` and,
when comparing, `state`) and a run-level
`properties.leaklens.graph` summary (schema, version, completeness, counts and
the optional comparison). Evidence bodies, link rationales and the full graph
are never duplicated into every result. SARIF is unchanged when graph options
are not used.

## Anchors

Every finding is anchored on something a reviewer can open and check. Anchors
are built from the shared detector manifest (`manifest.surface_for`) and the
fingerprint normaliser (`fingerprint.normalize_location`), so they agree with
fingerprints and baselines.

| Detector surface | Anchor | Example |
|---|---|---|
| dataset | `dataset:<feature>` | `dataset:chargeback_flag` |
| split (column-scoped) | `split:<feature>` | `split:patient_id` |
| split (whole-split check) | `split` | `split` |
| code / notebook | `file:<path>` | `file:train.py` |
| image | `image` | `image` |
| unknown kind | `finding:<kind>` | `finding:some_new_kind` |

Code findings group by **file**, not by line, so three suspicious calls in one
training script form one cluster. Dataset findings group by feature; related
split checks share the `split` anchor. Unknown kinds deliberately fall back to
a per-kind anchor instead of guessing a surface.

## Confidence

The score is a transparent heuristic, **not a calibrated probability**.

Whole-report confidence keeps its historical formula and input: one primary
record per finding. Detector detail and corroboration are reflected in each
cluster's confidence instead, so the existing `intelligence.confidence` output
does not change when a detector starts attaching evidence.

```
score = 0.25
      + 0.15 * min(supporting, 4)
      - 0.12 * min(contradicting, 4)
      + 0.10 if supporting evidence covers more than one kind
```

Cluster confidence starts from that same formula over one primary record per
finding, then adds only corroboration that is visible in the graph:

```
+ 0.03 * min(distinct_finding_kinds - 1, 3)
+ 0.02 * min(detector_records, 3)      # max +0.06
+ 0.02 if evidence comes from more than one surface
```

The result is clamped to `0.95`. A single finding with one detector record
therefore stays "low": one observation is weak evidence, even when its
severity is high. Severity and confidence answer different questions, and the
cluster keeps both instead of collapsing them.

## Detector evidence preservation

Findings already carry validated `Evidence` records (bounded, privacy-safe,
aggregate only). The intelligence layer now keeps all of them:

- one **primary** record per finding, with the historical `scanner-000000` id
  and `{"severity": ...}` metadata, so existing consumers keep working;
- one **detector** record per attached item, with a stable
  `scanner-000000-e01` id, `role=detector` metadata and the observation's
  `observed`/`expected` values.

`AnalysisContext.primary_evidence`, `.detector_evidence` and
`.evidence_for_finding(index)` expose the split. `finding_count` still counts
distinct findings, not evidence records.

## Links

Links explain why records are shown together. They never add a claim:

- `supports` - detector-level detail explains a primary finding; the link
  targets the primary record of the same finding, not the cluster
  representative;
- `corroborates` - two findings of different kinds share an anchor;
- `same_anchor` - two findings of the same kind share an anchor (repeated
  pattern in one file).

A detector record with no known finding index produces no link rather than an
invented target. Links are emitted per cluster, capped at 400 per report, and
only reference ids that exist in the same report. Ordering is deterministic:
clusters sort by priority, then confidence, then anchor; ids are slugs with a
numeric suffix on collision.

## Measured impact

When a test frame is provided (`--test`), `DatasetImpact` measurements are
attached to the cluster they belong to instead of a separate global list:

- duplicate test rows -> the `split` cluster;
- shared group values -> the `split:<group column>` or `dataset:<group column>`
  cluster;
- temporal overlap -> the `split:<time column>` or `dataset:<time column>`
  cluster.

Numbers are computed by the existing impact code; the graph only routes them.
Without `--test` every `measured_impact` list is empty.

## Guarantees

- **No new findings.** Every cluster member is an existing finding index and
  evidence id.
- **No score or exit change.** `leakage.score()` and the default exit gate are
  untouched and intelligence output is not part of either.
- **Deterministic.** No RNG, no wall clock, no network; repeated runs produce
  identical JSON and identical Mermaid/DOT text.
- **Portable exports.** Mermaid, DOT and JSON use only stable node ids and
  escaped labels; empty graphs and unknown ids are handled without crashing.
- **Bounded.** Links are capped at 400; evidence length comes from the existing
  per-finding cap; cluster ids are short slugs.
- **Offline.** No model, no service, no credential.

## Limitations

- Anchors are one-level: a cluster groups by feature, split or file but does
  not traverse joins, notebooks or cross-file data flow.
- Cluster confidence rewards corroboration, not ground truth. Two correlated
  detectors on the same column raise it even if both share the same blind spot.
- Same-kind findings in different files never link; only their own anchors do.
- Exports are a view of the graph, not a proof: an edge means two existing
  records share an anchor or one explains another. It does not establish
  causality or confirm leakage.
- Mermaid and DOT are emitted as text; LeakLens does not bundle a renderer. Use
  any Mermaid or Graphviz viewer to display the file.
- The graph is one statement/report wide. Snapshots compare separate scans but
  never merge their evidence into one graph, and a comparison is only as good as
  the two runs being complete and context-compatible.
- Measured impact is descriptive. A 100% duplicate rate is a fact about the two
  frames supplied, not proof that a model leaked.
- Unknown detector kinds get a coarse `finding:<kind>` anchor until their
  surface is added to the manifest.

## Relation to the AI layer

Intelligence is the deterministic layer that runs with `--intelligence`. The
experimental AI layer (`--ai`) is separate, off by default and documented in
[AI_INTELLIGENCE.md](AI_INTELLIGENCE.md); it may *cite* intelligence output but
never changes the graph, a severity or the trust score.
