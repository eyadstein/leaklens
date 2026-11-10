# Interactive scan workspace

`leaklens workspace` serves a local, read-only page for scanning a project or
auditing a dataset, and exploring the results: start a scan, watch live progress,
filter findings, read the evidence behind each one, open the surrounding source,
reopen retained sessions, compare compatible results, and export the same HTML
report the CLI writes.

It is a thin layer over the existing engine. Detection, fingerprints, trust
score, evidence clusters, fix advice and the HTML report are the same code paths
the CLI and the public API use; the workspace adds live updates, history,
comparison, navigation and a browser view.

```sh
pip install -e ".[app]"          # fastapi + uvicorn
leaklens workspace               # root = current folder, port 8765
leaklens workspace notebooks/ --port 9000 --no-open
```

The server binds `127.0.0.1` by default. `--host` is available for a deliberate
LAN setup, but note that anyone who can reach the port can read files under the
workspace root.

## What one source scan does

1. Enumerates the same files `leaklens --scan PATH` would scan
   (`scanner.scan_targets`), so both entry points agree on scope.
2. Scans them one at a time in a background thread, updating
   `done / total`, the current file, elapsed time and severity counts.
3. On completion, builds one payload from the existing deterministic layers:
   findings (`Finding.to_dict(root=...)`), the evidence graph
   (`intelligence.analyze_findings`), cluster confidence, root causes,
   recommendations, `policy.apply_policy()` and `leakage.score()`.
4. Renders the normal static HTML report with `report.render`.

An optional AI pass is requested per scan (`include_ai`, default off). It runs only after
`policy.apply_policy()` has classified the findings, so suppressed findings are never handed
to the AI layer as active; the AI source reader keeps its own 200-file / 512 KB caps on top of
the workspace limits. The payload is the same `AIReport` the CLI prints (schema `1.2`),
rendered by `report._ai_section`; nothing is re-implemented in the workspace. It never changes
the trust score, findings, policy labels or exit semantics, and a failure is reported as
`ai_unavailable: true` with the deterministic result intact.

Scans never write to disk and never execute the scanned code.

## Live updates

Each running session has a server-sent events stream:
`GET /api/scans/{id}/events` and `GET /api/datasets/{id}/events`. The stream reads
the same in-memory session snapshot the polling endpoint returns; it does not
queue results, so a slow or disconnected client cannot grow memory.

- The first event is a `snapshot`; later `snapshot` events are emitted only when
  the session actually changes (state, done/total, current file, counts, request,
  errors). A `failure` event carries an engine failure, and `done` is always the
  last event. Every event has an increasing `seq` and its `type`.
- Heartbeat comments keep idle streams alive. The browser uses `EventSource`
  when available and falls back to the existing 400 ms polling on an unsupported
  browser or a broken stream, showing `reconnecting with polling`.
- Reconnecting to a finished session replays its terminal snapshot and `done`
  once. Unknown or trimmed session ids return 404 before the stream starts;
  streams are scoped to exactly one session and never subscribe across sessions.
- Cancellation is still between files for a source scan and only while queued
  for a dataset audit. The stream reports those states honestly; it never
  implies the engine can be interrupted mid-call.

## Session history

`GET /api/history` returns the retained sessions, newest first: id, type, target,
state, start/completion wall-clock times, duration, result/report availability
and a small summary (findings, severities, trust score, files or dataset rows and
target).

- History is **in memory only**. It does not survive a server restart, and the
  endpoint and UI say so explicitly. No database was added and no source content
  is copied into a history index.
- Retention is the existing bound: the last 20 source scans and the last 20
  dataset audits. Starting a new session drops the oldest finished session of
  that kind first; running sessions are never dropped.
- The UI's history table can reopen results, the evidence graph and the HTML
  report, and can prefill a comparison. Reopening a retained result does not
  rescan anything.

## Comparing retained sessions

`GET /api/compare/scans?previous=ID&current=ID` and
`GET /api/compare/datasets?previous=ID&current=ID` compare two retained sessions
of the same kind. Compatibility is checked before any classification:

- A source scan is compatible with another source scan of the same target and
  the same SQL setting.
- A dataset audit is compatible with another audit of the same path, target,
  threshold, held-out file, time column and group column.

An incompatible pair returns `available: false` with the mismatched settings in
`checks` and an explicit reason; it is never partially compared.

The comparison reuses the previous session's findings as a
`Baseline.from_findings()` identity set and the existing
`baseline.compare()` classification. Changes are reported as:

- `added` — the fingerprint exists only in the current complete session.
- `unchanged` — the fingerprint exists in both with the same severity.
- `changed` — the fingerprint exists in both with a different severity; the
  previous and current findings are both included.
- `no_longer_present` — the fingerprint existed previously and is absent from a
  complete current session. Only these entries have `resolved: true`.
- `inconclusive` — the conclusion depends on a partial scan and is not claimed.
  A finding absent from a cancelled or failed current session is inconclusive,
  never resolved; a current finding absent from a partial previous session is
  inconclusive, never newly added.

Each entry keeps the current finding's policy output unchanged: `status`,
`suppression` reason/fingerprint and `requires_attention` are copied, and a
finding without a status stays unclassified. The summary counts every category,
plus `suppressed` and `unclassified`, and reports `counts_consistent` and
`truncated` (the list is capped at 500 entries while totals remain complete).
Entries include the current finding index and previous/current details, so the
UI can open the finding or its evidence graph.

## Baselines and suppressions

Both scan types use the existing policy engine. Nothing in the workspace defines
its own status or suppression semantics.

- The baseline is the file in the scan form (relative to the workspace root) or,
  when that field is blank, `[baseline].path` from `.leaklens.toml` at the root.
  A path outside the root is rejected, and a missing, malformed, incompatible or
  non-UTF-8 baseline fails the request with the `Baseline.load` message. It is
  never replaced with an empty baseline.
- `[[suppressions]]` from `.leaklens.toml` apply to source scans and dataset
  audits. Suppressed findings stay in the results with their original severity,
  detail and evidence; they carry `status: "suppressed"`, the matching rule's
  reason/fingerprint/expiry and `requires_attention: false`, and they stop
  counting toward `score()`.
- With a baseline, every finding gets the policy engine's `new`, `unchanged`,
  `changed` or `suppressed` status exactly as `policy.apply_policy()` returned
  it. `changed` keeps the previous baseline entry under `previous`. `resolved`
  entries are returned separately. Without a baseline, only suppressed findings
  have a status; the others are reported as `unclassified`, never invented as
  `unchanged`.
- `policy.statuses` counts are derived from the returned findings and
  `policy.unclassified` covers findings that had no baseline comparison. The
  UI filters on those exact statuses and keeps the severity/kind filters.
- A cancelled source scan is not compared with a baseline: a partial scan would
  make missing findings look resolved. Suppressions still label findings.
- The workspace is read-only: it loads and compares baselines but never creates
  or updates one. Use `leaklens --scan . --write-baseline` or
  `--update-baseline`; those writes stay atomic and explicit.

## Evidence graph

Both scan types expose the evidence graph that `intelligence` already built for
their findings. The workspace projects that graph; it never builds a second one
and never adds a node, edge, rationale or provenance record.

- `GET /api/scans/{id}/graph` reads the source result's top-level
  `evidence`/`clusters`/`links`; `GET /api/datasets/{id}/graph` reads the same
  fields from the dataset result's nested `intelligence` payload. Both support
  the same filters and paging.
- Nodes keep the engine's `evidence_id`, kind, source, description, location,
  metadata, `finding_index` and cluster anchor. Each node carries its finding
  view (including fingerprint, severity and policy `status`), its connected
  component, degree, `isolated` flag and bidirectional neighbour list with the
  recorded relation, direction and rationale.
- Filters are server-side: `kind`, `relation`, `severity`, `status`,
  `q` (evidence id, kind, source, description, location, finding fields),
  `offset`, `limit` (default 100, capped at 500), plus `focus` (an evidence id)
  and `finding` (a finding index). `relation` keeps only nodes incident to that
  relation.
- `focus` returns the selected node even when it is outside the current page;
  an unknown id is reported as `focus: null`, never fabricated. A finding focus
  prefers the finding's primary evidence over detector detail records.
- Invalid nodes/links are skipped and counted (`invalid_nodes`, `invalid_links`,
  `dangling_links`); duplicate and self-referencing links are not repaired.
  Missing or malformed graph data returns `available: false` with a reason, and
  an empty graph is `available: true` with zero nodes.
- `relationships_truncated` mirrors the engine's link limit for source results.
  Dataset intelligence does not serialise that flag, so it is `null` there
  (shown as unknown, not as complete). The UI explains both cases.
- Policy is untouched: graph responses use the findings' existing `status`
  values, suppressed findings remain inspectable with their evidence, and no
  graph read changes results, trust score, baselines or suppressions.
- The UI shares one graph panel for both scan types: filters, search, paging,
  node details, neighbour inspection, meta/observation details, and
  finding-to-evidence / evidence-to-finding navigation. Reading the graph has no
  path input, so it cannot read files outside the workspace root; unknown
  sessions return 404 and unfinished sessions 409.

## What one dataset audit does

A dataset audit is not a source scan with a different label: it is the public
API's audited pipeline, called in a background thread.

1. Resolves the path inside the workspace root and reads the bytes bounded by
   the same `GuardLimits` the public API applies (4 MiB, 5,000 rows, 200
   columns, 750,000 cells).
2. Runs `upload_guard.guard_upload` on it, so unsupported extensions, disabled
   formats, missing optional parsers, empty files and content that does not
   match its extension are rejected with the public API's status codes and
   messages.
3. Calls `public_api.run_audit_job` with the target, threshold, optional
   held-out file, and optional time/group columns. That function owns parsing,
   target validation, `leakage.scan`, split/feature/near-duplicate/temporal
   overlap, group risk, `audit()`, `analyze_dataset_impact()`, `score()`, the
   intelligence payload and `report.render`.
4. Adds two presentation-only keys to each finding (`index`, `advice`), applies
   the same `policy.apply_policy()` step as source scans to the serialized
   findings, and keeps the rest of the engine payload as the API returns it.
   Baseline statuses therefore match the CLI for the same dataset root and
   baseline; the audit engine itself is unchanged.

`POST /api/datasets/inspect` parses once with `run_inspect_job` to list columns,
preview rows and target candidates, so the UI can offer real choices instead of
free text. Nothing is cached between calls; a full audit re-reads the file once.

Supported input is whatever the installed parsers allow: CSV and JSON always,
`.xlsx` / `.parquet` / `.feather` only when their extra is installed **and** the
workspace limits enable them (they do not by default, matching the public API).
`GET /api/info` reports each format's `configured` and `installed` flags, and the
UI explains which ones are unavailable and why.

Exposed engine options are exactly those the audit job accepts: target,
threshold (`0.5`–`1.0`, rounded to 4 decimals), held-out dataset, time column,
group column and the opt-in `include_ai` flag (default off). The AI pass runs
inside `run_audit_job`, exactly as the public API runs it; a failure sets
`ai_unavailable: true` and leaves the deterministic audit untouched. There is
no snapshot control in this milestone; the baseline and suppression controls
described above are shared with source scans. Dataset AI blocks are computed
by the engine in that single call, before the workspace applies its policy
labels to the serialized findings; they never change a finding, a severity, a
status or the trust score.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /api/info` | version, root, scan limits, dataset limits/formats, SQL availability, config baseline/suppressions |
| `GET /api/history` | bounded in-memory session list with metadata and result/report availability |
| `GET /api/compare/scans?previous=&current=` | source/source comparison using stable fingerprints |
| `GET /api/compare/datasets?previous=&current=` | dataset/dataset comparison with configuration compatibility checks |
| `POST /api/scans` | `{"path": "...", "sql": false, "baseline": null, "include_ai": false}` → `{"scan_id"}` (202) |
| `GET /api/scans/{id}` | state, `done`, `total`, `current`, counts, errors, elapsed |
| `GET /api/scans/{id}/events` | `text/event-stream` snapshots → optional `failure` → `done` |
| `POST /api/scans/{id}/cancel` | request cancellation (409 if not running) |
| `GET /api/scans/{id}/results` | findings + statuses + policy + clusters + links + next actions (409 until finished) |
| `GET /api/scans/{id}/graph` | bounded evidence-graph page with filters, search, focus and finding navigation (409 until finished) |
| `GET /api/scans/{id}/report` | the static HTML report |
| `GET /api/source?path=...&line=N` | read-only source window around a line |
| `POST /api/datasets/inspect` | `{"path": "train.csv"}` → columns, preview, target candidates |
| `POST /api/datasets` | `{"path", "target", "threshold", "test_path", "time_col", "group_col", "baseline", "include_ai"}` → `{"dataset_id"}` (202) |
| `GET /api/datasets/{id}` | state, echoed request, error, elapsed |
| `GET /api/datasets/{id}/events` | `text/event-stream` snapshots → optional `failure` → `done` |
| `POST /api/datasets/{id}/cancel` | cancel before the engine starts (409 once it runs) |
| `GET /api/datasets/{id}/results` | the audit payload (409 until finished, or with the failure message) |
| `GET /api/datasets/{id}/graph` | bounded evidence-graph page from the audit's nested `intelligence` (409 until finished) |
| `GET /api/datasets/{id}/report` | the static HTML report for the audit |

Scan states are `running`, `done`, `cancelled` and `failed`. Audit states are
`queued`, `auditing`, `done`, `cancelled` and `failed`. Old sessions are kept in
memory (last 20 of each kind) so history, results, reports and comparisons stay
reachable; they are lost when the server restarts.

## Limits and safety

- One activity at a time: a source scan and a dataset audit share a single slot,
  so starting either while the other runs returns 409.
- Cancellation of a scan is checked between files. A single very large file
  finishes first: Python threads cannot be killed safely.
- A dataset audit can be cancelled while it is `queued`. Once `run_audit_job` is
  running there is no progress hook to interrupt, so cancelling returns 409 with
  that explanation instead of pretending to stop.
- All paths are resolved and must stay inside the workspace root; anything else
  returns 422.
- At most 5,000 files per scan, 1 MB source previews, and the public API's
  dataset limits for audits.
- A file that fails to read is reported per file; the rest of the scan continues.
  A failed audit keeps its engine error (`code` + message) and surfaces it on
  the status endpoint, the results endpoint and in the UI.

## Differences from the other surfaces

- `leaklens --scan` prints and exits: same findings, no progress or navigation.
- `leaklens FILE --target COL` runs the same audit with no progress, validity
  file, preview or browser view.
- The Streamlit UI audits a pasted snippet, an uploaded dataset, or the arena;
  it does not walk a project tree.
- The public live-scan API accepts untrusted uploads with rate limits; the
  workspace reads local files, has no upload path, and keeps the same guards.

## Tests

`tests/test_workspace.py` covers the scan lifecycle, progress fields, payload
shape and fingerprint stability, cluster coverage, path-escape rejection,
single-active-scan refusal (409), cancellation semantics, source windows and
report rendering, plus the dataset workflow: inspect, successful audit, engine
parity with `run_audit_job` and the CLI trust score, target and threshold
validation, held-out datasets and impact, format/optional-parser reporting,
traversal and absolute escapes, empty and malformed files, serialisation,
failure reporting, cross-kind locking, cancellation rules and report rendering.
Baseline coverage adds config and explicit selection, valid/missing/invalid/
non-UTF-8 baselines, traversal for both scan types, all four finding statuses,
suppression reason/rule identity, count consistency, resolved entries, source
and dataset parity with the CLI policy result, and suppression-only scans that
do not invent baseline statuses. Graph coverage adds shared UI controls, source
and dataset serialisation, evidence/link identity and neighbour symmetry,
finding/evidence navigation in both directions, filters, search, paging and
size caps, empty and isolated nodes, malformed/missing/dangling records,
session boundaries (404/409), and preserved suppression status/evidence.
Phase 11.5 adds SSE progress/completion/failure ordering, reconnect and trimmed
session 404s, polling fallback, in-memory history metadata/retention/restart
behaviour, compatible source and dataset comparisons,
added/unchanged/changed/no-longer-present/suppressed/inconclusive
classification, partial-scan safeguards, fingerprint/evidence consistency and
comparison count consistency/navigation.
Phase 12 M8 adds default-off AI (no AI call and no extra payload keys), AI-on
trust/finding/policy invariance for both scan types, failure isolation
(`ai_unavailable`), policy-active findings in the source-scan AI pass,
single-slot refusal with AI on, and HTML escaping of AI text in the workspace
report.
