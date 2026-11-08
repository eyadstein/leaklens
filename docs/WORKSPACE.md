# Interactive scan workspace

`leaklens workspace` serves a local, read-only page for scanning a project or
auditing a dataset, and exploring the results: start a scan, watch it progress,
filter findings, read the evidence behind each one, open the surrounding source,
and export the same HTML report the CLI writes.

It is a thin layer over the existing engine. Detection, fingerprints, trust
score, evidence clusters, fix advice and the HTML report are the same code paths
the CLI and the public API use; the workspace adds progress, navigation and a
browser view.

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
   recommendations and `leakage.score()`.
4. Renders the normal static HTML report with `report.render`.

Scans never write to disk and never execute the scanned code.

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
4. Adds two presentation-only keys to each finding (`index`, `advice`) and keeps
   the rest of the engine payload byte-for-byte as the API returns it.

`POST /api/datasets/inspect` parses once with `run_inspect_job` to list columns,
preview rows and target candidates, so the UI can offer real choices instead of
free text. Nothing is cached between calls; a full audit re-reads the file once.

Supported input is whatever the installed parsers allow: CSV and JSON always,
`.xlsx` / `.parquet` / `.feather` only when their extra is installed **and** the
workspace limits enable them (they do not by default, matching the public API).
`GET /api/info` reports each format's `configured` and `installed` flags, and the
UI explains which ones are unavailable and why.

Exposed engine options are exactly those the audit job accepts: target,
threshold (`0.5`–`1.0`, rounded to 4 decimals), held-out dataset, time column
and group column. There is no AI, baseline or snapshot control in this
milestone.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /api/info` | version, root, scan limits, dataset limits/formats, SQL availability |
| `POST /api/scans` | `{"path": "...", "sql": false}` → `{"scan_id"}` (202) |
| `GET /api/scans/{id}` | state, `done`, `total`, `current`, counts, errors, elapsed |
| `POST /api/scans/{id}/cancel` | request cancellation (409 if not running) |
| `GET /api/scans/{id}/results` | findings + clusters + links + next actions (409 until finished) |
| `GET /api/scans/{id}/report` | the static HTML report |
| `GET /api/source?path=...&line=N` | read-only source window around a line |
| `POST /api/datasets/inspect` | `{"path": "train.csv"}` → columns, preview, target candidates |
| `POST /api/datasets` | `{"path", "target", "threshold", "test_path", "time_col", "group_col"}` → `{"dataset_id"}` (202) |
| `GET /api/datasets/{id}` | state, echoed request, error, elapsed |
| `POST /api/datasets/{id}/cancel` | cancel before the engine starts (409 once it runs) |
| `GET /api/datasets/{id}/results` | the audit payload (409 until finished, or with the failure message) |
| `GET /api/datasets/{id}/report` | the static HTML report for the audit |

Scan states are `running`, `done`, `cancelled` and `failed`. Audit states are
`queued`, `auditing`, `done`, `cancelled` and `failed`. Old sessions are kept in
memory (last 20 of each kind) so their report and results stay reachable.

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
