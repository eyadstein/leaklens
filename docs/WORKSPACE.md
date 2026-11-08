# Interactive scan workspace

`leaklens workspace` serves a local, read-only page for scanning a project and
exploring the results: start a scan, watch it progress file by file, filter
findings, read the evidence behind each one, and open the surrounding source.

It is a thin layer over the existing engine. Detection, fingerprints, trust
score, evidence clusters, fix advice and the HTML report are the same code paths
the CLI uses; the workspace adds progress, navigation and a browser view.

```sh
pip install -e ".[app]"          # fastapi + uvicorn
leaklens workspace               # root = current folder, port 8765
leaklens workspace notebooks/ --port 9000 --no-open
```

The server binds `127.0.0.1` by default. `--host` is available for a deliberate
LAN setup, but note that anyone who can reach the port can read files under the
workspace root.

## What one scan does

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

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | the single-page UI |
| `GET /api/info` | version, workspace root, limits, SQL availability |
| `POST /api/scans` | `{"path": "...", "sql": false}` → `{"scan_id"}` (202) |
| `GET /api/scans/{id}` | state, `done`, `total`, `current`, counts, errors, elapsed |
| `POST /api/scans/{id}/cancel` | request cancellation (409 if not running) |
| `GET /api/scans/{id}/results` | findings + clusters + links + next actions (409 until finished) |
| `GET /api/scans/{id}/report` | the static HTML report |
| `GET /api/source?path=...&line=N` | read-only source window around a line |

States are `running`, `done`, `cancelled` and `failed`. Old sessions are kept in
memory (last 20) so their report and results stay reachable.

## Limits and safety

- One scan at a time; a second start returns 409 while one is running.
- Cancellation is checked between files. A single very large file finishes
  first: Python threads cannot be killed safely.
- All paths are resolved and must stay inside the workspace root; anything else
  returns 422.
- At most 5,000 files per scan, and source previews are capped at 1 MB.
- A file that fails to read is reported per file; the rest of the scan continues.

## Differences from the other surfaces

- `leaklens --scan` prints and exits: same findings, no progress or navigation.
- The Streamlit UI audits a pasted snippet, an uploaded dataset, or the arena;
  it does not walk a project tree.
- The public live-scan API accepts untrusted uploads with rate limits; the
  workspace is a local developer tool and has no upload path.

## Tests

`tests/test_workspace.py` covers the scan lifecycle, progress fields, payload
shape and fingerprint stability, cluster coverage, path-escape rejection,
single-active-scan refusal (409), cancellation semantics, source windows and
report rendering.
