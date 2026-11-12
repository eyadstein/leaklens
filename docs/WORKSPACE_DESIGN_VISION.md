# Workspace design vision

**Status:** conceptual. No frontend is implemented by this document. The shipped workspace is
described in [WORKSPACE.md](WORKSPACE.md); this file describes how that surface should grow.

**Constraint that shapes everything here:** the workspace is a thin local layer over the engine.
FastAPI request handlers call the same functions the CLI calls, the page is server-rendered HTML
plus small vanilla-JS modules, and there is no bundler, no JS framework and no client-side
database. Every view below must be reachable from that constraint, or it is not designed yet.

---

## 1. Design principles

1. **One payload, many views.** A view is a projection of the assembled audit payload (ADR-U1
   rejects a typed `AuditResult`), never a second analysis.
   If a view needs a fact the payload does not contain, the payload gains a field — not the view
   a computation.
2. **Evidence before conclusion.** Every claim on screen can be opened to the evidence that
   supports it, in one click, without leaving the page.
3. **State in the URL.** Filters, selected finding, expanded cluster and comparison baseline live
   in the query string, so a review can be linked into an issue or a chat and reproduced.
4. **Honest defaults.** Diagnostic findings are visibly separate from scored findings. AI content
   is labelled with its basis (`heuristic`, `uncalibrated ranking`, `unavailable`) next to the
   claim, not in a footnote. No bare percentage ever appears without its basis.
5. **No silent work.** Long operations show which engine is running, which were skipped and why
   (`unavailable: extras[vision]` is a first-class state, not an error toast).
6. **Read-only by default.** The workspace never writes to the scanned tree, never executes
   scanned code and never applies a remediation without an explicit copy destination.
7. **Keyboard first, colour never alone.** WCAG 2.2 AA as the target. Severity is carried by
   label and icon as well as colour; live updates never steal focus.

---

## 2. Information architecture

Six areas. Each maps to an existing or planned engine output, not to a marketing page.

```
┌ Audit ────────────── start a scan or dataset audit; watch progress; detector status
├ Findings ─────────── filter, sort, read, triage; scored vs diagnostic
├ Evidence ─────────── graph explorer, clusters, provenance, missing evidence
├ Lineage ──────────── pipeline DAG, feature trace, SQL/code joins (Phase 14+)
├ Compare ──────────── two runs: findings delta, score delta, graph diff, regression
└ Export ───────────── HTML report, SARIF, audit manifest, governance pack
```

Persistent chrome: audit target + revision, detector set and versions, policy/baseline in effect,
and a "show diagnostic findings" toggle that is visible (so nobody mistakes a filtered view for a
clean result).

---

## 3. Views

### 3.1 Audit launcher

- Target picker (project root, dataset file, pipeline definition, artifact path) with the
  accepted-formats list rendered from the same dispatcher the engine uses.
- Options grouped by engine: tabular / code / SQL / pipeline / text / vision / DL / artifacts.
  Each group shows whether its extra is installed; an uninstalled extra is disabled with the
  exact `pip install` line, not hidden.
- One primary action. The last configuration is remembered per target.
- A pre-flight line: files in scope, estimated size, detectors that will run, detectors that will
  not and why.

### 3.2 Run monitor

- Per-detector rows: `ok | skipped | unavailable | error`, duration, findings by severity, and a
  reason string for anything not `ok`.
- Live progress reuses the existing SSE stream and its `done`/`failure` contract; no new
  transport.
- Cancellation semantics are stated where the button is, not in a tooltip: between files for a
  source scan, only while queued for a dataset audit.

### 3.3 Findings explorer

- Server-side paging and sorting; filters: engine, kind, severity, status
  (`new|unchanged|changed|resolved`), policy label (active/suppressed with reason), scoring
  (`scored` / `diagnostic`), file, and free text over detail and evidence.
- Each row: severity chip, kind, one-line detail, location, fingerprint short id, and badges for
  baseline status and evidence count.
- Saved filter sets are URL fragments (a "leakage triage" preset is just a link).
- Bulk actions are limited to review-safe ones: copy fingerprints, export selection as SARIF or
  CSV, mark for a remediation proposal. CSV export prefixes cells beginning with `= + - @` so a
  finding detail cannot become a spreadsheet formula, and quotes per RFC 4180. No "hide" action
  without a reason field.

### 3.4 Finding detail

Four stacked panels, all server-rendered:

1. **Claim** — kind, severity, detail, remediation text, fingerprint, and why the fingerprint
   moved (when comparing).
2. **Evidence** — the evidence records behind the finding, rendered as labelled facts (column,
   rates, counts, source spans). Aggregate only; never raw rows.
3. **Context** — surrounding source (existing viewer), related findings sharing an anchor, and the
   cluster it belongs to.
4. **Next test** — deterministic suggestions ("audit this dataset with `--test`", "check the
   group column"), and, if AI is enabled, hypotheses linked to the evidence ids above with their
   basis and sufficiency labels.

### 3.5 Evidence graph

Extends the shipped graph view. Node = anchor (feature, split, file), edge = shared evidence or
lineage. Interactions: click to focus, expand one hop (server query, node budget enforced),
filter by detector, export the same `.mmd`/`.dot`/`.json` the CLI exports. A "missing evidence"
overlay marks anchors where no detector produced evidence — the absence is information.

### 3.6 Lineage view (Phase 14+, not shipped)

- Left: pipeline DAG (notebooks, SQL, DAG files, adapters) with stage badges for what was
  inspected, what was inferred and what is unresolved.
- Right: selected node's inputs/outputs, column mapping, and a "trace to model" path.
- Unresolved edges are drawn differently and listed, because an unresolved lineage claim must not
  look resolved. Confirmed `sql_target_as_feature` paths are highlighted; low-confidence
  `sql_unresolved_column` paths are not.
- No graph database. The DAG is computed per audit and kept in the session payload.

### 3.7 Modality lenses (Phases 15–17)

Same shell, different projection:

- **Text:** near-duplicate clusters (exact, shingle, optional embedding), side-by-side document
  snippets with the matching spans highlighted, and an eval/train contamination summary at the
  top.
- **Images:** a bounded thumbnail grid grouped by perceptual hash cluster, with split origin
  badges (train/val/test) and an "inspect pair" view showing both images and their hashes.
  Thumbnails are generated lazily, capped in count and size, and never cached to the scanned
  tree.
- **Video:** frame-sample strip per clip with the sampled timestamps listed, so a temporal split
  suspicion can be checked visually.
- **DL/artifacts:** training configuration table, checkpoint metadata, and split provenance, with
  the reproducibility checks (`seed`, `deterministic`, `eval split source`) rendered as
  pass/fail/unknown — unknown is a real state, not a failure.

### 3.8 Investigation panel (Phase 18)

A conversation-shaped panel over a **fixed** set of deterministic tools, not free-form execution.
The plan is deterministic and works with the AI extra absent; only the narration is optional:

- The investigator shows its plan (which questions it will answer, which tests it will run),
  then results, then what remains unknown.
- Every sentence that makes a claim links to evidence ids; a sentence without an anchor is not
  renderable.
- The panel distinguishes "the engine found", "the model suspects" and "you confirmed". A human
  confirmation is stored in the session only; it never mutates the findings or the score.
- When the LLM is unavailable, the panel falls back to the deterministic plan and says so.

### 3.9 Remediation review (Phase 19)

- A proposed patch per finding group: split correction, deduplication plan, preprocessing order
  fix, config change.
- Diff-first review: the patch is shown against a *copy* path the user names. Nothing is applied
  in place; the workspace never writes to the audited tree.
- Each proposal carries its risk note, the findings it would resolve, and a "verify" action that
  re-runs the relevant detectors on the copy and shows the delta in Compare.

### 3.10 Compare

- Inputs: two retained sessions (existing) or a session and a baseline.
- Panels: score delta with the exact formula output, findings delta by status
  (`new|unchanged|changed|resolved`), graph diff (added/removed/changed clusters), detector
  coverage delta (a detector that stopped running is a regression, not a clean result).
- Guard against the common misreading: comparing results from different detector sets shows a
  banner that the comparison is not like-for-like.

### 3.11 Export and governance

- HTML (existing), SARIF (existing), evidence graph exports (existing), JSON payload download.
- Phase 18: an **audit manifest** — input digests, tool version, detector versions, policy and
  baseline digests, evidence digests, the score and the exit-gate decision — downloadable and
  verifiable offline. The manifest is evidence, not a certificate.
- Phase 18: governance pages generated from the manifest (dataset card stub, model card stub)
  with every field marked `asserted`, `measured` or `unknown`.

---

## 4. Key workflows

**Triage a fresh scan (target: under 10 minutes).**
Land on Findings with severity descending and diagnostic collapsed. Read the top cluster, open
evidence, accept or suppress with a reason. Leave with a link to the saved filter and a SARIF
export for the PR.

**Investigate one suspicious result.**
Finding → evidence → graph focus → related findings → "next test" → re-run that test → Compare.
The AI panel, if enabled, proposes the tests; the human chooses which to run.

**Gate a pull request.**
CLI in CI produces the exit code and SARIF; the workspace is not in the gate path. When a gate
fails, the developer opens the compare view against `main`'s baseline, not a fresh scan.

**Review near-duplicate images.**
CV audit → image clusters sorted by cluster size → inspect a pair → confirm or dismiss. Each
dismissal requires a reason and produces a suppression candidate with the pair's fingerprint.

**Audit a pipeline before a launch.**
Lineage view → unresolved edges → detector status → "what we could not check" list on the export
page. The list is a required part of the report, not an appendix.

---

## 5. What the interface must never do

- Show a numeric confidence without its basis.
- Present a diagnostic finding as scored, or vice versa, without a visible label.
- Hide a skipped or unavailable detector.
- Present a hypothesis as a finding, or a human note as engine output.
- Compare two runs with different detector sets or versions as if they were like-for-like.
- Imply a compliance certification.
- Write to the audited tree, or execute audited code.

---

## 6. Delivery plan

| Phase | Workspace work | Why then |
|---|---|---|
| 13 | Detector status panel, `scored`/`diagnostic` split in Findings, engine filter | The payload gains detector results; the UI must not hide them |
| 14 | Lineage view over the pipeline DAG | Needs the pipeline graph payload |
| 15–17 | Text, image and DL lenses on the same shell | Each ships with its engine, never before |
| 18 | Investigation panel, audit manifest, governance pages | Needs evidence-anchored investigation output |
| 19 | Remediation review and verify-on-copy | Needs the patch contract |

No new frontend framework, no build step, no client state store at any phase. If a view cannot be
done in server-rendered HTML plus a small module, it waits or it is cut.

---

## 7. Tool note

OpenDesign was checked for this mission and is not installed in this environment
(`opendesign` is not on `PATH` and not importable). No design artifacts were generated by it;
this document is the design output, produced from the shipped workspace code and its existing
constraints.
