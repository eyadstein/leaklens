# Deterministic intelligence

`--intelligence` turns the existing scanner findings into a deterministic
evidence graph: confidence, root-cause explanations, remediation steps and -
since Wave 4 - **clusters, links and measured impact**. It is opt-in, offline
and CPU-only, and it is available in the base install with no new dependency.

```sh
leaklens train.csv --target label --intelligence
leaklens train.csv --target label --test test.csv --group patient_id --intelligence
leaklens --scan src/ --intelligence --json
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
top cluster, and the first remediation action and verification step. `--json`
exposes the full graph under `intelligence`, and `--html` renders an *Evidence
clusters* table.

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

- `supports` - detector-level detail explains a primary finding;
- `corroborates` - two findings of different kinds share an anchor;
- `same_anchor` - two findings of the same kind share an anchor (repeated
  pattern in one file).

Links are emitted per cluster, capped at 400 per report, and only reference
ids that exist in the same report. Ordering is deterministic: clusters sort by
priority, then confidence, then anchor; ids are slugs with a numeric suffix on
collision.

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
  identical JSON.
- **Bounded.** Links are capped at 400; evidence length comes from the existing
  per-finding cap; cluster ids are short slugs.
- **Offline.** No model, no service, no credential.

## Limitations

- Anchors are one-level: a cluster groups by feature, split or file but does
  not traverse joins, notebooks or cross-file data flow.
- Cluster confidence rewards corroboration, not ground truth. Two correlated
  detectors on the same column raise it even if both share the same blind spot.
- Same-kind findings in different files never link; only their own anchors do.
- Measured impact is descriptive. A 100% duplicate rate is a fact about the two
  frames supplied, not proof that a model leaked.
- Unknown detector kinds get a coarse `finding:<kind>` anchor until their
  surface is added to the manifest.

## Relation to the AI layer

Intelligence is the deterministic layer that runs with `--intelligence`. The
experimental AI layer (`--ai`) is separate, off by default and documented in
[AI_INTELLIGENCE.md](AI_INTELLIGENCE.md); it may *cite* intelligence output but
never changes the graph, a severity or the trust score.
