# Investigation plans and monitoring intake

Phase 18 adds a deterministic investigation plan (M1), detector-mapped suggested
tests (M2), a central missing-evidence report (M3) and a read-only intake for
caller-supplied monitoring reports (M5). All four are additive: they never
create, raise or score a finding.

## Investigation plan (M1)

`--intelligence` adds an `investigation` block to the JSON payload. It is built
by `leaklens/intelligence/investigation.py` from the evidence graph, so the
same findings always produce the same plan.

```json
{
  "engine": "leaklens.intelligence.investigation",
  "narration": "unavailable",
  "finding_count": 2,
  "steps": [
    {
      "step_type": "inspect_evidence",
      "ordinal": 1,
      "evidence_ids": ["scanner-000000"],
      "finding_ids": ["det-0"],
      "rationale": "Inspect the evidence behind the target_leakage finding(s) on 'dataset:amount'."
    }
  ]
}
```

### Step types

| `step_type` | Class | Purpose |
|---|---|---|
| `inspect_evidence` | `InspectEvidence` | Look at the evidence a cluster already carries. |
| `suggested_test` | `SuggestedTest` | A registered detector to re-run, with the question it answers. |
| `missing_evidence` | `MissingEvidence` | The finding fired without structured detector evidence. |
| `narration_slot` | `NarrationSlot` | Optional M4 rephrasing; it cannot add evidence. |

Each evidence cluster gets `inspect_evidence`, `suggested_test` and
`narration_slot` steps, plus `missing_evidence` when no detector-role evidence
exists for its findings. If none of the cluster's finding kinds is registered,
the `suggested_test` step is omitted and listed in `omitted_tests` instead.

### Id contract

- `evidence_ids` always name records in the payload's evidence graph
  (`scanner-<index>` and `scanner-<index>-e<ordinal>`).
- `finding_ids` always use `det-<index>`, where `<index>` is a valid position in
  the finding list.
- `ordinal` is contiguous from `1`; the order follows the deterministic cluster
  order.
- Before serialization `_ground()` rejects any step that cites an unknown
  evidence id or an out-of-range finding id, so an ungrounded step cannot be
  emitted.

### Limits

`MAX_STEPS` is 200 and `MAX_RATIONALE` is 300 characters. Clusters are capped
in build order, never sorted by an unstable key.

### AI-absent behaviour

`leaklens/intelligence` never imports `leaklens.ai`. The plan, including
evidence and remediation context, is complete without the AI layer;
`narration` is reported as `"unavailable"`. `leaklens.ai.schemas` re-exports
`ReasoningPlan`, `RootCauseGroup`, `Hypothesis`, `Remediation` and
`EvidenceSpan` unchanged, so `from leaklens.ai.schemas import ReasoningPlan`
still works (ADR-025).

### Optional narration (M4)

`--intelligence --ai-llm` asks the configured provider to rephrase each cluster
into its `narration_slot`. Narration is additive:

- The only entry point is `enrich_plan`, reached through
  `ai.reasoning.narrate_investigation`. The AI layer sees the plan through a
  structural `NarrationTarget` protocol, so `ai` still never imports
  `intelligence.investigation`.
- Each narration group id is the slot's own first evidence id, so the response
  allowlist is exactly "evidence ids already in the plan". A note naming
  anything else, malformed JSON, an empty note list or an unsupported safety
  claim discards the whole response.
- `narration` becomes `"llm"` only when at least one slot was filled. Any
  validation failure yields `"deterministic_fallback"`; an unreachable provider
  yields `"ai_unavailable"`. Steps, evidence, findings, `missing_evidence`,
  the trust score and the exit code never change.
- A slot's `text` is omitted while empty, so a plan with narration removed is
  byte-identical to a plan built with narration disabled
  (`InvestigationPlan.without_narration()`).
- Every `prompt_injection` corpus case is replayed as a provider response and
  cannot change the plan (`tests/test_narration.py`).

### Suggested tests (M2)

A `suggested_test` step is a reference to a registered detector, never free
text. It adds two fields:

```json
{
  "step_type": "suggested_test",
  "ordinal": 2,
  "detector": "target_leakage",
  "question": "Does re-running the target_leakage detector still reproduce the high finding on dataset:amount?",
  "evidence_ids": ["scanner-000000"],
  "finding_ids": ["det-0"],
  "rationale": "Re-run the target_leakage check after remediation and confirm the cited evidence no longer reproduces."
}
```

- `detector` is chosen from the cluster's own finding kinds and must exist in
  `manifest.DETECTORS`; no check is invented. Clusters are already ordered by
  priority and confidence by the evidence graph, so the step order is stable.
- `question` names the detector, the finding severity and the anchor kind
  (`feature`, `file`, `split`, `dataset` or `image`).
- `evidence_ids` are the records the test would discriminate.
- An unknown kind cannot produce a step at runtime. It is omitted and listed in
  `omitted_tests` with reason `unregistered_detector` and the affected finding
  ids. Constructing `SuggestedTest` with an unregistered kind raises
  `ValueError`, so a development mistake fails loudly.

## Missing-evidence report (M3)

When something could not be checked, the payload gains one `missing_evidence`
collection. The key is absent when nothing was skipped, so a clean payload is
unchanged. It is sourced from the engine `skipped` maps already in the payload
(`missingness`, `drift`, `pipeline`, `monitoring`, `dl`), per-split document
counts (`text`), split `unresolved` counters and `limitations`, a caught
`EngineUnavailable` and the plan's `omitted_tests`.

```json
{
  "entries": [
    {
      "area": "monitoring",
      "reason": "malformed",
      "effect": "The monitoring report was not evaluated; its drift or skew metrics are absent from the plan.",
      "source": "monitoring.skipped",
      "count": 1
    }
  ],
  "counts": {"malformed": 1},
  "total": 1
}
```

- Every reason keeps its own entry and count; nothing is collapsed into "some
  checks were skipped".
- `area` names the engine or subject, `reason` is the engine's stable code,
  `effect` says what the gap means for the review and `source` says where the
  skip was recorded. `finding_ids` appears when the gap belongs to findings
  (for example an omitted test).
- `counts` aggregates by reason for a quick view; `total` sums the counts.
- `MAX_ENTRIES` is 200. An overflow keeps the first 199 entries and adds one
  `entry_cap` record naming how many were dropped.
- The collection never creates, raises or scores a finding and never changes
  the exit gate. `leaklens/missing_evidence.py` builds it; it imports no
  optional extra and never imports `leaklens.ai`.

## Monitoring report intake (M5)

`--monitoring-report PATH` (repeatable) reads a caller-supplied drift or skew
report as data:

```json
{
  "kind": "drift",
  "metrics": [
    {"name": "amount", "value": 0.42, "threshold": 0.2},
    {"name": "region", "value": 0.05, "breached": false}
  ]
}
```

`kind` is `drift` or `skew`. Each metric needs a `name` and a finite numeric
`value`; `threshold` or a boolean `breached` says whether it is a problem. A
metric with neither is skipped as `no_breach_signal` rather than guessed.

Each breached metric becomes a `low` diagnostic finding
(`monitoring_drift_report` / `monitoring_skew_report`) whose evidence names the
report's relative path, the metric and the observed value. Findings are
`diagnostic: true`, so they never change `leakage.score()` or the default exit
gate.

The path is resolved with `paths.resolve_within` against the working directory.
Files over 1 MiB, more than 500 metrics, malformed JSON, unknown `kind`, and
out-of-root paths become counted skips (`oversized`, `too_many_metrics`,
`malformed`, `unsupported_kind`, `outside_root`) instead of tracebacks. The
intake never contacts a network, database or monitoring service.

```sh
leaklens train.csv --target label --monitoring-report reports/drift.json
leaklens --scan src --monitoring-report reports/drift.json --json
```
