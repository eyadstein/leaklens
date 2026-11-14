# Investigation plans and monitoring intake

Phase 18 adds a deterministic investigation plan (M1) and a read-only intake
for caller-supplied monitoring reports (M5). Both are additive: they never
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
| `suggested_test` | `SuggestedTest` | A follow-up check; M2 maps it to a registered detector kind. |
| `missing_evidence` | `MissingEvidence` | The finding fired without structured detector evidence. |
| `narration_slot` | `NarrationSlot` | Reserved for M4 narration; it cannot add evidence. |

Each evidence cluster gets `inspect_evidence`, `suggested_test` and
`narration_slot` steps, plus `missing_evidence` when no detector-role evidence
exists for its findings.

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
still works (ADR-025). `ai.reasoning.build_plan` remains the narration grouping
that M4 will re-point at the investigation plan; M1 adds only one investigation
builder.

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
