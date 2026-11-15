# Governance pack

Phase 18 M7 adds an opt-in governance pack: one provenance-labelled payload that
renders as JSON and Markdown. It records what an audit observed and what it was
never told. It is **not** a compliance claim, a certification or legal advice.

```sh
leaklens train.csv --target label --governance --json
leaklens train.csv --target label --governance-out pack.md
leaklens --scan src --governance-out pack.json
```

- `--governance` adds one `governance` block to the JSON payload. The key is
  absent unless the flag (or `--governance-out`) is used, so default output is
  unchanged.
- `--governance-out PATH` writes the pack to `.json`, `.md` or `.markdown`
  inside the working directory and implies `--governance`. Any other suffix, or
  a path outside the working directory, exits `2`.
- The pack never creates, raises, rescales or suppresses a finding, and it never
  changes the trust score or the exit gate.

## Provenance vocabulary

Every field carries exactly one label and a source.

| Label | Meaning |
|---|---|
| `asserted` | Supplied by the caller on the command line; not verified by this audit. |
| `measured` | Computed by this audit run from the payload it produced. |
| `unknown` | Not supplied and not verified. The value is `null`; the source says why. |

An `unknown` field never carries a value and a non-`unknown` field always does,
so a reader cannot mistake an empty label for a measurement. Each unknown is
also listed in the payload's `unknowns` array.

## Fields

| Field | Label | Source | Meaning |
|---|---|---|---|
| `dataset.train` | `asserted` | `cli.dataset` | Training dataset path given on the command line. |
| `dataset.test` | `asserted` | `cli.test` | Evaluation dataset path; `unknown` for a single-dataset or code-only audit. |
| `dataset.target` | `asserted` | `cli.target` | Target column; `unknown` for a code-only audit. |
| `audit.trust_score` | `measured` | `leakage.score()` | The single trust-score value for this run. |
| `audit.findings.high` | `measured` | `assemble.severity_counts` | High findings in the payload, diagnostics included. |
| `audit.findings.medium` | `measured` | `assemble.severity_counts` | Medium findings in the payload. |
| `audit.findings.low` | `measured` | `assemble.severity_counts` | Low findings in the payload, diagnostics included. |
| `audit.findings.total` | `measured` | `assemble.severity_counts` | Every serialized finding. |
| `audit.missing_evidence` | `measured` | `missing_evidence.total` | Counted skips; `unknown` when the collection is absent. |
| `investigation.steps` | `measured` | `investigation.steps` | Plan steps; `unknown` without `--intelligence`. |
| `investigation.narration` | `measured` | `investigation.narration` | Narration state (`unavailable`, `llm`, `deterministic_fallback`, `ai_unavailable`); `unknown` without `--intelligence`. |
| `model.name` | `unknown` | not supplied | LeakLens audits data and code, not a trained model. |
| `model.version` | `unknown` | not supplied | Same. |
| `data.license` | `unknown` | not supplied | LeakLens does not inspect licensing. |
| `data.collection` | `unknown` | not supplied | LeakLens does not inspect collection or consent. |
| `evaluation.protocol` | `unknown` | not supplied | LeakLens does not inspect the evaluation protocol. |

## JSON payload

```json
{
  "engine": "leaklens.governance",
  "card_version": 1,
  "fields": [
    {
      "field": "dataset.train",
      "label": "asserted",
      "value": "train.csv",
      "source": "cli.dataset"
    },
    {
      "field": "audit.trust_score",
      "label": "measured",
      "value": 95,
      "source": "leakage.score()"
    },
    {
      "field": "model.name",
      "label": "unknown",
      "value": null,
      "source": "not supplied: LeakLens audits data and code, not a model"
    }
  ],
  "unknowns": ["model.name"],
  "non_claims": [
    "This pack records what one audit observed; it is not a certification, an attestation, a guarantee or legal advice."
  ]
}
```

## Markdown export

`export.to_markdown()` renders the **same** payload: a field table, the
`unknowns` list and the `non_claims` list. Both renderers take the payload, so
the formats cannot drift. String values are JSON-quoted and pipes are escaped,
so a path cannot split the table.

## Non-claims

The pack always states them, and they are part of the payload:

- It records what one audit observed; it is not a certification, an
  attestation, a guarantee or legal advice.
- A clean or partial pack is not proof that a dataset or model is safe, lawful
  or compliant.
- An unknown field was not supplied to LeakLens and was not verified.

## Guarantees

- Standard library only; no network access, no new dependency.
- Deterministic: the same inputs produce byte-identical JSON and Markdown.
- `leaklens/governance` never imports `leaklens.ai`; it reads the assembled
  payload and the `leaklens.paths` containment helper only.
- Writes resolve through `paths.resolve_within` and replace the target
  atomically, so a failed write leaves no half-written pack.
- Limits: at most 100 fields and 300 characters per source string; asserted
  values are reported as supplied, never truncated.
