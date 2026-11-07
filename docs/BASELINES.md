# Baselines and suppressions

LeakLens identifies a finding by a stable fingerprint, not by line number or message text. That makes it possible to remember which findings you have already reviewed, and to make CI fail only when something new appears.

There are three distinct concepts:

| Concept | What it means | Where it appears |
|---|---|---|
| **Baseline** | Findings accepted as known, recorded by fingerprint | `policy.baseline`, `status: new/unchanged/changed` |
| **Suppression** | One exact fingerprint accepted, with a reason and optional expiry | `status: "suppressed"`, `suppression: {reason, expires}` |
| **Resolved** | A baseline entry the current scan no longer produces | top-level `resolved` list |

A suppression is not a resolution: the finding is still produced and still reported, it just stops counting toward the trust score and the default exit gate. A resolution is the absence of a finding that the baseline expected.

## Baseline format

```json
{
  "schema": "leaklens-baseline",
  "version": 1,
  "fingerprint_schema": 1,
  "tool_version": "0.13.0",
  "findings": [
    {
      "fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "kind": "target_leakage",
      "severity": "high",
      "column": "chargeback_flag"
    }
  ]
}
```

- Entries are sorted by fingerprint, so repeated runs produce byte-identical files.
- Only identity metadata is stored: no rows, feature values, credentials, source text, timestamps or machine paths.
- `fingerprint_schema` is the schema LeakLens used to hash findings. A baseline with a different value is refused (exit 2) instead of being compared, because the identities are not equivalent. Regenerate it with `--write-baseline`.
- `version` describes this JSON format. Unknown versions, unknown keys, duplicate fingerprints and malformed JSON are refused with an actionable message.
- `tool_version` is informational only; comparing across LeakLens versions is allowed as long as the fingerprint schema matches.

The file is written atomically to a temporary file in the same directory, flushed, then `os.replace`d, so an interrupted run cannot leave a partial baseline. Writing refuses symlinks, directories and missing parent directories.

## Workflow

Create a baseline from a run whose findings you have reviewed:

```sh
leaklens train.csv --target label --write-baseline --baseline leaklens-baseline.json
```

Compare later runs:

```sh
leaklens train.csv --target label --baseline leaklens-baseline.json
```

Human output shows `Baseline: N new, N unchanged, N changed, N resolved` and a status suffix per finding. JSON output adds:

```json
"policy": {
  "baseline": {
    "path": "leaklens-baseline.json",
    "fail_on_new": false,
    "new": 1, "unchanged": 4, "changed": 0, "resolved": 2
  },
  "diagnostics": []
}
```

and each finding gets a `status`. Changed means the fingerprint is unchanged but the recorded severity differs; this is how a detector reclassification or a policy upgrade surfaces.

Update explicitly after review:

```sh
leaklens train.csv --target label --update-baseline --baseline leaklens-baseline.json
```

A second `--write-baseline` does **not** overwrite: it exits 2 and names `--update-baseline` as the explicit alternative.

Make new findings a CI failure (opt-in):

```sh
leaklens train.csv --target label --baseline leaklens-baseline.json --fail-on-new
```

or in `.leaklens.toml`:

```toml
[baseline]
path = "leaklens-baseline.json"
fail_on_new = true
```

## Suppressions

```toml
[[suppressions]]
fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
reason = "accepted: recorded before prediction, confirmed with the data owner"
expires = "2027-06-30"
```

Rules match one exact fingerprint. There are no wildcards, so a rule cannot silently cover a new finding. Every rule needs a reason, which is copied into JSON/SARIF/HTML next to the finding. The optional expiry is inclusive; from the next day the rule is reported as expired and is not applied.

Diagnostics are printed to stderr and included in `policy.diagnostics`:

```text
leaklens: warning: suppression rule <fingerprint> expired on 2026-01-01 and was not applied
leaklens: warning: suppression rule <fingerprint> matched no current finding
```

Malformed rules (missing reason, invalid fingerprint, bad date, duplicate, unknown key) abort the run with exit 2 before the scan. Expired and unmatched rules are warnings only.

Suppression never edits the underlying finding: the fingerprint, kind, severity, detail and evidence are byte-identical to an unsuppressed run. It only adds `status` and `suppression` and removes the finding from the active set used by `leakage.score()` and the default exit gate.

## API

`POST /scan` accepts optional JSON fields:

```json
{
  "source": "...",
  "baseline": { "schema": "leaklens-baseline", "version": 1, "fingerprint_schema": 1, "tool_version": "0.13.0", "findings": [] },
  "suppressions": [
    {"fingerprint": "...", "reason": "reviewed", "expires": "2027-01-01"}
  ]
}
```

`POST /audit` accepts the same values as optional JSON **form fields** (`baseline`, `suppressions`). The response adds `policy` and, when a baseline is used, `resolved`. Invalid policy returns HTTP 422. Without a policy the payload is unchanged; `/audit` additionally always includes the additive `missingness` report.

## Exit codes

| Situation | Exit |
|---|---|
| an **active** (not suppressed) high-severity finding | 1 |
| trust score below `--fail-under` | 1 |
| `--fail-on-new` and at least one new finding | 1 |
| suppressed high findings only (no other failure) | 0 |
| invalid baseline, suppression or configuration | 2 |
| usage error (bad arguments) | 2 |

The default exit behavior is unchanged from previous releases. Only `--fail-on-new` / `[baseline].fail_on_new` adds a failure mode, and it is opt-in.

## Security and privacy

- Baselines contain no dataset rows, feature values, credentials, source text or absolute paths. Columns and kinds are metadata you already see in reports.
- Suppressions are part of the project configuration, so review them in code review exactly like code. The required reason is there to make silent acceptance visible.
- A suppression cannot change severity, evidence or the fingerprint, so it cannot disguise a different finding as the accepted one.
- The configuration never executes code, loads plugins or accesses the network.
