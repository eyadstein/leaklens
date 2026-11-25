# Report schema

Every audit that writes JSON (`--json`, and the `text`, `dl` and `images`
surfaces) writes one document. That document has a published JSON Schema, so a
consumer can validate a report before trusting it, and a track over months can
tell which contract a stored report was written against.

- Schema: [`leaklens/schemas/report-1.schema.json`](../leaklens/schemas/report-1.schema.json)
- Dialect: JSON Schema 2020-12
- Version: `report-1`

```sh
leaklens schema            # prints the schema document
leaklens data.csv --target y --json | jq -r '."$schema"'   # the URL a report claims
```

The schema is packaged with the wheel and `leaklens schema` reads the packaged
copy, so the document you validate against is the one your install ships.

## What the schema guarantees

It describes the **shape** of a report, not whether its findings are correct:

- `trust_score` — integer, 0–100, computed from the findings that survived
  policy (baseline and suppressions).
- `findings` — every finding the detectors produced, suppressed ones included,
  each with `severity` (`high`/`medium`/`low`), `kind`, `column`, `detail`,
  `evidence` and a versioned SHA-256 `fingerprint`. `diagnostic: true`, `status`
  and `suppression` appear only when they apply, and unknown keys are refused
  inside a finding so a typo in a detector is caught rather than ignored.
  `fix` is the optional remediation text: the live-scan API (`/audit`, `/scan`)
  always carries it (empty when the kind has no advice), the CLI does not.
- Optional sections: `counts` (the severity tally carried by the text, `dl` and
  image reports; the data audit leaves it to the consumer), `policy` (what the
  baseline and suppressions did), `resolved` (baseline entries no longer
  produced), `field`/`documents`/`skipped` (text reports).
- `policy.suppressions` is `{suppressed, rules, diagnostics}`, and a finding's
  `status: "suppressed"` carries the matched rule as `fingerprint` + `reason`,
  plus `expires` only when the rule has an expiry date. Both objects refuse
  unknown keys, so a consumer can rely on that shape.

Top-level unknown keys are **allowed**: a command that adds its own section
(`audit`, `missingness`, `drift`, `clustering`, `dl`, `pipeline`, `ai`) is
additive and stays valid.

The finding contract itself — which detector emits which `kind`, and what its
evidence means — is in [DETECTORS.md](DETECTORS.md). The schema only says how
that contract is spelled in JSON.

## Version policy

`report-1` is frozen. A change that removes a field, retypes one, or makes an
optional field required needs a new `report-2` schema; adding a section or an
optional field does not. A new version ships as a new file beside this one, and
`$schema` is how a consumer picks between them — never guess from the contents.

## Validating a report

The schema is not a runtime dependency of LeakLens. To validate, use any
2020-12 implementation:

```python
import json
import jsonschema
import urllib.request

schema = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/eyadstein/leaklens/main/leaklens/schemas/report-1.schema.json"))
report = json.load(open("report.json"))
jsonschema.Draft202012Validator(schema).validate(report)
```

Validation is a shape check: a report that validates may still contain a
finding you disagree with, and a report that fails validation should not be
read at all.

## Limits

- The schema describes reports only. The audit manifest (`--manifest`), the
  SARIF output, the baseline file, the badge JSON, `doctor --json`, the
  workspace envelopes and the SBOM each have their own documented format
  ([AUDIT_MANIFEST.md](AUDIT_MANIFEST.md), [BASELINES.md](BASELINES.md),
  [WORKSPACE.md](WORKSPACE.md), and the SARIF mapping in
  [DETECTORS.md](DETECTORS.md)). SARIF names its own `$schema`; the others
  carry `schema`, `schemaVersion` or nothing, and are validated against their
  own contract, not this one.
- Evidence fields are bounded strings. A report validated against the schema
  says nothing about whether a string was bounded correctly — that is the
  detector's contract and its tests.
- Older reports written before `report-1` was published carry no `$schema` key
  and are otherwise unchanged; they still validate, because the key is optional.
