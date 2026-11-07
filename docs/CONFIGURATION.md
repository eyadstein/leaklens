# Configuration (`.leaklens.toml`)

LeakLens reads `./.leaklens.toml` from the current directory when it exists, or the file passed with `--config PATH`. The file is optional; without it, behavior is exactly the previous CLI default.

Precedence, highest last:

1. built-in defaults,
2. the configuration file,
3. explicit CLI flags.

A CLI flag that is not passed is `None`, so the file value applies. Passing the flag always wins, including for booleans.

Invalid configuration fails closed: LeakLens prints `leaklens: configuration error: ...` with the file (and the key where possible) and exits **2** before scanning anything. Unknown keys are rejected so a typo cannot silently disable a check.

## Schema

Top-level keys mirror CLI flags (dashes become underscores):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `target` | string | — | label column (`--target`) |
| `test` | string path | — | held-out split (`--test`), resolved relative to the config file |
| `time` | string | — | timestamp column (`--time`) |
| `group` | string | — | entity column (`--group`) |
| `threshold` | number 0–1 | `0.95` | single-feature leakage threshold (`--threshold`) |
| `scan` | string or array | — | files/folders to scan (`--scan`) |
| `explain` | boolean | `false` | print a fix under each finding (`--explain`) |
| `fail_under` | number 0–100 | — | exit 1 below this trust score (`--fail-under`) |
| `sarif` | relative path | — | SARIF output (`--sarif`) |
| `html` | relative path | — | HTML output (`--html`) |
| `badge` | relative path | — | shields.io endpoint output (`--badge`) |
| `ai` | boolean | `false` | experimental AI layer (`--ai`) |
| `ai_llm` | boolean | `false` | rephrase the AI plan with the LLM (`--ai-llm`) |

```toml
target = "label"
test = "data/test.csv"
scan = ["src", "notebooks"]
threshold = 0.95
fail_under = 80
html = "reports/leaklens.html"
```

Input paths (`test`, `scan`) are resolved relative to the configuration file, so running from another working directory does not change them. Output paths (`sarif`, `html`, `badge`, `[baseline].path`) must be relative and must stay inside the configuration directory; a missing parent directory or an escaping path is an error. This keeps a scan from overwriting arbitrary files.

## `[baseline]`

```toml
[baseline]
path = "leaklens-baseline.json"
fail_on_new = false
```

- `path` is the baseline used for comparison (and written by `--write-baseline` / `--update-baseline` when no path is passed on the CLI).
- `fail_on_new` makes new findings a CI failure. It requires `path` in the same table. It is off by default.

The CLI path wins over the file. `--write-baseline` and `--update-baseline` are mutually exclusive.

## `[[suppressions]]`

```toml
[[suppressions]]
fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
reason = "accepted: the field is recorded before prediction"
expires = "2027-01-01"   # optional
```

- `fingerprint` must be exactly 64 lowercase hexadecimal characters; copy it from a JSON finding (`findings[].fingerprint`).
- `reason` is required, non-empty, at most 500 characters.
- `expires` is an optional `YYYY-MM-DD` date. A rule is active through that day and expired from the next day.
- The same fingerprint may appear only once. Unknown keys, invalid fingerprints, missing reasons and malformed dates are configuration errors (exit 2).
- There are no wildcards or prefix matches. A rule that matches nothing is reported as a warning and changes nothing.
- A disabled or malformed file never silently suppresses findings.

## `[missingness]`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `true` | run the missingness audit |
| `min_rows` | integer >= 10 | `200` | minimum usable rows for the target check |
| `min_group_rows` | integer >= 2 | `30` | minimum rows per target class / split / group |
| `min_missing` | integer >= 1 | `30` | minimum missing values before a column is examined |
| `min_rate_delta` | number 0.01–1 | `0.15` | minimum missing-rate difference (target/split) |
| `min_group_delta` | number 0.01–1 | `0.25` | minimum missing-rate difference for a group |
| `min_correlation` | number 0.01–1 | `0.30` | minimum \|point-biserial r\| for a numeric target |

All thresholds are deterministic. Lowering them increases sensitivity and false positives; the documented defaults favor silence over noise. See [MISSINGNESS_BENCHMARK.md](MISSINGNESS_BENCHMARK.md) for measured behavior on labeled synthetic cases.

## `[sql]`

Opt-in SQL lineage analysis. It needs the optional `sql` extra; without SQLGlot, enabling it raises an actionable error instead of silently skipping SQL.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `false` | also scan `.sql` files in folders (`--sql`); an explicit `.sql` file is always analyzed |
| `dialect` | string | `"ansi"` | one of `ansi`, `postgres`, `bigquery`, `snowflake` (`--sql-dialect`) |
| `split_column` | string | — | flag queries that read a table with this column but never filter on it |
| `schema` | table of tables | — | `[sql.schema.<table>]` maps column names to type names for lineage resolution |

```toml
[sql]
enabled = true
dialect = "postgres"
split_column = "split"

[sql.schema.orders]
id = "INT"
amount = "DOUBLE"
label = "INT"
split = "TEXT"
```

There is no silent dialect fallback. Unknown keys, unsupported dialects, an empty `schema` and non-string identifiers are configuration errors (exit 2). The schema map is bounded (200 tables, 1,000 columns per table) and never contacted over the network. See [SQL_LINEAGE.md](SQL_LINEAGE.md).

## `[drift]`

Opt-in train/test distribution-drift diagnostics. Findings are low severity and marked `diagnostic`, so they never change `leakage.score()` or the default exit gate.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | boolean | `false` | run the audit (`--drift`) |
| `min_rows` | integer >= 20 | `200` | minimum rows in each split |
| `min_effect` | number 0.01–1 | `0.20` | minimum KS/TVD/missingness effect to report |
| `max_rows` | integer >= 1000 | `100000` | deterministic subsample cap per column |
| `max_columns` | integer >= 1 | `50` | columns checked per audit |
| `max_categories` | integer >= 2 | `20` | top-k categories before the "other" bucket |

```toml
[drift]
enabled = true
min_rows = 200
min_effect = 0.20
```

Lowering `min_effect` or `min_rows` increases sensitivity and false positives. The target column is never compared, constant and fully missing columns are skipped, and no data or split is modified. See [DRIFT.md](DRIFT.md).

## Starter file

```sh
leaklens init            # writes ./.leaklens.toml
leaklens init mydir      # writes mydir/.leaklens.toml
leaklens init mydir --force
```

`init` refuses to overwrite an existing file unless `--force` is passed. The starter is entirely commented out, so it parses as an empty configuration and changes nothing until you enable a key. `leaklens doctor` validates `.leaklens.toml` against this schema; `leaklens doctor --config PATH` validates a specific file and exits 1 if it is invalid.

## Security and portability

- Parsing uses `tomllib` (Python 3.11+) or the already-declared `tomli` dependency (3.10). No new dependency.
- The file is data only: no plugins, imports, command substitution or network access.
- Paths are handled with `pathlib` and work on Windows, Linux and macOS; output confinement uses resolved paths.
- Diagnostics never echo secret values, dataset rows or source content.
