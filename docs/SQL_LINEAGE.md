# SQL lineage analysis

LeakLens can inspect SQL files for features built from the target and for
windows that read rows the model would not have at prediction time. The
analysis is **optional**, **offline** and never executes SQL or connects to a
database. It needs the `sql` extra:

```sh
pip install "leaklens-ml[sql]"
```

SQLGlot (MIT, evaluated at 30.17.0; the extra requires `>=25`) parses the query
and resolves column lineage through CTEs, derived tables, joins, unions and
aggregate aliases.

## Feasibility gate

The gate was evaluated on the sanitized corpus in
[`SQL_BENCHMARK.md`](SQL_BENCHMARK.md) before implementation:

- every expected parse outcome matched, including malformed SQL and a
  non-SELECT `MERGE` statement;
- target detection was TP=5, FP=0, FN=0 on the labeled cases;
- ambiguous join columns surfaced as unresolved (`?`) rather than invented
  lineage;
- the scanner stayed offline, deterministic and importable with or without
  SQLGlot.

**Gate result: passed with the limits below.** Broad dialect coverage with
uncertain lineage was rejected; the scanner only reports confirmed lineage.

## Supported dialects

| `--sql-dialect` | Notes |
|---|---|
| `ansi` (default) | SQLGlot's default parser; use it when the query is standard. |
| `postgres` | PostgreSQL syntax. |
| `bigquery` | BigQuery syntax, including backticked `project.dataset.table`. |
| `snowflake` | Snowflake syntax, including `QUALIFY`. |

There is **no silent dialect fallback**. Parse with the dialect the query is
written for; an unsupported `--sql-dialect` is rejected at the CLI/config
boundary. A query that fails to parse under the declared dialect produces a
single low `parse_error` finding and no lineage claims.

## Rules

| Kind | Severity | Condition |
|---|---|---|
| `sql_target_as_feature` | high | A non-bare projection whose expression references the target by name **and** whose SQLGlot lineage confirms the target column. |
| `sql_future_window` | medium | A window frame ends with `FOLLOWING`, so it can read later rows. |
| `sql_split_predicate_missing` | low | A configured `split_column` exists on a referenced table (declared schema) and no split predicate filters the query. |
| `sql_unresolved_column` | low, diagnostic | Lineage was ambiguous or unavailable (`SELECT *` without a schema, an unqualified column over several tables, an unsupported statement). |
| `parse_error` | low | The declared dialect cannot parse the input. Reuses the existing scanner kind. |

Deliberately **not** reported:

- a bare target projection such as `SELECT label` or a rename
  `SELECT label AS outcome`;
- a target used only as a filter (`WHERE label = 1`);
- a column whose *name* merely resembles the target - lineage has to confirm
  it;
- anything after unresolved lineage. The scanner emits the low diagnostic and
  stops instead of guessing.

## Usage

An explicit `.sql` file is always analyzed; folders need `--sql` (or
`[sql].enabled`):

```sh
leaklens --scan query.sql --target label --sql-dialect postgres
leaklens --scan warehouse/sql --sql --target label --sql-dialect bigquery
```

A schema map resolves `SELECT *` and ambiguous joins:

```toml
[sql]
enabled = true
dialect = "postgres"
split_column = "split"

[sql.schema.orders]
id = "INT"
customer_id = "INT"
amount = "DOUBLE"
label = "INT"
split = "TEXT"
created_at = "TIMESTAMP"
```

API: `POST /scan` with `"name": "query.sql"` and optional `dialect`, `target`,
`split_column` and `schema_map`. A missing SQLGlot install returns HTTP 422
with the install command; nothing else about `/scan` changes.

## Limitations

- SQLGlot resolves lineage inside one statement. It does not know view
  definitions, dbt models, UDF bodies, temporary tables or catalog metadata
  beyond the schema map you supply.
- `SELECT *` cannot be expanded without a schema; it becomes an unresolved
  diagnostic.
- An unqualified column over multiple source tables is treated as unresolved
  rather than assigned to a guessed table.
- Renames introduced by an upstream CTE are resolved by lineage, but a
  target referenced only through a computed alias that no longer contains the
  target name is not flagged. That is a deliberate false-negative.
- Non-SELECT statements (`MERGE`, `INSERT`, DDL) are parsed and skipped with a
  low diagnostic. Only `SELECT` and set operations are analyzed.
- The scanner analyzes at most 20 statements and 200 projections per
  statement; larger queries are reported as unresolved.
- `sql_future_window` looks at the frame direction only. It does not model
  watermarking, event time or the exact availability lag of a warehouse.
- Nested subqueries are analysed at the top-level projection list. A target
  aggregate computed in an inner subquery but not selected outside is not
  reported.

## Privacy and identity

The scanner reads SQL text only. Evidence contains table/column names, rule
names and counts; it never includes string literals, credentials, customer
data or rows. Findings use `path:line` for SARIF/HTML and a structural
`stable_key` (`target:<alias>`, `split:<table>`, `window:<alias>`) for
fingerprints, so moving a query by a line keeps its baseline identity.

## Performance

On the evaluation machine, parsing a 100-projection / 20-join query took about
9 ms and one lineage query about 15 ms. Lineage runs only for projections that
syntactically reference the target and only up to the projection cap, so
ordinary queries stay in the millisecond range.
