# SQL lineage benchmark

Sanitized, deterministic synthetic SQL for the optional lineage scanner. Times are wall-clock on the machine that generated this file.

## Metrics

| Metric | Value |
|---|---:|
| Cases | 19 |
| Leaky / clean | 5 / 14 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| False-positive rate | 0.00 |
| True positives | 5 |
| False positives | 0 |
| True negatives | 14 |
| False negatives | 0 |
| Total runtime | 115 ms |
| Median case | 0 ms |
| Slowest case | 45 ms |

## Cases

| Case | Category | Dialect | Leaky | Detected | Findings | ms |
|---|---|---|:---:|:---:|---|---:|
| target expression in projection | target | postgres | yes | yes | sql_target_as_feature | 32.8 |
| target through a CTE | target | postgres | yes | yes | sql_target_as_feature | 2.3 |
| target aggregate alias | target | ansi | yes | yes | sql_target_as_feature | 1.3 |
| bare target projection is clean | clean | ansi | no | no | - | 0.3 |
| target rename is clean | clean | postgres | no | no | - | 0.3 |
| target filter is clean | clean | ansi | no | no | - | 0.3 |
| join with schema is clean | clean | postgres | no | no | - | 0.5 |
| ambiguous join is a diagnostic | diagnostic | postgres | no | no | sql_unresolved_column | 0.4 |
| select star without schema is a diagnostic | diagnostic | ansi | no | no | sql_unresolved_column | 0.2 |
| union is clean | clean | ansi | no | no | - | 0.4 |
| nested derived feature is clean | clean | postgres | no | no | - | 0.6 |
| window reads following rows | temporal | postgres | yes | yes | sql_future_window | 0.6 |
| split predicate missing | split | postgres | yes | yes | sql_split_predicate_missing | 0.3 |
| split predicate present | clean | postgres | no | no | - | 0.4 |
| bigquery dialect parses | dialect | bigquery | no | no | - | 27.8 |
| snowflake dialect parses | dialect | snowflake | no | no | - | 45.3 |
| ansi dialect parses | dialect | ansi | no | no | - | 0.3 |
| malformed SQL fails cleanly | invalid | ansi | no | no | parse_error | 0.1 |
| non-SELECT statement is skipped | diagnostic | postgres | no | no | sql_unresolved_column | 0.3 |

## Misclassified cases

- none

## Caveats

These are hand-written synthetic queries with clear labels; they do not estimate real-warehouse precision or recall. Lineage depends on declared dialects and optional schema maps, and unresolved lineage is reported as a low diagnostic instead of a guessed target relationship.
