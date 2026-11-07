"""Sanitized, labeled benchmark for optional SQL lineage analysis.

The corpus is hand-written synthetic SQL that covers the supported dialects,
CTEs, derived tables, joins, unions, aggregates, windows, split filters and
deliberately ambiguous input. It guards against regressions; it is not an
estimate of real-warehouse accuracy.

Usage: ``python -m leaklens.bench_sql [--write docs/SQL_BENCHMARK.md]``
Requires the optional ``sql`` extra; without SQLGlot the benchmark reports
``unavailable`` and exits 0 so the default CI matrix stays green.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .sqlscan import available, scan_sql

SCHEMA = {
    "orders": {
        "id": "INT",
        "customer_id": "INT",
        "amount": "DOUBLE",
        "label": "INT",
        "split": "TEXT",
        "created_at": "TIMESTAMP",
    },
    "customers": {"id": "INT", "country": "TEXT"},
}

LEAKAGE_KINDS = {
    "sql_target_as_feature",
    "sql_future_window",
    "sql_split_predicate_missing",
}


@dataclass(frozen=True)
class SqlCase:
    name: str
    category: str
    dialect: str
    sql: str
    leaky: bool
    expected: str = ""
    schema: dict | None = field(default_factory=lambda: SCHEMA)
    split_column: str | None = None
    expect_parse: bool = True
    expect_diagnostic: bool = False


def _build_cases() -> list[SqlCase]:
    return [
        SqlCase(
            "target expression in projection",
            "target",
            "postgres",
            "SELECT id, label * 2 AS label_score FROM orders",
            True,
            "sql_target_as_feature",
        ),
        SqlCase(
            "target through a CTE",
            "target",
            "postgres",
            "WITH f AS (SELECT id, amount, label FROM orders) "
            "SELECT id, CASE WHEN label = 1 THEN amount ELSE 0 END AS spend_pos FROM f",
            True,
            "sql_target_as_feature",
        ),
        SqlCase(
            "target aggregate alias",
            "target",
            "ansi",
            "SELECT customer_id, AVG(label) AS label_avg FROM orders "
            "GROUP BY customer_id",
            True,
            "sql_target_as_feature",
        ),
        SqlCase(
            "bare target projection is clean",
            "clean",
            "ansi",
            "SELECT id, label FROM orders",
            False,
        ),
        SqlCase(
            "target rename is clean",
            "clean",
            "postgres",
            "SELECT id, label AS outcome FROM orders",
            False,
        ),
        SqlCase(
            "target filter is clean",
            "clean",
            "ansi",
            "SELECT id, amount FROM orders WHERE label = 1",
            False,
        ),
        SqlCase(
            "join with schema is clean",
            "clean",
            "postgres",
            "SELECT o.id, c.country, o.amount FROM orders o "
            "JOIN customers c ON o.customer_id = c.id",
            False,
        ),
        SqlCase(
            "ambiguous join is a diagnostic",
            "diagnostic",
            "postgres",
            "SELECT id, amount FROM orders o JOIN customers c ON o.customer_id = c.id",
            False,
            expect_diagnostic=True,
        ),
        SqlCase(
            "select star without schema is a diagnostic",
            "diagnostic",
            "ansi",
            "SELECT * FROM orders",
            False,
            schema=None,
            expect_diagnostic=True,
        ),
        SqlCase(
            "union is clean",
            "clean",
            "ansi",
            "SELECT id, amount FROM orders UNION ALL SELECT id, amount FROM orders",
            False,
        ),
        SqlCase(
            "nested derived feature is clean",
            "clean",
            "postgres",
            "SELECT id, amount FROM (SELECT id, amount, "
            "AVG(label) OVER (PARTITION BY customer_id) AS label_avg FROM orders) AS q",
            False,
        ),
        SqlCase(
            "window reads following rows",
            "temporal",
            "postgres",
            "SELECT id, SUM(amount) OVER (ORDER BY created_at "
            "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS f FROM orders",
            True,
            "sql_future_window",
        ),
        SqlCase(
            "split predicate missing",
            "split",
            "postgres",
            "SELECT id, amount FROM orders",
            True,
            "sql_split_predicate_missing",
            split_column="split",
        ),
        SqlCase(
            "split predicate present",
            "clean",
            "postgres",
            "SELECT id, amount FROM orders WHERE split = 'train'",
            False,
            split_column="split",
        ),
        SqlCase(
            "bigquery dialect parses",
            "dialect",
            "bigquery",
            "SELECT id, amount FROM `project.dataset.orders`",
            False,
            schema=None,
        ),
        SqlCase(
            "snowflake dialect parses",
            "dialect",
            "snowflake",
            "SELECT id, label FROM orders QUALIFY ROW_NUMBER() "
            "OVER (PARTITION BY id ORDER BY created_at) = 1",
            False,
        ),
        SqlCase(
            "ansi dialect parses",
            "dialect",
            "ansi",
            "SELECT id, amount FROM orders",
            False,
        ),
        SqlCase(
            "malformed SQL fails cleanly",
            "invalid",
            "ansi",
            "SELEC id FROM orders",
            False,
            expect_parse=False,
        ),
        SqlCase(
            "non-SELECT statement is skipped",
            "diagnostic",
            "postgres",
            "MERGE INTO orders o USING customers c ON o.customer_id = c.id "
            "WHEN MATCHED THEN UPDATE SET amount = 0",
            False,
            schema=None,
            expect_diagnostic=True,
        ),
    ]


def evaluate() -> dict:
    if not available():
        return {"unavailable": True, "rows": [], "errors": []}
    rows, errors = [], []
    tp = fp = tn = fn = 0
    for case in _build_cases():
        start = time.perf_counter()
        try:
            findings = scan_sql(
                case.sql,
                f"bench:{case.name}",
                dialect=case.dialect,
                target="label",
                schema=case.schema,
                split_column=case.split_column,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark reports the failure
            findings = []
            errors.append(f"{case.name}: raised {type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        kinds = {finding.kind for finding in findings}
        parsed = "parse_error" not in kinds
        diagnostic = "sql_unresolved_column" in kinds
        detected = any(kind in LEAKAGE_KINDS for kind in kinds)
        correct = (
            parsed == case.expect_parse
            and diagnostic == case.expect_diagnostic
            and (
                (case.leaky and case.expected in kinds)
                or (not case.leaky and not detected)
            )
        )
        if case.leaky and correct:
            tp += 1
        elif case.leaky:
            fn += 1
            errors.append(f"missed: {case.name} (expected {case.expected})")
        elif not correct:
            fp += 1
            errors.append(f"false alarm: {case.name} -> {sorted(kinds)}")
        else:
            tn += 1
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "dialect": case.dialect,
                "leaky": case.leaky,
                "detected": detected,
                "kinds": ", ".join(sorted(kinds)) or "-",
                "ms": elapsed * 1000.0,
            }
        )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    times = [row["ms"] for row in rows]
    return {
        "unavailable": False,
        "n": len(rows),
        "leaky": tp + fn,
        "clean": tn + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp / (fp + tn) if fp + tn else 0.0,
        "rows": rows,
        "errors": errors,
        "runtime_total_ms": sum(times),
        "runtime_median_ms": statistics.median(times) if times else 0.0,
        "runtime_max_ms": max(times) if times else 0.0,
    }


def markdown(metrics: dict) -> str:
    if metrics.get("unavailable"):
        return (
            "# SQL lineage benchmark\n\n"
            "SQLGlot is not installed, so this benchmark did not run. Install the "
            'optional extra with `pip install "leaklens-ml[sql]"`.\n'
        )
    table = [
        "| Case | Category | Dialect | Leaky | Detected | Findings | ms |",
        "|---|---|---|:---:|:---:|---|---:|",
    ]
    for row in metrics["rows"]:
        table.append(
            f"| {row['name']} | {row['category']} | {row['dialect']} | "
            f"{'yes' if row['leaky'] else 'no'} | "
            f"{'yes' if row['detected'] else 'no'} | {row['kinds']} | "
            f"{row['ms']:.1f} |"
        )
    miss = "\n".join(f"- {error}" for error in metrics["errors"]) or "- none"
    return (
        "# SQL lineage benchmark\n\n"
        "Sanitized, deterministic synthetic SQL for the optional lineage scanner. "
        "Times are wall-clock on the machine that generated this file.\n\n"
        "## Metrics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| Cases | {metrics['n']} |\n"
        f"| Leaky / clean | {metrics['leaky']} / {metrics['clean']} |\n"
        f"| Precision | {metrics['precision']:.2f} |\n"
        f"| Recall | {metrics['recall']:.2f} |\n"
        f"| F1 | {metrics['f1']:.2f} |\n"
        f"| False-positive rate | {metrics['fp_rate']:.2f} |\n"
        f"| True positives | {metrics['tp']} |\n"
        f"| False positives | {metrics['fp']} |\n"
        f"| True negatives | {metrics['tn']} |\n"
        f"| False negatives | {metrics['fn']} |\n"
        f"| Total runtime | {metrics['runtime_total_ms']:.0f} ms |\n"
        f"| Median case | {metrics['runtime_median_ms']:.0f} ms |\n"
        f"| Slowest case | {metrics['runtime_max_ms']:.0f} ms |\n\n"
        "## Cases\n\n" + "\n".join(table) + "\n\n"
        "## Misclassified cases\n\n" + miss + "\n\n"
        "## Caveats\n\n"
        "These are hand-written synthetic queries with clear labels; they do not "
        "estimate real-warehouse precision or recall. Lineage depends on declared "
        "dialects and optional schema maps, and unresolved lineage is reported as "
        "a low diagnostic instead of a guessed target relationship.\n"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    metrics = evaluate()
    text = markdown(metrics)
    print(text)
    if len(argv) >= 2 and argv[0] == "--write":
        out = Path(argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
    return 1 if metrics["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
