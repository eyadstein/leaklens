"""Optional SQL lineage: conservative detection, diagnostics, privacy, identity."""

import importlib.util
import sys

import pytest

from leaklens.leakage import score
from leaklens.sqlscan import (
    SUPPORTED_DIALECTS,
    SqlDialectError,
    SqlUnavailable,
    available,
    scan_sql,
)

HAS_SQLGLOT = importlib.util.find_spec("sqlglot") is not None
requires_sql = pytest.mark.skipif(not HAS_SQLGLOT, reason="sqlglot extra not installed")

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


def _scan(sql, **kwargs):
    kwargs.setdefault("target", "label")
    kwargs.setdefault("schema", SCHEMA)
    return scan_sql(sql, "query.sql", **kwargs)


def _kinds(findings):
    return {finding.kind for finding in findings}


def test_missing_optional_dependency_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlglot", None)
    with pytest.raises(SqlUnavailable, match="pip install"):
        scan_sql("SELECT 1", dialect="ansi")


def test_unknown_dialect_is_rejected_without_parsing():
    with pytest.raises(SqlDialectError, match="postgres"):
        scan_sql("SELECT 1", dialect="mysql")


@requires_sql
def test_dialect_matrix_parses_every_supported_dialect():
    for dialect in SUPPORTED_DIALECTS:
        findings = scan_sql("SELECT id, amount FROM orders", dialect=dialect)
        assert "parse_error" not in _kinds(findings), dialect
    assert available() is True


@requires_sql
def test_target_expression_is_high_with_evidence():
    findings = _scan("SELECT id, label * 2 AS label_score FROM orders")
    finding = next(f for f in findings if f.kind == "sql_target_as_feature")
    assert finding.severity == "high"
    assert finding.column == "query.sql:1"
    assert finding.evidence and finding.evidence[0].kind == "pattern"
    assert "label" in finding.detail
    assert score([finding]) == 65


@requires_sql
def test_target_through_cte_and_aggregate_alias():
    cte = _scan(
        "WITH f AS (SELECT id, amount, label FROM orders) "
        "SELECT id, CASE WHEN label = 1 THEN amount ELSE 0 END AS spend_pos FROM f"
    )
    assert "sql_target_as_feature" in _kinds(cte)
    aggregate = _scan(
        "SELECT customer_id, AVG(label) AS label_avg FROM orders GROUP BY customer_id"
    )
    assert "sql_target_as_feature" in _kinds(aggregate)


@requires_sql
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, label FROM orders",
        "SELECT id, label AS outcome FROM orders",
        "SELECT id, amount FROM orders WHERE label = 1",
        "SELECT o.id, c.country, o.amount FROM orders o "
        "JOIN customers c ON o.customer_id = c.id",
        "SELECT id, amount FROM orders UNION ALL SELECT id, amount FROM orders",
        "SELECT id, amount FROM (SELECT id, amount, "
        "AVG(label) OVER (PARTITION BY customer_id) AS label_avg FROM orders) AS q",
    ],
)
def test_bare_target_rename_filter_join_union_and_nested_stay_clean(sql):
    assert "sql_target_as_feature" not in _kinds(_scan(sql, dialect="postgres"))


@requires_sql
def test_ambiguous_join_is_a_low_diagnostic_not_a_guess():
    findings = _scan(
        "SELECT id, amount FROM orders o JOIN customers c ON o.customer_id = c.id",
        dialect="postgres",
    )
    assert "sql_target_as_feature" not in _kinds(findings)
    diagnostic = next(f for f in findings if f.kind == "sql_unresolved_column")
    assert diagnostic.severity == "low"
    assert diagnostic.diagnostic is True
    assert score(findings) == 100


@requires_sql
def test_select_star_without_schema_is_unresolved():
    findings = _scan("SELECT * FROM orders", schema=None)
    assert "sql_unresolved_column" in _kinds(findings)
    assert all(f.severity == "low" for f in findings)


@requires_sql
def test_statement_cap_reports_unresolved():
    findings = _scan("SELECT id FROM orders;\n" * 21)
    assert _kinds(findings) == {"sql_unresolved_column"}


@requires_sql
def test_future_window_is_medium_ordering_signal():
    findings = _scan(
        "SELECT id, SUM(amount) OVER (ORDER BY created_at "
        "ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS f FROM orders",
        dialect="postgres",
    )
    finding = next(f for f in findings if f.kind == "sql_future_window")
    assert finding.severity == "medium"
    assert finding.evidence[0].kind == "ordering"


@requires_sql
def test_split_predicate_rule_requires_schema_and_is_low():
    missing = _scan(
        "SELECT id, amount FROM orders", dialect="postgres", split_column="split"
    )
    finding = next(f for f in missing if f.kind == "sql_split_predicate_missing")
    assert finding.severity == "low"
    present = _scan(
        "SELECT id, amount FROM orders WHERE split = 'train'",
        dialect="postgres",
        split_column="split",
    )
    assert "sql_split_predicate_missing" not in _kinds(present)
    assert "sql_split_predicate_missing" not in _kinds(
        _scan("SELECT id, amount FROM orders", dialect="postgres")
    )


@requires_sql
def test_malformed_sql_is_a_low_parse_error():
    findings = _scan("SELEC id FROM orders")
    finding = next(f for f in findings if f.kind == "parse_error")
    assert finding.severity == "low"
    assert finding.column == "query.sql:1"


@requires_sql
def test_non_select_statement_is_skipped_with_a_diagnostic():
    findings = _scan(
        "MERGE INTO orders o USING customers c ON o.customer_id = c.id "
        "WHEN MATCHED THEN UPDATE SET amount = 0",
        schema=None,
    )
    assert _kinds(findings) == {"sql_unresolved_column"}
    assert findings[0].diagnostic is True


@requires_sql
def test_evidence_never_echoes_string_literals():
    findings = _scan(
        "SELECT id, CASE WHEN label = 1 THEN 'customer-secret' ELSE 'other' "
        "END AS bucket FROM orders"
    )
    assert "sql_target_as_feature" in _kinds(findings)
    text = str([finding.to_dict() for finding in findings])
    assert "customer-secret" not in text


@requires_sql
def test_fingerprints_are_deterministic_and_line_independent():
    first = _scan("SELECT id, label * 2 AS label_score FROM orders")
    second = _scan("\n\nSELECT id, label * 2 AS label_score FROM orders")
    assert first[0].fingerprint == second[0].fingerprint
    assert first[0].column.endswith(":1")
    assert second[0].column.endswith(":3")
    assert (
        first[0].fingerprint
        == _scan("SELECT id, label * 2 AS label_score FROM orders")[0].fingerprint
    )


@requires_sql
def test_diagnostic_flag_is_serialized_only_when_true():
    normal = _scan("SELECT id, label * 2 AS label_score FROM orders")[0]
    diagnostic = _scan("SELECT * FROM orders", schema=None)[0]
    assert "diagnostic" not in normal.to_dict()
    assert diagnostic.to_dict()["diagnostic"] is True
