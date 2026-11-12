"""Optional, offline SQL lineage analysis.

Column-level lineage is provided by SQLGlot (MIT) only when the ``sql`` extra
is installed. Nothing here executes SQL, opens a database connection or reads
warehouse credentials: the input is a piece of SQL text and the output is
findings built from the parsed AST.

Conservative contract:

* a target is only reported when SQLGlot's confirmed lineage for a non-bare
  projection includes that exact target column;
* unresolved or ambiguous lineage never produces a target finding - it
  produces at most one low-severity ``sql_unresolved_column`` diagnostic;
* source locations are ``path:line`` so SARIF/HTML show the query line, while
  fingerprints use a structural stable key rather than the line number.

Supported dialects are explicit (``ansi``, ``postgres``, ``bigquery``,
``snowflake``). There is no silent dialect fallback: pass the dialect the SQL
is written for. See ``docs/SQL_LINEAGE.md`` for limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .availability import EngineUnavailable
from .evidence import Evidence
from .leakage import Finding

SUPPORTED_DIALECTS = ("ansi", "postgres", "bigquery", "snowflake")
# SQLGlot has no dialect literally named "ansi"; its default parser is ANSI-ish.
_DIALECTS = {
    "ansi": None,
    "postgres": "postgres",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
}

MAX_STATEMENTS = 20
MAX_PROJECTIONS = 200


class SqlUnavailable(EngineUnavailable):
    """SQLGlot is not installed; SQL analysis cannot run."""


class SqlDialectError(ValueError):
    """The requested dialect is not one of the supported SQL dialects."""


@dataclass(frozen=True, slots=True)
class SqlSettings:
    """Opt-in SQL analysis settings (``[sql]`` in ``.leaklens.toml``)."""

    enabled: bool = False
    dialect: str = "ansi"
    split_column: str | None = None
    schema: dict[str, dict[str, str]] = field(default_factory=dict)


DEFAULT_SETTINGS = SqlSettings()

_MESSAGE = 'SQL scanning needs the optional SQL extra: pip install "leaklens-ml[sql]"'


def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.lineage import lineage
        from sqlglot.tokens import Tokenizer, TokenType
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise SqlUnavailable(_MESSAGE) from exc
    return sqlglot, exp, lineage, Tokenizer, TokenType


def available() -> bool:
    """True when SQLGlot can be imported (used by docs and benchmarks)."""
    try:
        _sqlglot()
    except SqlUnavailable:
        return False
    return True


def _read_dialect(dialect: str) -> str | None:
    try:
        return _DIALECTS[dialect]
    except KeyError:
        raise SqlDialectError(
            f"unsupported SQL dialect {dialect!r}; choose one of "
            f"{', '.join(SUPPORTED_DIALECTS)}"
        ) from None


def _normalise_schema(schema: Any) -> dict[str, dict[str, str]] | None:
    if not schema:
        return None
    out: dict[str, dict[str, str]] = {}
    for table, columns in schema.items():
        if not isinstance(columns, dict):
            raise TypeError("SQL schema tables must map to column tables")
        out[str(table).casefold()] = {
            str(column).casefold(): str(kind) for column, kind in columns.items()
        }
    return out


def _statement_lines(text: str, dialect: str | None) -> list[int]:
    _, _, _, Tokenizer, TokenType = _sqlglot()
    try:
        tokens = Tokenizer(dialect=dialect).tokenize(text)
    except Exception:  # noqa: BLE001 - line numbers are best effort, never fatal
        return []
    lines: list[int] = []
    start = None
    for token in tokens:
        if token.token_type == TokenType.SEMICOLON:
            if start is not None:
                lines.append(start)
            start = None
        elif start is None:
            start = token.line
    if start is not None:
        lines.append(start)
    return lines


def _selects_for(statement, exp) -> list:
    """Every SELECT branch of a statement, including nested set operations."""
    if isinstance(statement, exp.Select):
        return [statement]
    if isinstance(statement, (exp.Union, exp.Intersect, exp.Except)):
        return _selects_for(statement.this, exp) + _selects_for(
            statement.expression, exp
        )
    if isinstance(statement, (exp.Subquery, exp.Paren)):
        return _selects_for(statement.this, exp)
    return []


# Identifier quoting supported by the declared dialects. Single quotes are
# string literals, not identifiers, so they are deliberately absent.
_QUOTE_PAIRS = {'"': '"', "`": "`", "[": "]"}


def _identifier_name(value: str) -> str:
    """Decode one identifier segment, including doubled-quote escapes."""
    text = value.strip()
    if len(text) < 2 or text[0] not in _QUOTE_PAIRS:
        return text
    close = _QUOTE_PAIRS[text[0]]
    if text[-1] != close:
        return text
    return text[1:-1].replace(close + close, close)


def _unqualified(name: str) -> str:
    """Drop a table/alias prefix without splitting inside quoted identifiers."""
    text = name.strip()
    close = ""
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if close:
            if char == close:
                if index + 1 < len(text) and text[index + 1] == close:
                    index += 2
                    continue
                close = ""
        elif char in _QUOTE_PAIRS:
            close = _QUOTE_PAIRS[char]
        elif char == ".":
            start = index + 1
        index += 1
    return text[start:].strip()


def _references_target(expression, target: str, exp) -> bool:
    return any(
        isinstance(column, exp.Column) and column.name.casefold() == target
        for column in expression.find_all(exp.Column)
    )


def _leaf_state(node) -> tuple[set[str], bool]:
    """Return (confirmed leaf column names, unresolved?) for one lineage node."""
    names: set[str] = set()
    unresolved = False
    for leaf in node.walk():
        if leaf.downstream:
            continue
        expression = getattr(leaf, "expression", None)
        try:
            raw = expression.sql() if expression is not None else ""
        except Exception:  # noqa: BLE001 - a rendered leaf is optional metadata
            raw = ""
        name = getattr(leaf, "name", "") or ""
        if expression is None or raw in {"?", "*"} or name in {"", "*"}:
            unresolved = True
            continue
        base = _identifier_name(_unqualified(name))
        if base in {"", "*", "?"}:
            unresolved = True
            continue
        names.add(base.casefold())
    if not names:
        unresolved = True
    return names, unresolved


def _has_split_predicate(statement, split: str, exp) -> bool:
    for column in statement.find_all(exp.Column):
        if column.name.casefold() != split:
            continue
        parent = column.parent
        while parent is not None:
            if isinstance(parent, (exp.Where, exp.Having, exp.Join)):
                return True
            parent = parent.parent
    return False


def _location(label: str, line: int) -> str:
    return f"{label}:{line}"


def _sql_finding(
    severity: str,
    kind: str,
    label: str,
    line: int,
    detail: str,
    evidence: Evidence,
    *,
    stable_key: str,
    diagnostic: bool = False,
) -> Finding:
    return Finding(
        severity,
        kind,
        _location(label, line),
        detail,
        evidence=(evidence,),
        stable_key=stable_key,
        diagnostic=diagnostic,
    )


@dataclass(frozen=True, slots=True)
class SqlAnalysis:
    """Findings plus the column map resolved in the same parse (Phase 14 M2).

    ``column_map`` maps each output alias to the sorted source columns its
    expression derives from. It is built once here so downstream lineage (the
    pipeline engine) never re-parses the query.
    """

    findings: tuple[Finding, ...]
    column_map: dict[str, tuple[str, ...]] = field(default_factory=dict)


def scan_sql(
    text: str,
    label: str = "<sql>",
    *,
    dialect: str = "ansi",
    target: str | None = None,
    schema: Any = None,
    split_column: str | None = None,
) -> list[Finding]:
    """Analyze SQL text for target-derived features and temporal leakage.

    Kept as the findings-only entry point; ``analyze_sql`` also exposes the
    resolved column map.
    """
    return list(
        analyze_sql(
            text,
            label,
            dialect=dialect,
            target=target,
            schema=schema,
            split_column=split_column,
        ).findings
    )


def analyze_sql(
    text: str,
    label: str = "<sql>",
    *,
    dialect: str = "ansi",
    target: str | None = None,
    schema: Any = None,
    split_column: str | None = None,
) -> SqlAnalysis:
    """Analyze SQL text; return findings and the resolved output column map.

    ``schema`` is an optional ``{table: {column: type}}`` map. Without it,
    ``SELECT *`` and ambiguous joins stay unresolved and are reported as a low
    diagnostic instead of guessing lineage.
    """
    read = _read_dialect(dialect)
    sqlglot, exp, lineage, _, _ = _sqlglot()
    target_cf = target.casefold() if target else None
    split_cf = split_column.casefold() if split_column else None
    schema_map = _normalise_schema(schema)
    try:
        statements = sqlglot.parse(text, read=read)
    except Exception as exc:  # noqa: BLE001 - any parser error is a syntax finding
        reason = " ".join(str(exc).split())[:300]
        return SqlAnalysis(
            (
                Finding(
                    "low",
                    "parse_error",
                    _location(label, 1),
                    f"SQL could not be parsed as {dialect}: {reason}",
                    evidence=(
                        Evidence(
                            source="code",
                            kind="parse",
                            description="the SQL parser rejected this input",
                            observed=f"dialect={dialect}",
                            expected="valid SQL for the declared dialect",
                        ),
                    ),
                    stable_key=f"parse:{dialect}",
                ),
            )
        )

    lines = _statement_lines(text, read)
    findings: list[Finding] = []
    column_map: dict[str, tuple[str, ...]] = {}
    unresolved_count = 0
    unresolved_line = 1
    analysed = 0
    skipped = 0

    def mark_unresolved(at: int) -> None:
        nonlocal unresolved_count, unresolved_line
        if unresolved_count == 0:
            unresolved_line = at
        unresolved_count += 1

    if len(statements) > MAX_STATEMENTS:
        # Statements past the cap are not analysed: fail safe instead of silent.
        mark_unresolved(lines[MAX_STATEMENTS] if MAX_STATEMENTS < len(lines) else 1)

    for index, statement in enumerate(statements[:MAX_STATEMENTS]):
        line = lines[index] if index < len(lines) else 1
        selects = _selects_for(statement, exp)
        if not selects:
            skipped += 1
            continue
        analysed += 1
        projections = [
            (select, projection)
            for select in selects
            for projection in select.expressions
        ]
        if len(projections) > MAX_PROJECTIONS:
            mark_unresolved(line)
            continue
        statement_sql = statement.sql(dialect=read)
        has_star = False

        for select, projection in projections:
            sources = []
            if select.args.get("from_") is not None:
                sources.append(select.args["from_"].this)
            for join in select.args.get("joins") or ():
                if join.this is not None:
                    sources.append(join.this)
            multiple_sources = (
                len([source for source in sources if source is not None]) > 1
            )
            expression = (
                projection.this if isinstance(projection, exp.Alias) else projection
            )
            alias = projection.alias_or_name
            if isinstance(expression, exp.Star):
                has_star = True
                continue
            if (
                isinstance(expression, exp.Column)
                and not expression.table
                and multiple_sources
            ):
                # An unqualified column over several tables may come from either
                # source; fail safe instead of guessing.
                mark_unresolved(line)
                continue
            relevant = bool(target_cf) and _references_target(
                expression, target_cf, exp
            )
            if isinstance(expression, exp.Column):
                # A bare target projection (including a rename) is not leakage;
                # it is still useful identity lineage for downstream consumers.
                if alias:
                    column_map[alias] = (expression.name.casefold(),)
                continue
            if not alias:
                # No stable output name: lineage cannot be requested safely.
                if relevant:
                    mark_unresolved(line)
                continue
            try:
                node = lineage(alias, statement_sql, schema=schema_map, dialect=read)
            except Exception:  # noqa: BLE001 - unresolved lineage is a diagnostic
                if relevant:
                    mark_unresolved(line)
                continue
            leaves, unresolved = _leaf_state(node)
            if unresolved:
                if relevant:
                    mark_unresolved(line)
                continue
            column_map[alias] = tuple(sorted(leaves))
            if relevant and target_cf in leaves:
                findings.append(
                    _sql_finding(
                        "high",
                        "sql_target_as_feature",
                        label,
                        line,
                        f"projection {alias!r} derives from the target column "
                        f"{target!r}; it cannot be known before the outcome",
                        Evidence(
                            source="code",
                            kind="pattern",
                            description=(
                                "confirmed column lineage links a feature expression "
                                "to the target column"
                            ),
                            location=_location(label, line),
                            observed=f"lineage includes {target!r}",
                            expected="feature lineage excludes the target",
                        ),
                        stable_key=f"target:{alias}",
                    )
                )

        if has_star and schema_map is None:
            mark_unresolved(line)

        if split_cf and schema_map is not None:
            tables = {
                table.name.casefold()
                for table in statement.find_all(exp.Table)
                if table.name
            }
            known = {
                table
                for table in tables
                if table in schema_map and split_cf in schema_map[table]
            }
            if known and not _has_split_predicate(statement, split_cf, exp):
                table = min(known)
                findings.append(
                    _sql_finding(
                        "low",
                        "sql_split_predicate_missing",
                        label,
                        line,
                        f"query reads {table!r} without a {split_column!r} predicate; "
                        f"train and test rows may be mixed",
                        Evidence(
                            source="code",
                            kind="pattern",
                            description=(
                                "the configured split column is known on a source "
                                "table but no split predicate filters it"
                            ),
                            location=_location(label, line),
                            observed=f"table {table!r} has no {split_column!r} filter",
                            expected=f"a predicate on {split_column!r}",
                        ),
                        stable_key=f"split:{table}",
                    )
                )

        for window in statement.find_all(exp.Window):
            spec = window.args.get("spec")
            end_side = spec.args.get("end_side") if spec is not None else None
            if end_side != "FOLLOWING":
                continue
            alias = window.parent.alias_or_name if window.parent is not None else ""
            findings.append(
                _sql_finding(
                    "medium",
                    "sql_future_window",
                    label,
                    line,
                    "window frame extends into following rows, so the feature can "
                    "use values that are not available at prediction time",
                    Evidence(
                        source="code",
                        kind="ordering",
                        description="the window frame ends after the current row",
                        location=_location(label, line),
                        observed="frame end: FOLLOWING",
                        expected="frame ends at CURRENT ROW",
                    ),
                    stable_key=f"window:{alias or 'window'}",
                )
            )

    if unresolved_count:
        findings.append(
            _sql_finding(
                "low",
                "sql_unresolved_column",
                label,
                unresolved_line,
                f"{unresolved_count} projection(s) could not be resolved to a "
                "source column; target-derived features may be missed",
                Evidence(
                    source="code",
                    kind="parse",
                    description=(
                        "lineage was ambiguous or unavailable, so the analysis "
                        "stayed conservative instead of guessing"
                    ),
                    location=_location(label, unresolved_line),
                    observed=f"{unresolved_count} unresolved projection(s)",
                    expected="all projections resolve to a source column",
                ),
                stable_key="unresolved",
                diagnostic=True,
            )
        )

    if analysed == 0 and skipped:
        findings.append(
            _sql_finding(
                "low",
                "sql_unresolved_column",
                label,
                1,
                "no SELECT statement was found; only SELECT/UNION queries are analysed",
                Evidence(
                    source="code",
                    kind="parse",
                    description="the parsed statement is not a SELECT query",
                    observed=f"{skipped} non-SELECT statement(s)",
                    expected="a SELECT or UNION query",
                ),
                stable_key="unresolved",
                diagnostic=True,
            )
        )

    # Repeating the same query must not double-count one structural issue.
    unique: dict[str, Finding] = {}
    for finding in findings:
        unique.setdefault(finding.fingerprint, finding)
    return SqlAnalysis(tuple(unique.values()), dict(sorted(column_map.items())))


def scan_sql_file(path, *, settings: SqlSettings | None = None, target=None):
    """Read a ``.sql`` file and analyze it with explicit settings."""
    settings = settings or DEFAULT_SETTINGS
    text = path.read_text(encoding="utf-8", errors="ignore")
    return scan_sql(
        text,
        str(path),
        dialect=settings.dialect,
        target=target,
        schema=settings.schema,
        split_column=settings.split_column,
    )
