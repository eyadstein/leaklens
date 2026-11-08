"""Strict, offline project configuration (``.leaklens.toml``).

Precedence is deterministic: built-in defaults, then this file, then explicit
CLI arguments. Unknown keys, wrong types, invalid values and unsafe output
paths fail closed with the file location in the message. No option here loads
plugins, executes code or performs network access.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .drift import DEFAULT_SETTINGS as DEFAULT_DRIFT_SETTINGS
from .drift import DriftSettings
from .missingness import DEFAULT_SETTINGS, MissingnessSettings
from .sqlscan import DEFAULT_SETTINGS as DEFAULT_SQL_SETTINGS
from .sqlscan import SUPPORTED_DIALECTS, SqlSettings
from .suppression import SuppressionError, SuppressionRule, parse_rules

CONFIG_NAME = ".leaklens.toml"
_TOP_LEVEL = {
    "target",
    "test",
    "time",
    "group",
    "threshold",
    "scan",
    "explain",
    "fail_under",
    "sarif",
    "html",
    "badge",
    "ai",
    "ai_llm",
    "baseline",
    "suppressions",
    "missingness",
    "sql",
    "drift",
    "graph",
}
_BASELINE_KEYS = {"path", "fail_on_new"}
_GRAPH_KEYS = {"enabled", "baseline", "snapshot", "update_snapshot", "fail_on_new"}
_SQL_KEYS = {"enabled", "dialect", "split_column", "schema"}
_DRIFT_KEYS = {
    "enabled",
    "min_rows",
    "min_effect",
    "max_rows",
    "max_columns",
    "max_categories",
}
_MISSINGNESS_KEYS = {
    "enabled",
    "min_rows",
    "min_group_rows",
    "min_missing",
    "min_rate_delta",
    "min_group_delta",
    "min_correlation",
}


class ConfigError(ValueError):
    """The project configuration is unreadable, unknown or invalid."""


@dataclass(frozen=True, slots=True)
class Config:
    """Validated configuration with unset values represented as ``None``."""

    path: Path | None = None
    target: str | None = None
    test: Path | None = None
    time: str | None = None
    group: str | None = None
    threshold: float | None = None
    scan: tuple[Path, ...] = ()
    explain: bool | None = None
    fail_under: float | None = None
    sarif: Path | None = None
    html: Path | None = None
    badge: Path | None = None
    ai: bool | None = None
    ai_llm: bool | None = None
    baseline_path: Path | None = None
    fail_on_new: bool | None = None
    suppressions: tuple[SuppressionRule, ...] = ()
    missingness: MissingnessSettings = DEFAULT_SETTINGS
    sql: SqlSettings = DEFAULT_SQL_SETTINGS
    drift: DriftSettings = DEFAULT_DRIFT_SETTINGS
    graph_enabled: bool | None = None
    graph_baseline_path: Path | None = None
    graph_snapshot_path: Path | None = None
    graph_update_snapshot: bool | None = None
    graph_fail_on_new: bool | None = None


DEFAULT_CONFIG = Config()


def _toml_loads(text: str) -> Any:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    return tomllib.loads(text)


def _string(data: dict, key: str, path: Path, *, limit: int = 300) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{path}: {key} must be a string")
    text = value.strip()
    if not text:
        raise ConfigError(f"{path}: {key} must not be empty")
    if len(text) > limit:
        raise ConfigError(f"{path}: {key} exceeds {limit} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ConfigError(f"{path}: {key} contains control characters")
    return text


def _boolean(data: dict, key: str, path: Path) -> bool:
    value = data[key]
    if type(value) is not bool:
        raise ConfigError(f"{path}: {key} must be true or false")
    return value


def _integer(data: dict, key: str, path: Path, *, low: int) -> int:
    value = data[key]
    if type(value) is not int:
        raise ConfigError(f"{path}: {key} must be an integer")
    if value < low:
        raise ConfigError(f"{path}: {key} must be >= {low}")
    return value


def _number(data: dict, key: str, path: Path, *, low: float, high: float) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: {key} must be a number")
    number = float(value)
    if not low <= number <= high:
        raise ConfigError(f"{path}: {key} must be between {low} and {high}")
    return number


def _input_path(data: dict, key: str, path: Path, base: Path) -> Path:
    candidate = Path(_string(data, key, path))
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _output_path(data: dict, key: str, path: Path, base: Path) -> Path:
    text = _string(data, key, path)
    candidate = Path(text)
    if candidate.is_absolute():
        raise ConfigError(
            f"{path}: {key} must be relative to the configuration directory"
        )
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ConfigError(
            f"{path}: {key} must stay inside the project directory {base}"
        ) from None
    if not resolved.parent.is_dir():
        raise ConfigError(
            f"{path}: {key} parent directory does not exist: {resolved.parent}"
        )
    return resolved


def _parse_baseline(
    data: Any, path: Path, base: Path
) -> tuple[Path | None, bool | None]:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: baseline must be a table")
    unknown = sorted(set(data) - _BASELINE_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unsupported baseline keys: {', '.join(unknown)}")
    baseline_path = _output_path(data, "path", path, base) if "path" in data else None
    fail_on_new = _boolean(data, "fail_on_new", path) if "fail_on_new" in data else None
    if fail_on_new and baseline_path is None:
        raise ConfigError(
            f"{path}: baseline.fail_on_new needs baseline.path in the same table"
        )
    return baseline_path, fail_on_new


def _parse_graph(data: Any, path: Path, base: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: graph must be a table")
    unknown = sorted(set(data) - _GRAPH_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unsupported graph keys: {', '.join(unknown)}")
    enabled = _boolean(data, "enabled", path) if "enabled" in data else None
    baseline = _input_path(data, "baseline", path, base) if "baseline" in data else None
    snapshot = (
        _output_path(data, "snapshot", path, base) if "snapshot" in data else None
    )
    update = (
        _boolean(data, "update_snapshot", path) if "update_snapshot" in data else None
    )
    fail_on_new = _boolean(data, "fail_on_new", path) if "fail_on_new" in data else None
    if baseline is not None and not baseline.is_file():
        raise ConfigError(f"{path}: graph.baseline not found: {baseline}")
    if (baseline or snapshot or update or fail_on_new) and enabled is not True:
        raise ConfigError(
            f"{path}: graph.enabled must be true when other graph settings are present"
        )
    if update and snapshot is None:
        raise ConfigError(f"{path}: graph.update_snapshot needs graph.snapshot")
    if fail_on_new and baseline is None:
        raise ConfigError(f"{path}: graph.fail_on_new needs graph.baseline")
    return {
        "graph_enabled": enabled,
        "graph_baseline_path": baseline,
        "graph_snapshot_path": snapshot,
        "graph_update_snapshot": update,
        "graph_fail_on_new": fail_on_new,
    }


def _parse_missingness(data: Any, path: Path) -> MissingnessSettings:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: missingness must be a table")
    unknown = sorted(set(data) - _MISSINGNESS_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unsupported missingness keys: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    if "enabled" in data:
        values["enabled"] = _boolean(data, "enabled", path)
    if "min_rows" in data:
        values["min_rows"] = _integer(data, "min_rows", path, low=10)
    if "min_group_rows" in data:
        values["min_group_rows"] = _integer(data, "min_group_rows", path, low=2)
    if "min_missing" in data:
        values["min_missing"] = _integer(data, "min_missing", path, low=1)
    for key in ("min_rate_delta", "min_group_delta", "min_correlation"):
        if key in data:
            values[key] = _number(data, key, path, low=0.01, high=1.0)
    return replace(DEFAULT_SETTINGS, **values)


def _name(value: Any, what: str, path: Path) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: sql.schema {what} must be a string")
    text = value.strip()
    if (
        not text
        or len(text) > 200
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in text)
    ):
        raise ConfigError(f"{path}: sql.schema {what} is not a valid identifier")
    return text


def _parse_sql(data: Any, path: Path) -> SqlSettings:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: sql must be a table")
    unknown = sorted(set(data) - _SQL_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unsupported sql keys: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    if "enabled" in data:
        values["enabled"] = _boolean(data, "enabled", path)
    if "dialect" in data:
        dialect = _string(data, "dialect", path)
        if dialect not in SUPPORTED_DIALECTS:
            raise ConfigError(
                f"{path}: sql.dialect must be one of {', '.join(SUPPORTED_DIALECTS)}"
            )
        values["dialect"] = dialect
    if "split_column" in data:
        values["split_column"] = _string(data, "split_column", path)
    if "schema" in data:
        raw = data["schema"]
        if not isinstance(raw, dict) or not raw:
            raise ConfigError(f"{path}: sql.schema must be a non-empty table")
        if len(raw) > 200:
            raise ConfigError(f"{path}: sql.schema has too many tables (max 200)")
        schema: dict[str, dict[str, str]] = {}
        for table, columns in raw.items():
            name = _name(table, "table", path)
            if not isinstance(columns, dict) or not columns:
                raise ConfigError(f"{path}: sql.schema.{name} must be a column table")
            if len(columns) > 1000:
                raise ConfigError(f"{path}: sql.schema.{name} has too many columns")
            schema[name] = {
                _name(column, f"column in {name}", path): _string_value(
                    kind, f"sql.schema.{name}.{column}", path
                )
                for column, kind in columns.items()
            }
        values["schema"] = schema
    return replace(DEFAULT_SQL_SETTINGS, **values)


def _string_value(value: Any, what: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {what} must be a non-empty string")
    if len(value) > 100 or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ConfigError(f"{path}: {what} is not a valid type name")
    return value.strip()


def _parse_drift(data: Any, path: Path) -> DriftSettings:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: drift must be a table")
    unknown = sorted(set(data) - _DRIFT_KEYS)
    if unknown:
        raise ConfigError(f"{path}: unsupported drift keys: {', '.join(unknown)}")
    values: dict[str, Any] = {}
    if "enabled" in data:
        values["enabled"] = _boolean(data, "enabled", path)
    if "min_rows" in data:
        values["min_rows"] = _integer(data, "min_rows", path, low=20)
    if "max_rows" in data:
        values["max_rows"] = _integer(data, "max_rows", path, low=1000)
    if "max_columns" in data:
        values["max_columns"] = _integer(data, "max_columns", path, low=1)
    if "max_categories" in data:
        values["max_categories"] = _integer(data, "max_categories", path, low=2)
    if "min_effect" in data:
        values["min_effect"] = _number(data, "min_effect", path, low=0.01, high=1.0)
    return replace(DEFAULT_DRIFT_SETTINGS, **values)


def _parse(data: Any, path: Path) -> Config:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: configuration must be a table")
    unknown = sorted(set(data) - _TOP_LEVEL)
    if unknown:
        raise ConfigError(
            f"{path}: unknown configuration keys: {', '.join(unknown)}; "
            f"supported keys: {', '.join(sorted(_TOP_LEVEL))}"
        )
    base = path.parent.resolve()
    values: dict[str, Any] = {"path": path}
    for key in ("target", "time", "group"):
        if key in data:
            values[key] = _string(data, key, path)
    if "threshold" in data:
        values["threshold"] = _number(data, "threshold", path, low=0.0, high=1.0)
    if "test" in data:
        test = _input_path(data, "test", path, base)
        if not test.is_file():
            raise ConfigError(f"{path}: test file not found: {test}")
        values["test"] = test
    if "scan" in data:
        raw = data["scan"]
        items = [raw] if isinstance(raw, str) else raw
        if not isinstance(items, list) or not items:
            raise ConfigError(f"{path}: scan must be a non-empty string or array")
        paths: list[Path] = []
        for index, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(f"{path}: scan[{index}] must be a non-empty string")
            candidate = Path(item.strip())
            resolved = (
                candidate if candidate.is_absolute() else (base / candidate).resolve()
            )
            if not resolved.exists():
                raise ConfigError(f"{path}: scan path not found: {resolved}")
            paths.append(resolved)
        values["scan"] = tuple(paths)
    for key in ("explain", "ai", "ai_llm"):
        if key in data:
            values[key] = _boolean(data, key, path)
    if "fail_under" in data:
        values["fail_under"] = _number(data, "fail_under", path, low=0.0, high=100.0)
    for key in ("sarif", "html", "badge"):
        if key in data:
            values[key] = _output_path(data, key, path, base)
    if "baseline" in data:
        baseline_path, fail_on_new = _parse_baseline(data["baseline"], path, base)
        if baseline_path is not None:
            values["baseline_path"] = baseline_path
        if fail_on_new is not None:
            values["fail_on_new"] = fail_on_new
    if "suppressions" in data:
        try:
            values["suppressions"] = parse_rules(data["suppressions"])
        except SuppressionError as exc:
            raise ConfigError(f"{path}: {exc}") from None
    if "missingness" in data:
        values["missingness"] = _parse_missingness(data["missingness"], path)
    if "sql" in data:
        values["sql"] = _parse_sql(data["sql"], path)
    if "drift" in data:
        values["drift"] = _parse_drift(data["drift"], path)
    if "graph" in data:
        values.update(_parse_graph(data["graph"], path, base))
    return Config(**values)


def load_config(
    path: str | Path | None = None, *, cwd: str | Path | None = None
) -> Config:
    """Load and validate a config file, or return defaults when none exists."""
    directory = Path(cwd) if cwd is not None else Path.cwd()
    if path is None:
        candidate = directory / CONFIG_NAME
        if not candidate.exists():
            return DEFAULT_CONFIG
    else:
        candidate = Path(path)
    if candidate.is_dir():
        raise ConfigError(f"configuration path is a directory: {candidate}")
    try:
        text = candidate.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {candidate}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {candidate}: {exc}") from None
    try:
        data = _toml_loads(text)
    except ValueError as exc:
        raise ConfigError(f"invalid TOML in {candidate}: {exc}") from None
    return _parse(data, candidate.resolve())
