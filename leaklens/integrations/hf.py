"""Read a local HuggingFace snapshot (parquet/Arrow) without importing ``datasets``.

The reader is an integration: it produces text inputs and never findings. A
missing ``pyarrow`` is a recorded ``EngineUnavailable`` skip, not a traceback.
Every file is resolved inside the snapshot root before it is opened.
"""

from __future__ import annotations

from pathlib import Path

from ..availability import EngineUnavailable, probe
from ..paths import PathOutsideRoot, resolve_within

DEFAULT_MAX_ROWS = 10_000
SUFFIXES = frozenset({".parquet", ".arrow"})


class SnapshotUnavailable(EngineUnavailable):
    """pyarrow is not installed; a local snapshot cannot be read."""


def _pyarrow(module: str):
    return probe(
        module,
        "parquet",
        purpose="HuggingFace snapshot reading",
        label="pyarrow",
        error=SnapshotUnavailable,
    )


def _files(root: Path, split: str | None) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        try:
            resolved = resolve_within(root, path.relative_to(root).as_posix())
        except (PathOutsideRoot, ValueError):
            continue
        if split is not None and split not in path.relative_to(root).parts[:-1]:
            continue
        files.append(resolved)
    return files


def _column(table, field: str | None, pa):
    names = list(table.column_names)
    if field is None:
        field = "text" if "text" in names else None
        if field is None:
            for name in names:
                if pa.types.is_string(table.schema.field(name).type) or (
                    hasattr(pa.types, "is_large_string")
                    and pa.types.is_large_string(table.schema.field(name).type)
                ):
                    field = name
                    break
    if field is None or field not in names:
        raise ValueError(f"column {field!r} not found in the snapshot")
    return table.column(field).to_pylist()


def read_snapshot(
    root: str | Path,
    *,
    field: str | None = None,
    split: str | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> list[str | None]:
    """Return raw document strings from every snapshot file, in sorted path order."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    base = Path(root)
    if base.is_file():
        files = [base.resolve()]
    else:
        base = base.resolve()
        files = _files(base, split)
    if not files:
        raise ValueError("no parquet or arrow files under the snapshot root")

    values: list[str | None] = []
    for path in files:
        if path.suffix.lower() == ".parquet":
            parquet = _pyarrow("pyarrow.parquet")
            table = parquet.read_table(path)
            pa = _pyarrow("pyarrow")
        else:
            ipc = _pyarrow("pyarrow.ipc")
            table = ipc.open_stream(str(path)).read_all()
            pa = _pyarrow("pyarrow")
        values.extend(_column(table, field, pa))
        if len(values) >= max_rows:
            break
    return values[:max_rows]
