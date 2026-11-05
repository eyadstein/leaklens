"""Validation for untrusted dataset uploads on the public scan API.

Nothing here trusts a filename, a ``Content-Type`` header or a client-supplied
extension.  The guard sanitises the display name, verifies that the bytes look
like the claimed format, and validates the parsed frame against hard row,
column and cell limits before LeakLens runs any detector.

The module is deliberately independent of FastAPI so the checks can be unit
tested without an HTTP server.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .readers import DatasetLimitError, read_dataset

# suffix -> (human format, optional module needed, extra that provides it)
SUPPORTED_FORMATS: dict[str, tuple[str, str | None, str | None]] = {
    ".csv": ("CSV", None, None),
    ".json": ("JSON records", None, None),
    ".xlsx": ("Excel workbook", "openpyxl", "excel"),
    ".parquet": ("Parquet", "pyarrow", "parquet"),
    ".feather": ("Feather", "pyarrow", "feather"),
}

# The public deployment enables the dependency-free text formats by default.
DEFAULT_FORMATS: tuple[str, ...] = ("csv", "json")

_PREVIEW_ROWS = 5
_PREVIEW_CHARS = 80
_MAX_COLUMN_NAME_CHARS = 120
_MAX_CELL_CHARS = 512
_CSV_SAMPLE = 64 * 1024
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Filenames and column names also reject tabs/newlines/carriage returns.
_CONTROL_NAME = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_NAME = re.compile(r"[^\w.\- ]+", re.UNICODE)
_SPACE = re.compile(r"\s+")


class UploadRejected(ValueError):
    """A user-safe rejection raised before or during a scan."""

    def __init__(self, message: str, status_code: int = 422, code: str = "bad_upload"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class GuardLimits:
    """Hard limits applied to every public dataset upload."""

    max_bytes: int = 4 * 1024 * 1024
    max_rows: int = 5_000
    max_columns: int = 200
    max_cells: int = 750_000
    max_name_chars: int = 120
    max_uncompressed_bytes: int = 24 * 1024 * 1024
    max_zip_entries: int = 1_500
    formats: tuple[str, ...] = DEFAULT_FORMATS

    def allows(self, suffix: str) -> bool:
        return suffix.lstrip(".").lower() in self.formats

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "max_rows": self.max_rows,
            "max_columns": self.max_columns,
            "max_cells": self.max_cells,
            "max_uncompressed_bytes": self.max_uncompressed_bytes,
            "formats": list(self.formats),
        }


@dataclass(frozen=True)
class GuardedUpload:
    """A validated byte payload plus its sanitised display name."""

    data: bytes
    filename: str
    suffix: str
    content_type: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _module_available(module: str | None) -> bool:
    return module is None or importlib.util.find_spec(module) is not None


def sanitize_filename(name: str | None, max_chars: int = 120) -> str:
    """Return a safe display name, rejecting path components and NUL bytes.

    Modern browsers send only the base name, so a path separator is a strong
    signal of a hand-crafted request rather than a normal upload.
    """
    raw = (name or "").strip()
    if not raw:
        return "upload.csv"
    if "\x00" in raw:
        raise UploadRejected(
            "The filename contains a NUL byte.", 400, "invalid_filename"
        )
    if _CONTROL_NAME.search(raw):
        raise UploadRejected(
            "The filename contains control characters.", 400, "invalid_filename"
        )
    if "/" in raw or "\\" in raw:
        raise UploadRejected(
            "The filename must not contain a path.", 400, "invalid_filename"
        )
    normalised = unicodedata.normalize("NFKC", raw)
    safe = _UNSAFE_NAME.sub("_", normalised)
    safe = _SPACE.sub(" ", safe).strip(" .")
    if not safe:
        safe = "upload"
    if len(safe) > max_chars:
        stem, dot, suffix = safe.rpartition(".")
        if dot and len(suffix) <= 12:
            keep = max(1, max_chars - len(suffix) - 1)
            safe = f"{stem[:keep]}.{suffix}"
        else:
            safe = safe[:max_chars]
    if _CONTROL_NAME.search(safe):
        raise UploadRejected(
            "The filename contains control characters.", 400, "invalid_filename"
        )
    return safe


def _suffix(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot > 0 else ""


def _text_encoding(data: bytes) -> str | None:
    """Return the text encoding for a byte payload, or ``None`` if binary."""
    if not data:
        return None
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return None  # UTF-16 CSVs are not supported by the public reader
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    if _CONTROL.search(text.replace("\t", "").replace("\r", "").replace("\n", "")):
        return None
    return text


def _looks_like_csv(data: bytes) -> tuple[bool, str]:
    text = _text_encoding(data)
    if text is None:
        return False, "the file is binary or not UTF-8 text"
    if not text.strip():
        return False, "the file is empty"
    sample = text[:_CSV_SAMPLE]
    lines = [line for line in sample.splitlines() if line.strip()]
    if len(lines) < 2:
        return False, "a CSV needs a header line and at least one data row"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        # A single-column CSV is syntactically valid but unusable as an audit
        # target/feature pair; report the clearer reason below.
        dialect = None
    if dialect is None:
        return (
            False,
            "no CSV delimiter was found (expected comma, semicolon, tab or pipe)",
        )
    if len(lines[0].split(dialect.delimiter)) < 2:
        return False, "a dataset audit needs at least two columns"
    return True, ""


def _looks_like_json(data: bytes) -> tuple[bool, str]:
    text = _text_encoding(data)
    if text is None:
        return False, "the file is binary or not UTF-8 text"
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped:
        return False, "the file is empty"
    if stripped[0] not in "[{":
        return False, "JSON must start with an object or an array"
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return False, "the JSON is malformed"
    if isinstance(parsed, list):
        if not parsed or not all(isinstance(row, dict) for row in parsed):
            return False, "JSON arrays must contain at least one object record"
    elif isinstance(parsed, dict):
        if not parsed:
            return False, "the JSON object is empty"
        values = list(parsed.values())
        if not all(isinstance(value, list) for value in values):
            return False, "JSON objects must map column names to value arrays"
        if not all(len(value) == len(values[0]) for value in values):
            return False, "JSON column arrays must have equal lengths"
        if not values[0]:
            return False, "the JSON object has no records"
    else:  # pragma: no cover - defensive; json only returns list/dict/scalars
        return False, "JSON must contain records"
    return True, ""


def _looks_like_xlsx(data: bytes, limits: GuardLimits) -> tuple[bool, str]:
    if not data.startswith(b"PK\x03\x04"):
        return False, "an .xlsx file must be a ZIP container"
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            if len(infos) > limits.max_zip_entries:
                return False, "the workbook contains too many archive entries"
            total = 0
            names: set[str] = set()
            for info in infos:
                name = info.filename
                if name.startswith(("/", "\\")) or ".." in name.replace(
                    "\\", "/"
                ).split("/"):
                    return False, "the workbook contains an unsafe archive path"
                if info.flag_bits & 0x1:
                    return False, "encrypted workbooks are not supported"
                total += info.file_size
                if total > limits.max_uncompressed_bytes:
                    return False, "the workbook expands beyond the decompression limit"
                names.add(name)
            if "[Content_Types].xml" not in names or not any(
                name.startswith("xl/") for name in names
            ):
                return False, "the ZIP container is not an Excel workbook"
    except (zipfile.BadZipFile, OSError):
        return False, "the workbook is corrupt"
    return True, ""


def _looks_like_parquet(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"PAR1" and data[-4:] == b"PAR1"


def _looks_like_feather(data: bytes) -> bool:
    return data.startswith((b"FEA1", b"ARROW1"))


def sniff_upload(upload: GuardedUpload, limits: GuardLimits) -> None:
    """Verify that the bytes match the claimed format, not just the suffix."""
    data = upload.data
    suffix = upload.suffix
    if suffix == ".csv":
        ok, reason = _looks_like_csv(data)
    elif suffix == ".json":
        ok, reason = _looks_like_json(data)
    elif suffix == ".xlsx":
        ok, reason = _looks_like_xlsx(data, limits)
    elif suffix == ".parquet":
        ok, reason = (_looks_like_parquet(data), "the file is not a Parquet file")
    elif suffix == ".feather":
        ok, reason = (_looks_like_feather(data), "the file is not a Feather file")
    else:  # pragma: no cover - guarded by guard_upload
        ok, reason = False, "unsupported format"
    if not ok:
        raise UploadRejected(
            f"The upload does not look like a valid {suffix.lstrip('.').upper()} file: {reason}.",
            status_code=415,
            code="content_mismatch",
        )


def guard_upload(
    filename: str | None,
    data: bytes,
    limits: GuardLimits,
    content_type: str = "",
) -> GuardedUpload:
    """Sanitise, size-check and content-sniff one uploaded file."""
    display = sanitize_filename(filename, limits.max_name_chars)
    suffix = _suffix(display)
    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join("." + item for item in limits.formats)
        raise UploadRejected(
            f"Unsupported file type. This deployment accepts: {supported}.",
            status_code=415,
            code="unsupported_format",
        )
    if not limits.allows(suffix):
        supported = ", ".join("." + item for item in limits.formats)
        raise UploadRejected(
            f"This deployment has not enabled {suffix} uploads. Accepted formats: {supported}.",
            status_code=415,
            code="format_disabled",
        )
    module = SUPPORTED_FORMATS[suffix][1]
    if not _module_available(module):
        raise UploadRejected(
            f"{suffix} uploads are unavailable on this server because its optional parser is not installed.",
            status_code=415,
            code="format_unavailable",
        )
    if not data:
        raise UploadRejected("The uploaded file is empty.", 422, "empty_upload")
    if len(data) > limits.max_bytes:
        mib = limits.max_bytes / (1024 * 1024)
        raise UploadRejected(
            f"The upload is larger than the {mib:.1f} MiB limit.",
            status_code=413,
            code="upload_too_large",
        )
    upload = GuardedUpload(
        data=bytes(data),
        filename=display,
        suffix=suffix,
        content_type=(content_type or "")[:120],
    )
    sniff_upload(upload, limits)
    return upload


def _safe_cell(value: Any) -> Any:
    """Convert one dataframe cell to bounded JSON-friendly data."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except (TypeError, ValueError):
            return str(value)
    text = str(value)
    return text[:_MAX_CELL_CHARS]


def _column_metadata(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = max(len(df), 1)
    for name in df.columns:
        series = df[name]
        out.append(
            {
                "name": str(name)[:_MAX_COLUMN_NAME_CHARS],
                "dtype": str(series.dtype)[:40],
                "null_rate": round(float(series.isna().sum()) / rows, 4),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return out


def describe_dataset(df: pd.DataFrame, limits: GuardLimits) -> dict[str, Any]:
    """Return the bounded metadata and preview used by ``POST /inspect``."""
    preview = [
        {str(column): _safe_cell(value) for column, value in row.items()}
        for row in df.head(_PREVIEW_ROWS).to_dict(orient="records")
    ]
    columns = _column_metadata(df)
    candidates = [
        item["name"]
        for item in columns
        if 1 < item["unique"] <= 20
        and item["unique"] < max(2, int(len(df) * 0.95))
        and item["null_rate"] < 0.5
    ][:12]
    return {
        "rows": len(df),
        "columns": columns,
        "preview": preview,
        "target_candidates": candidates,
        "limits": limits.to_public_dict(),
    }


def validate_dataframe(
    df: pd.DataFrame,
    limits: GuardLimits,
    target: str | None = None,
    *,
    require_target: bool = False,
) -> None:
    """Enforce row/column/cell limits and safe column names on a parsed frame."""
    rows, cols = int(df.shape[0]), int(df.shape[1])
    if rows < 1:
        raise UploadRejected("The dataset has no data rows.", 422, "empty_dataset")
    if rows > limits.max_rows:
        raise UploadRejected(
            f"The dataset has {rows:,} rows; this server allows at most {limits.max_rows:,}.",
            status_code=413,
            code="too_many_rows",
        )
    if cols > limits.max_columns:
        raise UploadRejected(
            f"The dataset has {cols:,} columns; this server allows at most {limits.max_columns:,}.",
            status_code=413,
            code="too_many_columns",
        )
    if rows * cols > limits.max_cells:
        raise UploadRejected(
            f"The dataset is larger than the {limits.max_cells:,}-cell limit.",
            status_code=413,
            code="too_many_cells",
        )
    seen: set[str] = set()
    for raw in df.columns:
        name = str(raw)
        if not name or _CONTROL_NAME.search(name):
            raise UploadRejected(
                "The dataset contains a column name with control characters.",
                422,
                "bad_column_name",
            )
        if len(name) > _MAX_COLUMN_NAME_CHARS:
            raise UploadRejected(
                f"Column names must be at most {_MAX_COLUMN_NAME_CHARS} characters.",
                422,
                "bad_column_name",
            )
        if name in seen:
            raise UploadRejected(
                "The dataset contains duplicate column names.",
                422,
                "duplicate_columns",
            )
        seen.add(name)
    if cols < 2 and require_target:
        raise UploadRejected(
            "A dataset audit needs at least two columns (features and a target).",
            422,
            "too_few_columns",
        )
    if require_target:
        if not target:
            raise UploadRejected(
                "Choose a target column before scanning.", 422, "missing_target"
            )
        if len(target) > _MAX_COLUMN_NAME_CHARS or _CONTROL_NAME.search(target):
            raise UploadRejected("The target name is not valid.", 422, "bad_target")
        if target not in seen:
            raise UploadRejected(
                "The target column was not found. Inspect the file to list its columns.",
                422,
                "unknown_target",
            )
        series = df[target]
        if int(series.nunique(dropna=True)) < 2:
            raise UploadRejected(
                "The target column needs at least two distinct values.",
                422,
                "constant_target",
            )
        if int(series.notna().sum()) < 2:
            raise UploadRejected(
                "The target column has too few labelled rows.",
                422,
                "empty_target",
            )


def read_guarded_dataset(
    upload: GuardedUpload,
    limits: GuardLimits,
    target: str | None = None,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    """Parse a guarded upload with bounded rows and validate the frame."""
    try:
        df = read_dataset(
            io.BytesIO(upload.data),
            upload.filename,
            max_rows=limits.max_rows,
        )
    except DatasetLimitError as exc:
        raise UploadRejected(str(exc), 413, "dataset_limit") from None
    except ImportError:
        raise UploadRejected(
            f"{upload.suffix} parsing is not installed on this server.",
            415,
            "format_unavailable",
        ) from None
    except (ValueError, TypeError, KeyError, OverflowError, OSError):
        raise UploadRejected(
            f"The {upload.suffix.lstrip('.').upper()} file could not be parsed.",
            422,
            "parse_failed",
        ) from None
    except Exception:  # noqa: BLE001 - fail closed without leaking parser internals
        raise UploadRejected(
            "The uploaded file could not be parsed safely.",
            422,
            "parse_failed",
        ) from None
    if not isinstance(df, pd.DataFrame):
        raise UploadRejected(
            "The upload did not produce a tabular dataset.", 422, "not_tabular"
        )
    validate_dataframe(df, limits, target, require_target=require_target)
    return df


def available_formats(limits: GuardLimits) -> list[dict[str, Any]]:
    """Describe which configured formats can actually be parsed here."""
    out: list[dict[str, Any]] = []
    for suffix, (label, module, extra) in SUPPORTED_FORMATS.items():
        enabled = limits.allows(suffix)
        installed = _module_available(module)
        out.append(
            {
                "extension": suffix,
                "label": label,
                "enabled": bool(enabled and installed),
                "configured": enabled,
                "installed": installed,
                "extra": extra,
            }
        )
    return out
