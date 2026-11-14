"""The published JSON Schema for the audit report (report-1)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schemas") / "report-1.schema.json"
REPORT_SCHEMA_VERSION = "report-1"
REPORT_SCHEMA_URL = (
    "https://raw.githubusercontent.com/eyadstein/leaklens/main/"
    "leaklens/schemas/report-1.schema.json"
)


@lru_cache(maxsize=1)
def report_schema() -> dict[str, Any]:
    """Return the packaged report schema.

    A missing file means the schema was not packaged with the wheel, which the
    installed CLI cannot fix, so it is reported as such rather than as a bare
    FileNotFoundError.
    """
    try:
        text = SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"report schema is missing from this install: {SCHEMA_PATH}"
        ) from exc
    return json.loads(text)
