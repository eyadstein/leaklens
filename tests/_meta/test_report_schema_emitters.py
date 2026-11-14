"""Every surface that writes a report names the schema it follows.

The ``$schema`` key is the only thing that lets a consumer find the contract, so
a new report surface that forgets it is a silent regression: the document still
parses, and nobody knows which version to validate against. This reads the
source rather than the behaviour because the surfaces need a dataset to run.
"""

from __future__ import annotations

import re
from pathlib import Path

from leaklens.report_schema import REPORT_SCHEMA_URL

CLI = Path(__file__).resolve().parents[2] / "leaklens" / "cli.py"


def _source() -> str:
    return CLI.read_text(encoding="utf-8")


def test_every_report_payload_names_the_schema() -> None:
    """One stamp per payload: images, text, dl and the data audit."""
    stamps = re.findall(r'"\$schema":\s*REPORT_SCHEMA_URL', _source())

    assert len(stamps) == 4, (
        f"expected 4 report payloads to name the schema, found {len(stamps)}"
    )


def test_the_schema_url_is_pinned_to_a_released_path() -> None:
    """A moved file must move the URL with it, or every report's $schema 404s."""
    assert REPORT_SCHEMA_URL.startswith("https://")
    assert REPORT_SCHEMA_URL.endswith("/leaklens/schemas/report-1.schema.json")


def test_no_payload_hard_codes_the_url() -> None:
    """Import the constant, never paste it: a rename must have one home."""
    assert "raw.githubusercontent.com" not in _source()
