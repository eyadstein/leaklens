"""The published ``report-1`` schema describes what the CLI writes.

The schema is frozen: a change that removes or retypes a field needs a new
``report-2`` document. These tests hold both halves of that promise — a
document the CLI can emit validates, and a document it cannot emit is refused.
"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from leaklens.evidence import Evidence
from leaklens.leakage import Finding
from leaklens.report_schema import REPORT_SCHEMA_URL, report_schema


def _validator() -> Draft202012Validator:
    schema = report_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _finding(**over: object) -> dict:
    fields = {
        "severity": "high",
        "kind": "target_leakage",
        "column": "leaked",
        "detail": "The target is encoded in this column.",
        "evidence": (
            Evidence(
                source="data", kind="statistic", description="AUC 0.98 in one column"
            ),
        ),
    }
    fields.update(over)
    return Finding(**fields).to_dict()


def test_schema_is_published_and_identified() -> None:
    schema = report_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == REPORT_SCHEMA_URL
    assert schema["required"] == ["trust_score", "findings"]


def test_a_data_audit_document_validates() -> None:
    """The main ``--json`` payload: trust score and findings, no tally."""
    doc = {
        "$schema": REPORT_SCHEMA_URL,
        "trust_score": 65,
        "findings": [_finding()],
        "audit": {"class_balance": {"leaked": 0.5}},
        "missingness": {"columns": {}},
    }

    _validator().validate(doc)


def test_a_document_with_a_tally_validates() -> None:
    """The text, dl and image reports add the severity tally."""
    doc = {
        "$schema": REPORT_SCHEMA_URL,
        "trust_score": 100,
        "counts": {"high": 0, "medium": 1, "low": 2, "total": 3},
        "findings": [],
        "field": "text",
        "documents": {"train": 10, "test": 2},
        "skipped": {"train": 0, "test": 0},
        "policy": {
            "baseline": {
                "path": None,
                "fail_on_new": False,
                "new": 0,
                "unchanged": 0,
                "changed": 0,
                "resolved": 0,
            },
            "suppressions": {},
            "diagnostics": ["rule refused: reason missing"],
        },
        "resolved": [],
    }

    _validator().validate(doc)


def test_a_policy_annotated_finding_validates() -> None:
    finding = _finding()
    finding["status"] = "suppressed"
    finding["suppression"] = {"reason": "known duplicate", "expires": None}

    _validator().validate({"trust_score": 90, "findings": [finding]})


def test_a_diagnostic_finding_validates() -> None:
    _validator().validate(
        {"trust_score": 95, "findings": [_finding(severity="low", diagnostic=True)]}
    )


@pytest.mark.parametrize(
    "doc",
    [
        pytest.param({"trust_score": 101, "findings": []}, id="score-above-100"),
        pytest.param({"trust_score": -1, "findings": []}, id="score-below-0"),
        pytest.param({"trust_score": 100}, id="no-findings"),
        pytest.param(
            {"trust_score": 100, "findings": [_finding(severity="critical")]},
            id="unknown-severity",
        ),
        pytest.param(
            {"trust_score": 100, "findings": [{**_finding(), "diagnostic": False}]},
            id="diagnostic-false",
        ),
        pytest.param(
            {"trust_score": 100, "findings": [{**_finding(), "status": "resolved"}]},
            id="unknown-status",
        ),
        pytest.param(
            {"trust_score": 100, "findings": [{**_finding(), "untracked": 1}]},
            id="unknown-finding-field",
        ),
        pytest.param(
            {"trust_score": 100, "counts": {"high": 0, "medium": 0, "low": 0}},
            id="tally-without-total",
        ),
    ],
)
def test_a_document_the_cli_cannot_write_is_refused(doc: dict) -> None:
    with pytest.raises(ValidationError):
        _validator().validate(doc)


def test_the_finding_keys_the_schema_requires_are_the_ones_serialised() -> None:
    schema = report_schema()
    finding = schema["$defs"]["finding"]

    assert sorted(finding["required"]) == sorted(
        Finding("low", "k", "c", "d").to_dict()
    )


def test_a_missing_schema_file_is_reported_as_an_install_problem(
    monkeypatch, tmp_path
) -> None:
    from leaklens import report_schema as module

    monkeypatch.setattr(module, "SCHEMA_PATH", tmp_path / "absent.json")
    module.report_schema.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="missing from this install"):
            module.report_schema()
    finally:
        module.report_schema.cache_clear()


def test_the_schema_ships_inside_the_package() -> None:
    """Guards the package-data entry: the file must live beside the module."""
    from pathlib import Path

    from leaklens import report_schema as module

    assert module.SCHEMA_PATH.is_file()
    assert module.SCHEMA_PATH.parent == Path(module.__file__).parent / "schemas"
