"""Availability contract: one base error, one probe (ADR-023)."""

from __future__ import annotations

import pytest

from leaklens import images, sqlscan
from leaklens.availability import EngineUnavailable, probe


def test_probe_returns_the_module():
    assert probe("json", "images", purpose="tests").loads("{}") == {}


def test_probe_missing_module_names_the_extra():
    with pytest.raises(EngineUnavailable, match=r'pip install "leaklens-ml\[images\]"'):
        probe("leaklens_missing_module_for_tests", "images", purpose="image checks")


def test_probe_uses_the_requested_error_class():
    with pytest.raises(images.ImageUnavailable, match="Pillow"):
        probe(
            "leaklens_missing_module_for_tests",
            "images",
            purpose="image checks",
            label="Pillow",
            error=images.ImageUnavailable,
        )


def test_unavailable_errors_stay_import_errors():
    assert issubclass(EngineUnavailable, ImportError)
    assert issubclass(sqlscan.SqlUnavailable, ImportError)
    assert issubclass(images.ImageUnavailable, ImportError)
    assert issubclass(sqlscan.SqlUnavailable, EngineUnavailable)
    assert issubclass(images.ImageUnavailable, EngineUnavailable)
