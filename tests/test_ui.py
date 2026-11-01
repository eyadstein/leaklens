"""Checks the Streamlit UI wiring without opening a browser."""

import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from leaklens.fixes import advice
from leaklens.leakage import Finding


@pytest.fixture
def ui(monkeypatch):
    fake_st = ModuleType("streamlit")
    for name in (
        "set_page_config", "title", "caption", "tabs", "columns", "metric",
        "file_uploader", "text_area", "dataframe", "success", "markdown",
        "write", "download_button", "json",
    ):
        setattr(fake_st, name, Mock())

    fake_st.tabs.return_value = [nullcontext(), nullcontext(), nullcontext()]
    fake_st.file_uploader.return_value = None
    fake_st.text_area.return_value = ""
    fake_st.columns.return_value = (Mock(), Mock())
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    path = Path(__file__).resolve().parents[1] / "leaklens" / "ui.py"
    show = runpy.run_path(str(path))["show"]
    return show, fake_st


def test_ui_shows_fix_tip_and_downloads_dataset_report(ui):
    show, st = ui
    finding = Finding("high", "target_leakage", "suspicious_column", "looks like a label")
    show([finding], key="dataset", audit_data={"rows": 42, "cols": 2})

    st.markdown.assert_called_once()
    assert "target_leakage" in st.markdown.call_args.args[0]
    st.write.assert_called_once_with(advice("target_leakage"))

    kwargs = st.download_button.call_args.kwargs
    assert kwargs["file_name"] == "leaklens-report.html"
    assert kwargs["mime"] == "text/html"
    assert kwargs["key"] == "dataset_report"
    assert "65/100" in kwargs["data"]
    assert "suspicious_column" in kwargs["data"]
    assert "<b>rows</b> 42" in kwargs["data"]


def test_ui_download_works_with_no_findings(ui):
    show, st = ui
    show([], key="code")

    st.success.assert_called_once_with("No leakage found.")
    st.markdown.assert_not_called()
    kwargs = st.download_button.call_args.kwargs
    assert kwargs["key"] == "code_report"
    assert "100/100" in kwargs["data"]
    assert "No leakage found." in kwargs["data"]


def test_ui_skips_missing_fix_tip_but_keeps_report(ui):
    show, st = ui
    show([Finding("low", "unrecognized_kind", "example", "details")], key="code")

    st.markdown.assert_not_called()
    st.write.assert_not_called()
    assert "unrecognized_kind" in st.download_button.call_args.kwargs["data"]
