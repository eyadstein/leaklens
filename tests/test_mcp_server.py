"""Tests for the optional MCP scanner wrapper (no MCP service required)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from leaklens import mcp_server


class StubFinding:
    def to_dict(self):
        return {"severity": "high", "kind": "preprocess_before_split", "column": "input.py:3", "detail": "fitted on all rows"}


def test_scans_source_without_executing(monkeypatch):
    captured = []
    monkeypatch.setattr(mcp_server, "scan_source", lambda code, name: captured.append((code, name)) or [StubFinding()])
    monkeypatch.setattr(mcp_server, "score", lambda items: 100 - 35 * len(items))
    result = mcp_server.scan_python_code("raise RuntimeError('never execute')", "example.py")
    assert captured == [("raise RuntimeError('never execute')", "example.py")]
    assert result["trust_score"] == 65
    assert result["finding_count"] == 1
    assert result["findings"][0]["kind"] == "preprocess_before_split"


def test_rejects_oversized_source(monkeypatch):
    monkeypatch.setattr(mcp_server, "scan_source", lambda *_: pytest.fail("should not scan"))
    with pytest.raises(ValueError, match="512 KiB"):
        mcp_server.scan_python_code("x" * (512 * 1024 + 1))
    with pytest.raises(ValueError, match="filename"):
        mcp_server.scan_python_code("x", "")


def test_scans_local_python_file(monkeypatch, tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1", encoding="utf-8")
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(tmp_path))
    seen = []
    monkeypatch.setattr(mcp_server, "scan_file", lambda p: seen.append(p) or [])
    monkeypatch.setattr(mcp_server, "score", lambda fs: 100)
    assert mcp_server.scan_local_file("code.py") == {
        "trust_score": 100, "finding_count": 0, "findings": []
    }
    assert seen == [f.resolve()]


def test_scans_notebooks(monkeypatch, tmp_path):
    nb = tmp_path / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(tmp_path))
    monkeypatch.setattr(mcp_server, "scan_file", lambda p: [StubFinding()])
    monkeypatch.setattr(mcp_server, "score", lambda fs: 65)
    assert mcp_server.scan_local_file("notebook.ipynb")["finding_count"] == 1


def test_rejects_escape_and_symlink(monkeypatch, tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text("secret = 1", encoding="utf-8")
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(root))
    with pytest.raises(ValueError, match="outside"):
        mcp_server.scan_local_file(str(outside))
    with pytest.raises(ValueError, match="outside"):
        mcp_server.scan_local_file("../private.py")
    link = root / "link.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        mcp_server.scan_local_file("link.py")


def test_rejects_other_extensions_and_large_file(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(tmp_path))
    other = tmp_path / "credentials.txt"
    other.write_text("password", encoding="utf-8")
    with pytest.raises(ValueError, match="Only existing"):
        mcp_server.scan_local_file("credentials.txt")
    large = tmp_path / "large.py"
    large.write_bytes(b"x" * (mcp_server.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="2 MiB"):
        mcp_server.scan_local_file("large.py")


def test_registers_two_mcp_tools(monkeypatch):
    registered = []
    class FakeMCPServer:
        def __init__(self, name):
            assert name == "LeakLens"
        def tool(self):
            return lambda f: registered.append(f.__name__) or f
    mcp = ModuleType("mcp")
    server = ModuleType("mcp.server")
    server.MCPServer = FakeMCPServer
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", server)
    instance = mcp_server.create_server()
    assert isinstance(instance, FakeMCPServer)
    assert registered == ["scan_python_code", "scan_local_file"]


def test_rejects_invalid_root_and_empty_path(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="existing directory"):
        mcp_server.scan_local_file("code.py")
    monkeypatch.setenv("LEAKLENS_MCP_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        mcp_server.scan_local_file("")


def test_source_byte_limit_counts_unicode(monkeypatch):
    monkeypatch.setattr(mcp_server, "scan_source", lambda *_: pytest.fail("too large"))
    with pytest.raises(ValueError, match="512 KiB"):
        mcp_server.scan_python_code("☕" * (mcp_server.MAX_INPUT_BYTES // 3 + 1))


def test_main_uses_stdio_transport(monkeypatch):
    run_calls = []
    class FakeServer:
        def run(self, **kwargs):
            run_calls.append(kwargs)
    monkeypatch.setattr(mcp_server, "create_server", FakeServer)
    mcp_server.main()
    assert run_calls == [{"transport": "stdio"}]
