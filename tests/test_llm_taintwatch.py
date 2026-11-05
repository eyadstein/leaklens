"""TaintWatch optional-guard tests for leaklens/llm.py.

All tests run without the optional ``taintwatch`` dependency installed:
the guard is either disabled (default), or a fake guard object is injected
directly. Nothing here imports taintwatch.
"""

import sys
import types

import pytest

from leaklens import llm
from leaklens.leakage import Finding


class _FakeGuard:
    def __init__(self, *, blocks=False):
        self.blocks = blocks
        self.calls = []

    def check_call(self, sink_name, args):
        self.calls.append((sink_name, args))
        if self.blocks:
            raise RuntimeError("blocked by policy")


def _finding():
    return Finding("high", "target_leakage", "x.csv", "chargeback_flag predicts target")


def test_guard_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LEAKLENS_TAINTWATCH", raising=False)
    assert llm._taintwatch_guard() is None


def test_guard_disabled_unless_exactly_one(monkeypatch):
    for val in ("0", "true", "yes", "True", ""):
        monkeypatch.setenv("LEAKLENS_TAINTWATCH", val)
        assert llm._taintwatch_guard() is None, val


def test_guard_none_when_taintwatch_not_installed(monkeypatch):
    monkeypatch.setenv("LEAKLENS_TAINTWATCH", "1")
    # Ensure the import fails even if taintwatch happens to be installed.
    monkeypatch.setitem(sys.modules, "taintwatch", None)
    monkeypatch.setitem(sys.modules, "taintwatch.guard", None)
    assert llm._taintwatch_guard() is None


def test_check_noop_when_guard_none():
    # Must not raise and must not touch anything.
    llm._check_with_taintwatch(None, "http_post", "payload")


def test_check_allows_clean_payload():
    guard = _FakeGuard(blocks=False)
    llm._check_with_taintwatch(guard, "http_post", "finding: high target_leakage")
    assert guard.calls and guard.calls[0][0] == "http_post"


def test_check_blocks_confidential_payload():
    guard = _FakeGuard(blocks=True)
    with pytest.raises(llm.LLMError, match="TaintWatch blocked"):
        llm._check_with_taintwatch(guard, "http_post", "secret dataset row values")


def test_check_passes_llmerror_through():
    class _ErrGuard:
        def check_call(self, sink_name, args):
            raise llm.LLMError("custom")

    with pytest.raises(llm.LLMError, match="custom"):
        llm._check_with_taintwatch(_ErrGuard(), "http_post", "x")


def test_guard_from_fake_module(monkeypatch):
    """When LEAKLENS_TAINTWATCH=1 and taintwatch is importable, guard is used."""
    monkeypatch.setenv("LEAKLENS_TAINTWATCH", "1")
    fake_guard_mod = types.ModuleType("taintwatch.guard")

    class FakeGuardClass:
        @classmethod
        def default(cls):
            return _FakeGuard()

    fake_guard_mod.Guard = FakeGuardClass
    fake_pkg = types.ModuleType("taintwatch")
    monkeypatch.setitem(sys.modules, "taintwatch", fake_pkg)
    monkeypatch.setitem(sys.modules, "taintwatch.guard", fake_guard_mod)
    guard = llm._taintwatch_guard()
    assert isinstance(guard, _FakeGuard)
