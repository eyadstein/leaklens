"""The default exit gate ignores diagnostic findings (Phase 13 M9)."""

from __future__ import annotations

from types import SimpleNamespace

from leaklens.cli import _gate_failure
from leaklens.leakage import Finding


def _finding(severity: str, *, diagnostic: bool = False) -> Finding:
    return Finding(severity, "parse_error", "a.py", "detail", diagnostic=diagnostic)


def _outcome(*active: Finding, new: tuple[Finding, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(active=tuple(active), diff=SimpleNamespace(new=new))


def test_high_diagnostic_does_not_fail_the_gate():
    outcome = _outcome(_finding("high", diagnostic=True))
    assert not _gate_failure(outcome, 100, fail_under=None, fail_on_new=False)


def test_high_non_diagnostic_still_fails_the_gate():
    outcome = _outcome(_finding("high"))
    assert _gate_failure(outcome, 100, fail_under=None, fail_on_new=False)


def test_fail_under_still_applies():
    outcome = _outcome(_finding("low"))
    assert _gate_failure(outcome, 40, fail_under=50, fail_on_new=False)
    assert not _gate_failure(outcome, 60, fail_under=50, fail_on_new=False)


def test_new_diagnostic_is_not_a_new_failure():
    outcome = _outcome(new=(_finding("medium", diagnostic=True),))
    assert not _gate_failure(outcome, 100, fail_under=None, fail_on_new=True)


def test_new_non_diagnostic_fails_with_fail_on_new():
    outcome = _outcome(new=(_finding("medium"),))
    assert _gate_failure(outcome, 100, fail_under=None, fail_on_new=True)
    assert not _gate_failure(outcome, 100, fail_under=None, fail_on_new=False)
