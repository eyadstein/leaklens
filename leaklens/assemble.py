"""One payload assembly for the CLI, API and live-scan surfaces (Phase 13 M2).

This module owns the finding/counts/policy sections and nothing else. It calls
``leakage.score()`` and ``policy.apply_policy()`` unchanged; the workspace keeps
its root-relative serializer because that is a different contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .baseline import Baseline
from .leakage import Finding, score
from .policy import PolicyOutcome, apply_policy
from .suppression import SuppressionRule


@dataclass(frozen=True, slots=True)
class Sections:
    outcome: PolicyOutcome
    trust_score: int
    findings: list[dict[str, Any]]
    counts: dict[str, int]
    policy: dict[str, Any]
    resolved: list[dict[str, Any]]


def severity_counts(findings: Sequence[Finding]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        if finding.severity in counts:
            counts[finding.severity] += 1
    counts["total"] = len(findings)
    return counts


def from_outcome(
    outcome: PolicyOutcome,
    *,
    baseline_path: str | Path | None = None,
    fail_on_new: bool = False,
) -> Sections:
    """Serialize one already-applied policy outcome without re-running policy."""
    policy: dict[str, Any] = {}
    resolved: list[dict[str, Any]] = []
    if outcome.diff is not None or outcome.suppression is not None:
        policy = outcome.summary(baseline_path=baseline_path, fail_on_new=fail_on_new)
        if outcome.diff is not None:
            resolved = [entry.to_dict() for entry in outcome.diff.resolved]
    return Sections(
        outcome=outcome,
        trust_score=score(outcome.active),
        findings=[outcome.finding_payload(f) for f in outcome.findings],
        counts=severity_counts(outcome.findings),
        policy=policy,
        resolved=resolved,
    )


def build(
    findings: Sequence[Finding],
    *,
    baseline: Baseline | None = None,
    rules: tuple[SuppressionRule, ...] = (),
    today: date | None = None,
    baseline_path: str | Path | None = None,
    fail_on_new: bool = False,
) -> Sections:
    """Apply policy once and serialize the shared sections."""
    outcome = apply_policy(findings, baseline=baseline, rules=rules, today=today)
    return from_outcome(outcome, baseline_path=baseline_path, fail_on_new=fail_on_new)
