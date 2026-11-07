"""Apply baselines and suppressions to one audit result without mutating it.

Suppression only labels findings; it never edits the finding, its evidence, its
fingerprint or its severity. The outcome keeps every finding and exposes the
subset that still counts toward the trust score and the default exit code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .baseline import Baseline, BaselineDiff, compare
from .suppression import SuppressionResult, SuppressionRule, evaluate, parse_rules

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from .leakage import Finding


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """Every finding plus the policy metadata derived for one audit."""

    findings: tuple[Finding, ...]
    active: tuple[Finding, ...]
    status: dict[str, str]
    diff: BaselineDiff | None = None
    suppression: SuppressionResult | None = None

    @property
    def suppressed(self) -> frozenset[str]:
        return self.suppression.fingerprints if self.suppression else frozenset()

    def summary(
        self, *, baseline_path: str | Path | None = None, fail_on_new: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.diff is not None:
            payload["baseline"] = {
                "path": str(baseline_path) if baseline_path is not None else None,
                "fail_on_new": bool(fail_on_new),
                **self.diff.counts,
            }
        if self.suppression is not None:
            payload["suppressions"] = self.suppression.to_dict()
        payload["diagnostics"] = (
            list(self.suppression.diagnostics) if self.suppression else []
        )
        return payload

    def finding_payload(self, finding: Finding) -> dict[str, Any]:
        data = finding.to_dict()
        fingerprint = finding.fingerprint
        if self.diff is not None:
            data["status"] = self.status.get(fingerprint, "unchanged")
        elif fingerprint in self.suppressed:
            data["status"] = "suppressed"
        rule = self.suppression.rule_for(fingerprint) if self.suppression else None
        if rule is not None:
            data["suppression"] = rule.to_dict()
        return data


def parse_baseline(payload: Any) -> Baseline | None:
    if payload is None:
        return None
    return Baseline.from_dict(payload)


def parse_suppressions(items: Any) -> tuple[SuppressionRule, ...]:
    return parse_rules(items)


def apply_policy(
    findings: Any,
    *,
    baseline: Baseline | None = None,
    rules: tuple[SuppressionRule, ...] = (),
    today: date | None = None,
) -> PolicyOutcome:
    """Classify findings and suppress exact matches; order is preserved."""
    ordered = tuple(findings)
    suppression = evaluate(rules, ordered, today=today) if rules else None
    diff = compare(baseline, ordered) if baseline is not None else None
    suppressed = suppression.fingerprints if suppression else frozenset()
    status: dict[str, str] = {}
    if diff is not None:
        status.update({finding.fingerprint: "new" for finding in diff.new})
        status.update({finding.fingerprint: "unchanged" for finding in diff.unchanged})
        status.update({finding.fingerprint: "changed" for finding in diff.changed})
    for fingerprint in suppressed:
        status[fingerprint] = "suppressed"
    active = tuple(
        finding for finding in ordered if finding.fingerprint not in suppressed
    )
    return PolicyOutcome(
        findings=ordered,
        active=active,
        status=status,
        diff=diff,
        suppression=suppression,
    )
