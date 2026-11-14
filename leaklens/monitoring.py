"""Opt-in, read-only intake for caller-supplied monitoring reports (Phase 18 M5).

A report is a JSON object::

    {"kind": "drift" | "skew",
     "metrics": [{"name": "amount", "value": 0.42, "threshold": 0.2}]}

``breached`` may replace ``threshold``. Files are size-capped, parsed with the
standard library and resolved inside the working root. Every finding is
diagnostic: it never changes ``leakage.score()`` or the exit gate. A missing,
malformed, oversized or traversing report is counted as a skip, never raised.
No network, database or monitoring service is contacted.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evidence import Evidence
from .leakage import Finding
from .paths import PathOutsideRoot, resolve_within

MAX_REPORT_BYTES = 1_000_000
MAX_METRICS = 500
MAX_NAME = 200


@dataclass(frozen=True, slots=True)
class MonitoringReport:
    """Findings plus an explicit account of which reports were skipped."""

    status: str
    findings: tuple[Finding, ...] = ()
    reports: int = 0
    metrics: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reports": self.reports,
            "metrics": self.metrics,
            "findings": len(self.findings),
            "skipped": {key: self.skipped[key] for key in sorted(self.skipped)},
        }


def _skip(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _valid_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or len(name) > MAX_NAME:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        return None
    return name


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _load(resolved: Path) -> tuple[dict | None, str | None]:
    try:
        if not resolved.is_file():
            return None, "missing"
        with resolved.open("rb") as handle:
            raw = handle.read(MAX_REPORT_BYTES + 1)
    except OSError:
        return None, "unreadable"
    if len(raw) > MAX_REPORT_BYTES:
        return None, "oversized"
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None, "malformed"
    if not isinstance(data, dict):
        return None, "malformed"
    return data, None


def _finding(
    rel: str, report_kind: str, name: str, value: float, threshold: float | None
) -> Finding:
    kind = "monitoring_skew_report"
    if report_kind == "drift":
        kind = "monitoring_drift_report"
    observed = f"{value:g}"
    expected = f"<= {threshold:g}" if threshold is not None else "not breached"
    description = _clip(f"{rel} reports {name}={observed} for {report_kind}", 480)
    return Finding(
        "low",
        kind,
        name,
        _clip(
            f"monitoring report {rel} flags {name}={observed}; "
            "investigate the cause before trusting the deployed model",
            280,
        ),
        evidence=(
            Evidence(
                source="data",
                kind="statistic",
                description=description,
                location=_clip(f"{rel}:{name}", 300),
                observed=observed,
                expected=expected,
            ),
        ),
        diagnostic=True,
    )


def audit_monitoring(
    paths: Sequence[str], *, root: Path | None = None
) -> MonitoringReport:
    """Read caller-supplied drift/skew reports; diagnostics only, never raise."""

    if not paths:
        return MonitoringReport(status="disabled")
    root = Path(root) if root is not None else Path.cwd()
    base = Path(os.path.realpath(str(root)))
    findings: list[Finding] = []
    skipped: dict[str, int] = {}
    reports = 0
    evaluated = 0

    for value in paths:
        try:
            resolved = resolve_within(root, value)
        except PathOutsideRoot:
            _skip(skipped, "outside_root")
            continue
        rel = Path(os.path.relpath(str(resolved), str(base))).as_posix()
        data, reason = _load(resolved)
        if data is None:
            _skip(skipped, reason or "malformed")
            continue
        report_kind = data.get("kind")
        if report_kind not in ("drift", "skew"):
            _skip(skipped, "unsupported_kind")
            continue
        metrics = data.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            _skip(skipped, "malformed_metrics")
            continue
        if len(metrics) > MAX_METRICS:
            _skip(skipped, "too_many_metrics")
            metrics = metrics[:MAX_METRICS]

        reports += 1
        for metric in metrics:
            if not isinstance(metric, dict):
                _skip(skipped, "invalid_metric")
                continue
            name = _valid_name(metric.get("name"))
            value = _number(metric.get("value"))
            if name is None or value is None:
                _skip(skipped, "invalid_metric")
                continue
            threshold_raw = metric.get("threshold")
            threshold = None if threshold_raw is None else _number(threshold_raw)
            if threshold_raw is not None and threshold is None:
                _skip(skipped, "invalid_metric")
                continue
            flag = metric.get("breached")
            if flag is not None and type(flag) is not bool:
                _skip(skipped, "invalid_metric")
                continue
            killed: bool | None = flag
            if killed is None:
                killed = value > threshold if threshold is not None else None
            if killed is None:
                _skip(skipped, "no_breach_signal")
                continue
            evaluated += 1
            if killed:
                findings.append(_finding(rel, report_kind, name, value, threshold))

    return MonitoringReport(
        status="ok" if reports else "skipped",
        findings=tuple(findings),
        reports=reports,
        metrics=evaluated,
        skipped=skipped,
    )
