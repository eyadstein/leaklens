"""Opt-in, conservative distribution-drift diagnostics.

These checks compare a training frame with a held-out frame and report columns
whose distribution changed enough to merit investigation. They are diagnostics,
not leakage findings: they are low severity, marked ``diagnostic`` so
``leakage.score()`` and the default CLI exit code ignore them, and they only
run when explicitly enabled (``--drift`` or ``[drift].enabled``).

Methods (numpy/pandas only, deterministic, no SciPy):

* numeric columns: two-sample Kolmogorov-Smirnov statistic;
* categorical columns: total-variation distance over the top-k categories plus
  an aggregate "other" bucket;
* every column: absolute difference in missing-value rate.

Sample floors, column caps and an effect-size floor keep small or noisy
samples quiet. No raw values, category labels or rows are copied into
findings - only aggregate statistics. Drift does not prove leakage: a shift
can come from time, sampling, a filter or a broken join.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .evidence import Evidence
from .leakage import Finding


@dataclass(frozen=True, slots=True)
class DriftSettings:
    """Thresholds for the drift audit (documented in docs/DRIFT.md)."""

    enabled: bool = False
    min_rows: int = 200
    min_effect: float = 0.20
    max_rows: int = 100_000
    max_columns: int = 50
    max_categories: int = 20


DEFAULT_SETTINGS = DriftSettings()


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Findings plus an explicit account of what was checked and skipped."""

    status: str
    rows: int = 0
    test_rows: int = 0
    findings: tuple[Finding, ...] = ()
    checked_columns: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": self.rows,
            "test_rows": self.test_rows,
            "checked_columns": self.checked_columns,
            "skipped": {key: self.skipped[key] for key in sorted(self.skipped)},
            "findings": len(self.findings),
        }


def _bounded(series: pd.Series, max_rows: int) -> pd.Series:
    if len(series) <= max_rows:
        return series
    index = np.linspace(0, len(series) - 1, max_rows).astype(np.int64)
    return series.iloc[index]


def _ks(train: np.ndarray, test: np.ndarray) -> float:
    grid = np.concatenate((train, test))
    train_cdf = np.searchsorted(train, grid, side="right") / len(train)
    test_cdf = np.searchsorted(test, grid, side="right") / len(test)
    return float(np.max(np.abs(train_cdf - test_cdf)))


def _top_k(counts: pd.Series, k: int) -> pd.Series:
    ordered = sorted(counts.items(), key=lambda item: (-float(item[1]), str(item[0])))
    return pd.Series({str(name): float(value) for name, value in ordered[:k]})


def _tvd(train: pd.Series, test: pd.Series, max_categories: int) -> float:
    a = _top_k(train.value_counts(normalize=True), max_categories)
    b = _top_k(test.value_counts(normalize=True), max_categories)
    categories = sorted(set(a.index) | set(b.index))
    return float(
        0.5
        * sum(
            abs(float(a.get(category, 0.0)) - float(b.get(category, 0.0)))
            for category in categories
        )
    )


def _skip(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


def audit_drift(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    target: str | None = None,
    settings: DriftSettings | None = None,
) -> DriftReport:
    """Compare train/test distributions; opt-in and diagnostics-only."""
    settings = settings or DEFAULT_SETTINGS
    if not settings.enabled:
        return DriftReport(status="disabled")
    if test is None or len(test) == 0:
        return DriftReport(status="skipped", rows=len(train))
    rows, test_rows = len(train), len(test)
    if rows < settings.min_rows or test_rows < settings.min_rows:
        return DriftReport(
            status="insufficient_data",
            rows=rows,
            test_rows=test_rows,
            skipped={"small_sample": 1},
        )

    findings: list[Finding] = []
    skipped: dict[str, int] = {}
    checked = 0
    candidates = [c for c in train.columns if c != target and c in test.columns]
    if len(candidates) > settings.max_columns:
        _skip(skipped, "max_columns")
        candidates = candidates[: settings.max_columns]

    for column in candidates:
        left = _bounded(train[column], settings.max_rows)
        right = _bounded(test[column], settings.max_rows)
        train_missing = float(left.isna().mean())
        test_missing = float(right.isna().mean())
        if train_missing == 1.0 and test_missing == 1.0:
            _skip(skipped, "constant")
            continue
        missing_delta = abs(train_missing - test_missing)

        numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(
            right
        )
        numeric = numeric and not pd.api.types.is_bool_dtype(left)
        method = "missingness"
        effect = missing_delta
        if numeric:
            a = left.dropna().to_numpy(dtype=float)
            b = right.dropna().to_numpy(dtype=float)
            if len(a) < settings.min_rows or len(b) < settings.min_rows:
                _skip(skipped, "small_sample")
                continue
            if float(np.ptp(a)) == 0.0 and float(np.ptp(b)) == 0.0:
                _skip(skipped, "constant")
                continue
            ks = _ks(np.sort(a), np.sort(b))
            if ks > effect:
                method, effect = "ks", ks
            checked += 1
        else:
            try:
                tvd = _tvd(left.dropna(), right.dropna(), settings.max_categories)
            except (TypeError, ValueError):
                _skip(skipped, "unsupported")
                continue
            if tvd > effect:
                method, effect = "tvd", tvd
            checked += 1

        if effect < settings.min_effect:
            continue
        findings.append(
            Finding(
                "low",
                "distribution_shift",
                column,
                f"{column!r} differs between train and test ({method}={effect:.2f}); "
                f"investigate the cause before trusting held-out metrics",
                evidence=(
                    Evidence(
                        source="split",
                        kind="statistic",
                        description=(
                            "the train and test distributions differ by more than "
                            "the configured effect floor"
                        ),
                        location=column,
                        observed=f"{method}={effect:.2f}",
                        expected=f"{method} < {settings.min_effect:.2f}",
                    ),
                ),
                diagnostic=True,
            )
        )

    return DriftReport(
        status="ok",
        rows=rows,
        test_rows=test_rows,
        findings=tuple(findings),
        checked_columns=checked,
        skipped=skipped,
    )
