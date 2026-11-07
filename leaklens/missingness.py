"""Conservative, split-aware missingness audit.

The audit looks for patterns where *why* a value is missing may carry
information the model would not legitimately have at prediction time:

* missingness associated with the target (a missing-value indicator can encode
  the outcome or a post-outcome process);
* missingness that differs substantially between train and test;
* missingness that differs across entities (``--group``).

All calculations are deterministic, use only pandas/numpy, and never copy raw
rows, cell values or group identifiers into findings. Ordinary missingness is
not reported; every signal needs both a minimum effect size and a minimum
sample size.

Statistical limitations: the numeric-target check is a point-biserial Pearson
correlation (it detects linear association only), and the categorical checks
use a two-proportion z approximation. There is no multiple-comparison
correction; instead the default effect-size floors and ``z >= 3`` keep the
false-positive rate low. A finding is a reason to investigate, not proof of a
leak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .evidence import Evidence
from .leakage import Finding

MIN_Z = 3.0
MAX_GROUPS = 500
_SAFE_LIMIT = 300


@dataclass(frozen=True, slots=True)
class MissingnessSettings:
    """Thresholds for the missingness audit (documented in docs/CONFIGURATION.md)."""

    enabled: bool = True
    min_rows: int = 200
    min_group_rows: int = 30
    min_missing: int = 30
    min_rate_delta: float = 0.15
    min_group_delta: float = 0.25
    min_correlation: float = 0.30


DEFAULT_SETTINGS = MissingnessSettings()


@dataclass(frozen=True, slots=True)
class MissingnessReport:
    """Findings plus an explicit account of what was checked and skipped."""

    status: str
    rows: int
    findings: tuple[Finding, ...] = ()
    checked_columns: int = 0
    target_checked: bool = False
    split_checked: bool = False
    group_checked: bool = False
    skipped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rows": self.rows,
            "checked_columns": self.checked_columns,
            "target_checked": self.target_checked,
            "split_checked": self.split_checked,
            "group_checked": self.group_checked,
            "skipped": {key: self.skipped[key] for key in sorted(self.skipped)},
            "findings": len(self.findings),
        }


def _safe(value: Any) -> str:
    text = " ".join(str(value).split())
    return text[:_SAFE_LIMIT]


def _mask(series: pd.Series) -> np.ndarray:
    return series.isna().to_numpy(dtype=bool)


def _two_proportion_z(m1: int, n1: int, m2: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p1, p2 = m1 / n1, m2 / n2
    pooled = (m1 + m2) / (n1 + n2)
    variance = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if variance <= 0:
        return 0.0 if p1 == p2 else math.inf
    return abs(p1 - p2) / math.sqrt(variance)


def _skip_reason(
    series: pd.Series, mask: np.ndarray, settings: MissingnessSettings
) -> str | None:
    if not mask.any():
        return "no_missing"
    if mask.all():
        return "all_missing"
    try:
        if series.nunique(dropna=True) <= 1:
            return "constant"
    except TypeError:  # unhashable values; treat as non-constant
        pass
    if int(mask.sum()) < settings.min_missing:
        return "too_few_missing"
    if int((~mask).sum()) < settings.min_group_rows:
        return "too_few_present"
    return None


def _finding(
    kind: str,
    severity: str,
    column: Any,
    detail: str,
    *,
    source: str,
    description: str,
    observed: str,
    expected: str,
) -> Finding:
    location = _safe(column)
    return Finding(
        severity,
        kind,
        str(column),
        detail,
        evidence=(
            Evidence(
                source=source,
                kind="statistic",
                description=description,
                location=location or None,
                observed=observed,
                expected=expected,
            ),
        ),
    )


def _target_findings(
    df: pd.DataFrame,
    target: str,
    features: list[Any],
    settings: MissingnessSettings,
    skipped: dict[str, int],
) -> tuple[list[Finding], bool]:
    y = df[target]
    valid = y.notna().to_numpy(dtype=bool)
    if int(valid.sum()) < settings.min_rows:
        skipped["target_too_few_rows"] = skipped.get("target_too_few_rows", 0) + 1
        return [], False
    numeric = pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20
    if numeric:
        values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        usable = valid & np.isfinite(values)
        classes: np.ndarray | None = None
    else:
        labels = y[valid].astype("string")
        codes, uniques = pd.factorize(labels, sort=True)
        if len(uniques) < 2:
            skipped["target_constant"] = skipped.get("target_constant", 0) + 1
            return [], False
        classes = codes
        usable = valid
        values = np.zeros(0)

    findings: list[Finding] = []
    for column in features:
        series = df[column]
        mask = _mask(series)
        reason = _skip_reason(series, mask, settings)
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        indicator = mask[usable]
        rows = int(usable.sum())
        if int(indicator.sum()) < settings.min_missing:
            skipped["too_few_missing"] = skipped.get("too_few_missing", 0) + 1
            continue
        if classes is None:
            target_values = values[usable]
            if target_values.size and float(np.std(target_values)) == 0.0:
                skipped["target_constant"] = skipped.get("target_constant", 0) + 1
                continue
            correlation = float(
                np.corrcoef(indicator.astype(float), target_values)[0, 1]
            )
            if (
                not np.isfinite(correlation)
                or abs(correlation) < settings.min_correlation
            ):
                continue
            mean_missing = float(target_values[indicator].mean())
            mean_present = float(target_values[~indicator].mean())
            kind = "missingness_target_signal"
            findings.append(
                _finding(
                    kind,
                    "medium",
                    column,
                    f"missingness is associated with the target (point-biserial "
                    f"r={correlation:.2f}); a missing-value indicator may encode "
                    f"information that is unavailable at prediction time",
                    source="data",
                    description="missingness is correlated with the numeric target",
                    observed=(
                        f"r={correlation:.2f}; target mean {mean_missing:.3f} when "
                        f"missing vs {mean_present:.3f} when present; "
                        f"{int(indicator.sum())} of {rows} rows missing"
                    ),
                    expected=(
                        f"|r| < {settings.min_correlation:.2f} and no target "
                        "association"
                    ),
                )
            )
            continue

        rates: list[tuple[int, int, float]] = []
        for code in np.unique(classes):
            selected = classes == code
            size = int(selected.sum())
            if size < settings.min_group_rows:
                continue
            missing = int(indicator[selected].sum())
            rates.append((missing, size, missing / size))
        if len(rates) < 2:
            skipped["target_classes_too_small"] = (
                skipped.get("target_classes_too_small", 0) + 1
            )
            continue
        rates.sort(key=lambda item: item[2])
        low, high = rates[0], rates[-1]
        delta = high[2] - low[2]
        z = _two_proportion_z(high[0], high[1], low[0], low[1])
        if delta < settings.min_rate_delta or z < MIN_Z:
            continue
        kind = "missingness_target_signal"
        findings.append(
            _finding(
                kind,
                "medium",
                column,
                f"missingness differs across target classes ({low[2]:.1%} vs "
                f"{high[2]:.1%}, delta {delta:.1%}); a missing-value indicator may "
                f"encode the target, so check whether the value is missing because "
                f"of the outcome or the labelling process",
                source="data",
                description="missingness differs across target classes",
                observed=(
                    f"missing rate {low[2]:.1%} vs {high[2]:.1%} across target "
                    f"classes (delta {delta:.1%}, z={z:.1f}); "
                    f"{int(indicator.sum())} of {rows} rows missing"
                ),
                expected=(
                    f"missing-rate difference < {settings.min_rate_delta:.0%} "
                    f"and z < {MIN_Z:g}"
                ),
            )
        )
    return findings, True


def _split_findings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    group: str | None,
    settings: MissingnessSettings,
    skipped: dict[str, int],
) -> list[Finding]:
    findings: list[Finding] = []
    shared = [
        column
        for column in train.columns
        if column in test.columns and column != target and column != group
    ]
    for column in shared:
        left = _mask(train[column])
        right = _mask(test[column])
        if len(left) < settings.min_group_rows or len(right) < settings.min_group_rows:
            skipped["split_too_small"] = skipped.get("split_too_small", 0) + 1
            continue
        rate_train = float(left.mean())
        rate_test = float(right.mean())
        delta = abs(rate_train - rate_test)
        missing_total = int(left.sum()) + int(right.sum())
        populated = max(int((~left).sum()), int((~right).sum()))
        complete_shift = (
            (rate_train == 1.0 and rate_test == 0.0)
            or (rate_train == 0.0 and rate_test == 1.0)
        ) and populated >= settings.min_missing
        if not complete_shift and (
            delta < settings.min_rate_delta or missing_total < settings.min_missing
        ):
            continue
        z = _two_proportion_z(int(left.sum()), len(left), int(right.sum()), len(right))
        if not complete_shift and z < MIN_Z:
            continue
        observed = f"train {rate_train:.1%} missing vs test {rate_test:.1%} missing" + (
            "; one split has no values at all"
            if complete_shift
            else f" (delta {delta:.1%}, z={z:.1f})"
        )
        kind = "missingness_split_shift"
        findings.append(
            _finding(
                kind,
                "medium",
                column,
                f"missingness differs between train ({rate_train:.1%}) and test "
                f"({rate_test:.1%}); if the split is random this suggests a "
                f"collection or join difference, and a missing indicator may behave "
                f"differently in production",
                source="split",
                description="missingness differs between the train and test splits",
                observed=observed,
                expected=(
                    f"missing-rate difference < {settings.min_rate_delta:.0%} "
                    f"and z < {MIN_Z:g}"
                ),
            )
        )
    return findings


def _group_findings(
    df: pd.DataFrame,
    features: list[Any],
    group: Any,
    settings: MissingnessSettings,
    skipped: dict[str, int],
) -> list[Finding]:
    codes, uniques = pd.factorize(df[group], sort=True)
    valid = codes >= 0
    if not valid.any() or len(uniques) < 2:
        skipped["group_too_few_groups"] = skipped.get("group_too_few_groups", 0) + 1
        return []
    sizes = np.bincount(codes[valid])
    qualifying = [
        index for index, size in enumerate(sizes) if size >= settings.min_group_rows
    ]
    if len(qualifying) < 2:
        skipped["group_too_small"] = skipped.get("group_too_small", 0) + 1
        return []
    if len(qualifying) > MAX_GROUPS:
        skipped["too_many_groups"] = skipped.get("too_many_groups", 0) + 1
        return []
    rows = int(valid.sum())
    findings: list[Finding] = []
    for column in features:
        series = df[column]
        mask = _mask(series)
        reason = _skip_reason(series, mask, settings)
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        missing_total = int(mask[valid].sum())
        if missing_total < settings.min_missing:
            skipped["too_few_missing"] = skipped.get("too_few_missing", 0) + 1
            continue
        best: tuple[float, float, float, float, int] | None = None
        for index in qualifying:
            selected = codes == index
            size = int(selected.sum())
            missing = int(mask[selected].sum())
            rest_size = rows - size
            rest_missing = missing_total - missing
            delta = abs(missing / size - rest_missing / rest_size)
            z = _two_proportion_z(missing, size, rest_missing, rest_size)
            if delta < settings.min_group_delta or z < MIN_Z:
                continue
            candidate = (
                delta,
                z,
                missing / size,
                rest_missing / rest_size,
                len(qualifying),
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            continue
        delta, z, rate_group, rate_rest, groups_checked = best
        kind = "missingness_group_signal"
        findings.append(
            _finding(
                kind,
                "low",
                column,
                f"missingness differs across {group} values "
                f"({rate_group:.1%} vs {rate_rest:.1%}); group membership may "
                f"determine whether a value is recorded. Group identifiers are not "
                f"included in this report",
                source="data",
                description="missingness is associated with entity/group membership",
                observed=(
                    f"{groups_checked} groups checked; missing rate differs by "
                    f"{delta:.1%} from the rest (z={z:.1f})"
                ),
                expected=(
                    f"group missing-rate difference < {settings.min_group_delta:.0%} "
                    f"and z < {MIN_Z:g}"
                ),
            )
        )
    return findings


def audit_missingness(
    df: pd.DataFrame,
    target: str,
    *,
    test: pd.DataFrame | None = None,
    group: str | None = None,
    settings: MissingnessSettings | None = None,
) -> MissingnessReport:
    """Run the conservative missingness audit; never mutates ``df`` or ``test``."""
    settings = settings or DEFAULT_SETTINGS
    rows = len(df)
    split_checked = test is not None
    group_checked = group is not None and group in df.columns
    if not settings.enabled:
        return MissingnessReport(
            "disabled", rows, split_checked=split_checked, group_checked=group_checked
        )
    if target not in df.columns:
        return MissingnessReport(
            "no_target", rows, split_checked=split_checked, group_checked=group_checked
        )
    if rows < settings.min_rows:
        return MissingnessReport(
            "insufficient_data",
            rows,
            split_checked=split_checked,
            group_checked=group_checked,
            skipped={"dataset_too_small": 1},
        )
    features = [column for column in df.columns if column != target and column != group]
    skipped: dict[str, int] = {}
    findings: list[Finding] = []
    target_found, target_checked = _target_findings(
        df, target, features, settings, skipped
    )
    findings.extend(target_found)
    if test is not None:
        findings.extend(_split_findings(df, test, target, group, settings, skipped))
    if group_checked:
        findings.extend(_group_findings(df, features, group, settings, skipped))
    return MissingnessReport(
        "ok",
        rows,
        findings=tuple(findings),
        checked_columns=len(features),
        target_checked=target_checked,
        split_checked=split_checked,
        group_checked=group_checked,
        skipped=skipped,
    )
