"""Deterministic, labeled benchmark for the opt-in drift diagnostics.

Usage: ``python -m leaklens.bench_drift [--write docs/DRIFT_BENCHMARK.md]``
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .drift import DriftReport, DriftSettings, audit_drift

SETTINGS = DriftSettings(enabled=True)


@dataclass(frozen=True)
class DriftCase:
    name: str
    category: str
    leaky: bool
    run: Callable[[], DriftReport]
    expected: str = "distribution_shift"
    status: str = "ok"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _numeric_shift() -> DriftReport:
    rng = _rng(1)
    train = pd.DataFrame({"x": rng.normal(0.0, 1.0, 500)})
    test = pd.DataFrame({"x": rng.normal(2.0, 1.0, 500)})
    return audit_drift(train, test, settings=SETTINGS)


def _categorical_shift() -> DriftReport:
    train = pd.DataFrame({"segment": ["a"] * 320 + ["b"] * 80})
    test = pd.DataFrame({"segment": ["a"] * 80 + ["b"] * 320})
    return audit_drift(train, test, settings=SETTINGS)


def _missingness_shift() -> DriftReport:
    rng = _rng(2)
    n = 500
    train = pd.DataFrame(
        {"x": np.where(rng.random(n) < 0.05, np.nan, rng.normal(size=n))}
    )
    test = pd.DataFrame(
        {"x": np.where(rng.random(n) < 0.60, np.nan, rng.normal(size=n))}
    )
    return audit_drift(train, test, settings=SETTINGS)


def _identical() -> DriftReport:
    rng = _rng(3)
    values = rng.normal(size=500)
    train = pd.DataFrame({"x": values})
    test = pd.DataFrame({"x": values.copy()})
    return audit_drift(train, test, settings=SETTINGS)


def _small_sample() -> DriftReport:
    rng = _rng(4)
    train = pd.DataFrame({"x": rng.normal(size=50)})
    test = pd.DataFrame({"x": rng.normal(3.0, 1.0, 50)})
    return audit_drift(train, test, settings=SETTINGS)


def _constant() -> DriftReport:
    train = pd.DataFrame({"x": np.ones(500)})
    test = pd.DataFrame({"x": np.ones(500)})
    return audit_drift(train, test, settings=SETTINGS)


def _all_missing() -> DriftReport:
    train = pd.DataFrame({"x": [np.nan] * 500})
    test = pd.DataFrame({"x": [np.nan] * 500})
    return audit_drift(train, test, settings=SETTINGS)


def _high_cardinality() -> DriftReport:
    train = pd.DataFrame({"code": [f"c{i}" for i in range(50)] * 10})
    test = pd.DataFrame({"code": [f"d{i}" for i in range(50)] * 10})
    return audit_drift(train, test, settings=SETTINGS)


def _target_ignored() -> DriftReport:
    rng = _rng(5)
    train = pd.DataFrame({"x": rng.normal(size=500), "y": [0] * 500})
    test = pd.DataFrame({"x": rng.normal(size=500), "y": [1] * 500})
    return audit_drift(train, test, target="y", settings=SETTINGS)


def _without_test() -> DriftReport:
    rng = _rng(6)
    train = pd.DataFrame({"x": rng.normal(size=500)})
    return audit_drift(train, None, settings=SETTINGS)


def _build_cases() -> list[DriftCase]:
    return [
        DriftCase("numeric shift", "numeric", True, _numeric_shift),
        DriftCase("categorical shift", "categorical", True, _categorical_shift),
        DriftCase("missingness shift", "missingness", True, _missingness_shift),
        DriftCase("high cardinality", "categorical", True, _high_cardinality),
        DriftCase("identical distributions", "clean", False, _identical),
        DriftCase(
            "small sample safeguard",
            "clean",
            False,
            _small_sample,
            status="insufficient_data",
        ),
        DriftCase("constant column", "clean", False, _constant),
        DriftCase("all-missing column", "clean", False, _all_missing),
        DriftCase("target column ignored", "clean", False, _target_ignored),
        DriftCase(
            "without test metadata", "clean", False, _without_test, status="skipped"
        ),
    ]


def evaluate() -> dict:
    rows, errors = [], []
    tp = fp = tn = fn = 0
    for case in _build_cases():
        start = time.perf_counter()
        try:
            report = case.run()
        except Exception as exc:  # noqa: BLE001 - benchmark reports the failure
            report = DriftReport(status="error")
            errors.append(f"{case.name}: raised {type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        kinds = {finding.kind for finding in report.findings}
        flagged = "distribution_shift" in kinds
        correct = (
            report.status == case.status
            and flagged == case.leaky
            and (not case.leaky or case.expected in kinds)
        )
        if case.leaky and correct:
            tp += 1
        elif case.leaky:
            fn += 1
            errors.append(f"missed: {case.name} (expected {case.expected})")
        elif not correct:
            fp += 1
            errors.append(f"false alarm: {case.name} -> {sorted(kinds)}")
        else:
            tn += 1
        rows.append(
            {
                "name": case.name,
                "category": case.category,
                "leaky": case.leaky,
                "flagged": flagged,
                "status": report.status,
                "kinds": ", ".join(sorted(kinds)) or "-",
                "ms": elapsed * 1000.0,
            }
        )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    times = [row["ms"] for row in rows]
    return {
        "n": len(rows),
        "leaky": tp + fn,
        "clean": tn + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp / (fp + tn) if fp + tn else 0.0,
        "rows": rows,
        "errors": errors,
        "runtime_total_ms": sum(times),
        "runtime_median_ms": statistics.median(times) if times else 0.0,
        "runtime_max_ms": max(times) if times else 0.0,
    }


def markdown(metrics: dict) -> str:
    table = [
        "| Case | Category | Leaky | Flagged | Status | Findings | ms |",
        "|---|---|:---:|:---:|---|---|---:|",
    ]
    for row in metrics["rows"]:
        table.append(
            f"| {row['name']} | {row['category']} | "
            f"{'yes' if row['leaky'] else 'no'} | "
            f"{'yes' if row['flagged'] else 'no'} | {row['status']} | "
            f"{row['kinds']} | {row['ms']:.1f} |"
        )
    miss = "\n".join(f"- {error}" for error in metrics["errors"]) or "- none"
    return (
        "# Drift benchmark\n\n"
        "Deterministic synthetic train/test pairs for the opt-in drift audit. "
        "Times are wall-clock on the machine that generated this file.\n\n"
        "## Metrics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| Cases | {metrics['n']} |\n"
        f"| Leaky / clean | {metrics['leaky']} / {metrics['clean']} |\n"
        f"| Precision | {metrics['precision']:.2f} |\n"
        f"| Recall | {metrics['recall']:.2f} |\n"
        f"| F1 | {metrics['f1']:.2f} |\n"
        f"| False-positive rate | {metrics['fp_rate']:.2f} |\n"
        f"| True positives | {metrics['tp']} |\n"
        f"| False positives | {metrics['fp']} |\n"
        f"| True negatives | {metrics['tn']} |\n"
        f"| False negatives | {metrics['fn']} |\n"
        f"| Total runtime | {metrics['runtime_total_ms']:.0f} ms |\n"
        f"| Median case | {metrics['runtime_median_ms']:.0f} ms |\n"
        f"| Slowest case | {metrics['runtime_max_ms']:.0f} ms |\n\n"
        "## Cases\n\n" + "\n".join(table) + "\n\n"
        "## Misclassified cases\n\n" + miss + "\n\n"
        "## Caveats\n\n"
        "These are hand-picked synthetic controls with strong effects; they do "
        "not estimate real-world precision or recall. Drift is a diagnostic, not "
        "a leakage finding: it is low severity, excluded from the trust score and "
        "the default exit code, and never modifies the data.\n"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    metrics = evaluate()
    text = markdown(metrics)
    print(text)
    if len(argv) >= 2 and argv[0] == "--write":
        out = Path(argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
    return 1 if metrics["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
