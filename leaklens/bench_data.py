"""Reproducible benchmark for the dataset-level detectors.

The scanner benchmark (``leaklens.bench``) covers code.  This module covers the
data checks: target leakage, identifier columns, categorical target leakage,
duplicate/near-duplicate overlap, group contamination, and temporal boundaries.

Every case is synthetic and deterministic, so precision/recall and runtime are
reproducible without shipping private data.

Usage: ``python -m leaklens.bench_data [--write docs/DATA_BENCHMARK.md]``
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

from .leakage import (
    feature_overlap,
    group_overlap,
    group_risk,
    near_duplicate_overlap,
    scan,
    split_overlap,
    temporal_overlap,
)


@dataclass(frozen=True)
class DataCase:
    name: str
    category: str
    leaky: bool
    finding: Callable[[], list]
    expected: str


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (x + rng.normal(scale=2.0, size=n) > 0).astype(int)
    return pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})


def _scan(df, target="y"):
    return scan(df, target)


def _categorical():
    n = 120
    df = pd.DataFrame(
        {
            "cat": [f"group_{i % 12}" for i in range(n)],
            "noise": np.arange(n) % 7,
            "y": [i % 12 % 2 for i in range(n)],
        }
    )
    return df


def _membership_cases():
    base = _toy()
    train, test = base.iloc[:300], base.iloc[300:]
    return train, test


def _build_cases() -> list[DataCase]:
    clean = _toy(seed=3)
    target = _toy()
    target = target.assign(leak=target["y"].astype(float) * 2.0)

    ids = _toy()
    ids["uid"] = [f"u{i}" for i in range(len(ids))]

    cat = _categorical()

    train, _test = _membership_cases()
    exact_test = train.iloc[:10]
    feature_conflict = pd.DataFrame(
        {
            "x": train["x"].to_numpy()[:5],
            "noise": train["noise"].to_numpy()[:5],
            "y": 1 - train["y"].to_numpy()[:5],
        }
    )
    near_dup = pd.DataFrame(
        {
            "city": ["paris", "LYON ", "Nice"],
            "x": [1.0, 2.0, 3.0],
            "y": [0, 1, 0],
        }
    )
    near_dup_train = pd.DataFrame(
        {
            "city": ["Paris", "Lyon", "Nice"],
            "x": [1.0, 2.00004, 3.0],
            "y": [0, 1, 0],
        }
    )
    clean_test = pd.DataFrame(
        {
            "x": [50.0, 51.0, 52.0],
            "noise": [9.0, 10.0, 11.0],
            "y": [0, 1, 0],
        }
    )
    grouped = pd.DataFrame(
        {
            "pid": [1, 1, 2, 2, 3, 3] * 20,
            "x": np.arange(120),
            "y": [0, 1] * 60,
        }
    )
    by_time = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120, freq="D"),
            "x": np.arange(120),
            "y": [0, 1] * 60,
        }
    )

    return [
        DataCase("clean numeric frame", "clean", False, lambda: _scan(clean), ""),
        DataCase(
            "target copied into a feature",
            "target",
            True,
            lambda: _scan(target),
            "target_leakage",
        ),
        DataCase(
            "identifier column", "identifier", True, lambda: _scan(ids), "id_like"
        ),
        DataCase(
            "high-cardinality categorical target encoding",
            "categorical",
            True,
            lambda: _scan(cat),
            "categorical_target_leakage",
        ),
        DataCase(
            "identical feature rows with conflicting labels",
            "duplicates",
            True,
            lambda: feature_overlap(train, feature_conflict, "y"),
            "feature_overlap",
        ),
        DataCase(
            "case/whitespace near duplicates",
            "duplicates",
            True,
            lambda: near_duplicate_overlap(near_dup_train, near_dup, "y"),
            "near_duplicate_overlap",
        ),
        DataCase(
            "exact rows in both splits",
            "duplicates",
            True,
            lambda: split_overlap(train, exact_test),
            "split_overlap",
        ),
        DataCase(
            "disjoint splits",
            "clean",
            False,
            lambda: (
                split_overlap(train, clean_test) + feature_overlap(train, clean_test)
            ),
            "",
        ),
        DataCase(
            "entity repeated across rows",
            "group",
            True,
            lambda: group_risk(grouped, "pid"),
            "group_leakage_risk",
        ),
        DataCase(
            "entity in both splits",
            "group",
            True,
            lambda: group_overlap(grouped.iloc[:40], grouped.iloc[30:], "pid"),
            "group_overlap",
        ),
        DataCase(
            "future rows before earlier rows",
            "temporal",
            True,
            lambda: temporal_overlap(by_time.iloc[60:], by_time.iloc[:60], "date"),
            "temporal_leakage",
        ),
        DataCase(
            "chronological split",
            "temporal",
            False,
            lambda: temporal_overlap(by_time.iloc[:60], by_time.iloc[60:], "date"),
            "",
        ),
    ]


def evaluate() -> dict:
    rows, errors = [], []
    tp = fp = tn = fn = 0
    for case in _build_cases():
        start = time.perf_counter()
        try:
            findings = case.finding()
        except Exception as exc:  # noqa: BLE001 - benchmark reports the failure
            findings = []
            errors.append(f"{case.name}: raised {type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        kinds = {f.kind for f in findings if f.severity in ("high", "medium")}
        flagged = bool(kinds)
        correct = (flagged == case.leaky) and (
            not case.leaky or not case.expected or case.expected in kinds
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
        "rows": rows,
        "errors": errors,
        "runtime_total_ms": sum(times),
        "runtime_median_ms": statistics.median(times) if times else 0.0,
        "runtime_max_ms": max(times) if times else 0.0,
    }


def markdown(metrics: dict) -> str:
    table = [
        "| Case | Category | Leaky | Flagged | Findings | ms |",
        "|---|---|:---:|:---:|---|---:|",
    ]
    for row in metrics["rows"]:
        table.append(
            f"| {row['name']} | {row['category']} | "
            f"{'yes' if row['leaky'] else 'no'} | "
            f"{'yes' if row['flagged'] else 'no'} | {row['kinds']} | "
            f"{row['ms']:.1f} |"
        )
    miss = "\n".join(f"- {error}" for error in metrics["errors"]) or "- none"
    return (
        "# Dataset detector benchmark\n\n"
        "Synthetic, deterministic cases for the dataset-level checks. "
        "Times are wall-clock on the machine that generated this file.\n\n"
        "## Metrics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| Cases | {metrics['n']} |\n"
        f"| Leaky / clean | {metrics['leaky']} / {metrics['clean']} |\n"
        f"| Precision | {metrics['precision']:.2f} |\n"
        f"| Recall | {metrics['recall']:.2f} |\n"
        f"| F1 | {metrics['f1']:.2f} |\n"
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
        "These are hand-written synthetic controls, not a sample of real "
        "datasets. They guard against regressions and make runtime visible; "
        "they do not establish real-world precision or recall.\n"
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
