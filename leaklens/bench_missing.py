"""Synthetic, labeled benchmark for the missingness audit.

Cases are deterministic and deliberately strong (or deliberately clean) so the
benchmark guards against regressions instead of estimating real-world accuracy.
All findings count, including the deliberately low-severity group signal.

Usage: ``python -m leaklens.bench_missing [--write docs/MISSINGNESS_BENCHMARK.md]``
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

from .missingness import audit_missingness


@dataclass(frozen=True)
class MissingCase:
    name: str
    category: str
    leaky: bool
    run: Callable[[], object]
    expected: str


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _random_target(n: int, seed: int) -> np.ndarray:
    return _rng(seed).integers(0, 2, n)


def _target_categorical() -> object:
    rng = _rng(1)
    n = 800
    y = rng.integers(0, 2, n)
    threshold = np.where(y == 1, 0.70, 0.03)
    missing = rng.random(n) < threshold
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    return audit_missingness(frame, "y")


def _target_string_class() -> object:
    rng = _rng(2)
    n = 800
    y = np.where(rng.integers(0, 2, n) == 1, "yes", "no")
    threshold = np.where(y == "yes", 0.65, 0.02)
    missing = rng.random(n) < threshold
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame({"x": x, "y": y})
    return audit_missingness(frame, "y")


def _target_numeric() -> object:
    rng = _rng(3)
    n = 600
    missing = rng.random(n) < 0.35
    y = 2.0 * missing + rng.normal(scale=1.0, size=n)
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": y})
    return audit_missingness(frame, "y")


def _split_shift(train_missing: float, test_missing: float, seed: int) -> object:
    rng = _rng(seed)
    n = 400
    train = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < train_missing, np.nan, rng.normal(size=n)),
            "noise": rng.normal(size=n),
            "y": _random_target(n, seed + 1),
        }
    )
    test = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < test_missing, np.nan, rng.normal(size=n)),
            "noise": rng.normal(size=n),
            "y": _random_target(n, seed + 2),
        }
    )
    return audit_missingness(train, "y", test=test)


def _one_side_empty() -> object:
    rng = _rng(6)
    n = 400
    train = pd.DataFrame(
        {
            "x": np.full(n, np.nan),
            "noise": rng.normal(size=n),
            "y": _random_target(n, 7),
        }
    )
    test = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "y": _random_target(n, 8),
        }
    )
    return audit_missingness(train, "y", test=test)


def _group_signal() -> object:
    rng = _rng(9)
    n = 600
    groups = np.where(np.arange(n) % 2 == 0, "A", "B")
    rate = np.where(groups == "A", 0.03, 0.60)
    missing = rng.random(n) < rate
    x = rng.normal(size=n)
    x[missing] = np.nan
    frame = pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=n),
            "site": groups,
            "y": _random_target(n, 10),
        }
    )
    return audit_missingness(frame, "y", group="site")


def _clean_random_missingness() -> object:
    rng = _rng(11)
    n = 400
    train = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "z": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "y": _random_target(n, 12),
        }
    )
    test = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "z": np.where(rng.random(n) < 0.10, np.nan, rng.normal(size=n)),
            "y": _random_target(n, 13),
        }
    )
    return audit_missingness(train, "y", test=test)


def _small_sample() -> object:
    rng = _rng(14)
    n = 80
    y = np.array([1] * 40 + [0] * 40)
    x = np.where(y == 1, np.nan, rng.normal(size=n))
    frame = pd.DataFrame({"x": x, "y": y})
    return audit_missingness(frame, "y")


def _constant_and_empty() -> object:
    rng = _rng(15)
    n = 400
    frame = pd.DataFrame(
        {
            "constant": np.full(n, 7.0),
            "empty": np.full(n, np.nan),
            "noise": rng.normal(size=n),
            "y": _random_target(n, 16),
        }
    )
    return audit_missingness(frame, "y")


def _shift_without_test() -> object:
    rng = _rng(17)
    n = 400
    frame = pd.DataFrame(
        {
            "x": np.where(rng.random(n) < 0.60, np.nan, rng.normal(size=n)),
            "y": _random_target(n, 18),
        }
    )
    return audit_missingness(frame, "y")


def _build_cases() -> list[MissingCase]:
    return [
        MissingCase(
            "binary target association",
            "target",
            True,
            _target_categorical,
            "missingness_target_signal",
        ),
        MissingCase(
            "string target association",
            "target",
            True,
            _target_string_class,
            "missingness_target_signal",
        ),
        MissingCase(
            "numeric target association",
            "target",
            True,
            _target_numeric,
            "missingness_target_signal",
        ),
        MissingCase(
            "train and test missingness differ",
            "split",
            True,
            lambda: _split_shift(0.05, 0.60, 4),
            "missingness_split_shift",
        ),
        MissingCase(
            "feature entirely absent from train",
            "split",
            True,
            _one_side_empty,
            "missingness_split_shift",
        ),
        MissingCase(
            "group-associated missingness",
            "group",
            True,
            _group_signal,
            "missingness_group_signal",
        ),
        MissingCase(
            "ordinary random missingness",
            "clean",
            False,
            _clean_random_missingness,
            "",
        ),
        MissingCase(
            "small sample safeguard",
            "clean",
            False,
            _small_sample,
            "",
        ),
        MissingCase(
            "constant and entirely missing columns",
            "clean",
            False,
            _constant_and_empty,
            "",
        ),
        MissingCase(
            "shift without test metadata",
            "clean",
            False,
            _shift_without_test,
            "",
        ),
    ]


def evaluate() -> dict:
    rows, errors = [], []
    tp = fp = tn = fn = 0
    for case in _build_cases():
        start = time.perf_counter()
        try:
            report = case.run()
            findings = report.findings
        except Exception as exc:  # noqa: BLE001 - benchmark reports the failure
            findings = []
            errors.append(f"{case.name}: raised {type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - start
        kinds = {finding.kind for finding in findings}
        flagged = bool(findings)
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
        "fp_rate": fp / (fp + tn) if fp + tn else 0.0,
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
        "# Missingness benchmark\n\n"
        "Synthetic, deterministic cases for the split-aware missingness audit. "
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
        "These are hand-written synthetic controls with deliberately strong "
        "signals; they do not estimate real-world precision or recall. The "
        "detector reports patterns that merit investigation, not confirmed "
        "leaks, and it never modifies the data or the split.\n"
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
