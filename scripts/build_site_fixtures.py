"""Generate the browser demo fixtures from the real LeakLens scanners.

The website cannot run Python, so ``docs/demo.html`` replays results that were
produced by the actual scanners at build time.  Nothing is hand-written: every
severity, evidence string and remediation comes from ``leaklens.scanner``,
``leaklens.leakage`` and ``leaklens.fixes``.

Usage:
    python scripts/build_site_fixtures.py           # write docs/assets/fixtures.mjs
    python scripts/build_site_fixtures.py --check   # fail if the file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leaklens import __version__  # noqa: E402
from leaklens.fixes import advice  # noqa: E402
from leaklens.leakage import (  # noqa: E402
    feature_overlap,
    near_duplicate_overlap,
    scan,
    score,
    split_overlap,
)
from leaklens.scanner import scan_source  # noqa: E402

OUT = ROOT / "docs" / "assets" / "fixtures.mjs"
EXAMPLES = ROOT / "examples"

SAFE_PIPELINE = """from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Split first: the held-out rows are untouched until the final score.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Every learned transform lives inside the pipeline, so it is fitted per fold.
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
"""


def _line_of(column: str) -> int | None:
    tail = column.rsplit(":", 1)
    if len(tail) == 2 and tail[1].isdigit():
        return int(tail[1])
    return None


def _finding(finding) -> dict:
    return {
        "severity": finding.severity,
        "kind": finding.kind,
        "column": finding.column,
        "detail": finding.detail,
        "fix": advice(finding.kind),
        "line": _line_of(finding.column),
    }


def _counts(findings) -> dict:
    return {
        "high": sum(f.severity == "high" for f in findings),
        "medium": sum(f.severity == "medium" for f in findings),
        "low": sum(f.severity == "low" for f in findings),
    }


def _code_fixture(
    fixture_id: str,
    filename: str,
    title: str,
    description: str,
    source: str,
    command: str,
) -> dict:
    findings = scan_source(source, filename)
    serialized = [_finding(f) for f in findings]
    return {
        "id": fixture_id,
        "group": "code",
        "name": filename,
        "title": title,
        "description": description,
        "language": "python",
        "source": source,
        "command": command,
        "findings": serialized,
        "counts": _counts(findings),
        "clean": not serialized,
        "trustScore": score(findings),
    }


def make_transactions(n: int = 600, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = np.round(rng.lognormal(3, 1, n), 2)
    hour = rng.integers(0, 24, n)
    distance = np.round(rng.exponential(20, n), 2)
    signal = 0.8 * np.log(amount) + 0.05 * distance + (hour < 5) * 1.2 - 7
    fraud = (rng.random(n) < 1 / (1 + np.exp(-signal))).astype(int)
    return pd.DataFrame(
        {
            "txn_id": [f"t{i:05d}" for i in range(n)],
            "amount": amount,
            "hour": hour,
            "dist_home_km": distance,
            "channel": rng.choice(["web", "app", "pos"], n),
            "chargeback_flag": fraud * 1.0,
            "is_fraud": fraud,
        }
    )


def _preview(frame: pd.DataFrame, rows: int = 6) -> list[dict]:
    return json.loads(frame.head(rows).to_json(orient="records"))


def _dataset_fixture(
    fixture_id: str,
    title: str,
    description: str,
    frame: pd.DataFrame,
    target: str,
    command: str,
) -> dict:
    findings = scan(frame, target)
    serialized = [_finding(f) for f in findings]
    return {
        "id": fixture_id,
        "group": "dataset",
        "name": f"{fixture_id}.csv",
        "title": title,
        "description": description,
        "target": target,
        "rows": len(frame),
        "columns": list(frame.columns),
        "preview": _preview(frame),
        "command": command,
        "findings": serialized,
        "counts": _counts(findings),
        "clean": not serialized,
        "trustScore": score(findings),
    }


def _normalised(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[features].copy()
    for column in features:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = pd.to_numeric(out[column], errors="coerce").round(4)
        else:
            out[column] = out[column].astype(str).str.strip().str.lower()
    return out


def _overlap_plot(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Project real rows into a 3-D field; mark the rows that also appear in train."""
    features = [
        c for c in ("amount", "hour", "dist_home_km", "channel") if c in train.columns
    ]
    train_hash = pd.util.hash_pandas_object(_normalised(train, features), index=False)
    test_hash = pd.util.hash_pandas_object(_normalised(test, features), index=False)
    hash_to_train = {int(h): i for i, h in enumerate(train_hash)}

    pairs: list[tuple[int, int]] = []
    for test_index, value in enumerate(test_hash):
        original = hash_to_train.get(int(value))
        if original is not None:
            pairs.append((test_index, original))

    duplicate_tests = [test_index for test_index, _ in pairs][:20]
    duplicate_trains: list[int] = []
    for test_index, original in pairs[:20]:
        if original not in duplicate_trains:
            duplicate_trains.append(original)

    test_idx = (
        duplicate_tests
        + [i for i in range(len(test)) if i not in duplicate_tests][
            : 90 - len(duplicate_tests)
        ]
    )
    train_idx = (
        duplicate_trains
        + [i for i in range(len(train)) if i not in duplicate_trains][
            : 90 - len(duplicate_trains)
        ]
    )

    amount_max = float(np.log1p(train["amount"]).max()) or 1.0
    dist_max = float(min(train["dist_home_km"].max(), 60.0)) or 1.0

    def point(row, index, duplicated=False, pair=None):
        return {
            "x": round(float(np.log1p(row["amount"]) / amount_max), 4),
            "y": round(float(row["hour"]) / 23.0, 4),
            "z": round(float(min(row["dist_home_km"], 60.0) / dist_max), 4),
            "dup": bool(duplicated),
            "pair": pair,
            "row": int(index),
        }

    train_slot = {}
    plot_train = []
    for slot, index in enumerate(train_idx):
        row = train.iloc[index]
        duplicated = index in duplicate_trains
        plot_train.append(point(row, index, duplicated))
        if duplicated:
            train_slot[int(index)] = slot

    pair_by_test = {test_index: original for test_index, original in pairs}
    plot_test = []
    for index in test_idx:
        row = test.iloc[index]
        original = pair_by_test.get(index)
        slot = train_slot.get(original) if original is not None else None
        plot_test.append(point(row, index, slot is not None, slot))

    duplicates = sum(1 for p in plot_test if p["dup"])
    return {
        "labels": {"x": "log amount", "y": "hour", "z": "distance"},
        "train": plot_train,
        "test": plot_test,
        "counts": {
            "train": len(plot_train),
            "test": len(plot_test),
            "duplicates": duplicates,
        },
        "caption": (
            "Actual rows from the checked-in split fixture. Red points appear in "
            "both train and test; the lines show the same record on both sides."
        ),
    }


def _overlap_fixture(
    fixture_id: str,
    title: str,
    description: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    command: str,
) -> dict:
    exact = split_overlap(train, test)
    if exact:
        findings = exact
        check = "exact row overlap"
    else:
        shared = feature_overlap(train, test, "is_fraud")
        if shared:
            findings = shared
            check = "feature-vector overlap"
        else:
            findings = near_duplicate_overlap(train, test, "is_fraud")
            check = "normalised near-duplicate overlap"
    serialized = [_finding(f) for f in findings]
    return {
        "id": fixture_id,
        "group": "overlap",
        "name": f"{fixture_id}.csv",
        "title": title,
        "description": description,
        "target": "is_fraud",
        "rows": int(len(train) + len(test)),
        "columns": list(train.columns),
        "preview": _preview(test),
        "command": command,
        "findings": serialized,
        "counts": _counts(findings),
        "clean": not serialized,
        "trustScore": score(findings),
        "check": check,
        "split": {"train": len(train), "test": len(test)},
        "plot": _overlap_plot(train, test),
    }


def build() -> dict:
    transactions = make_transactions()
    clean = transactions.drop(columns=["txn_id", "chargeback_flag"])

    train = transactions.iloc[:420].copy()
    test = transactions.iloc[400:].copy()

    near_train = transactions.iloc[:120].copy()
    near_test = transactions.iloc[120:220].copy()
    shifted = transactions.iloc[:8].copy()
    shifted["amount"] = shifted["amount"] + 0.00001
    shifted["channel"] = shifted["channel"].str.upper() + " "
    near_test = pd.concat([near_test, shifted], ignore_index=True)
    near_test = near_test[
        ["amount", "hour", "dist_home_km", "channel", "chargeback_flag", "is_fraud"]
    ]
    near_train = near_train[
        ["amount", "hour", "dist_home_km", "channel", "chargeback_flag", "is_fraud"]
    ]

    code = [
        _code_fixture(
            "advanced",
            "advanced_leaks.py",
            "Five leaks in one pipeline",
            "Target encoding, feature selection, tuning, time ordering and group "
            "CV, all in a single realistic script.",
            (EXAMPLES / "advanced_leaks.py").read_text(encoding="utf-8"),
            "leaklens --scan examples/advanced_leaks.py --explain",
        ),
        _code_fixture(
            "unsafe",
            "bad_pipeline.py",
            "Preprocessing before the split",
            "Scaling and SMOTE run before train_test_split, so the held-out rows "
            "shape the training data.",
            (EXAMPLES / "bad_pipeline.py").read_text(encoding="utf-8"),
            "leaklens --scan examples/bad_pipeline.py --explain",
        ),
        _code_fixture(
            "safe",
            "safe_pipeline.py",
            "The corrected version",
            "Split first, keep every learned transform inside a pipeline, and touch "
            "the test set once.",
            SAFE_PIPELINE,
            "leaklens --scan examples/safe_pipeline.py --explain",
        ),
    ]

    datasets = [
        _dataset_fixture(
            "leaky_transactions",
            "Leaky transaction table",
            "A copied target column and a unique identifier inflate the trust score "
            "to 50/100. This is the shape of most '99% accuracy' demos.",
            transactions,
            "is_fraud",
            "leaklens leaky_transactions.csv --target is_fraud",
        ),
        _dataset_fixture(
            "clean_transactions",
            "Same table, leaks removed",
            "Drop the copied target and the identifier: the audit is clean and the "
            "trust score reaches 100/100.",
            clean,
            "is_fraud",
            "leaklens clean_transactions.csv --target is_fraud",
        ),
    ]

    overlaps = [
        _overlap_fixture(
            "overlap_exact",
            "Exact rows on both sides",
            "The same transactions appear in train and test. The audit reports the "
            "overlap and the CLI exits with code 1.",
            train,
            test,
            "leaklens train.csv --test test.csv --target is_fraud",
        ),
        _overlap_fixture(
            "overlap_near",
            "Near duplicates after normalisation",
            "Exact matching finds nothing, but case, whitespace and decimal noise "
            "hide the same rows. The normalised check catches them.",
            near_train,
            near_test,
            "leaklens train.csv --test test.csv --target is_fraud",
        ),
    ]

    return {
        "meta": {
            "product": "LeakLens",
            "version": __version__,
            "generatedBy": "scripts/build_site_fixtures.py",
            "scanner": "leaklens.scanner.scan_source",
            "dataset": "leaklens.leakage.scan / score",
            "duplicateChecks": "leaklens.leakage.split_overlap / feature_overlap / near_duplicate_overlap",
            "prerecorded": True,
            "note": (
                "Every finding on the demo page is real output from the Python "
                "scanners, generated at build time. The browser never executes "
                "LeakLens; run the CLI for a live audit."
            ),
        },
        "code": code,
        "datasets": datasets,
        "overlaps": overlaps,
    }


def render(payload: dict) -> str:
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    return (
        "// Generated by scripts/build_site_fixtures.py -- do not edit by hand.\n"
        "// Findings come from the real LeakLens scanners at build time.\n"
        "// Regenerate with: python scripts/build_site.py\n"
        f"export const fixtures = {body};\n"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = render(build())
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"stale fixtures: {OUT.relative_to(ROOT)}")
            return 1
        print(f"fixtures up to date: {OUT.relative_to(ROOT)}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
