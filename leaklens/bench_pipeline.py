"""Labelled, synthetic benchmark for the pipeline/MLOps engine (Phase 14 M7).

Every case is a small repository written to a temporary directory and audited
with ``audit_pipeline``. Truth is the exact ``(kind, column)`` set each case
should emit, so a miss and a false alarm are both visible and precision/recall
can be reported per finding family. These are hand-written controls, not a
sample of real repositories: the numbers are a regression guard, never an
accuracy claim.

Usage: ``python -m leaklens.bench_pipeline [--write docs/PIPELINE_BENCHMARK.md]``
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from .pipeline import audit_pipeline


@dataclass(frozen=True)
class PipelineCase:
    name: str
    family: str
    files: dict[str, str]
    expected: frozenset[tuple[str, str]] = frozenset()
    formats: tuple[str, ...] = ()
    expect_skipped: str | None = None


def _data_files(*names: str) -> dict[str, str]:
    return {name: "a,b\n1,2\n" for name in names}


CLEAN_DVC = """stages:
  split:
    cmd: python split.py
    deps: [data/raw.csv]
    outs: [data/train.csv, data/eval.csv]
  train:
    cmd: python train.py
    deps: [data/train.csv]
    params:
    - params.yaml:
      - seed
      - n_estimators
    outs: [model.pkl]
  evaluate:
    cmd: python evaluate.py
    deps: [data/eval.csv, model.pkl]
    outs: [metrics.json]
"""


def _clean_dvc() -> dict[str, str]:
    return {
        "dvc.yaml": CLEAN_DVC,
        "params.yaml": "seed: 42\nn_estimators: 100\n",
        **_data_files("data/raw.csv", "data/train.csv", "data/eval.csv"),
    }


def _shared_input() -> dict[str, str]:
    return {
        "dvc.yaml": (
            "stages:\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: [data/all.csv]\n"
            "    params:\n"
            "    - params.yaml:\n"
            "      - seed\n"
            "    outs: [model.pkl]\n"
            "  evaluate:\n"
            "    cmd: python evaluate.py\n"
            "    deps: [data/all.csv, model.pkl]\n"
            "    outs: [metrics.json]\n"
        ),
        "params.yaml": "seed: 42\n",
        **_data_files("data/all.csv"),
    }


def _eval_reads_train_output() -> dict[str, str]:
    return {
        "dvc.yaml": (
            "stages:\n"
            "  split:\n"
            "    cmd: python split.py\n"
            "    deps: [data/raw.csv]\n"
            "    outs: [data/train.csv, data/eval.csv]\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: [data/train.csv]\n"
            "    params:\n"
            "    - params.yaml:\n"
            "      - seed\n"
            "    outs: [data/train_features.csv]\n"
            "  evaluate:\n"
            "    cmd: python evaluate.py\n"
            "    deps: [data/train_features.csv]\n"
            "    outs: [metrics.json]\n"
        ),
        "params.yaml": "seed: 42\n",
        **_data_files("data/raw.csv", "data/train.csv", "data/eval.csv"),
    }


def _unresolved() -> dict[str, str]:
    return {
        "dvc.yaml": (
            "stages:\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: [data/missing.csv]\n"
            "    params:\n"
            "    - params.yaml:\n"
            "      - seed\n"
            "    outs: [model.pkl]\n"
        ),
        "params.yaml": "seed: 42\n",
        **_data_files("data/train.csv"),
    }


def _dag_clean() -> dict[str, str]:
    return {
        "pipeline.dag.json": json.dumps(
            {
                "stages": [
                    {
                        "name": "split",
                        "deps": ["data/raw.csv"],
                        "outs": ["data/train.csv", "data/eval.csv"],
                    },
                    {
                        "name": "train",
                        "deps": ["data/train.csv"],
                        "outs": ["model.pkl"],
                    },
                    {
                        "name": "evaluate",
                        "deps": ["data/eval.csv"],
                        "outs": ["metrics.json"],
                    },
                ]
            }
        ),
        **_data_files("data/raw.csv", "data/train.csv", "data/eval.csv"),
    }


def _dag_unresolved() -> dict[str, str]:
    return {
        "pipeline.dag.json": json.dumps(
            {
                "stages": [
                    {
                        "name": "train",
                        "deps": ["data/absent.parquet"],
                        "outs": ["model.pkl"],
                    }
                ]
            }
        ),
        **_data_files("data/train.csv"),
    }


def _lock(params: dict[str, object]) -> dict[str, str]:
    lines = [
        "stages:",
        "  train:",
        "    cmd: python train.py",
        "    deps:",
        "    - path: data/train.csv",
        "      md5: 11111111111111111111111111111111",
        "    params:",
        "      params.yaml:",
    ]
    lines.extend(f"        {key}: {value}" for key, value in params.items())
    lines.extend(
        [
            "    outs:",
            "    - path: model.pkl",
            "      md5: 22222222222222222222222222222222",
        ]
    )
    return {"dvc.lock": "\n".join(lines) + "\n"}


def _lock_drift() -> dict[str, str]:
    return {
        **_lock({"seed": 42}),
        "params.yaml": "seed: 7\n",
        **_data_files("data/train.csv"),
    }


def _lock_matching() -> dict[str, str]:
    return {
        **_lock({"seed": 7}),
        "params.yaml": "seed: 7\n",
        **_data_files("data/train.csv"),
    }


def _lock_missing_key() -> dict[str, str]:
    return {
        **_lock({"seed": 7, "max_depth": 5}),
        "params.yaml": "seed: 7\n",
        **_data_files("data/train.csv"),
    }


def _lock_missing_digest() -> dict[str, str]:
    return {
        "dvc.lock": (
            "stages:\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: []\n"
            "    outs:\n"
            "    - path: model.pkl\n"
        ),
        **_data_files("data/train.csv"),
    }


def _dvc_file() -> dict[str, str]:
    return {
        "features.dvc": (
            "outs:\n"
            "- md5: 33333333333333333333333333333333\n"
            "  path: data/features.csv\n"
            "deps:\n"
            "- path: data/raw.csv\n"
        ),
        **_data_files("data/raw.csv", "data/features.csv"),
    }


def _unpinned_two_files() -> dict[str, str]:
    return {
        "requirements.txt": "pandas>=2\nscikit-learn==1.5.0\n",
        "requirements/dev.txt": "black\n",
    }


def _unpinned_no_version() -> dict[str, str]:
    return {"requirements.txt": "pandas\nnumpy==2.0.0\n"}


def _pinned_only() -> dict[str, str]:
    return {"requirements.txt": "pandas==2.2.1\nnumpy==2.0.0\n"}


def _training_without_seed() -> dict[str, str]:
    return {
        "dvc.yaml": (
            "stages:\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: [data/train.csv]\n"
            "    params:\n"
            "    - params.yaml:\n"
            "      - n_estimators\n"
            "      - max_depth\n"
            "    outs: [model.pkl]\n"
        ),
        "params.yaml": "n_estimators: 100\nmax_depth: 6\n",
        **_data_files("data/train.csv"),
    }


def _training_with_seed() -> dict[str, str]:
    return {
        "dvc.yaml": (
            "stages:\n"
            "  train:\n"
            "    cmd: python train.py\n"
            "    deps: [data/train.csv]\n"
            "    params:\n"
            "    - params.yaml:\n"
            "      - seed\n"
            "      - n_estimators\n"
            "    outs: [model.pkl]\n"
        ),
        "params.yaml": "seed: 42\nn_estimators: 100\n",
        **_data_files("data/train.csv"),
    }


def _notebook(source_by_count: list[tuple[int, str]]) -> str:
    cells = [
        {
            "cell_type": "code",
            "execution_count": count,
            "source": source,
        }
        for count, source in source_by_count
    ]
    return json.dumps({"cells": cells})


def _notebook_order() -> dict[str, str]:
    return {
        "pipeline.ipynb": _notebook(
            [
                (1, 'import pandas as pd\ndf = pd.read_csv("data/raw.csv")'),
                (2, 'df.to_csv("data/features.csv", index=False)'),
                (3, 'pd.read_csv("data/features.csv")'),
            ]
        ),
        **_data_files("data/raw.csv"),
    }


def _notebook_unresolved() -> dict[str, str]:
    return {
        "pipeline.ipynb": _notebook([(1, 'pd.read_csv("data/absent.csv")')]),
    }


def _malformed_yaml() -> dict[str, str]:
    return {"dvc.yaml": "stages: [unclosed\n  train: {cmd: ["}


def _malformed_json() -> dict[str, str]:
    return {"pipeline.dag.json": '{"stages": ['}


def _malformed_notebook() -> dict[str, str]:
    return {"pipeline.ipynb": '{"cells": [}'}


def _empty() -> dict[str, str]:
    return {}


def _build_cases() -> list[PipelineCase]:
    return [
        PipelineCase(
            "clean three stage dvc pipeline",
            "clean",
            _clean_dvc(),
            formats=("dvc",),
        ),
        PipelineCase(
            "train and eval share one dataset",
            "pipeline_split_overlap",
            _shared_input(),
            frozenset({("pipeline_split_overlap", "data/all.csv")}),
            ("dvc",),
        ),
        PipelineCase(
            "evaluation reads the training artifact",
            "pipeline_split_overlap",
            _eval_reads_train_output(),
            frozenset({("pipeline_split_overlap", "data/train_features.csv")}),
            ("dvc",),
        ),
        PipelineCase(
            "disjoint split in dag json",
            "clean",
            _dag_clean(),
            formats=("dag_json",),
        ),
        PipelineCase(
            "dag json dependency has no producer",
            "pipeline_unresolved_lineage",
            _dag_unresolved(),
            frozenset({("pipeline_unresolved_lineage", "data/absent.parquet")}),
            ("dag_json",),
        ),
        PipelineCase(
            "unresolved stage dependency",
            "pipeline_unresolved_lineage",
            _unresolved(),
            frozenset({("pipeline_unresolved_lineage", "data/missing.csv")}),
            ("dvc",),
        ),
        PipelineCase(
            "notebook reads a missing artifact",
            "pipeline_unresolved_lineage",
            _notebook_unresolved(),
            frozenset({("pipeline_unresolved_lineage", "data/absent.csv")}),
            ("notebook",),
        ),
        PipelineCase(
            "locked parameter drift",
            "pipeline_param_drift",
            _lock_drift(),
            frozenset({("pipeline_param_drift", "params.yaml:seed")}),
            ("dvc_lock",),
        ),
        PipelineCase(
            "locked parameter key removed",
            "pipeline_param_drift",
            _lock_missing_key(),
            frozenset({("pipeline_param_drift", "params.yaml:max_depth")}),
            ("dvc_lock",),
        ),
        PipelineCase(
            "locked parameters still match",
            "clean",
            _lock_matching(),
            formats=("dvc_lock",),
        ),
        PipelineCase(
            "locked output without digest",
            "pipeline_missing_digest",
            _lock_missing_digest(),
            frozenset({("pipeline_missing_digest", "model.pkl")}),
            ("dvc_lock",),
        ),
        PipelineCase(
            "unpinned requirements entry",
            "pipeline_unpinned_dependency",
            _unpinned_two_files(),
            frozenset(
                {
                    ("pipeline_unpinned_dependency", "requirements.txt"),
                    ("pipeline_unpinned_dependency", "requirements/dev.txt"),
                }
            ),
            ("requirements",),
        ),
        PipelineCase(
            "requirements without any version",
            "pipeline_unpinned_dependency",
            _unpinned_no_version(),
            frozenset({("pipeline_unpinned_dependency", "requirements.txt")}),
            ("requirements",),
        ),
        PipelineCase(
            "pinned requirements only",
            "clean",
            _pinned_only(),
            formats=("requirements",),
        ),
        PipelineCase(
            "training stage without a seed",
            "no_random_state",
            _training_without_seed(),
            frozenset({("no_random_state", "dvc.yaml:train")}),
            ("dvc",),
        ),
        PipelineCase(
            "seed makes the training stage reproducible",
            "clean",
            _training_with_seed(),
            formats=("dvc",),
        ),
        PipelineCase(
            "notebook cells in execution order",
            "clean",
            _notebook_order(),
            formats=("notebook",),
        ),
        PipelineCase(
            "dvc file declares outputs",
            "clean",
            _dvc_file(),
            formats=("dvc_file",),
        ),
        PipelineCase(
            "malformed dvc yaml is a skip",
            "clean",
            _malformed_yaml(),
            formats=("malformed",),
            expect_skipped="malformed_yaml",
        ),
        PipelineCase(
            "malformed dag json is a skip",
            "clean",
            _malformed_json(),
            formats=("malformed",),
            expect_skipped="malformed_json",
        ),
        PipelineCase(
            "malformed notebook is a skip",
            "clean",
            _malformed_notebook(),
            formats=("malformed",),
            expect_skipped="malformed_notebook",
        ),
        PipelineCase("empty repository", "clean", _empty(), formats=("empty",)),
    ]


def _family_metrics(
    pairs: list[tuple[frozenset[tuple[str, str]], set[tuple[str, str]]]],
) -> dict[str, dict[str, int]]:
    families = {kind for expected, _ in pairs for kind, _ in expected}
    families |= {kind for _, predicted in pairs for kind, _ in predicted}
    out = {}
    for family in sorted(families):
        tp = fp = fn = 0
        for expected, predicted in pairs:
            mine = {item for item in predicted if item[0] == family}
            theirs = {item for item in expected if item[0] == family}
            tp += len(mine & theirs)
            fp += len(mine - theirs)
            fn += len(theirs - mine)
        out[family] = {"tp": tp, "fp": fp, "fn": fn}
    return out


def _complete_evidence(finding) -> bool:
    return bool(finding.evidence) and all(
        getattr(finding.evidence[0], name) is not None
        for name in ("location", "observed", "expected")
    )


def evaluate() -> dict:
    rows = []
    errors = []
    pairs: list[tuple[frozenset[tuple[str, str]], set[tuple[str, str]]]] = []
    cases = _build_cases()
    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, text in case.files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            tracemalloc.start()
            start = time.perf_counter()
            try:
                report = audit_pipeline(root)
                elapsed = time.perf_counter() - start
                _current, peak = tracemalloc.get_traced_memory()
            except Exception as exc:  # noqa: BLE001 - benchmark reports failures
                errors.append(f"{case.name}: raised {type(exc).__name__}: {exc}")
                continue
            finally:
                tracemalloc.stop()
        predicted = {(finding.kind, finding.column) for finding in report.findings}
        pairs.append((case.expected, predicted))
        missing = case.expected - predicted
        extra = predicted - case.expected
        if missing:
            errors.append(f"missed: {case.name} -> {sorted(missing)}")
        if extra:
            errors.append(f"false alarm: {case.name} -> {sorted(extra)}")
        if case.expect_skipped and case.expect_skipped not in report.skipped:
            errors.append(
                f"{case.name}: expected skip {case.expect_skipped!r}, "
                f"got {sorted(report.skipped)}"
            )
        complete = sum(1 for finding in report.findings if _complete_evidence(finding))
        rows.append(
            {
                "name": case.name,
                "family": case.family,
                "formats": ", ".join(case.formats) or "none",
                "status": report.status,
                "findings": ", ".join(
                    f"{kind}:{column}" for kind, column in sorted(predicted)
                )
                or "-",
                "skipped": ", ".join(
                    f"{reason}={count}" for reason, count in report.skipped.items()
                )
                or "-",
                "evidence": f"{complete}/{len(report.findings)}"
                if report.findings
                else "-",
                "ms": elapsed * 1000.0,
                "peak_kib": peak / 1024.0,
            }
        )

    per_family = _family_metrics(pairs)
    tp = sum(row["tp"] for row in per_family.values())
    fp = sum(row["fp"] for row in per_family.values())
    fn = sum(row["fn"] for row in per_family.values())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    formats = sorted({fmt for case in cases for fmt in case.formats})
    total_findings = sum(len(predicted) for _, predicted in pairs)
    with_evidence = sum(
        int(row["evidence"].split("/")[0]) for row in rows if row["evidence"] != "-"
    )
    times = [row["ms"] for row in rows]
    peaks = [row["peak_kib"] for row in rows]
    return {
        "n": len(rows),
        "expected": sum(len(case.expected) for case in cases),
        "findings": total_findings,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "evidence_complete": with_evidence,
        "evidence_rate": with_evidence / total_findings if total_findings else 1.0,
        "families": per_family,
        "formats": formats,
        "rows": rows,
        "errors": errors,
        "runtime_total_ms": sum(times),
        "runtime_median_ms": statistics.median(times) if times else 0.0,
        "runtime_max_ms": max(times) if times else 0.0,
        "peak_max_kib": max(peaks) if peaks else 0.0,
    }


def markdown(metrics: dict) -> str:
    family_rows = [
        "| Family | TP | FP | FN | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, counts in metrics["families"].items():
        precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if counts["tp"] + counts["fp"]
            else 1.0
        )
        recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if counts["tp"] + counts["fn"]
            else 1.0
        )
        family_rows.append(
            f"| {family} | {counts['tp']} | {counts['fp']} | {counts['fn']} "
            f"| {precision:.2f} | {recall:.2f} |"
        )
    case_rows = [
        "| Case | Family | Formats | Status | Findings | Skipped | Evidence | ms | Peak KiB |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in metrics["rows"]:
        case_rows.append(
            f"| {row['name']} | {row['family']} | {row['formats']} | "
            f"{row['status']} | {row['findings']} | {row['skipped']} | "
            f"{row['evidence']} | {row['ms']:.1f} | {row['peak_kib']:.0f} |"
        )
    miss = "\n".join(f"- {error}" for error in metrics["errors"]) or "- none"
    return (
        "# Pipeline benchmark\n\n"
        "Synthetic, labelled pipeline repositories for the Phase 14 pipeline/MLOps "
        "engine. Truth is the exact `(kind, column)` set per case. These are "
        "hand-written controls, not a sample of real repositories, so the numbers "
        "are a regression guard and do not estimate real-world accuracy. Times and "
        "memory are measured on the machine that generated this file.\n\n"
        "## Metrics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| Cases | {metrics['n']} |\n"
        f"| Expected findings | {metrics['expected']} |\n"
        f"| Reported findings | {metrics['findings']} |\n"
        f"| True positives | {metrics['tp']} |\n"
        f"| False positives | {metrics['fp']} |\n"
        f"| False negatives | {metrics['fn']} |\n"
        f"| Precision | {metrics['precision']:.2f} |\n"
        f"| Recall | {metrics['recall']:.2f} |\n"
        f"| F1 | {metrics['f1']:.2f} |\n"
        f"| Findings with complete evidence | {metrics['evidence_complete']} "
        f"({metrics['evidence_rate']:.2f}) |\n"
        f"| Formats covered | {', '.join(metrics['formats'])} |\n"
        f"| Total runtime | {metrics['runtime_total_ms']:.0f} ms |\n"
        f"| Median case | {metrics['runtime_median_ms']:.0f} ms |\n"
        f"| Slowest case | {metrics['runtime_max_ms']:.0f} ms |\n"
        f"| Peak memory | {metrics['peak_max_kib']:.0f} KiB |\n\n"
        "## By family\n\n" + "\n".join(family_rows) + "\n\n"
        "## Cases\n\n" + "\n".join(case_rows) + "\n\n"
        "## Misclassified cases\n\n" + miss + "\n\n"
        "## Limitations\n\n"
        "- The corpus is synthetic and deliberately small; every positive case is a "
        "strong signal. Real DVC/DAG exports vary and can produce formats this "
        "corpus does not cover.\n"
        "- `pipeline_split_overlap` only compares stages whose names identify a "
        "train and an eval side, and only data-like artifacts. A differently named "
        "pipeline is not checked.\n"
        "- Pipeline findings ship `scored=False` (and are diagnostics), so none of "
        "the metrics above affect the trust score or the exit code.\n"
        "- Thresholds were not tuned against this corpus; the cases define expected "
        "output and any change to them is a deliberate detector change.\n"
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
