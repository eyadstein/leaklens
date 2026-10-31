from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from . import __version__, llm
from .audit import audit
from .leakage import group_overlap, group_risk, scan, score, split_overlap, temporal_overlap
from .fixes import advice
from .readers import read_dataset
from .scanner import scan_path

_SEVERITY_COLORS = {
    "high": "\033[31m",
    "medium": "\033[33m",
    "low": "\033[32m",
}
_RESET = "\033[0m"


def _severity_label(severity: str) -> str:
    label = f"[{severity.upper()}]"
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return label
    color = _SEVERITY_COLORS.get(severity.lower())
    return f"{color}{label}{_RESET}" if color else label


def _read_dataset(path):
    return read_dataset(path)


STARTER_CONFIG = """# LeakLens settings for this project.
# Every key mirrors a command-line flag (dashes become underscores).
# LeakLens does not read this file yet, so everything is commented out:
# uncomment what you use and pass the same values as flags until it does.

# target = "label"          # --target: label column of the training data
# test = "test.csv"         # --test: held-out split to compare against
# time = "date"             # --time: timestamp column for the temporal check
# group = "patient_id"      # --group: entity column for group leakage
# threshold = 0.95          # --threshold: single-feature score that counts as leakage
# scan = ["notebooks"]      # --scan: files or folders of .py/.ipynb to scan
# explain = false           # --explain: print a suggested fix under each finding
# fail_under = 80           # --fail-under: exit 1 when the trust score is below this
# sarif = "leaklens.sarif"  # --sarif: write SARIF 2.1.0 for GitHub code scanning
# html = "report.html"      # --html: write a single-file HTML report
# badge = "badge.json"      # --badge: write a shields.io endpoint JSON
"""

PRE_COMMIT_SNIPPET = """repos:
  - repo: https://github.com/eyadstein/leaklens
    rev: main
    hooks:
      - id: leaklens
"""


def init(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens init",
        description="Write a starter .leaklens.toml",
    )
    p.add_argument("dir", nargs="?", default=".", help="folder to write .leaklens.toml into")
    p.add_argument("--force", action="store_true", help="overwrite an existing .leaklens.toml")
    a = p.parse_args(argv)
    path = Path(a.dir) / ".leaklens.toml"
    if path.exists() and not a.force:
        p.error(f"{path} already exists (use --force to overwrite)")
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    print(f"Wrote {path}")
    print("\nAdd this to .pre-commit-config.yaml to scan notebooks and scripts on every commit:\n")
    print(PRE_COMMIT_SNIPPET, end="")
    return 0


def images(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens images",
        description="Find near-duplicate images across train and test folders",
    )
    p.add_argument("train_dir")
    p.add_argument("test_dir")
    p.add_argument("--max-distance", type=int, default=6,
                   help="largest dHash Hamming distance (0-64) that counts as a duplicate")
    p.add_argument("--samples", type=int, default=5,
                   help="number of example pairs to print")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    if not 0 <= a.max_distance <= 64:
        p.error("--max-distance must be between 0 and 64")
    if a.samples < 0:
        p.error("--samples must be non-negative")

    from .images import image_overlap

    try:
        fs, pairs, stats = image_overlap(a.train_dir, a.test_dir, a.max_distance)
    except ImportError as e:
        p.error(str(e))

    rep = {
        "trust_score": score(fs),
        **stats,
        "findings": [f.to_dict() for f in fs],
        "pairs": pairs,
    }

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"LeakLens trust score: {rep['trust_score']}/100")
        print(
            f"{stats['train_images']} train images, "
            f"{stats['test_images']} test images"
            + (f", {stats['unreadable']} unreadable" if stats["unreadable"] else "")
        )
        for f in fs:
            print(f"[{f.severity.upper()}] {f.kind} :: {f.column} :: {f.detail}")
        for m in pairs[:a.samples]:
            print(f"    {m['test']} ~ {m['train']} (distance {m['distance']})")
        if not fs:
            print("No leakage found.")

    return 1 if fs else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "init":
        return init(argv[1:])
    if argv and argv[0] == "images":
        return images(argv[1:])
    p = argparse.ArgumentParser(prog="leaklens", description="Audit datasets and ML code for leakage")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("train", nargs="?")
    p.add_argument("--target")
    p.add_argument("--test")
    p.add_argument("--time")
    p.add_argument("--group", help="entity column (patient, customer...) to check for group leakage")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--scan", nargs="+", help="file or folder of .py/.ipynb to scan for pipeline leaks")
    p.add_argument("--arena", action="store_true", help="compare 3 model families, tracked in MLflow")
    p.add_argument("--out", default="models")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--json", action="store_true")
    p.add_argument("--explain", action="store_true", help="print a suggested fix under each finding")
    p.add_argument("--llm", action="store_true", help="ask the configured provider to explain findings")
    p.add_argument("--llm-provider", choices=("ollama", "openai-compatible"), default="ollama")
    p.add_argument("--llm-base-url", help="LLM provider base URL")
    p.add_argument("--llm-model", help="LLM model name")
    p.add_argument("--llm-max-findings", type=int, default=10, help="maximum findings to explain (default: 10)")
    p.add_argument("--sarif", help="write SARIF 2.1.0 for GitHub code scanning")
    p.add_argument("--html", help="write a shareable single-file HTML report")
    p.add_argument("--badge", help="write a shields.io endpoint JSON with the trust score")
    p.add_argument("--fail-under", type=float, help="exit with code 1 if trust score is below N")
    a = p.parse_args(argv)
    if a.llm and a.llm_max_findings < 1:
        p.error("--llm-max-findings must be at least 1")
    if not a.train and not a.scan:
        p.error("give a CSV and/or --scan PATH")
    if a.train and not a.target:
        p.error("--target is required with a CSV")
    if a.arena and not a.train:
        p.error("--arena needs a CSV")
    fs, rep, res = [], {}, None
    if a.train:
        try:
            tr = _read_dataset(a.train)
        except ImportError as e:
            p.error(str(e))
        fs += scan(tr, a.target, a.threshold)
        if a.test:
            try:
                te = _read_dataset(a.test)
            except ImportError as e:
                p.error(str(e))
            fs += split_overlap(tr, te)
            if a.time:
                fs += temporal_overlap(tr, te, a.time)
        if a.group:
            fs += group_risk(tr, a.group)
            if a.test:
                fs += group_overlap(tr, te, a.group)
        rep["audit"] = audit(tr, a.target)
        if a.arena:
            try:
                from .arena import run
            except ImportError:
                p.error('install extras: pip install -e ".[arena]"')
            res = run(tr, a.target, a.time, a.out, a.tracking_uri)
            rep["arena"] = res.round(4).to_dict("records")
    if a.scan:
        for s in a.scan:
            fs += scan_path(s)
    explanations = None
    if a.llm:
        try:
            provider = llm.create_provider(a.llm_provider, a.llm_base_url, a.llm_model)
            explanations = llm.explain_findings(fs, provider, a.llm_max_findings)
        except llm.LLMError as exc:
            p.error(str(exc))
        if len(fs) > a.llm_max_findings:
            print(
                f"LLM explanations skipped {len(fs) - a.llm_max_findings} finding(s) "
                f"after the {a.llm_max_findings}-finding limit.",
                file=sys.stderr,
            )

    rep = {"trust_score": score(fs), **rep, "findings": [f.to_dict() for f in fs]}
    if explanations is not None:
        for finding_report, explanation in zip(rep["findings"], explanations):
            if explanation is not None:
                finding_report["llm_explanation"] = explanation
    if a.sarif:
        from .sarif import to_sarif
        with open(a.sarif, "w", encoding="utf-8") as fh:
            json.dump(to_sarif(fs, a.train or "."), fh, indent=2)
    if a.html:
        from .report import render
        with open(a.html, "w", encoding="utf-8") as fh:
            fh.write(render(rep))
    if a.badge:
        s = rep["trust_score"]
        col = "brightgreen" if s >= 85 else "yellow" if s >= 60 else "red"
        with open(a.badge, "w", encoding="utf-8") as fh:
            json.dump({"schemaVersion": 1, "label": "leakage trust", "message": f"{s}/100", "color": col}, fh)
    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"LeakLens trust score: {rep['trust_score']}/100")
        for index, f in enumerate(fs):
            print(f"{_severity_label(f.severity)} {f.kind} :: {f.column} :: {f.detail}")
            if explanations is not None and explanations[index] is not None:
                print(f"    LLM: {explanations[index]}")
            if a.explain and advice(f.kind):
                print(f"    fix: {advice(f.kind)}")
        if fs:
            print(", ".join(f"{sum(f.severity == lvl for f in fs)} {lvl}" for lvl in ("high", "medium", "low") if any(f.severity == lvl for f in fs)))
        else:
            print("No leakage found.")
        if res is not None:
            print(f"\nArena ranked by {res.attrs['primary']} (auto-dropped: {', '.join(map(str, res.attrs['dropped'])) or 'none'})")
            print(res.round(4).to_string(index=False))
    return 1 if any(f.severity == "high" for f in fs) or (a.fail_under is not None and rep["trust_score"] < a.fail_under) else 0
