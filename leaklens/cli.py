from __future__ import annotations

import argparse
import json

import pandas as pd

from .audit import audit
from .leakage import group_overlap, group_risk, scan, score, split_overlap, temporal_overlap
from .fixes import advice
from .scanner import scan_path


def _read_dataset(path):
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="leaklens", description="Audit datasets and ML code for leakage")
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
    p.add_argument("--llm", action="store_true", help="ask an LLM to explain findings (opt-in; sends finding details to the provider)")
    p.add_argument("--llm-provider", choices=("ollama", "openai-compatible"), default="ollama")
    p.add_argument("--llm-base-url", help="provider URL; defaults to local Ollama at http://localhost:11434")
    p.add_argument("--llm-model", help="model name (or LEAKLENS_LLM_MODEL)")
    p.add_argument("--sarif", help="write SARIF 2.1.0 for GitHub code scanning")
    p.add_argument("--html", help="write a shareable single-file HTML report")
    p.add_argument("--badge", help="write a shields.io endpoint JSON with the trust score")
    p.add_argument("--fail-under", type=float, help="exit with code 1 if trust score is below N")
    a = p.parse_args(argv)
    if not a.train and not a.scan:
        p.error("give a CSV and/or --scan PATH")
    if a.train and not a.target:
        p.error("--target is required with a CSV")
    if a.arena and not a.train:
        p.error("--arena needs a CSV")
    fs, rep, res = [], {}, None
    if a.train:
        tr = _read_dataset(a.train)
        fs += scan(tr, a.target, a.threshold)
        if a.test:
            te = _read_dataset(a.test)
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

    explanations = []
    if a.llm and fs:
        from .llm import LLMError, create_provider, explain_findings

        try:
            provider = create_provider(a.llm_provider, a.llm_base_url, a.llm_model)
            explanations = explain_findings(fs, provider)
        except LLMError as exc:
            p.error(str(exc))

    finding_reports = []
    for index, finding in enumerate(fs):
        finding_report = finding.to_dict()
        if explanations:
            finding_report["llm_explanation"] = explanations[index]
        finding_reports.append(finding_report)
    rep = {"trust_score": score(fs), **rep, "findings": finding_reports}
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
            print(f"[{f.severity.upper()}] {f.kind} :: {f.column} :: {f.detail}")
            if a.explain and advice(f.kind):
                print(f"    fix: {advice(f.kind)}")
            if explanations:
                print(f"    LLM: {explanations[index]}")
        if not fs:
            print("No leakage found.")
        if res is not None:
            print(f"\nArena ranked by {res.attrs['primary']} (auto-dropped: {', '.join(map(str, res.attrs['dropped'])) or 'none'})")
            print(res.round(4).to_string(index=False))
    return 1 if any(f.severity == "high" for f in fs) or (a.fail_under is not None and rep["trust_score"] < a.fail_under) else 0
