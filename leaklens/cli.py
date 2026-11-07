from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, llm
from .audit import audit
from .baseline import Baseline, BaselineError
from .config import ConfigError, load_config
from .fixes import advice
from .leakage import (
    feature_overlap,
    group_overlap,
    group_risk,
    near_duplicate_overlap,
    scan,
    score,
    split_overlap,
    temporal_overlap,
)
from .missingness import audit_missingness
from .policy import apply_policy
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
# Top-level keys mirror command-line flags (dashes become underscores).
# LeakLens reads .leaklens.toml from the current directory unless --config is
# passed. Explicit CLI flags win over this file. Uncomment only what you need.

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
# ai = false                # --ai: add experimental, evidence-bound AI hypotheses
# ai_llm = false            # --ai-llm: rephrase the AI plan with the configured LLM

# Project policy (not CLI flags). Output paths must stay inside this directory.
#
# [baseline]
# path = "leaklens-baseline.json"
# fail_on_new = false
#
# [[suppressions]]
# fingerprint = "64-lowercase-hex-characters-from-a-json-finding"
# reason = "why this finding is accepted"
# expires = "2026-12-31"
#
# [missingness]
# enabled = true
# min_rows = 200
# min_group_rows = 30
# min_missing = 30
# min_rate_delta = 0.15
# min_group_delta = 0.25
# min_correlation = 0.30
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
    p.add_argument(
        "dir", nargs="?", default=".", help="folder to write .leaklens.toml into"
    )
    p.add_argument(
        "--force", action="store_true", help="overwrite an existing .leaklens.toml"
    )
    a = p.parse_args(argv)
    path = Path(a.dir) / ".leaklens.toml"
    if path.exists() and not a.force:
        p.error(f"{path} already exists (use --force to overwrite)")
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    print(f"Wrote {path}")
    print(
        "\nAdd this to .pre-commit-config.yaml to scan notebooks and scripts on every commit:\n"
    )
    print(PRE_COMMIT_SNIPPET, end="")
    return 0


def images(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens images",
        description="Find near-duplicate images across train and test folders",
    )
    p.add_argument("train_dir")
    p.add_argument("test_dir")
    p.add_argument(
        "--max-distance",
        type=int,
        default=6,
        help="largest dHash Hamming distance (0-64) that counts as a duplicate",
    )
    p.add_argument(
        "--samples", type=int, default=5, help="number of example pairs to print"
    )
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
        for m in pairs[: a.samples]:
            print(f"    {m['test']} ~ {m['train']} (distance {m['distance']})")
        if not fs:
            print("No leakage found.")

    return 1 if fs else 0


def doctor(argv) -> int:
    from importlib import metadata as _meta

    EXTRAS = {
        "arena": "mlflow",
        "parquet": "pyarrow",
        "feather": "pyarrow",
        "excel": "openpyxl",
        "images": "PIL",
        "app": "fastapi",
        "taintwatch": "taintwatch",
    }

    def _import_ok(mod):
        import importlib.util

        return importlib.util.find_spec(mod) is not None

    def _pkg_version(pkg):
        try:
            return _meta.version(pkg)
        except _meta.PackageNotFoundError:
            return None

    report = {
        "python_version": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 10),
        "leaklens_version": __version__,
        "leaklens_location": __file__,
        "cli_available": True,
        "entry_point": "leaklens:main",
        "required_ok": all(_import_ok(m) for m in ("pandas", "numpy", "sklearn")),
        "extras": {extra: _import_ok(m) for extra, m in EXTRAS.items()},
        "extras_available": [],
        "toml_config_present": None,
    }
    report["extras_available"] = sorted(
        e for e, ok in report["extras"].items() if ok and e != "taintwatch"
    )
    report["config_valid"] = True
    report["config_error"] = None
    config_arg = None
    if "--config" in argv:
        index = argv.index("--config")
        config_arg = argv[index + 1] if index + 1 < len(argv) else ""
    if config_arg is not None:
        report["toml_config_present"] = config_arg or None
        try:
            if not config_arg:
                raise ConfigError("--config needs a path")
            load_config(config_arg)
        except (
            OSError,
            ValueError,
            UnicodeDecodeError,
        ) as exc:  # invalid config path or schema
            report["config_valid"] = False
            report["config_error"] = str(exc)
    else:
        for name in (".leaklens.toml", "pyproject.toml"):
            path = Path(name)
            if path.exists():
                report["toml_config_present"] = name
                try:
                    if name == ".leaklens.toml":
                        load_config(path)
                    else:
                        try:
                            import tomllib
                        except ModuleNotFoundError:  # Python 3.10
                            import tomli as tomllib

                        tomllib.loads(path.read_text(encoding="utf-8"))
                except (
                    OSError,
                    ValueError,
                    UnicodeDecodeError,
                ) as exc:  # invalid TOML/config in repo config
                    report["config_valid"] = False
                    report["config_error"] = str(exc)
                break
        else:
            report["toml_config_present"] = None
            report["config_valid"] = True
    if argv and argv[0] in ("--help", "-h"):
        print(
            "usage: leaklens doctor [--json] [--config PATH]\n\n"
            "Check Python/LeakLens versions, install location, required and "
            "optional dependencies, and config validity."
        )
        return 0
    text = [
        f"Python      : {report['python_version']} (required >=3.10) {'OK' if report['python_ok'] else 'FAIL: python<3.10'}",
        f"LeakLens    : {report['leaklens_version']} at {report['leaklens_location']}",
        f"CLI         : {report['entry_point']} OK",
        f"Core deps   : {'OK' if report['required_ok'] else 'FAIL: pandas/numpy/scikit-learn missing -> pip install --upgrade leaklens-ml'}",
        f"Extras      : {', '.join(report['extras_available']) or 'none'} (of {', '.join([e for e in EXTRAS if e != 'taintwatch'])})",
        f"Config      : {report['toml_config_present'] or 'none (ok)'}"
        + (f" INVALID: {report['config_error']}" if not report["config_valid"] else ""),
    ]
    if (
        not report["python_ok"]
        or not report["required_ok"]
        or not report["config_valid"]
    ):
        text.append(
            "Result      : DOCTOR FAILED - see FAIL rows above; install/reinstall leaklens-ml or fix config"
        )
        print("\n".join(text))
        return 1
    if argv and "--json" in argv:
        print(json.dumps(report, indent=2))
        return 0
    text.append("Result      : DOCTOR OK (no secrets, no network calls)")
    print("\n".join(text))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "init":
        return init(argv[1:])
    if argv and argv[0] == "images":
        return images(argv[1:])
    if argv and argv[0] == "doctor":
        return doctor(argv[1:])
    p = argparse.ArgumentParser(
        prog="leaklens",
        description="Audit datasets and ML code for leakage",
        epilog=(
            "Run 'leaklens doctor' to check your install, and "
            "'leaklens init' to write a starter .leaklens.toml config."
        ),
    )
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("train", nargs="?")
    p.add_argument("--target")
    p.add_argument("--test")
    p.add_argument("--time")
    p.add_argument(
        "--group",
        help="entity column (patient, customer...) to check for group leakage",
    )
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--scan",
        nargs="+",
        help="file or folder of .py/.ipynb to scan for pipeline leaks",
    )
    p.add_argument(
        "--arena",
        action="store_true",
        help="compare 3 model families, tracked in MLflow",
    )
    p.add_argument("--out", default="models")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--explain",
        action="store_true",
        default=None,
        help="print a suggested fix under each finding",
    )
    p.add_argument(
        "--intelligence",
        action="store_true",
        help="include deterministic confidence, root-cause, and remediation analysis",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="ask the configured provider to explain findings",
    )
    p.add_argument(
        "--llm-provider", choices=("ollama", "openai-compatible"), default="ollama"
    )
    p.add_argument("--llm-base-url", help="LLM provider base URL")
    p.add_argument("--llm-model", help="LLM model name")
    p.add_argument(
        "--llm-max-findings",
        type=int,
        default=10,
        help="maximum findings to explain (default: 10)",
    )
    p.add_argument(
        "--ai",
        action="store_true",
        default=None,
        help="add experimental, evidence-bound AI hypotheses (offline)",
    )
    p.add_argument(
        "--ai-llm",
        action="store_true",
        default=None,
        help="rephrase the AI plan with the configured LLM (safe fallback)",
    )
    p.add_argument(
        "--config",
        help="project configuration file (default: ./.leaklens.toml when present)",
    )
    p.add_argument(
        "--baseline",
        help="compare findings with this baseline (or use [baseline].path in config)",
    )
    baseline_action = p.add_mutually_exclusive_group()
    baseline_action.add_argument(
        "--write-baseline",
        action="store_true",
        help="write a new baseline; refuses to overwrite an existing file",
    )
    baseline_action.add_argument(
        "--update-baseline",
        action="store_true",
        help="explicitly replace an existing baseline",
    )
    p.add_argument(
        "--fail-on-new",
        action="store_true",
        help="opt in: exit 1 when the baseline comparison finds new findings",
    )
    p.add_argument("--sarif", help="write SARIF 2.1.0 for GitHub code scanning")
    p.add_argument("--html", help="write a shareable single-file HTML report")
    p.add_argument(
        "--badge", help="write a shields.io endpoint JSON with the trust score"
    )
    p.add_argument(
        "--fail-under", type=float, help="exit with code 1 if trust score is below N"
    )
    a = p.parse_args(argv)
    try:
        cfg = load_config(a.config)
    except ConfigError as exc:
        print(f"leaklens: configuration error: {exc}", file=sys.stderr)
        return 2

    def choose(cli_value, configured, default=None):
        return (
            cli_value
            if cli_value is not None
            else (configured if configured is not None else default)
        )

    target = a.target if a.target is not None else cfg.target
    test_path = (
        a.test
        if a.test is not None
        else (str(cfg.test) if cfg.test is not None else None)
    )
    time_col = a.time if a.time is not None else cfg.time
    group_col = a.group if a.group is not None else cfg.group
    threshold = choose(a.threshold, cfg.threshold, 0.95)
    explain = choose(a.explain, cfg.explain, False)
    fail_under = choose(a.fail_under, cfg.fail_under)
    sarif_path = choose(a.sarif, cfg.sarif)
    html_path = choose(a.html, cfg.html)
    badge_path = choose(a.badge, cfg.badge)
    use_ai = choose(a.ai, cfg.ai, False)
    use_ai_llm = choose(a.ai_llm, cfg.ai_llm, False)
    if use_ai_llm:
        use_ai = True
    scan_targets = a.scan if a.scan is not None else [str(path) for path in cfg.scan]
    baseline_path = (
        a.baseline
        if a.baseline is not None
        else (str(cfg.baseline_path) if cfg.baseline_path is not None else None)
    )
    fail_on_new = bool(a.fail_on_new or cfg.fail_on_new)
    writing_baseline = bool(a.write_baseline or a.update_baseline)
    if writing_baseline and baseline_path is None:
        baseline_path = "leaklens-baseline.json"
    if a.fail_on_new and not baseline_path:
        p.error("--fail-on-new needs --baseline PATH or [baseline].path in config")
    if a.llm and a.llm_max_findings < 1:
        p.error("--llm-max-findings must be at least 1")
    if not a.train and not scan_targets:
        p.error("give a CSV and/or --scan PATH")
    if a.train and not target:
        p.error("--target is required with a CSV")
    if a.arena and not a.train:
        p.error("--arena needs a CSV")

    baseline = None
    if baseline_path and not writing_baseline:
        try:
            baseline = Baseline.load(baseline_path)
        except BaselineError as exc:
            print(f"leaklens: {exc}", file=sys.stderr)
            return 2

    fs, rep, res = [], {}, None
    tr = te = None
    if a.train:
        try:
            tr = _read_dataset(a.train)
        except ImportError as e:
            p.error(str(e))
        fs += scan(tr, target, threshold)
        if test_path:
            try:
                te = _read_dataset(test_path)
            except ImportError as e:
                p.error(str(e))
            exact = split_overlap(tr, te)
            fs += exact
            if not exact:
                shared = feature_overlap(tr, te, target)
                fs += shared
                if not shared:
                    fs += near_duplicate_overlap(tr, te, target)
            if time_col:
                fs += temporal_overlap(tr, te, time_col)
        if group_col:
            fs += group_risk(tr, group_col)
            if test_path:
                fs += group_overlap(tr, te, group_col)
        missing_report = audit_missingness(
            tr, target, test=te, group=group_col, settings=cfg.missingness
        )
        fs += missing_report.findings
        rep["missingness"] = missing_report.to_dict()
        rep["audit"] = audit(tr, target)
        if a.arena:
            try:
                from .arena import run
            except ImportError:
                p.error('install extras: pip install -e ".[arena]"')
            res = run(tr, target, time_col, a.out, a.tracking_uri)
            rep["arena"] = res.round(4).to_dict("records")
    for source in scan_targets:
        fs += scan_path(source)

    outcome = apply_policy(fs, baseline=baseline, rules=cfg.suppressions)
    fs = list(outcome.findings)
    active = list(outcome.active)
    policy_active = baseline is not None or bool(cfg.suppressions)
    if outcome.suppression is not None:
        for message in outcome.suppression.diagnostics:
            print(f"leaklens: warning: {message}", file=sys.stderr)
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

    rep = {
        "trust_score": score(active),
        **rep,
        "findings": [outcome.finding_payload(f) for f in fs],
    }
    if policy_active:
        rep["policy"] = outcome.summary(
            baseline_path=baseline_path, fail_on_new=fail_on_new
        )
        if outcome.diff is not None and outcome.diff.resolved:
            rep["resolved"] = [entry.to_dict() for entry in outcome.diff.resolved]
    ai_report = None
    if use_ai:
        try:
            from .ai import DatasetInput, analyze, collect_code_sources

            dataset = (
                DatasetInput(
                    train=tr,
                    test=te if test_path else None,
                    target=target,
                    group=group_col,
                    time=time_col,
                )
                if a.train
                else None
            )
            provider = None
            if use_ai_llm:
                try:
                    provider = llm.create_provider(
                        a.llm_provider, a.llm_base_url, a.llm_model
                    )
                except llm.LLMError as exc:
                    print(
                        f"AI LLM unavailable; using the deterministic plan: {exc}",
                        file=sys.stderr,
                    )
            ai_report = analyze(
                fs,
                code=collect_code_sources(scan_targets) if scan_targets else (),
                dataset=dataset,
                llm_provider=provider,
                trust_score=rep["trust_score"],
            ).to_dict()
            rep["ai"] = ai_report
        except Exception as exc:  # noqa: BLE001 - optional AI must not break a scan
            print(
                f"AI analysis unavailable; deterministic results are unchanged: {exc}",
                file=sys.stderr,
            )
    intelligence_report = None
    if a.intelligence:
        from .intelligence import analyze_dataset_impact, analyze_findings

        impact = None
        if a.train and test_path:
            impact = analyze_dataset_impact(
                tr,
                te,
                group_col=(
                    group_col
                    if group_col and group_col in tr.columns and group_col in te.columns
                    else None
                ),
                time_col=(
                    time_col
                    if time_col and time_col in tr.columns and time_col in te.columns
                    else None
                ),
            )
        intelligence_report = analyze_findings(fs, impact=impact).to_dict()
        rep["intelligence"] = intelligence_report
    if explanations is not None:
        for finding_report, explanation in zip(rep["findings"], explanations):
            if explanation is not None:
                finding_report["llm_explanation"] = explanation
    if writing_baseline:
        baseline_doc = Baseline.from_findings(fs, tool_version=__version__)
        try:
            written = baseline_doc.write(baseline_path, update=bool(a.update_baseline))
        except BaselineError as exc:
            print(f"leaklens: {exc}", file=sys.stderr)
            return 2
        print(
            f"Wrote baseline {written} ({len(baseline_doc.entries)} findings)",
            file=sys.stderr,
        )
    if sarif_path:
        from .sarif import to_sarif

        with open(sarif_path, "w", encoding="utf-8") as fh:
            json.dump(to_sarif(fs, a.train or ".", outcome=outcome), fh, indent=2)
    if html_path:
        from .report import render

        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render(rep))
    if badge_path:
        s = rep["trust_score"]
        col = "brightgreen" if s >= 85 else "yellow" if s >= 60 else "red"
        with open(badge_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "schemaVersion": 1,
                    "label": "leakage trust",
                    "message": f"{s}/100",
                    "color": col,
                },
                fh,
            )
    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"LeakLens trust score: {rep['trust_score']}/100")
        if outcome.diff is not None:
            counts = outcome.diff.counts
            print(
                f"Baseline: {counts['new']} new, {counts['unchanged']} unchanged, "
                f"{counts['changed']} changed, {counts['resolved']} resolved"
                + (f" ({baseline_path})" if baseline_path else "")
            )
        if outcome.suppression is not None:
            print(f"Suppressions: {len(outcome.suppressed)} finding(s) suppressed")
        if intelligence_report is not None:
            confidence = intelligence_report["confidence"]
            print(
                f"Intelligence confidence: {confidence['level']} ({confidence['score']:.2f})"
            )
            for cause in intelligence_report["root_causes"]:
                print(f"    Root cause: {cause['title']}")
            for recommendation in intelligence_report["recommendations"]:
                print(f"    Recommendation: {recommendation['title']}")
        if ai_report is not None:
            hypotheses = ai_report["hypotheses"]
            groups = ai_report["plan"]["root_causes"]
            print(
                f"AI (experimental): {len(hypotheses)} hypothesis(es), "
                f"{len(groups)} root-cause group(s); deterministic findings unchanged."
            )
            for group in groups[:3]:
                print(
                    f"    AI root cause [{group['priority']}]: {group['title']} "
                    f"({group['source']})"
                )
            if ai_report["plan"].get("llm_summary"):
                print(f"    AI summary: {ai_report['plan']['llm_summary']}")
            for hypothesis in hypotheses[:3]:
                print(
                    f"    AI hypothesis [{hypothesis['confidence']:.2f}]: "
                    f"{hypothesis['title']} :: {hypothesis['location']}"
                )
        for index, f in enumerate(fs):
            suffix = ""
            rule = (
                outcome.suppression.rule_for(f.fingerprint)
                if outcome.suppression is not None
                else None
            )
            if rule is not None:
                suffix = f" [suppressed: {rule.reason}]"
            elif outcome.diff is not None:
                suffix = f" [{outcome.status.get(f.fingerprint, 'unchanged')}]"
            print(
                f"{_severity_label(f.severity)} {f.kind} :: {f.column} :: "
                f"{f.detail}{suffix}"
            )
            if explanations is not None and explanations[index] is not None:
                print(f"    LLM: {explanations[index]}")
            if explain and advice(f.kind):
                print(f"    fix: {advice(f.kind)}")
        if outcome.diff is not None and outcome.diff.resolved:
            print("Resolved since baseline:")
            for entry in outcome.diff.resolved:
                short = entry.fingerprint[:12]
                print(f"    {entry.kind} :: {entry.column} ({short})")
        if active:
            print(
                ", ".join(
                    f"{sum(f.severity == lvl for f in active)} {lvl}"
                    for lvl in ("high", "medium", "low")
                    if any(f.severity == lvl for f in active)
                )
            )
        else:
            print("No leakage found.")
        if res is not None:
            print(
                f"\nArena ranked by {res.attrs['primary']} (auto-dropped: {', '.join(map(str, res.attrs['dropped'])) or 'none'})"
            )
            print(res.round(4).to_string(index=False))
    gate = any(f.severity == "high" for f in active) or (
        fail_under is not None and rep["trust_score"] < fail_under
    )
    if fail_on_new and outcome.diff is not None and outcome.diff.new:
        gate = True
    return 1 if gate else 0
