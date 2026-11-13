from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path

from . import __version__, assemble, llm
from .audit import audit
from .baseline import Baseline, BaselineError
from .config import ConfigError, load_config
from .drift import audit_drift
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
from .readers import read_dataset
from .scanner import SKIP, scan_path

_SEVERITY_COLORS = {
    "high": "\033[31m",
    "medium": "\033[33m",
    "low": "\033[32m",
}
_RESET = "\033[0m"


_GRAPH_FORMATS = {
    ".mmd": "mermaid",
    ".mermaid": "mermaid",
    ".md": "mermaid",
    ".dot": "dot",
    ".gv": "dot",
    ".json": "json",
}


def _write_graph(graph, path: str, fmt: str | None) -> None:
    suffix = Path(path).suffix.lower()
    chosen = fmt or _GRAPH_FORMATS.get(suffix)
    if chosen is None:
        raise ValueError(f"cannot infer graph format from {path!r}; use --graph-format")
    if chosen not in {"mermaid", "dot", "json"}:
        raise ValueError(f"unsupported graph format: {chosen!r}")
    if chosen == "mermaid":
        text = graph.to_mermaid()
    elif chosen == "dot":
        text = graph.to_dot()
    else:
        text = json.dumps(graph.to_dict(), indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _severity_label(severity: str) -> str:
    label = f"[{severity.upper()}]"
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return label
    color = _SEVERITY_COLORS.get(severity.lower())
    return f"{color}{label}{_RESET}" if color else label


def _read_dataset(path):
    return read_dataset(path)


def _graph_completeness_reasons(scan_targets, findings, graph, *, sql=False):
    """Collect only conditions a real scan path reports as partial."""
    suffixes = {".py", ".ipynb"}
    if sql:
        suffixes.add(".sql")
    reasons: list[str] = []
    for target in scan_targets:
        path = Path(target)
        if not path.exists():
            reasons.append(f"source path was not found: {target}")
        elif path.is_dir() and not any(
            item.is_file() and item.suffix in suffixes and not (SKIP & set(item.parts))
            for item in path.rglob("*")
        ):
            reasons.append(f"no scannable source files under: {target}")
    if any(finding.kind == "parse_error" for finding in findings):
        reasons.append("a source file could not be parsed completely")
    if graph is not None and graph.links_truncated:
        reasons.append("evidence link limit was reached; links are truncated")
    return tuple(dict.fromkeys(reasons))


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
#
# [sql]
# enabled = false           # --sql: also scan .sql files (needs the sql extra)
# dialect = "ansi"          # --sql-dialect: ansi, postgres, bigquery, snowflake
# split_column = "split"    # optional: flag queries without a split predicate
#
# [sql.schema.orders]       # optional column types for lineage resolution
# id = "INT"
# amount = "DOUBLE"
# label = "INT"
# split = "TEXT"
#
# [graph]                   # opt-in baseline-aware evidence graph
# enabled = true
# baseline = "evidence-baseline.json"  # existing snapshot to compare against
# snapshot = "evidence-current.json"   # write/refresh this snapshot
# update_snapshot = false              # must be true to replace an existing file
# fail_on_new = false                  # exit 1 only when new evidence appears
#
# [drift]
# enabled = false           # --drift: opt-in train/test drift diagnostics
# min_rows = 200
# min_effect = 0.20
# max_rows = 100000
# max_columns = 50
# max_categories = 20
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


def workspace(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens workspace",
        description="Serve a local, read-only interactive scanning workspace",
    )
    p.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="folder to treat as the workspace root (default: .)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    a = p.parse_args(argv)
    root = Path(a.dir).resolve()
    if not root.is_dir():
        p.error(f"{root} is not a folder")
    try:
        import uvicorn

        from .workspace import create_workspace_app
    except ImportError:
        print(
            'leaklens workspace needs the app extra: pip install -e ".[app]"',
            file=sys.stderr,
        )
        return 2
    app = create_workspace_app(root)
    url = f"http://{a.host}:{a.port}/"
    print(f"LeakLens workspace: {url}  (root: {root})")
    print("Press Ctrl+C to stop.")
    if not a.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
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

    from .images import analyze_images

    try:
        audit = analyze_images(a.train_dir, a.test_dir, a.max_distance)
    except ImportError as e:
        p.error(str(e))

    stats = audit.stats
    rep = {
        "trust_score": score(audit.findings),
        **stats,
        "findings": [f.to_dict() for f in audit.findings],
        "pairs": audit.pairs,
        "clustering": {
            "max_distance": a.max_distance,
            "exact_duplicates": audit.exact_duplicates,
            "uninformative": audit.uninformative,
            "exact_groups": list(audit.exact_groups),
            "clusters": [cluster.to_dict() for cluster in audit.clusters],
        },
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
        if stats["refused"]:
            reasons = ", ".join(f"{k}: {v}" for k, v in stats["refused"].items())
            print(f"    refused {sum(stats['refused'].values())} image(s): {reasons}")
        for f in audit.findings:
            print(f"[{f.severity.upper()}] {f.kind} :: {f.column} :: {f.detail}")
        for m in audit.pairs[: a.samples]:
            print(f"    {m['test']} ~ {m['train']} (distance {m['distance']})")
        if audit.exact_duplicates:
            print(
                f"    {audit.exact_duplicates} test image(s) are byte-identical "
                "to a train image"
            )
        if audit.clusters:
            members = sum(len(cluster.members) for cluster in audit.clusters)
            print(
                f"    {len(audit.clusters)} perceptual cluster(s) across both folders, "
                f"{members} image(s); max dHash distance {a.max_distance}"
            )
            for cluster in audit.clusters[: a.samples]:
                print(
                    f"    cluster {cluster.key} "
                    f"train={cluster.train} test={cluster.test} "
                    f"exact_pairs={cluster.exact_pairs}"
                )
                for member in cluster.members:
                    print(f"        {member}")
        if not audit.findings:
            print("No leakage found.")

    return 1 if audit.findings else 0


def text(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens text",
        description=(
            "Check text corpora for cross-split duplicates, eval contamination "
            "and RAG index leakage"
        ),
    )
    p.add_argument(
        "train", help="training corpus (.jsonl, .json, .txt, .parquet, .arrow)"
    )
    p.add_argument("test", help="evaluation corpus")
    p.add_argument("--field", help="document field in JSON/JSONL/parquet rows")
    p.add_argument(
        "--min-chars",
        type=int,
        default=32,
        help="minimum normalised characters per document (default 32)",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=10_000,
        help="maximum documents per split (default 10000)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="near-duplicate similarity threshold (default 0.8)",
    )
    p.add_argument(
        "--shingle-size", type=int, default=5, help="word shingle size (default 5)"
    )
    p.add_argument(
        "--boilerplate", help="file of boilerplate lines to strip before hashing"
    )
    p.add_argument(
        "--eval-fingerprints", help="file of eval digests, one per line (hash-only)"
    )
    p.add_argument(
        "--write-fingerprints",
        help="write this evaluation split's digests for --eval-fingerprints",
    )
    p.add_argument(
        "--rag", metavar="ROOT", help="scan a Python root for RAG index leakage"
    )
    p.add_argument(
        "--eval-path", help="declared evaluation path for --rag (repo-relative)"
    )
    p.add_argument(
        "--embeddings",
        nargs=2,
        metavar=("TRAIN_VECTORS", "TEST_VECTORS"),
        help="optional .npy matrices of user-supplied vectors",
    )
    p.add_argument(
        "--embedding-model", help="model id that produced the --embeddings vectors"
    )
    p.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.95,
        help="cosine threshold for --embeddings (default 0.95)",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when any diagnostic text finding is reported",
    )
    a = p.parse_args(argv)

    if not 0.0 <= a.threshold <= 1.0:
        p.error("--threshold must be between 0 and 1")
    if not 0.0 <= a.embedding_threshold <= 1.0:
        p.error("--embedding-threshold must be between 0 and 1")
    if a.min_chars < 0:
        p.error("--min-chars must be non-negative")
    if a.max_docs < 1:
        p.error("--max-docs must be positive")
    if a.shingle_size < 1:
        p.error("--shingle-size must be positive")
    if bool(a.rag) != bool(a.eval_path):
        p.error("--rag and --eval-path must be used together")
    if bool(a.embeddings) != bool(a.embedding_model):
        p.error("--embeddings and --embedding-model must be used together")

    from .analysis.text.contamination import (
        contamination_findings,
        load_fingerprints,
        write_fingerprints,
    )
    from .analysis.text.dedup import exact_findings, near_findings
    from .analysis.text.embeddings import embedding_findings, load_vectors
    from .analysis.text.normalize import load_documents, normalize_text
    from .analysis.text.rag import scan_rag

    field = a.field or "text"
    boilerplate = frozenset()
    if a.boilerplate:
        try:
            lines = (
                Path(a.boilerplate)
                .read_text(encoding="utf-8-sig", errors="replace")
                .splitlines()
            )
        except OSError as exc:
            p.error(str(exc))
        boilerplate = frozenset(
            normalize_text(line) for line in lines if normalize_text(line)
        )

    try:
        train_docs, train_skipped = load_documents(
            a.train,
            "train",
            field=a.field,
            min_chars=a.min_chars,
            max_docs=a.max_docs,
            boilerplate=boilerplate,
        )
        test_docs, test_skipped = load_documents(
            a.test,
            "test",
            field=a.field,
            min_chars=a.min_chars,
            max_docs=a.max_docs,
            boilerplate=boilerplate,
        )
    except (OSError, ValueError, ImportError) as exc:
        p.error(str(exc))

    try:
        findings = [
            *exact_findings(train_docs, test_docs, field),
            *near_findings(
                train_docs,
                test_docs,
                field,
                shingle_size=a.shingle_size,
                threshold=a.threshold,
            ),
        ]
    except ValueError as exc:
        p.error(str(exc))
    extra: dict[str, object] = {}

    if a.eval_fingerprints:
        try:
            fingerprints = load_fingerprints(a.eval_fingerprints)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        findings.extend(contamination_findings(train_docs, fingerprints, field))

    if a.write_fingerprints:
        try:
            written = write_fingerprints(a.write_fingerprints, test_docs)
        except OSError as exc:
            p.error(str(exc))
        extra["fingerprints"] = {"path": a.write_fingerprints, "count": written}

    if a.rag:
        try:
            rag_findings, hits = scan_rag(a.rag, [a.eval_path])
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        findings.extend(rag_findings)
        extra["rag"] = {"hits": len(hits)}

    if a.embeddings:
        try:
            train_vectors = load_vectors(a.embeddings[0])
            test_vectors = load_vectors(a.embeddings[1])
            findings.extend(
                embedding_findings(
                    train_docs,
                    test_docs,
                    train_vectors,
                    test_vectors,
                    model=a.embedding_model,
                    threshold=a.embedding_threshold,
                    field=field,
                )
            )
        except (OSError, ValueError) as exc:
            p.error(str(exc))

    sections = assemble.build(findings)
    rep = {
        "trust_score": sections.trust_score,
        "counts": sections.counts,
        "findings": sections.findings,
        "policy": sections.policy,
        "resolved": sections.resolved,
        "field": field,
        "documents": {"train": len(train_docs), "test": len(test_docs)},
        "skipped": {"train": train_skipped, "test": test_skipped},
        **extra,
    }

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"LeakLens trust score: {rep['trust_score']}/100")
        print(
            f"{len(train_docs)} train documents, {len(test_docs)} test documents"
            + (
                f", {train_skipped + test_skipped} skipped"
                if train_skipped + test_skipped
                else ""
            )
        )
        for finding in sections.findings:
            print(
                f"[{finding['severity'].upper()}] {finding['kind']} :: "
                f"{finding['column']} :: {finding['detail']}"
            )
        if not findings:
            print("No leakage found.")

    return 1 if a.fail_on_findings and findings else 0


def dl(argv) -> int:
    p = argparse.ArgumentParser(
        prog="leaklens dl",
        description=(
            "Audit deep-learning training code, configs and model artifacts "
            "for split overlap, reproducibility and unsafe checkpoints"
        ),
    )
    p.add_argument("path", help="repository root to audit")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--max-files",
        type=int,
        default=5000,
        help="maximum files walked (default 5000)",
    )
    p.add_argument(
        "--max-artifact-mb",
        type=int,
        default=64,
        help="maximum artifact size in MiB (default 64)",
    )
    p.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 1 when any diagnostic deep-learning finding is reported",
    )
    a = p.parse_args(argv)
    if a.max_files < 1:
        p.error("--max-files must be positive")
    if a.max_artifact_mb < 1:
        p.error("--max-artifact-mb must be positive")
    root = Path(a.path)
    if not root.is_dir():
        p.error(f"not a directory: {a.path}")

    from .analysis.dl import audit_dl

    report = audit_dl(
        root,
        max_files=a.max_files,
        max_artifact_bytes=a.max_artifact_mb * 1024 * 1024,
    )
    findings = list(report.findings)
    sections = assemble.build(findings)
    rep = {
        "trust_score": sections.trust_score,
        "counts": sections.counts,
        "findings": sections.findings,
        "policy": sections.policy,
        "resolved": sections.resolved,
        "dl": report.to_dict(),
    }

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"LeakLens trust score: {rep['trust_score']}/100")
        print(
            f"{len(findings)} diagnostic findings, "
            f"{len(report.artifacts)} artifacts inspected"
            + (f", {sum(report.skipped.values())} skipped" if report.skipped else "")
        )
        if report.skipped.get("yaml_not_installed"):
            print(
                'warning: YAML configs were skipped; pip install "leaklens-ml[yaml]"',
                file=sys.stderr,
            )
        for finding in sections.findings:
            print(
                f"[{finding['severity'].upper()}] {finding['kind']} :: "
                f"{finding['column']} :: {finding['detail']}"
            )
        if not findings:
            print("No deep-learning leakage found.")

    return 1 if a.fail_on_findings and findings else 0


def doctor(argv) -> int:
    from importlib import metadata as _meta

    EXTRAS = {
        "arena": "mlflow",
        "parquet": "pyarrow",
        "feather": "pyarrow",
        "excel": "openpyxl",
        "images": "PIL",
        "app": "fastapi",
        "yaml": "yaml",
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


def verify(argv) -> int:
    """Verify an audit manifest offline with the standard library."""
    from . import audit_manifest

    p = argparse.ArgumentParser(
        prog="leaklens verify",
        description=(
            "Verify an audit manifest offline. Detects a manifest or declared "
            "input changed after it was written; does not prove who wrote it."
        ),
    )
    p.add_argument("manifest", help="audit manifest JSON to verify")
    p.add_argument(
        "--root",
        help=("containment root for declared inputs (default: the current directory)"),
    )
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        issues = audit_manifest.verify(a.manifest, root=a.root)
    except audit_manifest.ManifestError as exc:
        if a.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"leaklens verify: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"ok": not issues, "issues": issues}, indent=2))
    elif issues:
        print(f"manifest FAILED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"manifest OK: {a.manifest}")
    return 0 if not issues else 1


def _gate_failure(
    outcome, trust_score: int, *, fail_under: float | None, fail_on_new: bool
) -> bool:
    """Failure predicates ignore ``diagnostic=True``, matching ``leakage.score()``."""
    if any(
        f.severity == "high" and not getattr(f, "diagnostic", False)
        for f in outcome.active
    ):
        return True
    if fail_under is not None and trust_score < fail_under:
        return True
    return bool(
        fail_on_new
        and outcome.diff is not None
        and any(not getattr(f, "diagnostic", False) for f in outcome.diff.new)
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "init":
        return init(argv[1:])
    if argv and argv[0] == "images":
        return images(argv[1:])
    if argv and argv[0] == "text":
        return text(argv[1:])
    if argv and argv[0] == "dl":
        return dl(argv[1:])
    if argv and argv[0] == "doctor":
        return doctor(argv[1:])
    if argv and argv[0] == "workspace":
        return workspace(argv[1:])
    if argv and argv[0] == "verify":
        return verify(argv[1:])
    p = argparse.ArgumentParser(
        prog="leaklens",
        description="Audit datasets and ML code for leakage",
        epilog=(
            "Run 'leaklens doctor' to check your install, 'leaklens init' to write "
            "a starter .leaklens.toml config, 'leaklens workspace' for a "
            "local interactive scan view, and 'leaklens verify' to check an "
            "audit manifest offline."
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
        help="file or folder of .py/.ipynb (and .sql with --sql) to scan",
    )
    p.add_argument(
        "--sql",
        action="store_true",
        default=None,
        help="include .sql files in --scan folders (needs the sql extra)",
    )
    p.add_argument(
        "--sql-dialect",
        choices=("ansi", "postgres", "bigquery", "snowflake"),
        default=None,
        help="declared dialect for SQL lineage (default: ansi)",
    )
    p.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "audit pipeline metadata (dvc.yaml/.dvc/dvc.lock, *.dag.json, notebook "
            "cell order, requirements) under --scan paths; opt-in diagnostics, "
            "needs the yaml extra for DVC files"
        ),
    )
    p.add_argument(
        "--drift",
        action="store_true",
        default=None,
        help="opt-in train/test distribution-drift diagnostics",
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
        default=None,
        help="include deterministic confidence, root-cause, and remediation analysis",
    )
    p.add_argument(
        "--graph-out",
        help="write the evidence graph (.mmd/.md, .dot/.gv or .json); "
        "implies --intelligence",
    )
    p.add_argument(
        "--graph-format",
        choices=("mermaid", "dot", "json"),
        help="override the evidence-graph format inferred from --graph-out",
    )
    p.add_argument(
        "--graph-baseline",
        help="compare the current evidence graph with this snapshot; "
        "implies --intelligence",
    )
    p.add_argument(
        "--graph-snapshot",
        help="write a validated evidence-graph snapshot for future comparisons; "
        "implies --intelligence",
    )
    p.add_argument(
        "--update-graph-snapshot",
        action="store_true",
        help="explicitly replace an existing --graph-snapshot file",
    )
    p.add_argument(
        "--fail-on-graph-new",
        action="store_true",
        help="opt in: exit 1 when the graph comparison finds new evidence",
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
        "--llm-prompt-template",
        metavar="PATH",
        help="custom prompt template file for LLM explanations",
    )
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
        "--manifest",
        metavar="PATH",
        help="write an unsigned, offline-verifiable audit manifest (leaklens verify)",
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
    manifest_path = a.manifest
    use_ai = choose(a.ai, cfg.ai, False)
    use_ai_llm = choose(a.ai_llm, cfg.ai_llm, False)
    if use_ai_llm:
        use_ai = True
    sql_settings = replace(
        cfg.sql,
        enabled=choose(a.sql, cfg.sql.enabled, False),
        dialect=choose(a.sql_dialect, cfg.sql.dialect, "ansi"),
    )
    drift_settings = replace(
        cfg.drift, enabled=choose(a.drift, cfg.drift.enabled, False)
    )
    scan_targets = a.scan if a.scan is not None else [str(path) for path in cfg.scan]
    use_graph_intelligence = choose(a.intelligence, cfg.graph_enabled, False)
    graph_baseline_path = choose(
        a.graph_baseline,
        str(cfg.graph_baseline_path) if cfg.graph_baseline_path is not None else None,
    )
    graph_snapshot_path = choose(
        a.graph_snapshot,
        str(cfg.graph_snapshot_path) if cfg.graph_snapshot_path is not None else None,
    )
    update_graph_snapshot = bool(a.update_graph_snapshot or cfg.graph_update_snapshot)
    fail_on_graph_new = bool(a.fail_on_graph_new or cfg.graph_fail_on_new)
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
    if a.update_graph_snapshot and not graph_snapshot_path:
        p.error("--update-graph-snapshot needs --graph-snapshot PATH")
    if a.fail_on_graph_new and not graph_baseline_path:
        p.error("--fail-on-graph-new needs --graph-baseline PATH")
    if update_graph_snapshot and not graph_snapshot_path:
        p.error("graph.update_snapshot needs graph.snapshot in config")
    if fail_on_graph_new and not graph_baseline_path:
        p.error("graph.fail_on_new needs graph.baseline in config")
    if a.llm and a.llm_max_findings < 1:
        p.error("--llm-max-findings must be at least 1")
    if a.llm_prompt_template and not a.llm:
        p.error("--llm-prompt-template requires --llm")
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

    graph_baseline = None
    if graph_baseline_path:
        from .intelligence import GraphBaselineError, GraphSnapshot

        try:
            graph_baseline = GraphSnapshot.load(graph_baseline_path)
        except GraphBaselineError as exc:
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
        if drift_settings.enabled:
            drift_report = audit_drift(tr, te, target=target, settings=drift_settings)
            fs += drift_report.findings
            rep["drift"] = drift_report.to_dict()
        rep["audit"] = audit(tr, target)
        if a.arena:
            try:
                from .arena import run
            except ImportError:
                p.error('install extras: pip install -e ".[arena]"')
            res = run(tr, target, time_col, a.out, a.tracking_uri)
            rep["arena"] = res.round(4).to_dict("records")
    for source in scan_targets:
        try:
            if sql_settings.enabled or str(source).lower().endswith(".sql"):
                fs += scan_path(source, sql=sql_settings, target=target)
            else:
                fs += scan_path(source)
        except ImportError as exc:
            p.error(str(exc))
    if a.pipeline and not scan_targets:
        p.error("--pipeline audits --scan paths; add --scan PATH")
    if a.pipeline:
        from .pipeline import PipelineUnavailable, audit_pipeline, is_yaml_metadata

        reports = []
        for source in scan_targets:
            try:
                report = audit_pipeline(
                    source,
                    strict_yaml=Path(source).is_file() and is_yaml_metadata(source),
                )
            except PipelineUnavailable as exc:
                p.error(str(exc))
            fs += report.findings
            reports.append(report.to_dict())
            if report.skipped.get("yaml_not_installed"):
                print(
                    "leaklens: DVC metadata skipped; install it with "
                    'pip install "leaklens-ml[yaml]"',
                    file=sys.stderr,
                )
        rep["pipeline"] = reports[0] if len(reports) == 1 else {"reports": reports}

    sections = assemble.build(
        fs,
        baseline=baseline,
        rules=cfg.suppressions,
        baseline_path=baseline_path,
        fail_on_new=fail_on_new,
    )
    outcome = sections.outcome
    fs = list(outcome.findings)
    active = list(outcome.active)
    policy_active = baseline is not None or bool(cfg.suppressions)
    if outcome.suppression is not None:
        for message in outcome.suppression.diagnostics:
            print(f"leaklens: warning: {message}", file=sys.stderr)
    explanations = None
    if a.llm:
        try:
            provider = llm.create_provider(
                a.llm_provider,
                a.llm_base_url,
                a.llm_model,
                prompt_template=a.llm_prompt_template,
            )
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
        "trust_score": sections.trust_score,
        **rep,
        "findings": sections.findings,
    }
    if policy_active:
        rep["policy"] = sections.policy
        if sections.resolved:
            rep["resolved"] = sections.resolved
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
    evidence_graph = None
    graph_snapshot = None
    graph_diff = None
    if (
        use_graph_intelligence
        or a.graph_out
        or graph_baseline_path
        or graph_snapshot_path
    ):
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
        report = analyze_findings(fs, impact=impact)
        evidence_graph = report.to_graph()
        intelligence_report = report.to_dict()
        rep["intelligence"] = intelligence_report
    if graph_baseline_path or graph_snapshot_path:
        from .intelligence import (
            GraphSnapshot,
            compare_graph_snapshots,
            snapshot_context,
        )

        mode = (
            "mixed"
            if a.train and scan_targets
            else "dataset"
            if a.train
            else "code"
            if scan_targets
            else "empty"
        )
        graph_snapshot = GraphSnapshot.from_findings(
            fs,
            graph=evidence_graph,
            complete=not any(f.kind == "parse_error" for f in fs),
            incomplete_reasons=_graph_completeness_reasons(
                scan_targets,
                fs,
                evidence_graph,
                sql=sql_settings.enabled,
            ),
            context=snapshot_context(
                mode=mode,
                target=target or "",
                test=bool(test_path),
                group=group_col or "",
                time=time_col or "",
                split=sql_settings.split_column or "",
                dialect=sql_settings.dialect if sql_settings.enabled else "",
                threshold=threshold,
                drift=drift_settings.enabled,
            ),
        )
        if graph_baseline is not None:
            graph_diff = compare_graph_snapshots(graph_baseline, graph_snapshot)
            rep["graph_baseline"] = graph_diff.to_dict()
    if graph_snapshot_path and graph_snapshot is not None:
        from .intelligence import GraphBaselineError

        try:
            written = graph_snapshot.write(
                graph_snapshot_path, update=update_graph_snapshot
            )
        except GraphBaselineError as exc:
            print(f"leaklens: {exc}", file=sys.stderr)
            return 2
        print(
            f"Wrote graph snapshot {written} "
            f"({len(graph_snapshot.evidence)} evidence records)",
            file=sys.stderr,
        )
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
            json.dump(
                to_sarif(
                    fs,
                    a.train or ".",
                    outcome=outcome,
                    graph_snapshot=graph_snapshot,
                    graph_diff=graph_diff,
                ),
                fh,
                indent=2,
            )
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
    if a.graph_out and evidence_graph is not None:
        try:
            _write_graph(evidence_graph, a.graph_out, a.graph_format)
        except (OSError, ValueError) as exc:
            print(f"leaklens: cannot write evidence graph: {exc}", file=sys.stderr)
            return 2
    gate = _gate_failure(
        outcome, rep["trust_score"], fail_under=fail_under, fail_on_new=fail_on_new
    )
    if fail_on_graph_new and graph_diff is not None and graph_diff.counts["new"]:
        gate = True
    if manifest_path:
        from . import audit_manifest

        inputs = [path for path in (a.train, test_path, *scan_targets) if path]
        if baseline is not None and baseline_path:
            inputs.append(baseline_path)
        options = {
            "target": target,
            "test": bool(test_path),
            "group": group_col,
            "time": time_col,
            "threshold": threshold,
            "sql": sql_settings.enabled,
            "sql_dialect": sql_settings.dialect if sql_settings.enabled else None,
            "drift": drift_settings.enabled,
            "pipeline": bool(a.pipeline),
            "ai": bool(use_ai),
            "ai_llm": bool(use_ai_llm),
            "fail_under": fail_under,
            "fail_on_new": fail_on_new,
        }
        try:
            document = audit_manifest.build(
                sections,
                inputs=inputs,
                options=options,
                gate={
                    "failed": gate,
                    "fail_under": fail_under,
                    "fail_on_new": fail_on_new,
                },
                root=Path.cwd(),
            )
            with open(manifest_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(audit_manifest.dumps(document))
        except (OSError, ValueError) as exc:
            print(f"leaklens: cannot write audit manifest: {exc}", file=sys.stderr)
            return 2
        print(
            f"Wrote audit manifest {manifest_path} ({document['manifest_digest']})",
            file=sys.stderr,
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
        if graph_diff is not None:
            counts = graph_diff.counts
            print(
                f"Graph baseline: {counts['new']} new, "
                f"{counts['unchanged']} unchanged, {counts['changed']} changed, "
                f"{counts['resolved']} resolved, "
                f"{counts['uncertain']} uncertain"
            )
            relations = graph_diff.relationship_counts
            if relations["added"] or relations["removed"]:
                print(
                    f"Graph relationships: {relations['added']} added, "
                    f"{relations['removed']} removed"
                )
            for item in graph_diff.evidence:
                if item.state == "unchanged" and not item.location_changed:
                    continue
                changed = ", ".join(item.changes)
                print(
                    f"    Graph evidence [{item.state}] {item.kind} :: "
                    f"{item.identity}" + (f" ({changed})" if changed else "")
                )
                if item.location_changed:
                    print(
                        f"        moved: {item.baseline_location or 'unknown'} -> "
                        f"{item.current_location or 'unknown'}"
                    )
            for name, completeness in (
                ("baseline", graph_diff.baseline_completeness),
                ("current", graph_diff.current_completeness),
            ):
                if completeness.complete:
                    continue
                detail = "; ".join(completeness.reasons) or "no reason recorded"
                print(
                    f"    Graph completeness ({name}): {completeness.state} ({detail})"
                )
            for warning in graph_diff.warnings:
                print(f"    Graph baseline warning: {warning}")
        if outcome.suppression is not None:
            print(f"Suppressions: {len(outcome.suppressed)} finding(s) suppressed")
        if drift_settings.enabled and "drift" in rep:
            print(
                "Drift (diagnostic, ignored by the score): "
                f"{rep['drift']['findings']} column(s) shifted"
            )
        if intelligence_report is not None:
            confidence = intelligence_report["confidence"]
            print(
                f"Intelligence confidence: {confidence['level']} ({confidence['score']:.2f})"
            )
            clusters = intelligence_report.get("clusters", [])
            if clusters:
                print(f"Evidence clusters: {len(clusters)}")
                for cluster in clusters[:3]:
                    kinds = ", ".join(cluster["finding_kinds"])
                    local = cluster["confidence"]
                    print(
                        f"    Cluster [{cluster['priority']}] {cluster['anchor']}: "
                        f"{kinds} ({local['level']} {local['score']:.2f})"
                    )
                    for note in cluster.get("measured_impact", [])[:2]:
                        print(f"        Measured: {note}")
                top_kinds = set(clusters[0]["finding_kinds"])
                next_step = next(
                    (
                        item
                        for item in intelligence_report["recommendations"]
                        if top_kinds & set(item["related_finding_kinds"])
                    ),
                    None,
                )
                if next_step is not None:
                    print(f"    Next step: {next_step['actions'][0]}")
                    print(f"    Verify: {next_step['validation_steps'][0]}")
            links = intelligence_report.get("links", [])
            if links:
                relations: dict[str, int] = {}
                for link in links:
                    relations[link["relation"]] = relations.get(link["relation"], 0) + 1
                summary = ", ".join(
                    f"{relation} {count}"
                    for relation, count in sorted(relations.items())
                )
                print(f"Evidence links: {len(links)} ({summary})")
            for cause in intelligence_report["root_causes"]:
                print(f"    Root cause: {cause['title']}")
            for recommendation in intelligence_report["recommendations"]:
                print(f"    Recommendation: {recommendation['title']}")
        if ai_report is not None:
            from .ai.schemas import confidence_label

            hypotheses = ai_report["hypotheses"]
            groups = ai_report["plan"]["root_causes"]
            print(
                f"AI (experimental): {len(hypotheses)} hypothesis(es), "
                f"{len(groups)} root-cause group(s); deterministic findings unchanged."
            )
            bases: list[str] = []
            for hypothesis in hypotheses:
                label = confidence_label(hypothesis.get("confidence_basis"))
                if label not in bases:
                    bases.append(label)
            if bases:
                print(f"    AI confidence bases: {', '.join(bases)}")
            selection = ai_report.get("selection")
            if selection:
                threshold = float(selection.get("threshold", 0) or 0)
                state = (
                    f"applied at p>={threshold:.2f}"
                    if selection.get("applied")
                    else f"not applied (no candidate met the acceptance rule); "
                    f"unselective output at p>={threshold:.2f} kept"
                )
                low, high = selection.get("risk_interval", (0.0, 1.0))
                print(
                    f"    AI selective reporting: {state} · coverage "
                    f"{float(selection.get('coverage', 0)):.1%} of emitted "
                    f"({selection.get('accepted')} of {selection.get('emitted')}) · "
                    f"selective risk {float(selection.get('selective_risk', 0)):.3f} "
                    f"(95% cluster-bootstrap {float(low):.3f}-{float(high):.3f}) · "
                    f"unselective error "
                    f"{float(selection.get('unselective_error_rate', 0)):.3f}; "
                    "no distribution-free guarantee."
                )
            buckets = {"supported": 0, "weak": 0, "insufficient": 0}
            for hypothesis in hypotheses:
                value = hypothesis.get("sufficiency")
                buckets[value if value in buckets else "weak"] += 1
            words = {
                "supported": "supported",
                "weak": "weak",
                "insufficient": "ranking only (insufficient calibration data)",
            }
            summary = ", ".join(
                f"{count} {words[key]}" for key, count in buckets.items() if count
            )
            if summary:
                print(f"    AI sufficiency: {summary}")
            for group in groups[:3]:
                print(
                    f"    AI root cause [{group['priority']}]: {group['title']} "
                    f"({group['source']})"
                )
            if ai_report["plan"].get("llm_summary"):
                print(f"    AI summary: {ai_report['plan']['llm_summary']}")
            for hypothesis in hypotheses[:3]:
                print(
                    f"    AI hypothesis "
                    f"[{confidence_label(hypothesis.get('confidence_basis'))}: "
                    f"{hypothesis['confidence']:.2f}"
                    + (
                        ", ranking only"
                        if hypothesis.get("sufficiency") == "insufficient"
                        else ""
                    )
                    + f"]: {hypothesis['title']} :: {hypothesis['location']}"
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
            if getattr(f, "diagnostic", False):
                suffix += " [diagnostic]"
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
    return 1 if gate else 0
