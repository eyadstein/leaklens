"""Deterministic deep-learning code and artifact audit (Phase 16).

``audit_dl`` is one bounded pass: Python training code via ``ast``, documented
framework configs, requirement pins and model-artifact metadata. It never
imports a framework, unpickles, extracts an archive or writes anything. Every
finding is ``diagnostic=True``, so ``leakage.score()`` and existing exit gates
are unchanged; only ``leaklens dl --fail-on-findings`` turns them into a status.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...leakage import Finding
from ...paths import contained
from ..artifacts import formats
from . import config, training
from .checkpoints import inspect_artifacts

MAX_FILES = 5000
MAX_PY_FILES = 2000
MAX_REQUIREMENTS_BYTES = 1024 * 1024
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
        "dist",
        "build",
        "site-packages",
    }
)
CONFIG_TOKENS = frozenset({"config", "hparams", "hyperparams", "params"})
PYTHON_SKIPS = frozenset(
    {
        "python_file_too_large",
        "python_unreadable",
        "python_not_utf8",
        "python_syntax_error",
    }
)
ARTIFACT_JSON_TOKENS = frozenset({"model", "models", "metadata", "weights"})
ARTIFACT_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".onnx",
        ".pkl",
        ".pickle",
        ".joblib",
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".npz",
        ".xlsx",
        ".jar",
        ".whl",
    }
)


@dataclass(frozen=True, slots=True)
class DlReport:
    status: str
    findings: tuple[Finding, ...]
    signals: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    skipped: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "signals": list(self.signals),
            "artifacts": list(self.artifacts),
            "skipped": dict(sorted(self.skipped.items())),
        }


def _collect(base: Path, max_files: int) -> tuple[list[Path], dict[str, int]]:
    files: list[Path] = []
    skipped: dict[str, int] = {}
    for current, dirs, names in os.walk(base, followlinks=False):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in SKIP_DIRS and not directory.startswith(".")
        )
        for name in sorted(names):
            if name.startswith("."):
                continue
            path = Path(current) / name
            try:
                inside = contained(base, path.resolve())
            except OSError:
                skipped["unreadable_path"] = skipped.get("unreadable_path", 0) + 1
                continue
            if not inside:  # a symlink to a file outside the audited tree
                skipped["outside_root"] = skipped.get("outside_root", 0) + 1
                continue
            files.append(path)
            if len(files) > max_files:
                skipped["file_limit_reached"] = 1
                return files[:max_files], skipped
    return files, skipped


def _read_text(path: Path, limit: int) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def audit_dl(
    root: str | Path,
    *,
    max_files: int = MAX_FILES,
    max_artifact_bytes: int = formats.MAX_ARTIFACT_BYTES,
    max_config_bytes: int = config.MAX_CONFIG_BYTES,
) -> DlReport:
    """Audit one repository root, deterministically and without execution."""
    base = Path(root).resolve()
    files, skipped = _collect(base, max_files)
    python: list[Path] = []
    configs: list[Path] = []
    requirements: list[Path] = []
    artifacts: list[Path] = []
    for path in files:
        suffix = path.suffix.lower()
        tokens = training.tokens(path.stem)
        if suffix == ".py":
            python.append(path)
        elif suffix in {".yaml", ".yml", ".json"}:
            if tokens & CONFIG_TOKENS:
                configs.append(path)
            elif suffix == ".json" and tokens & ARTIFACT_JSON_TOKENS:
                artifacts.append(path)
        if path.name.startswith("requirements") and suffix == ".txt":
            requirements.append(path)
        if suffix in ARTIFACT_SUFFIXES:
            artifacts.append(path)
    if len(python) > MAX_PY_FILES:
        python = python[:MAX_PY_FILES]
        skipped["python_file_limit"] = 1

    signals = training.scan_python(base, python, skipped=skipped)
    python_parsed = len(python) - sum(skipped.get(reason, 0) for reason in PYTHON_SKIPS)
    configs_read = 0
    for path in configs:
        try:
            loaded = config.load_config(
                path,
                source=path.relative_to(base).as_posix(),
                max_bytes=max_config_bytes,
            )
        except config.DlUnavailable:
            skipped["yaml_not_installed"] = skipped.get("yaml_not_installed", 0) + 1
            continue
        if loaded is None:
            skipped["config_unreadable"] = skipped.get("config_unreadable", 0) + 1
            continue
        try:
            config_findings = config.config_signals(loaded)
        except RecursionError:
            skipped["config_unreadable"] = skipped.get("config_unreadable", 0) + 1
            continue
        configs_read += 1
        signals.extend(config_findings)

    pins: list[Finding] = []
    requirements_read = 0
    for path in requirements:
        text = _read_text(path, MAX_REQUIREMENTS_BYTES)
        if text is None:
            skipped["requirements_unreadable"] = (
                skipped.get("requirements_unreadable", 0) + 1
            )
            continue
        requirements_read += 1
        pins.extend(
            config.requirement_pin_findings(text, path.relative_to(base).as_posix())
        )

    checkpoint = inspect_artifacts(base, artifacts, max_bytes=max_artifact_bytes)
    for reason, count in checkpoint.skipped.items():
        skipped[reason] = skipped.get(reason, 0) + count

    findings = [
        *training.split_findings(signals),
        *training.training_findings(signals),
        *training.reproducibility_findings(signals),
        *pins,
        *checkpoint.findings,
    ]
    unique: dict[str, Finding] = {}
    for finding in findings:
        unique.setdefault(finding.fingerprint, finding)
    ordered = sorted(
        unique.values(),
        key=lambda finding: (
            finding.kind,
            finding.column,
            finding.stable_key or "",
            finding.detail,
        ),
    )
    parsed = bool(
        python_parsed or configs_read or requirements_read or checkpoint.artifacts
    )
    return DlReport(
        status="ok" if parsed else "no_artifacts",
        findings=tuple(ordered),
        signals=tuple(
            signal.to_dict()
            for signal in sorted(
                signals,
                key=lambda signal: (
                    signal.path,
                    signal.line,
                    signal.name,
                    signal.value,
                    signal.field,
                ),
            )
        ),
        artifacts=checkpoint.artifacts,
        skipped=dict(sorted(skipped.items())),
    )
