"""Orchestration for the optional, offline AI intelligence layer.

The deterministic scanner and dataset detectors remain the source of truth.
This module only produces an :class:`AIReport` containing separate hypotheses,
a root-cause plan, measured model cards and explicit limitations.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from leaklens.leakage import Finding, score

from .learned import get_model, learned_hypotheses
from .reasoning import CompletionProvider, build_plan, enrich_plan
from .schemas import GLOBAL_LIMITATIONS, AIReport, Hypothesis
from .semantics import CodeSource, SchemaInfo, analyze_semantics

MAX_CODE_FILES = 200
MAX_SOURCE_BYTES = 512 * 1024
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetInput:
    """Metadata-only dataset description; row values never enter the AI layer."""

    train: pd.DataFrame
    test: pd.DataFrame | None = None
    target: str | None = None
    group: str | None = None
    time: str | None = None

    def schema(self) -> SchemaInfo:
        columns = tuple(
            (str(name), str(dtype)) for name, dtype in self.train.dtypes.items()
        )
        test_columns = (
            tuple(str(name) for name in self.test.columns)
            if self.test is not None
            else ()
        )
        return SchemaInfo(
            columns=columns,
            target=self.target,
            group=self.group,
            time=self.time,
            test_columns=test_columns,
        )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return path.name


def _read_source(path: Path) -> CodeSource | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_SOURCE_BYTES:
        data = data[:MAX_SOURCE_BYTES]
    text = data.decode("utf-8", "replace")
    label = _display_path(path)
    if path.suffix.lower() != ".ipynb":
        return CodeSource(label=label, source=text)
    try:
        notebook = json.loads(text)
    except (TypeError, ValueError):
        return CodeSource(label=label, source=text)
    cells = []
    raw_cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    for cell in raw_cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            cells.append("".join(str(part) for part in source))
        elif isinstance(source, str):
            cells.append(source)
    return CodeSource(label=label, source=text, cells=tuple(cells))


def collect_code_sources(
    paths: Iterable[str | Path],
    *,
    max_files: int = MAX_CODE_FILES,
    max_bytes: int = MAX_SOURCE_BYTES,
) -> tuple[CodeSource, ...]:
    """Read Python files and notebook code cells without executing anything."""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if len(found) >= max_files:
                    break
                if any(part in _SKIP_DIRS for part in candidate.parts):
                    continue
                if (
                    candidate.suffix.lower() in {".py", ".ipynb"}
                    and candidate.is_file()
                ):
                    found.append(candidate)
        elif path.is_file():
            found.append(path)
        if len(found) >= max_files:
            break
    sources = []
    for path in found[:max_files]:
        source = _read_source(path)
        if source is not None:
            sources.append(source)
    return tuple(sources)


def analyze(
    findings: Sequence[Finding] = (),
    *,
    code: Sequence[CodeSource] = (),
    dataset: DatasetInput | None = None,
    llm_provider: CompletionProvider | None = None,
    trust_score: int | None = None,
) -> AIReport:
    """Build an AI report without modifying deterministic findings or scores."""

    findings = tuple(findings)
    schema = dataset.schema() if dataset is not None else None
    semantic = analyze_semantics(code=code, schema=schema, findings=findings)
    learned = learned_hypotheses(code, findings) if code else ()
    hypotheses: tuple[Hypothesis, ...] = tuple(dict.fromkeys(semantic + learned))

    plan = build_plan(findings, hypotheses, target=dataset.target if dataset else None)
    if llm_provider is not None:
        plan = enrich_plan(plan, llm_provider)

    _evaluation, model = get_model(compare_candidates=False)
    limitations = list(GLOBAL_LIMITATIONS)
    limitations.append(
        "Column names, comments and notebook text are treated as untrusted data."
    )
    if any(item.concept == "untrusted_instruction" for item in hypotheses):
        limitations.append(
            "Instruction-like text was reported as evidence but was never followed."
        )
    if plan.enrichment == "deterministic_fallback":
        limitations.append(
            "LLM enrichment failed validation; the deterministic plan is shown instead."
        )

    if code and schema is not None:
        scope = "mixed"
    elif code:
        scope = "code"
    else:
        scope = "dataset"
    deterministic_score = (
        int(trust_score) if trust_score is not None else int(score(findings))
    )

    return AIReport(
        scope=scope,
        hypotheses=hypotheses,
        plan=plan,
        models=(model.card,),
        deterministic_findings=len(findings),
        target=dataset.target if dataset else None,
        deterministic_trust_score=deterministic_score,
        limitations=tuple(limitations),
    )


def analyze_code(
    sources: Sequence[CodeSource],
    findings: Sequence[Finding] = (),
    *,
    llm_provider: CompletionProvider | None = None,
) -> AIReport:
    return analyze(findings, code=sources, llm_provider=llm_provider)


def analyze_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    target: str | None = None,
    group: str | None = None,
    time: str | None = None,
    findings: Sequence[Finding] = (),
    trust_score: int | None = None,
) -> AIReport:
    return analyze(
        findings,
        dataset=DatasetInput(
            train=train, test=test, target=target, group=group, time=time
        ),
        trust_score=trust_score,
    )


__all__ = [
    "DatasetInput",
    "analyze",
    "analyze_code",
    "analyze_dataset",
    "collect_code_sources",
]
