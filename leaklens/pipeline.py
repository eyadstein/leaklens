"""Read-only pipeline and MLOps intelligence (Phase 14 M1-M4).

Parses repository metadata only: DVC stages and locks, serialized DAG JSON,
notebook cell order and dependency declarations. It never executes a command,
imports an orchestrator or contacts anything.

Every finding this engine emits is a diagnostic: it ships ``scored=False`` in
the manifest and ``diagnostic=True`` on the finding, so the trust score, the
exit code and every existing finding stay exactly as they were. Pipeline
metadata is untrusted input: a malformed file is recorded as a skip, never
raised and never guessed around.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from .availability import EngineUnavailable, probe
from .evidence import Evidence
from .leakage import Finding
from .lineage import LineageEdge, sort_edges
from .paths import PathOutsideRoot, resolve_within

MAX_FILES = 5000
MAX_STAGES = 400
SKIP = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "graphify-out",
        "site-packages",
    }
)
YAML_NAMES = frozenset({"dvc.yaml", "dvc.lock", "params.yaml"})
READERS = frozenset(
    {
        "read_csv",
        "read_parquet",
        "read_excel",
        "read_json",
        "read_table",
        "read_feather",
        "read_pickle",
        "read_sql",
    }
)
WRITERS = frozenset(
    {
        "to_csv",
        "to_parquet",
        "to_excel",
        "to_json",
        "to_table",
        "to_feather",
        "to_pickle",
    }
)
MODEL_KEYS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "num_boost_round",
        "epochs",
        "batch_size",
        "hidden_size",
        "n_layers",
        "model",
        "model_name",
        "estimator",
    }
)
SEED_KEYS = frozenset({"seed", "random_state", "random_seed", "seeds"})
DATA_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".parquet",
        ".json",
        ".jsonl",
        ".feather",
        ".arrow",
        ".xlsx",
        ".xls",
    }
)
TRAIN_TOKENS = frozenset({"train", "training", "fit", "fitting"})
EVAL_TOKENS = frozenset(
    {
        "eval",
        "evaluate",
        "evaluation",
        "test",
        "testing",
        "score",
        "predict",
        "validate",
    }
)


class PipelineUnavailable(EngineUnavailable):
    """PyYAML is not installed; DVC metadata cannot be parsed."""


def available() -> bool:
    return importlib_util.find_spec("yaml") is not None


def require_yaml():
    """Import PyYAML or raise the typed, actionable unavailability error."""
    return probe(
        "yaml",
        "yaml",
        purpose="pipeline metadata parsing",
        label="PyYAML",
        error=PipelineUnavailable,
    )


def is_yaml_metadata(path: str | Path) -> bool:
    p = Path(path)
    return p.name in YAML_NAMES or p.suffix == ".dvc"


@dataclass(frozen=True, slots=True)
class Stage:
    """One pipeline stage read from metadata; paths are root-relative POSIX."""

    id: str
    source: str
    deps: tuple[str, ...] = ()
    outs: tuple[str, ...] = ()
    cmd: str = ""
    params: tuple[str, ...] = ()  # "params.yaml:key" references
    values: tuple[tuple[str, str], ...] = ()  # ("params.yaml:key", locked text)
    missing_digest: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "deps": list(self.deps),
            "outs": list(self.outs),
            "cmd": self.cmd,
            "params": list(self.params),
            "missing_digest": list(self.missing_digest),
        }


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """Stages, deterministic lineage edges and diagnostic findings."""

    status: str
    root: str
    stages: tuple[Stage, ...] = ()
    edges: tuple[LineageEdge, ...] = ()
    findings: tuple[Finding, ...] = ()
    unresolved: tuple[str, ...] = ()
    skipped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "root": self.root,
            "stages": [stage.to_dict() for stage in self.stages],
            "edges": [edge.to_dict() for edge in self.edges],
            "findings": len(self.findings),
            "unresolved": list(self.unresolved),
            "skipped": dict(sorted(self.skipped.items())),
        }


def _skip(skipped: dict[str, int], reason: str, count: int = 1) -> None:
    skipped[reason] = skipped.get(reason, 0) + count


def _clean(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _paths(value: Any) -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(value, str):
        found.add(_clean(value))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                found.add(_clean(item))
            elif isinstance(item, dict) and isinstance(item.get("path"), str):
                found.add(_clean(item["path"]))
    return tuple(sorted(item for item in found if item))


def _param_refs(value: Any) -> tuple[str, ...]:
    refs: set[str] = set()
    if isinstance(value, dict):
        for source, keys in value.items():
            if isinstance(keys, list):
                refs.update(
                    f"{_clean(str(source))}:{key}"
                    for key in keys
                    if isinstance(key, str)
                )
            elif isinstance(keys, str):
                refs.add(f"{_clean(str(source))}:{keys}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                refs.update(_param_refs(item))
            elif isinstance(item, str):
                refs.add(_clean(item))
    return tuple(sorted(refs))


def _param_values(value: Any) -> tuple[tuple[str, str], ...]:
    found: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        for source, keys in value.items():
            if not isinstance(keys, dict):
                continue
            for key, locked in keys.items():
                found.add((f"{_clean(str(source))}:{key}", str(locked)))
    return tuple(sorted(found))


def _load_yaml(path: Path, yaml_mod: Any, skipped: dict[str, int]) -> Any | None:
    try:
        data = yaml_mod.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - malformed metadata degrades to a skip
        _skip(skipped, "malformed_yaml")
        return None
    if data is None:
        return {}
    if not isinstance(data, dict):
        _skip(skipped, "unexpected_yaml")
        return None
    return data


def _dvc_stages(rel: str, data: dict, skipped: dict[str, int]) -> list[Stage]:
    stages = data.get("stages")
    if not isinstance(stages, dict):
        _skip(skipped, "no_stages")
        return []
    out = []
    for name in sorted(stages):
        body = stages[name]
        if not isinstance(body, dict):
            _skip(skipped, "unexpected_stage")
            continue
        out.append(
            Stage(
                id=f"{rel}:{name}",
                source=rel,
                deps=_paths(body.get("deps")),
                outs=_paths(body.get("outs")),
                cmd=str(body.get("cmd") or "")[:200],
                params=_param_refs(body.get("params")),
            )
        )
    return out


def _lock_stages(rel: str, data: dict, skipped: dict[str, int]) -> list[Stage]:
    stages = data.get("stages")
    if not isinstance(stages, dict):
        _skip(skipped, "no_stages")
        return []
    out = []
    for name in sorted(stages):
        body = stages[name]
        if not isinstance(body, dict):
            _skip(skipped, "unexpected_stage")
            continue
        missing = []
        outs = body.get("outs")
        if isinstance(outs, list):
            for item in outs:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("path"), str)
                    and not (item.get("md5") or item.get("hash"))
                ):
                    missing.append(_clean(item["path"]))
        out.append(
            Stage(
                id=f"{rel}:{name}",
                source=rel,
                deps=_paths(body.get("deps")),
                outs=_paths(outs),
                cmd=str(body.get("cmd") or "")[:200],
                params=_param_refs(body.get("params")),
                values=_param_values(body.get("params")),
                missing_digest=tuple(sorted(set(missing))),
            )
        )
    return out


def _dvc_file_stage(rel: str, data: dict, skipped: dict[str, int]) -> list[Stage]:
    if not isinstance(data.get("outs"), list) and not isinstance(
        data.get("deps"), list
    ):
        _skip(skipped, "unexpected_dvc_file")
        return []
    return [
        Stage(
            id=f"{rel}:{Path(rel).stem}",
            source=rel,
            deps=_paths(data.get("deps")),
            outs=_paths(data.get("outs")),
        )
    ]


def _dag_stages(rel: str, data: dict, skipped: dict[str, int]) -> list[Stage]:
    stages = data.get("stages")
    if not isinstance(stages, list):
        _skip(skipped, "dag_shape")
        return []
    out = []
    for item in stages:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            _skip(skipped, "dag_shape")
            continue
        out.append(
            Stage(
                id=f"{rel}:{item['name']}",
                source=rel,
                deps=_paths(item.get("deps")),
                outs=_paths(item.get("outs")),
                cmd=str(item.get("cmd") or "")[:200],
            )
        )
    return out


def _literal_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _cell_paths(tree: ast.AST) -> tuple[set[str], set[str]]:
    reads: set[str] = set()
    writes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        path = _literal_path(node.args[0])
        if path is None:
            continue
        if name in READERS:
            reads.add(_clean(path))
        elif name in WRITERS:
            writes.add(_clean(path))
        elif name == "open":
            mode = (
                _literal_path(node.args[1])
                if len(node.args) > 1
                else next(
                    (
                        _literal_path(item.value)
                        for item in node.keywords
                        if item.arg == "mode"
                    ),
                    None,
                )
            )
            (writes if mode and any(ch in mode for ch in "wa+") else reads).add(
                _clean(path)
            )
    return reads, writes


def _notebook_stages(rel: str, path: Path, skipped: dict[str, int]) -> list[Stage]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        _skip(skipped, "malformed_notebook")
        return []
    cells = doc.get("cells") if isinstance(doc, dict) else None
    if not isinstance(cells, list):
        _skip(skipped, "malformed_notebook")
        return []
    ordered = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        count = cell.get("execution_count")
        rank = (
            count if isinstance(count, int) and not isinstance(count, bool) else 10**9
        )
        ordered.append((rank, index, cell))
    ordered.sort(key=lambda item: (item[0], item[1]))
    stages = []
    for position, (_, index, cell) in enumerate(ordered, start=1):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            _skip(skipped, "malformed_cell")
            continue
        reads, writes = _cell_paths(tree)
        if not reads and not writes:
            continue
        stages.append(
            Stage(
                id=f"{rel}#exec-{position}",
                source=f"{rel}[cell {index}]",
                deps=tuple(sorted(reads)),
                outs=tuple(sorted(writes)),
                cmd="notebook cell",
            )
        )
    return stages


def _is_remote(path: str) -> bool:
    return "://" in path or (
        len(path) > 1 and path[1] == ":" and not path[2:3].isdigit()
    )


def _side(stage: Stage) -> str | None:
    name = stage.id.rsplit(":", 1)[-1].casefold()
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    if any(token in TRAIN_TOKENS for token in tokens):
        return "train"
    if any(token in EVAL_TOKENS for token in tokens):
        return "eval"
    return None


def _finding(
    kind: str,
    column: str,
    detail: str,
    evidence: Evidence,
    stable_key: str,
) -> Finding:
    return Finding(
        "low" if kind != "pipeline_split_overlap" else "high",
        kind,
        column,
        detail,
        evidence=(evidence,),
        stable_key=stable_key,
        diagnostic=True,
    )


def _unique(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    unique: dict[str, Finding] = {}
    for finding in findings:
        unique.setdefault(finding.fingerprint, finding)
    return tuple(sorted(unique.values(), key=lambda f: (f.kind, f.column, f.detail)))


def _files(anchor: Path, target: Path | None, skipped: dict[str, int]) -> list[Path]:
    if target is not None:
        return [target]
    paths = []
    for path in sorted(anchor.rglob("*")):
        if len(paths) >= MAX_FILES:
            _skip(skipped, "file_cap")
            break
        if not path.is_file() or SKIP & set(path.parts):
            continue
        paths.append(path)
    return paths


def _read_params(
    anchor: Path, file: str, yaml_mod: Any, skipped: dict[str, int]
) -> dict | None:
    try:
        path = resolve_within(anchor.resolve(), file)
    except PathOutsideRoot:
        # A lock must never make the audit read (and echo) a file outside the root.
        _skip(skipped, "params_outside_root")
        return None
    if not path.is_file():
        _skip(skipped, "missing_params_file")
        return None
    data = _load_yaml(path, yaml_mod, skipped)
    return data if isinstance(data, dict) else None


def _nested(data: Any, key: str) -> Any:
    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING = object()


def audit_pipeline(
    root: str | Path, *, strict_yaml: bool = False, max_stages: int = MAX_STAGES
) -> PipelineReport:
    """Read repository metadata under ``root``; never execute or contact anything.

    Without PyYAML the audit degrades to the formats that need no parser and
    records ``yaml_not_installed`` in ``skipped``. An explicitly named DVC file
    (or ``strict_yaml=True``) raises the typed ``PipelineUnavailable`` instead,
    because silently skipping the one artifact the caller named would be wrong.
    """
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"pipeline target not found: {base}")
    target = base if base.is_file() else None
    anchor = base.parent if target is not None else base
    skipped: dict[str, int] = {}
    yaml_missing = not available()
    if yaml_missing and (
        strict_yaml or (target is not None and is_yaml_metadata(target))
    ):
        require_yaml()
    yaml_mod = None if yaml_missing else require_yaml()
    if yaml_missing:
        _skip(skipped, "yaml_not_installed")

    files = _files(anchor, target, skipped)
    stages: list[Stage] = []
    findings: list[Finding] = []
    requirements: list[Path] = []

    for path in files:
        rel = path.relative_to(anchor).as_posix()
        name = path.name.casefold()
        try:
            if name in YAML_NAMES or path.suffix.casefold() == ".dvc":
                if yaml_mod is None:
                    continue
                data = _load_yaml(path, yaml_mod, skipped)
                if data is None:
                    continue
                if name == "dvc.yaml":
                    stages.extend(_dvc_stages(rel, data, skipped))
                elif name == "dvc.lock":
                    stages.extend(_lock_stages(rel, data, skipped))
                elif name == "params.yaml":
                    continue
                else:
                    stages.extend(_dvc_file_stage(rel, data, skipped))
            elif name == "dag.json" or name.endswith(".dag.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, ValueError):
                    _skip(skipped, "malformed_json")
                    continue
                if not isinstance(data, dict):
                    _skip(skipped, "dag_shape")
                    continue
                stages.extend(_dag_stages(rel, data, skipped))
            elif path.suffix.casefold() == ".ipynb":
                stages.extend(_notebook_stages(rel, path, skipped))
            elif path.suffix.casefold() == ".txt" and (
                name.startswith("requirements")
                or path.parent.name.casefold() == "requirements"
            ):
                requirements.append(path)
        except Exception:  # noqa: BLE001 - one bad file must not stop the audit
            _skip(skipped, "unreadable_file")

    stages = sorted(stages, key=lambda stage: stage.id)
    if len(stages) > max_stages:
        _skip(skipped, "stage_cap", len(stages) - max_stages)
        stages = stages[:max_stages]

    producers: dict[str, str] = {}
    for stage in stages:
        for out in stage.outs:
            producers.setdefault(out, stage.id)

    edges: list[LineageEdge] = []
    unresolved: set[str] = set()
    for stage in stages:
        for dep in stage.deps:
            producer = producers.get(dep)
            if producer is not None:
                edges.append(
                    LineageEdge(
                        stage=stage.id,
                        column=dep,
                        transform=producer,
                        locator=stage.source,
                        confidence="confirmed",
                    )
                )
            elif _is_remote(dep) or (anchor / dep).exists():
                edges.append(
                    LineageEdge(
                        stage=stage.id,
                        column=dep,
                        transform="remote input" if _is_remote(dep) else "root input",
                        locator=stage.source,
                        confidence="confirmed",
                    )
                )
            else:
                unresolved.add(dep)
                edges.append(
                    LineageEdge(
                        stage=stage.id,
                        column=dep,
                        transform="unresolved",
                        locator=stage.source,
                        confidence="unresolved",
                    )
                )

    findings.extend(_split_findings(stages))
    findings.extend(
        _finding(
            "pipeline_unresolved_lineage",
            dep,
            f"no parsed stage produces {dep!r} and it is not present in the "
            f"repository; the producer is unresolved, so any lineage claim "
            f"through it is unverified",
            Evidence(
                source="code",
                kind="parse",
                description=(
                    "a declared pipeline dependency has no producer stage and no "
                    "file on disk"
                ),
                location=dep,
                observed=f"declared dependency {dep!r}",
                expected="a producing stage or an existing input artifact",
            ),
            stable_key=f"unresolved:{dep}"[:180],
        )
        for dep in sorted(unresolved)
    )
    findings.extend(_digest_findings(stages))
    findings.extend(_drift_findings(anchor, stages, yaml_mod, skipped))
    findings.extend(_seed_findings(stages))
    findings.extend(_unpinned_findings(anchor, requirements, skipped))

    parsed = bool(stages or requirements)
    return PipelineReport(
        status="ok" if parsed else "no_artifacts",
        root=str(anchor),
        stages=tuple(stages),
        edges=sort_edges(edges),
        findings=_unique(findings),
        unresolved=tuple(sorted(unresolved)),
        skipped=dict(sorted(skipped.items())),
    )


def _is_data(path: str) -> bool:
    return Path(path).suffix.casefold() in DATA_SUFFIXES


def _split_findings(stages: list[Stage]) -> list[Finding]:
    trains = [stage for stage in stages if _side(stage) == "train"]
    evals = [stage for stage in stages if _side(stage) == "eval"]
    out = []
    for train in trains:
        for evaluate in evals:
            if train.id == evaluate.id:
                continue
            produced = sorted(
                artifact
                for artifact in set(train.outs) & set(evaluate.deps)
                if _is_data(artifact)
            )
            for artifact in produced:
                out.append(
                    _finding(
                        "pipeline_split_overlap",
                        artifact,
                        f"evaluation stage {evaluate.id!r} reads {artifact!r}, "
                        f"which training stage {train.id!r} produces; the evaluation "
                        f"set is not independent of the training run",
                        Evidence(
                            source="code",
                            kind="overlap",
                            description=(
                                "a training stage output is consumed directly by an "
                                "evaluation stage"
                            ),
                            location=evaluate.source,
                            observed=f"{evaluate.id} deps include {artifact}",
                            expected="evaluation reads data the training stage never saw",
                        ),
                        stable_key=f"out:{train.id}:{evaluate.id}:{artifact}"[:180],
                    )
                )
            shared = sorted(
                artifact
                for artifact in (set(train.deps) & set(evaluate.deps)) - set(produced)
                if _is_data(artifact)
            )
            for artifact in shared:
                out.append(
                    _finding(
                        "pipeline_split_overlap",
                        artifact,
                        f"training stage {train.id!r} and evaluation stage "
                        f"{evaluate.id!r} both read {artifact!r}; the two sides may "
                        f"not derive from disjoint data",
                        Evidence(
                            source="code",
                            kind="overlap",
                            description=(
                                "train and eval stages share an input artifact in the "
                                "pipeline graph"
                            ),
                            location=evaluate.source,
                            observed=f"shared input {artifact}",
                            expected="disjoint train and eval inputs",
                        ),
                        stable_key=f"in:{train.id}:{evaluate.id}:{artifact}"[:180],
                    )
                )
    return out


def _digest_findings(stages: list[Stage]) -> list[Finding]:
    out = []
    for stage in stages:
        for artifact in stage.missing_digest:
            out.append(
                _finding(
                    "pipeline_missing_digest",
                    artifact,
                    f"dvc.lock records output {artifact!r} for stage {stage.id!r} "
                    f"without a content digest; the artifact identity cannot be "
                    f"verified when the pipeline is re-run",
                    Evidence(
                        source="code",
                        kind="pattern",
                        description="a locked stage output has no md5 or hash",
                        location=stage.source,
                        observed=f"{stage.id} out {artifact!r} has no digest",
                        expected="every locked output carries a content digest",
                    ),
                    stable_key=f"digest:{stage.id}:{artifact}"[:180],
                )
            )
    return out


def _drift_findings(
    anchor: Path, stages: list[Stage], yaml_mod: Any, skipped: dict[str, int]
) -> list[Finding]:
    if yaml_mod is None:
        return []
    out = []
    cache: dict[str, dict | None] = {}
    for stage in stages:
        files = sorted({ref.split(":", 1)[0] for ref, _ in stage.values})
        for file in files:
            if file not in cache:
                cache[file] = _read_params(anchor, file, yaml_mod, skipped)
            current = cache[file]
            if current is None:
                continue
            for ref, locked in stage.values:
                ref_file, _, key = ref.partition(":")
                if ref_file != file:
                    continue
                observed = _nested(current, key)
                if observed is _MISSING:
                    detail = (
                        f"dvc.lock records {ref!r} = {locked!r} for stage "
                        f"{stage.id!r}, but the key is absent from {file!r}; the "
                        f"locked run cannot be reproduced from the current config"
                    )
                elif str(observed) == locked:
                    continue
                else:
                    detail = (
                        f"dvc.lock records {ref!r} = {locked!r} for stage "
                        f"{stage.id!r}, but {file!r} now has {observed!r}; the "
                        f"pipeline configuration drifted from the locked run"
                    )
                out.append(
                    _finding(
                        "pipeline_param_drift",
                        ref,
                        detail,
                        Evidence(
                            source="code",
                            kind="pattern",
                            description=(
                                "a locked parameter value differs from the current "
                                "parameter file"
                            ),
                            location=stage.source,
                            observed=f"lock {locked!r}",
                            expected=f"{file} value {observed!r}",
                        ),
                        stable_key=f"drift:{stage.id}:{ref}"[:180],
                    )
                )
    return out


def _seed_findings(stages: list[Stage]) -> list[Finding]:
    out = []
    for stage in stages:
        keys = {
            ref.partition(":")[2].rsplit(".", 1)[-1].casefold() for ref in stage.params
        }
        keys |= {
            ref.partition(":")[2].rsplit(".", 1)[-1].casefold()
            for ref, _ in stage.values
        }
        if not keys:
            # No parameter metadata: a seed may still be hard-coded in code, so
            # reporting this stage would be a guess, not evidence.
            continue
        looks_training = _side(stage) == "train" or bool(keys & MODEL_KEYS)
        if not looks_training or keys & SEED_KEYS:
            continue
        out.append(
            _finding(
                "no_random_state",
                stage.id,
                f"pipeline stage {stage.id!r} configures training parameters "
                f"without a seed; the same metadata can produce a different model, "
                f"so a re-run cannot be compared with the evaluated run",
                Evidence(
                    source="code",
                    kind="pattern",
                    description="a training stage declares model parameters but no seed",
                    location=stage.source,
                    observed=f"params {sorted(keys) or 'none'}",
                    expected="a seed or random_state in the stage parameters",
                ),
                stable_key=f"seed:{stage.id}"[:180],
            )
        )
    return out


def _unpinned_findings(
    anchor: Path, files: list[Path], skipped: dict[str, int]
) -> list[Finding]:
    out = []
    for path in files:
        rel = path.relative_to(anchor).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            _skip(skipped, "unreadable_file")
            continue
        seen: set[str] = set()
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            package = line.split(";", 1)[0].strip()
            if "==" in package or "://" in package:
                continue
            name = package.split("[", 1)[0]
            for marker in ("===", ">=", "<=", "~=", "!=", ">", "<", "="):
                name = name.split(marker, 1)[0]
            name = name.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(
                _finding(
                    "pipeline_unpinned_dependency",
                    rel,
                    f"dependency {name!r} in {rel!r} has no exact version pin; "
                    f"re-running the pipeline can resolve a different version and "
                    f"change the trained model",
                    Evidence(
                        source="code",
                        kind="pattern",
                        description=("a requirements entry has no exact version pin"),
                        location=rel,
                        observed=package[:200],
                        expected="package==version",
                    ),
                    stable_key=f"pin:{rel}:{name}"[:180],
                )
            )
    return out
