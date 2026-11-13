"""Static RAG wiring analysis: loader -> splitter -> index -> retriever (Phase 15 M5).

The check is wiring recognition, not a dataflow proof. It reports a file that
constructs a retrieval index and loads documents from a literal path that
contains a declared evaluation path. Literals are compared as normalised path
parts; ``..`` and glob suffixes cannot widen the match. Chunk-file overlap is the
deterministic dedup audit run on the chunk files, not this static pass.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ...evidence import Evidence
from ...leakage import Finding
from ...paths import PathOutsideRoot, resolve_within

MAX_FILES = 500
MAX_EXAMPLES = 8
GLOB_CHARS = frozenset("*?[")
LOADER_NAMES = frozenset(
    {
        "DirectoryLoader",
        "TextLoader",
        "CSVLoader",
        "JSONLoader",
        "PyPDFLoader",
        "UnstructuredLoader",
        "SimpleDirectoryReader",
        "load_documents",
    }
)
INDEX_NAMES = frozenset(
    {
        "VectorStoreIndex",
        "FAISS",
        "Chroma",
        "Pinecone",
        "Qdrant",
        "Weaviate",
        "from_documents",
        "from_texts",
        "add_documents",
        "add_texts",
        "add_nodes",
    }
)


@dataclass(frozen=True, slots=True)
class RagHit:
    file: str
    line: int
    loader: str
    index: str
    path: str


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _has_traversal(value: str) -> bool:
    return ".." in value.replace("\\", "/").split("/")


def _parts(value: str) -> list[str]:
    text = value.replace("\\", "/").strip()
    parts = []
    for part in text.split("/"):
        if part in {"", "."}:
            continue
        cut = min(
            (part.index(char) for char in GLOB_CHARS if char in part), default=len(part)
        )
        if cut:
            parts.append(part[:cut])
    return parts


def _contains(parent: list[str], child: list[str]) -> bool:
    return bool(parent) and len(child) >= len(parent) and child[: len(parent)] == parent


def _loaders(tree: ast.AST) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in LOADER_NAMES:
            continue
        values = [*node.args, *(keyword.value for keyword in node.keywords)]
        for value in values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.append((node.lineno, _call_name(node), value.value))
    return found


def _indexes(tree: ast.AST) -> list[tuple[int, str]]:
    return [
        (node.lineno, _call_name(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in INDEX_NAMES
    ]


def scan_rag(
    root: str | Path,
    eval_paths: list[str],
    *,
    max_files: int = MAX_FILES,
) -> tuple[list[Finding], list[RagHit]]:
    """Return findings and hits for every parsed file with a shared index root."""
    base = Path(root)
    if not base.is_dir():
        raise ValueError(f"RAG root is not a directory: {base}")
    declared = [_parts(value) for value in eval_paths if not _has_traversal(value)]
    declared = [parts for parts in declared if parts]
    if not declared:
        return [], []

    findings: list[Finding] = []
    hits: list[RagHit] = []
    scanned = 0
    for path in sorted(base.rglob("*.py")):
        if scanned >= max_files:
            break
        if not path.is_file():
            continue
        try:
            resolve_within(base, path.relative_to(base).as_posix())
        except (PathOutsideRoot, ValueError):
            continue
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            continue
        loaders = _loaders(tree)
        indexes = _indexes(tree)
        if not loaders or not indexes:
            continue
        relative = path.relative_to(base).as_posix()
        matched: list[tuple[int, str, str, str]] = []
        seen: set[tuple[int, str]] = set()
        for line, loader, literal in loaders:
            if _has_traversal(literal):
                continue
            loader_parts = _parts(literal)
            for eval_path in eval_paths:
                eval_parts = _parts(eval_path)
                if not _contains(loader_parts, eval_parts):
                    continue
                key = (line, literal)
                if key in seen:
                    continue
                seen.add(key)
                matched.append((line, loader, literal, eval_path))
        if not matched:
            continue
        index_line, index_name = indexes[0]
        hits.extend(
            RagHit(relative, line, loader, index_name, literal)
            for line, loader, literal, _ in matched
        )
        summary = Evidence(
            source="code",
            kind="pattern",
            description="retrieval index is built over a loader root that contains eval data",
            location=f"file={relative}",
            observed=(
                f"matched={len(matched)} index={index_name} line={index_line} "
                f"loader={matched[0][1]} line={matched[0][0]}"
            ),
            expected="eval documents stay outside every indexed loader root",
        )
        examples = [
            Evidence(
                source="code",
                kind="pattern",
                description=f"{loader} loads documents for the index",
                observed=f"path={literal[:120]} contains eval={eval_path[:80]}",
            )
            for line, loader, literal, eval_path in matched[:MAX_EXAMPLES]
        ]
        findings.append(
            Finding(
                severity="high",
                kind="rag_index_leakage",
                column=relative,
                detail=(
                    f"{len(matched)} loader path(s) in {relative} include the declared "
                    "evaluation path, so the index can return eval documents."
                ),
                evidence=(summary, *examples),
                stable_key=relative,
                diagnostic=True,
            )
        )
    return findings, hits
