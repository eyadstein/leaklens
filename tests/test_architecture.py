"""The one-way import direction Phase 13 promises (M5).

The graph is read statically with ``ast``, so function-local imports count too.
Modules are grouped into layers; an edge that crosses a forbidden boundary must
be listed in ``ALLOWED`` with the phase that removes it. A new edge fails the
test with both module names in the message.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "leaklens"

CORE = {
    "assemble",
    "availability",
    "baseline",
    "config",
    "evidence",
    "fingerprint",
    "fixes",
    "leakage",
    "manifest",
    "paths",
    "policy",
    "source",
    "suppression",
}
ENGINES = {
    "audit",
    "cvcheck",
    "drift",
    "flow",
    "images",
    "llm",
    "missingness",
    "partition",
    "readers",
    "scanner",
    "sqlscan",
    "upload_guard",
}
AI = "ai"
INTELLIGENCE = "intelligence"
SURFACES = {
    "api",
    "cli",
    "public_api",
    "report",
    "sarif",
    "ui",
    "workspace",
    "workspace_ui",
    "__main__",
}

# (importer, imported) -> (removal phase, reason). Every entry must still exist,
# and every allowed crossing must be listed: stale entries fail the test too.
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    ("config", "drift"): (
        19,
        "config reads [drift].enabled; the engine owns its defaults",
    ),
    ("config", "missingness"): (
        19,
        "config reads [missingness].enabled; the engine owns its defaults",
    ),
    ("config", "sqlscan"): (
        19,
        "config validates the SQL dialect with the engine's own parser",
    ),
    ("ai.artifact", "ai.learned"): (
        18,
        "load_artifact constructs LearnedModel; M4 deferred (docs/superpowers/plans)",
    ),
    ("ai.learned", "ai.artifact"): (
        18,
        "get_model loads through artifact.load_artifact; M4 deferred",
    ),
}

# Deferred M4 is the only cycle left in the non-surface graph.
ALLOWED_CYCLES: dict[frozenset[str], tuple[int, str]] = {
    frozenset({"ai.artifact", "ai.learned"}): (18, "M4 deferred; see ALLOWED"),
    frozenset({"fingerprint", "leakage"}): (
        18,
        "Finding lives in leakage and fingerprint_for needs it; splitting the "
        "dataclass out is the same leaf-module work as M4",
    ),
}

FORBIDDEN = {
    "core": {"engines", "ai", "intelligence", "surfaces"},
    "engines": {"ai", "intelligence", "surfaces"},
    "intelligence": {"ai"},
    "ai": {"intelligence"},
}


def _layer(module: str) -> str:
    head = module.split(".")[0]
    if head in CORE:
        return "core"
    if head in ENGINES:
        return "engines"
    if head in SURFACES:
        return "surfaces"
    if head == AI:
        return "ai"
    if head == INTELLIGENCE:
        return "intelligence"
    # Benchmarks, arena, __main__ and future one-off tools: nothing may import
    # them, and they may import anything.
    return "infra"


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    name = ".".join(parts)
    return name.removesuffix(".__init__")


def _local_imports(tree: ast.AST, module: str, modules: set[str]) -> set[str]:
    package = module.split(".")
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - node.level]
            base += [node.module] if node.module else []
            names = [".".join([*base, alias.name]) for alias in node.names]
        else:
            continue
        for name in names:
            name = name.removeprefix("leaklens.")
            parts = name.split(".")
            while parts and ".".join(parts) not in modules:
                parts.pop()
            if parts:
                found.add(".".join(parts))
    return {name for name in found if name != module}


def build_graph(root: Path = PACKAGE) -> dict[str, set[str]]:
    paths = {_module_name(path): path for path in root.rglob("*.py")}
    modules = set(paths)
    return {
        module: _local_imports(
            ast.parse(path.read_text(encoding="utf-8")), module, modules
        )
        for module, path in paths.items()
    }


def _violations(graph: dict[str, set[str]]) -> list[str]:
    found = []
    for importer, imported in sorted(
        (source, target) for source, targets in graph.items() for target in targets
    ):
        source_layer, target_layer = _layer(importer), _layer(imported)
        if target_layer not in FORBIDDEN.get(source_layer, set()):
            continue
        if (importer, imported) in ALLOWED:
            continue
        found.append(f"{importer} ({source_layer}) -> {imported} ({target_layer})")
    return found


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Cycles inside the non-surface graph, each reported as its module path."""
    nodes = {m for m in graph if _layer(m) not in {"surfaces", "infra"}}
    state: dict[str, int] = {}
    stack: list[str] = []
    found: list[list[str]] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt not in nodes:
                continue
            if state.get(nxt) == 1:
                found.append([*stack[stack.index(nxt) :], nxt])
            elif nxt not in state:
                visit(nxt)
        stack.pop()
        state[node] = 2

    for node in sorted(nodes):
        if node not in state:
            visit(node)
    return found


def test_no_forbidden_import_edge():
    violations = _violations(build_graph())
    assert not violations, "forbidden import edge(s):\n" + "\n".join(violations)


def test_no_new_cycle_outside_the_surfaces():
    cycles = [c for c in _cycles(build_graph()) if frozenset(c) not in ALLOWED_CYCLES]
    assert not cycles, "import cycle(s): " + "; ".join(" -> ".join(c) for c in cycles)


def test_every_allowed_cycle_still_exists():
    cycles = {frozenset(c) for c in _cycles(build_graph())}
    missing = [sorted(c) for c in ALLOWED_CYCLES if c not in cycles]
    assert not missing, f"stale ALLOWED_CYCLES entries: {missing}"


def test_every_allowance_is_still_real():
    graph = build_graph()
    stale = [edge for edge in ALLOWED if edge[1] not in graph.get(edge[0], set())]
    assert not stale, f"stale ALLOWED entries: {stale}"


def test_a_new_edge_fails_with_a_readable_message():
    graph = {"leakage": {"scanner", "fingerprint"}}
    messages = _violations(graph)
    assert messages == ["leakage (core) -> scanner (engines)"]
    assert ("leakage", "scanner") not in ALLOWED


@pytest.mark.parametrize("module, layer", [("source", "core"), ("scanner", "engines")])
def test_layers_cover_the_phase13_modules(module, layer):
    assert _layer(module) == layer
