"""M-S3 acceptance: the manifest verifier imports only safe modules.

The graph is read statically with ``ast``, so function-local imports count too.
Verification must not be able to deserialize or execute project content, and it
must work without a third-party dependency.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "leaklens" / "audit_manifest.py"

# Modules that could execute, deserialize or reach the network. None belongs in
# a verifier that reads untrusted documents and untrusted project files.
FORBIDDEN = frozenset(
    {
        "ctypes",
        "http",
        "importlib",
        "marshal",
        "pickle",
        "runpy",
        "socket",
        "subprocess",
        "urllib",
    }
)


def imports(source: str) -> list[tuple[str, int]]:
    """Absolute top-level module names and the import's level, for one source."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((alias.name.split(".")[0], 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(((node.module or "").split(".")[0], node.level))
    return found


def test_the_resolver_sees_function_local_and_relative_imports():
    assert imports("import json") == [("json", 0)]
    assert imports("def f():\n    import pickle\n") == [("pickle", 0)]
    assert imports("from .paths import resolve_within") == [("paths", 1)]
    assert imports("from . import __version__") == [("", 1)]


def test_the_verifier_imports_only_the_standard_library_and_first_party_modules():
    bad = [
        (name, level)
        for name, level in imports(MODULE.read_text(encoding="utf-8"))
        if level == 0 and name not in sys.stdlib_module_names
    ]
    assert bad == []


def test_the_verifier_imports_no_execution_or_deserialization_module():
    used = {name for name, _ in imports(MODULE.read_text(encoding="utf-8"))}
    assert used & FORBIDDEN == set()
