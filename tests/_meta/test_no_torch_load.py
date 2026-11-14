"""Phase 16 acceptance: the audit path imports no deep-learning framework.

The graph is read statically with ``ast``, so function-local imports count too.
The resolver is proven on fixtures first, then applied to every production
module: LeakLens must audit a project without importing torch, TensorFlow, Keras
or an ONNX runtime, and must never call ``torch.load``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "leaklens"

FORBIDDEN = frozenset(
    {
        "torch",
        "torchaudio",
        "torchvision",
        "tensorflow",
        "keras",
        "tf_keras",
        "onnx",
        "onnxruntime",
    }
)


def imports(source: str) -> list[tuple[str, str | None]]:
    """Absolute ``(module, alias)`` pairs, including function-local imports."""
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((alias.name, alias.asname) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.append((node.module, None))
            found.extend(
                (f"{node.module}.{alias.name}", alias.asname) for alias in node.names
            )
    return found


def violations(source: str) -> set[str]:
    """Forbidden top-level modules imported by ``source``."""
    return {
        module.split(".")[0]
        for module, _ in imports(source)
        if module.split(".")[0] in FORBIDDEN
    }


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def torch_load_calls(source: str) -> list[int]:
    """Line numbers of ``<torch>.load(...)`` calls in ``source``."""
    tree = ast.parse(source)
    aliases = {
        alias or "torch" for module, alias in imports(source) if module == "torch"
    }
    loaded = {
        alias or "load" for module, alias in imports(source) if module == "torch.load"
    }
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        head, _, tail = name.rpartition(".")
        if (
            tail == "load" and (head in aliases or name in aliases | loaded)
        ) or name in loaded:
            lines.append(node.lineno)
    return sorted(lines)


def test_the_resolver_flags_every_import_form():
    assert violations("import torch") == {"torch"}
    assert violations("import torch as T") == {"torch"}
    assert violations("from torch import load") == {"torch"}
    assert violations("from tensorflow.keras import layers") == {"tensorflow"}
    assert violations("def f():\n    import onnxruntime as ort\n") == {"onnxruntime"}


def test_relative_and_safe_imports_are_not_flagged():
    assert violations("from . import onnx\nfrom ..artifacts import onnx") == set()
    assert violations("import numpy as np\nimport json") == set()


def test_the_resolver_flags_torch_load_aliases():
    assert torch_load_calls("import torch\ntorch.load('a.pt')") == [2]
    assert torch_load_calls("import torch as T\nT.load('a.pt')") == [2]
    assert torch_load_calls("from torch import load\nload('a.pt')") == [2]
    assert torch_load_calls("from torch import load as L\nL('a.pt')") == [2]
    assert torch_load_calls(
        "def f():\n    import torch as T\n    return T.load('a.pt')\n"
    ) == [3]


def test_safe_loads_are_not_flagged():
    assert torch_load_calls("import numpy as np\nnp.load('a.npy')") == []
    assert torch_load_calls("from torch import save\nsave(1)") == []


def test_no_production_module_imports_a_deep_learning_framework():
    bad = []
    for path in sorted(PACKAGE.rglob("*.py")):
        found = violations(path.read_text(encoding="utf-8"))
        if found:
            bad.append((path.relative_to(PACKAGE).as_posix(), sorted(found)))
    assert bad == []


def test_no_production_module_calls_torch_load():
    bad = []
    for path in sorted(PACKAGE.rglob("*.py")):
        lines = torch_load_calls(path.read_text(encoding="utf-8"))
        if lines:
            bad.append((path.relative_to(PACKAGE).as_posix(), lines))
    assert bad == []
