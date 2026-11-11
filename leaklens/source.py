"""Shared scanner vocabulary and AST helpers (ADR-019).

``cvcheck`` and ``flow`` import these from here, so the dependency direction is
one-way: analyzers -> source, and ``scanner`` re-exports them for compatibility.
Nothing in this module imports an engine, the AI layer or the intelligence layer.
"""

from __future__ import annotations

import ast
import hashlib

# Builders that only wrap the step which really learns from the data: the leak, if any,
# is inside them, so the class name has to be read from the wrapped steps.
WRAPPERS = {"make_pipeline", "Pipeline"}
HINTS = (
    "Scaler",
    "Imputer",
    "Encoder",
    "PCA",
    "SMOTE",
    "ADASYN",
    "Vectorizer",
    "Normalizer",
    "Transformer",
    "Selector",
    "SelectK",
    "SelectPercentile",
    "SelectFromModel",
    "RFE",
    "SequentialFeatureSelector",
    "RandomOverSampler",
    "RandomUnderSampler",
    "Binarizer",
    "Discretizer",
)
RESAMPLE = ("SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler")
BENIGN = ("LabelEncoder", "OrdinalEncoder", "OneHotEncoder")
UNSURE = ("ColumnTransformer",)
# Feature selectors learn which columns are useful from the label. Fitting them
# before the split uses held-out labels to choose the feature set.
FEATURE_SELECTION = (
    "Selector",
    "SelectK",
    "SelectPercentile",
    "SelectFromModel",
    "RFE",
    "SequentialFeatureSelector",
)
TARGET_ENCODERS = ("TargetEncoder", "CatBoostEncoder", "WOEEncoder", "CountEncoder")
FUNCS = {
    "scale",
    "minmax_scale",
    "robust_scale",
    "maxabs_scale",
    "quantile_transform",
    "power_transform",
}
SKIP = {".venv", "venv", ".git", "node_modules", "site-packages", "__pycache__"}


def _name(c: ast.Call) -> str:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _sev(cls: str) -> str:
    if any(b in cls for b in BENIGN):
        return "low"
    if any(u in cls for u in UNSURE):
        return "medium"
    return "high"


def _all_names(expr) -> set:
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}


def _tnames(t) -> set:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out = set()
        for e in t.elts:
            out |= _tnames(e)
        return out
    return set()


def _ast_digest(value, memo: dict[int, str]) -> str:
    """Structural digest of an AST node, stable across Python versions.

    ``ast.dump`` omits empty optional fields on newer versions, so it produces
    different text for the same code on 3.10 and 3.14. This walker hashes every
    declared field explicitly, never includes source positions, and hashes each
    subtree once (memoized by node identity).
    """
    if isinstance(value, ast.AST):
        known = memo.get(id(value))
        if known is not None:
            return known
        parts = [type(value).__name__]
        for name in value._fields:
            parts.append(name)
            parts.append(_ast_digest(getattr(value, name, None), memo))
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        memo[id(value)] = digest
        return digest
    if isinstance(value, (list, tuple)):
        items = "\x1e".join(_ast_digest(item, memo) for item in value)
        return hashlib.sha256(items.encode("utf-8")).hexdigest()
    return repr(value)


def _finding_keys(tree: ast.AST, nodes=None) -> dict[int, str]:
    """Line-independent stable keys for findings attached to an AST node.

    The key is a structural digest of the node (no source positions) plus the
    enclosing function/class name. Repeated identical statements in one scope
    get a deterministic ``#n`` suffix. Keys never contain line numbers,
    messages, timestamps or machine paths.
    """
    scopes: dict[int, str] = {}

    def scoped(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                for inner in ast.walk(child):
                    scopes[id(inner)] = name
                scoped(child, name + ".")
            else:
                scoped(child, prefix)

    scoped(tree, "")
    if nodes is None:
        nodes = tuple(ast.walk(tree))
    nodes = [n for n in nodes if isinstance(n, (ast.Call, ast.BinOp))]
    nodes.sort(key=lambda n: (n.lineno, n.col_offset, type(n).__name__))
    memo: dict[int, str] = {}
    keys: dict[int, str] = {}
    seen: dict[str, int] = {}
    for node in nodes:
        digest = _ast_digest(node, memo)[:16]
        scope = scopes.get(id(node), "")[:120]
        base = f"{scope}:{digest}" if scope else digest
        seen[base] = seen.get(base, 0) + 1
        keys[id(node)] = base if seen[base] == 1 else f"{base}#{seen[base]}"
    return keys


def _node_index(nodes) -> dict[type, list[ast.AST]]:
    """Nodes grouped by exact type, keeping ``ast.walk`` order within each group.

    The scanner passes only filter the tree by node type, so one grouped pass
    replaces a full ``isinstance`` scan per pass. Subclasses are not folded in:
    every loop below matches exact AST classes, as before.
    """
    index: dict[type, list[ast.AST]] = {}
    for node in nodes:
        index.setdefault(type(node), []).append(node)
    return index
