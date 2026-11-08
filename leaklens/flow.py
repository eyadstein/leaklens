from __future__ import annotations

import ast
import warnings

from .leakage import Finding
from .scanner import (
    BENIGN,
    FEATURE_SELECTION,
    FUNCS,
    HINTS,
    RESAMPLE,
    TARGET_ENCODERS,
    _all_names,
    _finding_keys,
    _name,
    _sev,
)

REPLACEABLE = {
    "fit_on_full_data",
    "transformer_before_split",
    "preprocess_before_split",
    "resample_before_split",
    "encoder_before_split",
}


def _targets(t) -> set:
    if isinstance(t, ast.Name):
        return {t.id}
    if isinstance(t, (ast.Tuple, ast.List)):
        out: set = set()
        for e in t.elts:
            out |= _targets(e)
        return out
    if isinstance(t, (ast.Subscript, ast.Attribute)):
        b = t.value
        while isinstance(b, (ast.Subscript, ast.Attribute)):
            b = b.value
        return {b.id} if isinstance(b, ast.Name) else set()
    return set()


def _inner_cls(node, cls: str) -> str:
    """Classify a ColumnTransformer by the steps inside it."""
    names = [
        _name(n) for n in ast.walk(node) if isinstance(n, ast.Call) and n is not node
    ]
    hints = [n for n in names if any(h in n for h in HINTS)]
    if not hints:
        return cls
    bad = [n for n in hints if not any(b in n for b in BENIGN)]
    return bad[0] if bad else hints[0]


def _transform(c, ctor, node_of, funcs):
    f = c.func
    if isinstance(f, ast.Name) and f.id in funcs:
        return f.id, f.id
    if isinstance(f, ast.Attribute) and f.attr in ("fit_transform", "fit_resample"):
        b = f.value
        if isinstance(b, ast.Name):
            cls, node = ctor.get(b.id, ""), node_of.get(b.id)
        elif isinstance(b, ast.Call):
            cls, node = _name(b), b
        else:
            return None
        if not any(h in cls for h in HINTS):
            return None
        if node is not None and "ColumnTransformer" in cls:
            cls = _inner_cls(node, cls)
        return cls, f"{cls}.{f.attr}"
    return None


def scan_flow(
    src: str,
    label: str = "<src>",
    cells=None,
    *,
    tree: ast.AST | None = None,
    keys: dict[int, str] | None = None,
    nodes: tuple[ast.AST, ...] | None = None,
) -> list[Finding]:
    """Transformed data that later feeds a split, wherever the transform sits in the source."""
    if tree is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(src)
        except SyntaxError:
            return []
    if nodes is None:
        nodes = tuple(ast.walk(tree))
    ctor, node_of, assigns, calls, funcs = {}, {}, [], [], set()
    for n in nodes:
        if isinstance(n, ast.Assign):
            assigns.append(n)
            if isinstance(n.value, ast.Call):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        ctor[t.id] = _name(n.value)
                        node_of[t.id] = n.value
        elif isinstance(n, ast.Call):
            calls.append(n)
        elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith(
            "sklearn.preprocessing"
        ):
            for al in n.names:
                if al.name in FUNCS:
                    funcs.add(al.asname or al.name)
    assigns.sort(key=lambda a: a.lineno)

    def is_split(c):
        if _name(c) == "train_test_split":
            return True
        f = c.func
        return (
            isinstance(f, ast.Attribute)
            and f.attr == "split"
            and isinstance(f.value, ast.Name)
            and any(k in ctor.get(f.value.id, "") for k in ("Fold", "Split"))
        )

    splits = sorted((c for c in calls if is_split(c)), key=lambda c: c.lineno)
    if not splits:
        return []
    if keys is None:
        keys = _finding_keys(tree, nodes)

    def loc(line):
        if cells:
            h = [(s, i) for s, i in cells if s <= line]
            if h:
                s, i = h[-1]
                return f"{label}[cell {i}]:{line - s + 1}"
        return f"{label}:{line}"

    out = []
    for a in assigns:
        for c in ast.walk(a.value):
            if not isinstance(c, ast.Call):
                continue
            res = _transform(c, ctor, node_of, funcs)
            if not res:
                continue
            cls, call = res
            taint = set()
            for t in a.targets:
                taint |= _targets(t)
            if not taint:
                continue
            events = sorted(
                [("a", b.lineno, b) for b in assigns if b.lineno > a.lineno]
                + [("s", s.lineno, s) for s in splits if s.lineno > a.lineno],
                key=lambda e: (e[1], e[0] == "s"),
            )
            hit = None
            for kind, _, node in events:
                if kind == "a":
                    used = _all_names(node.value) & taint
                    tg = set()
                    for t in node.targets:
                        tg |= _targets(t)
                    if used:
                        taint |= tg
                    else:
                        taint -= tg
                else:
                    args = list(node.args) + [
                        kw.value for kw in node.keywords if kw.arg in ("X", "y")
                    ]
                    if any(_all_names(x) & taint for x in args):
                        hit = node
                        break
            if hit is None:
                continue
            if any(r in cls for r in RESAMPLE):
                kd = "resample_before_split"
            elif any(b in cls for b in BENIGN):
                kd = "encoder_before_split"
            elif any(t in cls for t in TARGET_ENCODERS):
                kd = "target_encode_before_split"
            elif any(h in cls for h in FEATURE_SELECTION):
                kd = "feature_selection_before_split"
            elif "ColumnTransformer" in cls:
                kd = "transformer_before_split"
            else:
                kd = "preprocess_before_split"
            out.append(
                Finding(
                    _sev(cls),
                    kd,
                    loc(c.lineno),
                    f"{call}() output feeds the split at {loc(hit.lineno)}, so the split rows already shaped it",
                    stable_key=keys[id(c)],
                )
            )
    return out


def merge(core, extra):
    """Prefer the dataflow finding when both report the same call (in place); drop exact duplicates."""
    emap = {f.column: f for f in extra if f.kind in REPLACEABLE}
    refs = " ".join(f.detail for f in extra if f.kind.endswith("_outside_cv"))
    out, used = [], set()
    for f in core:
        if f.kind == "fit_on_full_data" and f"at {f.column};" in refs:
            continue
        if f.kind in REPLACEABLE and f.column in emap:
            if f.column not in used:
                used.add(f.column)
                out.append(emap[f.column])
        else:
            out.append(f)
    out += [f for f in extra if not (f.kind in REPLACEABLE and f.column in used)]
    seen, res = set(), []
    for f in out:
        k = (f.kind, f.column)
        if k not in seen:
            seen.add(k)
            res.append(f)
    return res
