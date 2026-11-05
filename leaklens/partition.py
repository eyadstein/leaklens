"""Infer which variables hold train/test partitions.

Variable names are only a hint: ``X_train, X_test, y_train, y_test =
train_test_split(...)`` puts the held-out rows in the *second* returned value,
so a model tuned on ``X_holdout`` or on an arbitrarily named second element is
just as leaky as one tuned on ``X_test``.  This module derives roles from

1. explicit name hints (``*_test``, ``*_holdout``, ``*_train``, ...),
2. the positional convention of ``train_test_split`` when the assign targets
   look like an ``(X_train, X_test, y_train, y_test)`` unpacking, and
3. simple aliases such as ``X_te = X_test.copy()``.

It deliberately refuses to guess when a validation partition is present: tuning
on a validation split is a design choice, not the test-set leak this scanner
reports.
"""

from __future__ import annotations

import ast
import re

TEST_RE = re.compile(r"(?:^|[_\W])(test|holdout|hold_out)(?:$|[_\W])", re.IGNORECASE)
_TE_RE = re.compile(r"(?:^|[_\W])te(?:$|[_\W])", re.IGNORECASE)
TRAIN_RE = re.compile(r"(?:^|[_\W])(train|training)(?:$|[_\W])", re.IGNORECASE)
VALID_RE = re.compile(r"(?:^|[_\W])(val|valid|validation)(?:$|[_\W])", re.IGNORECASE)

_ALIAS_METHODS = {
    "copy",
    "values",
    "to_numpy",
    "astype",
    "reset_index",
    "set_index",
    "loc",
    "iloc",
}


def call_name(c: ast.Call) -> str:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def target_names(target) -> list[str]:
    """Flatten an assignment target (``a, b = ...`` or ``obj[c] = ...``)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: list[str] = []
        for element in target.elts:
            out.extend(target_names(element))
        return out
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        base = target.value
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            return [base.id]
    return []


def name_role(name: str) -> str | None:
    """Classify a variable name without looking at its definition."""
    if TEST_RE.search(name) or _TE_RE.search(name):
        return "test"
    if VALID_RE.search(name):
        return "validation"
    if TRAIN_RE.search(name):
        return "train"
    return None


def _alias_base(value):
    """The source name of a cheap alias expression, else ``None``."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        return value.value.id
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in _ALIAS_METHODS
        and isinstance(value.func.value, ast.Name)
    ):
        return value.func.value.id
    return None


def infer_roles(assigns) -> dict[str, str]:
    """Map variable names to ``train`` / ``test`` / ``validation`` roles."""
    roles: dict[str, str] = {}

    for assignment in assigns:
        value = assignment.value
        if not (isinstance(value, ast.Call) and call_name(value) == "train_test_split"):
            continue
        names: list[str] = []
        for target in assignment.targets:
            names.extend(target_names(target))
        if not names:
            continue

        explicit = {n: name_role(n) for n in names if name_role(n)}
        positional: dict[str, str] = {}
        # The conventional unpack is (X_train, X_test[, y_train, y_test[, ...]]).
        # Skip it when a validation split is present or name hints contradict it:
        # guessing there would turn legitimate validation tuning into a finding.
        has_validation = any(role == "validation" for role in explicit.values())
        if len(names) in (2, 4, 6) and not has_validation:
            candidate = {
                n: ("train" if i % 2 == 0 else "test") for i, n in enumerate(names)
            }
            if not any(n in explicit and explicit[n] != candidate[n] for n in names):
                positional = candidate

        for n in names:
            role = explicit.get(n) or positional.get(n)
            if role:
                roles.setdefault(n, role)

    # Propagate cheap aliases a bounded number of times so X_te = X_test.copy()
    # is recognised without attempting general dataflow analysis.
    for _ in range(3):
        changed = False
        for assignment in assigns:
            base = _alias_base(assignment.value)
            if base is None:
                continue
            role = roles.get(base) or name_role(base)
            if not role:
                continue
            for target in assignment.targets:
                for name in target_names(target):
                    if name not in roles and name_role(name) in (None, role):
                        roles[name] = role
                        changed = True
        if not changed:
            break
    return roles


def is_test(name: str, roles: dict[str, str] | None = None) -> bool:
    """Whether ``name`` denotes the held-out test partition."""
    if roles and name in roles:
        return roles[name] == "test"
    return name_role(name) == "test"


def any_test(names, roles: dict[str, str] | None = None) -> bool:
    return any(is_test(name, roles) for name in names)
