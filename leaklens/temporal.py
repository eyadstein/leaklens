from __future__ import annotations

import ast

from .leakage import Finding

_TARGET_NAMES = {"y", "target", "targets", "label", "labels", "outcome"}


def _names(node: ast.AST) -> set[str]:
    return {item.id.lower() for item in ast.walk(node) if isinstance(item, ast.Name)}


def _target_expression(node: ast.AST, aliases: dict[str, set[str]]) -> bool:
    names = _names(node)
    if names & _TARGET_NAMES:
        return True
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            key = child.slice
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.lower() in _TARGET_NAMES
            ):
                return True
        if (
            isinstance(child, ast.Name)
            and child.id.lower() in aliases
            and aliases[child.id.lower()] & _TARGET_NAMES
        ):
            return True
    return False


def _negative_shift(call: ast.Call) -> bool:
    values = list(call.args)
    values.extend(item.value for item in call.keywords if item.arg == "periods")
    for value in values:
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            return True
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, (int, float))
            and value.value < 0
        ):
            return True
    return False


def _includes_current_or_future(call: ast.Call) -> bool:
    center = next((item.value for item in call.keywords if item.arg == "center"), None)
    return isinstance(center, ast.Constant) and center.value is True


def scan_temporal(src: str, label: str = "<src>") -> list[Finding]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    split_lines = [
        item.lineno
        for item in ast.walk(tree)
        if isinstance(item, ast.Call)
        and (
            (isinstance(item.func, ast.Name) and item.func.id == "train_test_split")
            or (isinstance(item.func, ast.Attribute) and item.func.attr == "split")
        )
    ]
    if not split_lines:
        return []
    split_line = min(split_lines)

    aliases: dict[str, set[str]] = {}
    for item in ast.walk(tree):
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            value = item.value
            if value is None or not _target_expression(value, aliases):
                continue
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    aliases[target.id.lower()] = _names(value) | _TARGET_NAMES

    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for item in ast.walk(tree):
        if not isinstance(item, ast.Call) or item.lineno >= split_line:
            continue
        func = item.func
        if not isinstance(func, ast.Attribute):
            continue
        receiver = func.value
        if func.attr == "shift" and _negative_shift(item):
            kind = "future_target_feature_before_split"
            detail = "Negative shift reads a later row before the train/test split."
        elif func.attr == "rolling" and _includes_current_or_future(item):
            kind = "future_target_feature_before_split"
            detail = "Rolling window includes the current or a future row before the train/test split."
        else:
            continue
        if not _target_expression(receiver, aliases):
            continue
        marker = (kind, item.lineno)
        if marker in seen:
            continue
        seen.add(marker)
        findings.append(Finding("high", kind, label, f"line {item.lineno}: {detail}"))
    return findings
