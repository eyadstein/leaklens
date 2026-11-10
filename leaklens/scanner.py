from __future__ import annotations

import ast
import hashlib
import json
import re
import threading
import warnings
from collections import OrderedDict
from pathlib import Path

from .leakage import Finding

FIT = {"fit", "fit_transform", "fit_resample"}
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
# Splitters whose folds are random only when shuffle=True (fold order is deterministic without it).
SHUFFLED_SPLITS = {"KFold", "StratifiedKFold", "StratifiedGroupKFold"}
# Repeated splitters always shuffle, so a missing random_state always makes them non-reproducible.
ALWAYS_SHUFFLED = {
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "GroupShuffleSplit",
}
FUNCS = {
    "scale",
    "minmax_scale",
    "robust_scale",
    "maxabs_scale",
    "quantile_transform",
    "power_transform",
}
EVAL = re.compile(r"(^|[_\W])(test|val|valid|validation|eval)([_\W]|$)", re.IGNORECASE)
SKIP = {".venv", "venv", ".git", "node_modules", "site-packages", "__pycache__"}
# Process-local parsed-source cache. Keyed by exact content, so edits and file
# replacements miss; no mtime/size heuristic can return a stale tree. Bounded by
# entry count and total source characters.
_PARSE_CACHE_MAX_ENTRIES = 128
_PARSE_CACHE_MAX_CHARS = 2_000_000
_PARSE_CACHE_MAX_SOURCE = 1_000_000
_parse_cache: OrderedDict[
    str, tuple[ast.AST, dict[int, str], tuple[ast.AST, ...], int]
] = OrderedDict()
_parse_cache_chars = 0
_parse_lock = threading.Lock()


def _name(c: ast.Call) -> str:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def _step(expr, aliases: dict[str, str] | None = None) -> str:
    """Class that learns behind an expression, through import aliases and pipeline builders."""
    if not isinstance(expr, ast.Call):
        return ""
    aliases = aliases or {}
    names = [_name(expr)]
    if names[0] in WRAPPERS:
        for a in expr.args:
            names += [_name(x) for x in ast.walk(a) if isinstance(x, ast.Call)]
    for n in names:
        real = aliases.get(n, n)
        if any(h in real for h in HINTS):
            return real
    return aliases.get(names[0], names[0])


def _sev(cls: str) -> str:
    if any(b in cls for b in BENIGN):
        return "low"
    if any(u in cls for u in UNSURE):
        return "medium"
    return "high"


def _all_names(expr) -> set:
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}


def _strings(expr) -> set:
    return {
        s.value
        for s in ast.walk(expr)
        if isinstance(s, ast.Constant) and isinstance(s.value, str)
    }


def _names(expr) -> set:
    """Names in an expression, or none if it indexes (X[idx] is usually a partition, not the full data)."""
    if any(isinstance(n, ast.Subscript) for n in ast.walk(expr)):
        return set()
    return _all_names(expr)


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


def _parse_source(
    src: str,
) -> tuple[
    ast.AST | None, dict[int, str] | None, tuple[ast.AST, ...] | None, str | None
]:
    """Parse and index one source once; repeated scans reuse the bounded cache."""
    global _parse_cache_chars
    key = hashlib.sha256(src.encode("utf-8")).hexdigest()
    with _parse_lock:
        cached = _parse_cache.get(key)
        if cached is not None:
            _parse_cache.move_to_end(key)
            return cached[0], cached[1], cached[2], None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError as exc:
        return None, None, None, str(exc)
    nodes = tuple(ast.walk(tree))
    keys = _finding_keys(tree, nodes)
    if len(src) <= _PARSE_CACHE_MAX_SOURCE:
        with _parse_lock:
            previous = _parse_cache.pop(key, None)
            if previous is not None:
                _parse_cache_chars -= previous[3]
            _parse_cache[key] = (tree, keys, nodes, len(src))
            _parse_cache_chars += len(src)
            while _parse_cache and (
                len(_parse_cache) > _PARSE_CACHE_MAX_ENTRIES
                or _parse_cache_chars > _PARSE_CACHE_MAX_CHARS
            ):
                _, (_, _, _, size) = _parse_cache.popitem(last=False)
                _parse_cache_chars -= size
    return tree, keys, nodes, None


def _is_tts(v) -> bool:
    return isinstance(v, ast.Call) and _name(v) == "train_test_split"


def _kw_true(c, name: str) -> bool:
    """Whether a keyword argument is the literal True."""
    return any(
        kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in c.keywords
    )


def _has_seed(c) -> bool:
    """random_state passed, or a **kwargs expansion that might carry it (assume it does)."""
    return any(kw.arg == "random_state" or kw.arg is None for kw in c.keywords)


def _kw_false(c, name: str) -> bool:
    """Whether a keyword argument is the literal False."""
    return any(
        kw.arg == name
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
        for kw in c.keywords
    )


# Calls that seed torch's global generator (what random_split uses when it is given no generator).
GLOBAL_SEEDERS = {"seed_everything", "set_seed"}


def _imports_torch(index: dict[type, list[ast.AST]]) -> bool:
    """Whether the file imports torch (keeps random_split from matching unrelated libraries)."""
    for n in index.get(ast.Import, ()):
        if any(al.name.split(".")[0] == "torch" for al in n.names):
            return True
    return any(
        (n.module or "").split(".")[0] == "torch" for n in index.get(ast.ImportFrom, ())
    )


def _seeds_torch_globally(c: ast.Call) -> bool:
    """torch.manual_seed(...) or a seed_everything()/set_seed() helper: seeds the global generator."""
    f = c.func
    if _name(c) in GLOBAL_SEEDERS:
        return True
    if _name(c) != "manual_seed":
        return False
    if isinstance(f, ast.Name):  # from torch import manual_seed
        return True
    root = f.value if isinstance(f, ast.Attribute) else None
    while isinstance(root, ast.Attribute):
        root = root.value
    return isinstance(root, ast.Name) and root.id == "torch"


def _random_split_unseeded(c: ast.Call) -> bool:
    """random_split(dataset, lengths) with no generator: the split comes from the global RNG.

    generator is the third positional parameter. A *args/**kwargs expansion might carry it,
    so assume it does (same rule as _has_seed). An explicit generator is the caller's choice
    and is never flagged; only a missing generator or a literal generator=None is.
    """
    if any(isinstance(a, ast.Starred) for a in c.args):
        return False
    if any(kw.arg is None for kw in c.keywords):
        return False
    gen = c.args[2] if len(c.args) >= 3 else None
    for kw in c.keywords:
        if kw.arg == "generator":
            gen = kw.value
    return gen is None or (isinstance(gen, ast.Constant) and gen.value is None)


def scan_source(src: str, label: str = "<src>", cells=None) -> list[Finding]:
    from .cvcheck import scan_cv
    from .flow import merge, scan_flow

    tree, keys, nodes, error = _parse_source(src)
    if tree is None or keys is None or nodes is None:
        return [Finding("low", "parse_error", label, error or "invalid syntax")]
    index = _node_index(nodes)
    return merge(
        _scan_core(src, label, cells, tree=tree, keys=keys, nodes=nodes, index=index),
        scan_flow(src, label, cells, tree=tree, keys=keys, nodes=nodes, index=index)
        + scan_cv(src, label, cells, tree=tree, keys=keys, nodes=nodes, index=index),
    )


def _scan_core(
    src: str,
    label: str = "<src>",
    cells=None,
    *,
    tree: ast.AST | None = None,
    keys: dict[int, str] | None = None,
    nodes: tuple[ast.AST, ...] | None = None,
    index: dict[type, list[ast.AST]] | None = None,
) -> list[Finding]:
    if tree is None or keys is None:
        tree, keys, nodes, error = _parse_source(src)
        if tree is None or keys is None:
            return [Finding("low", "parse_error", label, error or "invalid syntax")]
    if nodes is None:
        nodes = tuple(ast.walk(tree))
    if index is None:
        index = _node_index(nodes)
    ctor, calls, funcs, assigns = {}, [], set(), []
    # Estimators imported under a local name (`from ... import KBinsDiscretizer as KBD`).
    # Matching a class on its local name only hides it from the hint lists (issue #333).
    sk_aliases: dict[str, str] = {}
    for n in index.get(ast.ImportFrom, ()):
        if (n.module or "").startswith("sklearn"):
            for al in n.names:
                if al.asname:
                    sk_aliases[al.asname] = al.name

    pandas_aliases, get_dummies_aliases = set(), set()

    for n in index.get(ast.Assign, ()):
        assigns.append(n)
        if isinstance(n.value, ast.Call):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    ctor[t.id] = _step(n.value, sk_aliases)
    calls = list(index.get(ast.Call, ()))
    for n in index.get(ast.ImportFrom, ()):
        if (n.module or "").startswith("sklearn.preprocessing"):
            for al in n.names:
                if al.name in FUNCS:
                    funcs.add(al.asname or al.name)
        elif n.module == "pandas":
            for al in n.names:
                if al.name == "get_dummies":
                    get_dummies_aliases.add(al.asname or al.name)
    for n in index.get(ast.Import, ()):
        for al in n.names:
            if al.name == "pandas":
                pandas_aliases.add(al.asname or "pandas")

    assigns.sort(key=lambda n: n.lineno)

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

    splits = [c.lineno for c in calls if is_split(c)]
    first = min(splits) if splits else None

    # dataflow: which names hold the FULL dataset that train_test_split divides?
    tts = sorted(
        (c for c in calls if _name(c) == "train_test_split"), key=lambda c: c.lineno
    )
    tts_line, full = None, set()
    if tts:
        tts_line = tts[0].lineno
        for a in tts[0].args:
            full |= _all_names(a)
        defs = {}
        for a in assigns:
            if a.lineno < tts_line and not _is_tts(a.value):
                for t in a.targets:
                    for nm in _tnames(t):
                        defs.setdefault(nm, []).append(a.value)
        work = list(full)
        while work:
            nm = work.pop()
            for v in defs.get(nm, []):
                for x in _all_names(v):
                    if x in defs and x not in full:
                        full.add(x)
                        work.append(x)
        for a in assigns:
            if _is_tts(a.value):
                continue
            if _names(a.value) & full:
                for t in a.targets:
                    full |= _tnames(t)

    # names and string columns the split treats as the label (label argument(s) only, args[1:]):
    # train_test_split(X, y) -> y; df["target"] -> "target"; X's columns are features, never the label
    target: set = set()
    if tts:
        for a in tts[0].args[1:]:
            target |= _all_names(a) | _strings(a)

    # Issue #4: detect future-looking features derived from the target.
    # Track target variables and aliases assigned from target-derived expressions.
    # Only a bare target variable is a target name. For df["target"],
    # track the column name, not the dataframe variable "df".
    target_names = {
        arg.id for arg in (tts[0].args[1:] if tts else []) if isinstance(arg, ast.Name)
    }
    target_columns = {
        s for arg in (tts[0].args[1:] if tts else []) for s in _strings(arg)
    }
    target_derived = set(target_names)
    changed = True
    while changed:
        changed = False
        for assignment in assigns:
            if assignment.lineno >= first if first is not None else False:
                continue
            value_names = _all_names(assignment.value)
            value_columns = _strings(assignment.value)
            if value_names & target_derived or value_columns & target_columns:
                for target_node in assignment.targets:
                    for name in _tnames(target_node):
                        if name not in target_derived:
                            target_derived.add(name)
                            changed = True

    def uses_target(expr):
        return bool(
            (_all_names(expr) & target_derived) or (_strings(expr) & target_columns)
        )

    def loc(line):
        if cells:
            hit = [(s, i) for s, i in cells if s <= line]
            if hit:
                s, i = hit[-1]
                return f"{label}[cell {i}]:{line - s + 1}"
        return f"{label}:{line}"

    out = []

    # Future-looking target features leak information from later observations.
    if first is not None:
        for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
            if c.lineno >= first or not isinstance(c.func, ast.Attribute):
                continue
            method = c.func.attr
            receiver = c.func.value
            if not uses_target(receiver):
                continue

            shift_period = (
                c.args[0]
                if c.args
                else next(
                    (kw.value for kw in c.keywords if kw.arg in {"periods", "period"}),
                    None,
                )
            )
            future_shift = (
                method == "shift"
                and isinstance(shift_period, ast.UnaryOp)
                and isinstance(shift_period.op, ast.USub)
                and isinstance(shift_period.operand, ast.Constant)
                and isinstance(shift_period.operand.value, (int, float))
                and shift_period.operand.value > 0
            )
            centered_rolling = method == "rolling" and any(
                kw.arg == "center"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in c.keywords
            )
            if future_shift or centered_rolling:
                reason = (
                    "negative shift exposes future target values"
                    if future_shift
                    else "centered rolling windows include future target values"
                )
                out.append(
                    Finding(
                        "high",
                        "future_target_feature_before_split",
                        loc(c.lineno),
                        f"{ast.unparse(c)} creates a future-looking target feature before "
                        f"the split at {loc(first)}: {reason}",
                        stable_key=keys[id(c)],
                    )
                )

    # Issue #12: get_dummies on the full dataset before splitting learns the
    # category vocabulary from held-out rows. This is low severity because it
    # usually affects feature-column consistency rather than target information.
    if first is not None:
        for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
            if _name(c) != "get_dummies":
                continue
            nested_in_split = bool(tts and any(c is node for node in ast.walk(tts[0])))
            if c.lineno >= first and not nested_in_split:
                continue
            result_flows_into_split = any(
                c is node
                and bool(
                    set().union(*(_tnames(target) for target in assignment.targets))
                    & full
                )
                for assignment in assigns
                if assignment.lineno < first
                for node in ast.walk(assignment.value)
            )
            if (
                result_flows_into_split
                or nested_in_split
                or (c.args and _all_names(c.args[0]) & full)
            ):
                out.append(
                    Finding(
                        "low",
                        "get_dummies_before_split",
                        loc(c.lineno),
                        f"{ast.unparse(c)} encodes the full dataset before the split "
                        f"at {loc(first)}; category columns are learned using held-out rows",
                        stable_key=keys[id(c)],
                    )
                )

    # random_split() draws from torch's global RNG unless given a generator, so a
    # torch.manual_seed()/seed_everything() call anywhere in the file makes it reproducible.
    torch_unseeded_global = _imports_torch(index) and not any(
        _seeds_torch_globally(c) for c in calls
    )

    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        nm = _name(c)

        is_get_dummies = (isinstance(f, ast.Name) and f.id in get_dummies_aliases) or (
            isinstance(f, ast.Attribute)
            and f.attr == "get_dummies"
            and isinstance(f.value, ast.Name)
            and f.value.id in pandas_aliases
        )

        if is_get_dummies and first is not None and c.lineno < first:
            flows_to_split = any(
                any(isinstance(n, ast.Call) and n is c for n in ast.walk(a.value))
                and bool(set().union(*(_tnames(t) for t in a.targets)) & full)
                for a in assigns
                if a.lineno == c.lineno
            )
            if flows_to_split:
                out.append(
                    Finding(
                        "low",
                        "get_dummies_before_split",
                        loc(c.lineno),
                        f"pandas.get_dummies() runs before the split at {loc(first)}, "
                        "so held-out categories can shape the encoded columns",
                        stable_key=keys[id(c)],
                    )
                )

        # low severity: reproducibility - a random split nobody can rerun
        if (
            nm == "train_test_split"
            and not _has_seed(c)
            and not _kw_false(c, "shuffle")
        ):
            out.append(
                Finding(
                    "low",
                    "no_random_state",
                    loc(c.lineno),
                    "train_test_split() without random_state: a different split on every run, so results are not reproducible",
                    stable_key=keys[id(c)],
                )
            )
        elif nm in SHUFFLED_SPLITS and not _has_seed(c) and _kw_true(c, "shuffle"):
            out.append(
                Finding(
                    "low",
                    "no_random_state",
                    loc(c.lineno),
                    f"{nm}(shuffle=True) without random_state: folds differ on every run, so results are not reproducible",
                    stable_key=keys[id(c)],
                )
            )
        elif nm in ALWAYS_SHUFFLED and not _has_seed(c):
            out.append(
                Finding(
                    "low",
                    "no_random_state",
                    loc(c.lineno),
                    f"{nm}() without random_state: it always shuffles, so folds differ on every run, so results are not reproducible",
                    stable_key=keys[id(c)],
                )
            )
        elif (
            nm == "random_split" and torch_unseeded_global and _random_split_unseeded(c)
        ):
            out.append(
                Finding(
                    "low",
                    "no_random_state",
                    loc(c.lineno),
                    "random_split() without a generator: the split comes from torch's global RNG, "
                    "so it differs on every run, so results are not reproducible",
                    stable_key=keys[id(c)],
                )
            )
        if isinstance(f, ast.Name) and f.id in funcs:
            cls, call = f.id, f.id
        elif isinstance(f, ast.Attribute) and f.attr in FIT:
            b = f.value
            if isinstance(b, ast.Name):
                cls = ctor.get(b.id, "")
            elif isinstance(b, ast.Call):
                cls = _step(b, sk_aliases)
            else:
                cls = ""
            if not any(h in cls for h in HINTS):
                continue
            call = f"{cls}.{f.attr}"
        else:
            continue
        args = ", ".join(ast.unparse(a) for a in c.args)
        if EVAL.search(args):
            sev = "medium" if any(b in cls for b in BENIGN) else "high"
            out.append(
                Finding(
                    sev,
                    "fit_on_eval_data",
                    loc(c.lineno),
                    f"{call}({args}) learns from evaluation data",
                    stable_key=keys[id(c)],
                )
            )
        elif first is not None and c.lineno < first:
            if any(r in cls for r in RESAMPLE):
                kind = "resample_before_split"
            elif any(b in cls for b in BENIGN):
                kind = "encoder_before_split"
            elif any(t in cls for t in TARGET_ENCODERS):
                kind = "target_encode_before_split"
            elif any(h in cls for h in FEATURE_SELECTION):
                kind = "feature_selection_before_split"
            elif any(u in cls for u in UNSURE):
                kind = "transformer_before_split"
            else:
                kind = "preprocess_before_split"
            out.append(
                Finding(
                    _sev(cls),
                    kind,
                    loc(c.lineno),
                    f"{call}() runs before the split at {loc(first)}",
                    stable_key=keys[id(c)],
                )
            )
        elif (
            tts_line is not None
            and c.lineno > tts_line
            and any(_names(a) & full for a in c.args)
        ):
            out.append(
                Finding(
                    "medium",
                    "fit_on_full_data",
                    loc(c.lineno),
                    f"{call}({args}) fits on the full dataset after the split at {loc(tts_line)}; "
                    "fine only for a final refit",
                    stable_key=keys[id(c)],
                )
            )

    # issue #20: features built from the target or from group statistics computed on all rows.
    # .agg/.apply usually build summary tables (EDA), so only a result written into a column
    # (e.g. df['m'] = df.groupby(...).agg(...)) counts as a feature; .transform is row-aligned already.
    assigned_cols = set()
    for a in assigns:
        if a.targets and isinstance(a.targets[0], ast.Subscript):
            for n in ast.walk(a.value):
                if isinstance(n, ast.Call):
                    assigned_cols.add(id(n))
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        if not (
            isinstance(f, ast.Attribute) and f.attr in ("transform", "agg", "apply")
        ):
            continue
        if f.attr in ("agg", "apply") and id(c) not in assigned_cols:
            continue
        if first is None or c.lineno >= first:
            continue
        sel, base = None, None
        if (
            isinstance(f.value, ast.Subscript)
            and isinstance(f.value.value, ast.Call)
            and _name(f.value.value) == "groupby"
        ):
            sel, base = f.value.slice, f.value.value
        elif isinstance(f.value, ast.Call) and _name(f.value) == "groupby":
            base = f.value
        if base is None:
            continue
        used = (_all_names(sel) | _strings(sel)) if sel is not None else set()
        group_keys = set()
        for a in base.args:
            group_keys |= _all_names(a) | _strings(a)
        if used & target or group_keys & target:
            out.append(
                Finding(
                    "high",
                    "target_encode_before_split",
                    loc(c.lineno),
                    f"{ast.unparse(c)} builds a feature from the target across all rows "
                    f"before the split at {loc(first)}",
                    stable_key=keys[id(c)],
                )
            )
        else:
            out.append(
                Finding(
                    "medium",
                    "group_stat_before_split",
                    loc(c.lineno),
                    f"{ast.unparse(c)} uses group statistics from all rows before the "
                    f"split at {loc(first)}; held-out rows help build the feature",
                    stable_key=keys[id(c)],
                )
            )
    # issue #77: hand-rolled scaling with full-data statistics before the split,
    # e.g. (X - X.mean()) / X.std() or X / X.max(). Statistics taken from a train
    # partition only occur after the split, so they are never reported.
    STATS = {"mean", "std", "var", "min", "max", "median", "quantile"}

    def stat_recv(n):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in STATS
        ):
            if (
                n.args
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id in ("np", "numpy")
            ):
                return set().union(*(_all_names(a) for a in n.args))
            return _all_names(n.func.value)
        return None

    if first is not None:
        inner = {
            id(k)
            for b in nodes
            if isinstance(b, ast.BinOp)
            for k in (b.left, b.right)
            if isinstance(k, ast.BinOp)
        }
        binops = [b for b in nodes if isinstance(b, ast.BinOp)]
        for b in sorted(binops, key=lambda x: (x.lineno, x.col_offset)):
            if id(b) in inner or b.lineno >= first:
                continue
            recv, skip = set(), set()
            for n in ast.walk(b):
                r = stat_recv(n)
                if r is not None:
                    recv |= r
                    skip |= {id(m) for m in ast.walk(n)}
            if not recv:
                continue
            outside = {
                m.id
                for m in ast.walk(b)
                if isinstance(m, ast.Name) and id(m) not in skip
            }
            if outside & recv:
                out.append(
                    Finding(
                        "high",
                        "manual_scale_before_split",
                        loc(b.lineno),
                        f"{ast.unparse(b)} scales with statistics from all rows "
                        f"before the split at {loc(first)}",
                        stable_key=keys[id(b)],
                    )
                )

    # fillna with a statistic computed on all rows before the split (e.g. df.fillna(df.mean())).
    if first is not None:
        for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
            if _name(c) != "fillna" or c.lineno >= first:
                continue
            vals = list(c.args) + [kw.value for kw in c.keywords]
            uses_stat = False
            for v in vals:
                for n in ast.walk(v):
                    if (
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr in ("mean", "median", "mode")
                    ):
                        uses_stat = True
                        break
                if uses_stat:
                    break
            if not uses_stat:
                continue
            out.append(
                Finding(
                    "medium",
                    "fillna_before_split",
                    loc(c.lineno),
                    f"{ast.unparse(c)} imputes with a statistic from all rows "
                    f"before the split at {loc(first)}",
                    stable_key=keys[id(c)],
                )
            )

    # Target-derived aggregates pasted back with map/merge, and frequency
    # encodings built from the full column, both before the split.
    if first is not None:
        column_assign_calls = set()
        for a in assigns:
            if a.lineno >= first:
                continue
            targets = (
                set().union(*(_tnames(t) for t in a.targets)) if a.targets else set()
            )
            if any(isinstance(t, ast.Subscript) for t in a.targets) or targets & full:
                for n in ast.walk(a.value):
                    if isinstance(n, ast.Call):
                        column_assign_calls.add(id(n))

        target_stats, freq_stats = set(), set()
        for a in assigns:
            if a.lineno >= first:
                continue
            value_calls = [n for n in ast.walk(a.value) if isinstance(n, ast.Call)]
            named = (
                set().union(*(_tnames(t) for t in a.targets)) if a.targets else set()
            )
            aggregates = any(
                isinstance(n.func, ast.Attribute)
                and n.func.attr in ("mean", "median", "agg", "sum", "count", "size")
                for n in value_calls
            )
            if uses_target(a.value) and aggregates:
                target_stats |= named
            if any(_name(n) == "value_counts" for n in value_calls):
                freq_stats |= named

        seen_target_lines = set()
        for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
            if c.lineno >= first or not isinstance(c.func, ast.Attribute):
                continue
            method = c.func.attr
            if method not in ("map", "replace", "merge"):
                continue
            if id(c) not in column_assign_calls:
                continue
            exprs = list(c.args) + [
                kw.value
                for kw in c.keywords
                if kw.arg not in ("on", "how", "suffixes", "validate", "copy")
            ]
            expr_names = (
                set().union(*(_all_names(e) for e in exprs)) if exprs else set()
            )
            direct_target = any(uses_target(e) for e in exprs)
            direct_freq = any(
                isinstance(n, ast.Call) and _name(n) == "value_counts"
                for e in exprs
                for n in ast.walk(e)
            )
            if expr_names & target_stats or direct_target:
                if c.lineno in seen_target_lines:
                    continue
                seen_target_lines.add(c.lineno)
                out.append(
                    Finding(
                        "high",
                        "target_encode_before_split",
                        loc(c.lineno),
                        f"{ast.unparse(c)} turns a target-derived statistic into a feature "
                        f"before the split at {loc(first)}; held-out labels shape the encoding",
                        stable_key=keys[id(c)],
                    )
                )
            elif expr_names & freq_stats or direct_freq:
                out.append(
                    Finding(
                        "low",
                        "frequency_encode_before_split",
                        loc(c.lineno),
                        f"{ast.unparse(c)} encodes category frequency using every row before "
                        f"the split at {loc(first)}; held-out rows help set the values",
                        stable_key=keys[id(c)],
                    )
                )

    # Time-ordered data split with shuffling enabled: adjacent timestamps leak
    # across the boundary even when no column is literally named "target".
    if first is not None:
        time_like = re.compile(
            r"(date|time|timestamp|datetime|month|year|week|day)", re.IGNORECASE
        )
        sorted_names = set()
        for a in assigns:
            if a.lineno >= first:
                continue
            value = a.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "sort_values"
            ):
                continue
            by = list(value.args) + [
                kw.value for kw in value.keywords if kw.arg in ("by", "key")
            ]
            labels = set().union(*(_strings(x) for x in by)) if by else set()
            receiver = _all_names(value.func.value)
            if any(time_like.search(s) for s in labels) or any(
                time_like.search(n) for n in receiver
            ):
                sorted_names |= set().union(*(_tnames(t) for t in a.targets))
        split_names = set()
        for c in tts:
            for arg in list(c.args) + [
                kw.value for kw in c.keywords if kw.arg in ("X", "y")
            ]:
                split_names |= _all_names(arg)
        if sorted_names & split_names:
            for c in tts:
                shuffle = next(
                    (kw.value for kw in c.keywords if kw.arg == "shuffle"), None
                )
                if isinstance(shuffle, ast.Constant) and shuffle.value is False:
                    continue
                explicit = isinstance(shuffle, ast.Constant) and shuffle.value is True
                out.append(
                    Finding(
                        "high" if explicit else "medium",
                        "temporal_split_shuffle",
                        loc(c.lineno),
                        "train_test_split() shuffles data that was sorted by a time "
                        "column; use shuffle=False or a time-based split so future rows "
                        "cannot land in the training set",
                        stable_key=keys[id(c)],
                    )
                )
    return out


def scan_file(path) -> list[Finding]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix != ".ipynb":
        return scan_source(txt, str(p))
    try:
        nb = json.loads(txt)
    except ValueError as e:
        return [Finding("low", "parse_error", str(p), str(e))]
    parts, cells, line = [], [], 1
    executed = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        count = c.get("execution_count")
        if isinstance(count, int) and not isinstance(count, bool):
            executed.append((i, count))
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines = []
        for l in src.splitlines():
            if l.lstrip().startswith(("%", "!")):
                l = l[: len(l) - len(l.lstrip())] + "pass"
            lines.append(l)
        cells.append((line, i))
        parts.append("\n".join(lines))
        line += max(len(lines), 1)
    out = scan_source("\n".join(parts), str(p), cells)
    order_finding = None
    if len(executed) >= 2:
        counts = [count for _, count in executed]
        if counts != sorted(counts):
            order_finding = Finding(
                "low",
                "notebook_out_of_order_execution",
                str(p),
                "code cells were executed out of order (execution_count "
                + ", ".join(str(count) for count in counts)
                + "); runtime state may not match cell order, so a split or "
                "preprocessing step can run in a different order than it appears",
            )
    if not any(f.kind == "parse_error" for f in out):
        return out + ([order_finding] if order_finding else [])
    # A single bad cell (e.g. a "%%writefile" body that is not Python) makes the whole
    # notebook unparseable, hiding the leaks in every other cell. Skip only the cells that
    # fail to parse on their own and keep scanning the rest with unchanged line offsets.
    bad = []
    for idx, part in enumerate(parts):
        try:
            ast.parse(part)
        except SyntaxError as e:
            bad.append((idx, e))
    if not bad:
        return out
    for idx, _ in bad:
        parts[idx] = "\n".join(["pass"] * (parts[idx].count("\n") + 1))
    out = scan_source("\n".join(parts), str(p), cells)
    skipped = [f"cell {cells[idx][1]} skipped: {e}" for idx, e in bad]
    return [
        *out,
        *([order_finding] if order_finding else []),
        Finding("low", "parse_error", str(p), "; ".join(skipped)),
    ]


def scan_targets(path, *, sql=None) -> list[Path]:
    """Files ``scan_path`` would scan, in deterministic order.

    Shared so callers that need progress (or a total count) enumerate exactly
    the same set the one-shot scan uses.
    """
    p = Path(path)
    suffixes = {".py", ".ipynb"}
    if sql is not None and sql.enabled:
        suffixes.add(".sql")
    if p.is_file():
        return [p]
    return [
        f
        for f in sorted(p.rglob("*"))
        if f.suffix in suffixes and not (SKIP & set(f.parts))
    ]


def scan_path(path, *, sql=None, target=None) -> list[Finding]:
    """Scan Python/notebook files, plus ``.sql`` files when enabled or explicit.

    Directories include ``.sql`` only when SQL settings are enabled; an explicit
    ``.sql`` file is always analyzed because the caller named it.
    """
    files = scan_targets(path, sql=sql)
    out = []
    for f in files:
        if f.suffix == ".sql":
            from .sqlscan import scan_sql_file

            out += scan_sql_file(f, settings=sql, target=target)
        else:
            out += scan_file(f)
    return out
