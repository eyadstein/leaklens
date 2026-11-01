from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

from .leakage import Finding

FIT = {"fit", "fit_transform", "fit_resample"}
HINTS = ("Scaler", "Imputer", "Encoder", "PCA", "SMOTE", "ADASYN", "Vectorizer", "Normalizer",
         "Transformer", "Selector", "SelectK", "SelectPercentile", "SelectFromModel",
         "RandomOverSampler", "RandomUnderSampler", "Binarizer", "Discretizer")
RESAMPLE = ("SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler")
BENIGN = ("LabelEncoder", "OrdinalEncoder", "OneHotEncoder")
UNSURE = ("ColumnTransformer",)
# Splitters whose folds are random only when shuffle=True (fold order is deterministic without it).
SHUFFLED_SPLITS = {"KFold", "StratifiedKFold", "StratifiedGroupKFold"}
# Repeated splitters always shuffle, so a missing random_state always makes them non-reproducible.
ALWAYS_SHUFFLED = {"RepeatedKFold", "RepeatedStratifiedKFold", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit"}
FUNCS = {"scale", "minmax_scale", "robust_scale", "maxabs_scale", "quantile_transform", "power_transform"}
EVAL = re.compile(r"(^|[_\W])(test|val|valid|validation|eval)([_\W]|$)", re.I)
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


def _strings(expr) -> set:
    return {s.value for s in ast.walk(expr) if isinstance(s, ast.Constant) and isinstance(s.value, str)}


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


def _is_tts(v) -> bool:
    return isinstance(v, ast.Call) and _name(v) == "train_test_split"


def _kw_true(c, name: str) -> bool:
    """Whether a keyword argument is the literal True."""
    return any(kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True
               for kw in c.keywords)


def _has_seed(c) -> bool:
    """random_state passed, or a **kwargs expansion that might carry it (assume it does)."""
    return any(kw.arg == "random_state" or kw.arg is None for kw in c.keywords)


def _kw_false(c, name: str) -> bool:
    """Whether a keyword argument is the literal False."""
    return any(kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is False
               for kw in c.keywords)


def scan_source(src: str, label: str = "<src>", cells=None) -> list[Finding]:
    from .cvcheck import scan_cv
    from .flow import merge, scan_flow

    return merge(_scan_core(src, label, cells), scan_flow(src, label, cells) + scan_cv(src, label, cells))


def _scan_core(src: str, label: str = "<src>", cells=None) -> list[Finding]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError as e:
        return [Finding("low", "parse_error", label, str(e))]
    ctor, calls, funcs, assigns = {}, [], set(), []

    pandas_aliases, get_dummies_aliases = set(), set()

    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            assigns.append(n)
            if isinstance(n.value, ast.Call):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        ctor[t.id] = _name(n.value)
        elif isinstance(n, ast.Call):
            calls.append(n)
        elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sklearn.preprocessing"):
            for al in n.names:
                if al.name in FUNCS:
                    funcs.add(al.asname or al.name)
        elif isinstance(n, ast.Import):
            for al in n.names:
                if al.name == "pandas":
                    pandas_aliases.add(al.asname or "pandas")
        elif isinstance(n, ast.ImportFrom) and n.module == "pandas":
            for al in n.names:
                if al.name == "get_dummies":
                    get_dummies_aliases.add(al.asname or al.name)

    assigns.sort(key=lambda n: n.lineno)

    def is_split(c):
        if _name(c) == "train_test_split":
            return True
        f = c.func
        return (isinstance(f, ast.Attribute) and f.attr == "split" and isinstance(f.value, ast.Name)
                and any(k in ctor.get(f.value.id, "") for k in ("Fold", "Split")))

    splits = [c.lineno for c in calls if is_split(c)]
    first = min(splits) if splits else None

    # dataflow: which names hold the FULL dataset that train_test_split divides?
    tts = sorted((c for c in calls if _name(c) == "train_test_split"), key=lambda c: c.lineno)
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

    def loc(line):
        if cells:
            hit = [(s, i) for s, i in cells if s <= line]
            if hit:
                s, i = hit[-1]
                return f"{label}[cell {i}]:{line - s + 1}"
        return f"{label}:{line}"

    out = []
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        nm = _name(c)

        is_get_dummies = (
    isinstance(f, ast.Name) and f.id in get_dummies_aliases
) or (
    isinstance(f, ast.Attribute) and f.attr == "get_dummies"
    and isinstance(f.value, ast.Name) and f.value.id in pandas_aliases
)

        if is_get_dummies and first is not None and c.lineno < first:
            flows_to_split = any(
        any(isinstance(n, ast.Call) and n is c for n in ast.walk(a.value))
        and bool(set().union(*(_tnames(t) for t in a.targets)) & full)
        for a in assigns if a.lineno == c.lineno
    )
            if flows_to_split:
               out.append(Finding(
            "low", "get_dummies_before_split", loc(c.lineno),
            f"pandas.get_dummies() runs before the split at {loc(first)}, "
            "so held-out categories can shape the encoded columns",
            ))

        # low severity: reproducibility - a random split nobody can rerun
        if nm == "train_test_split" and not _has_seed(c) and not _kw_false(c, "shuffle"):
            out.append(Finding("low", "no_random_state", loc(c.lineno),
                               "train_test_split() without random_state: a different split on every run, so results are not reproducible"))
        elif nm in SHUFFLED_SPLITS and not _has_seed(c) and _kw_true(c, "shuffle"):
            out.append(Finding("low", "no_random_state", loc(c.lineno),
                               f"{nm}(shuffle=True) without random_state: folds differ on every run, so results are not reproducible"))
        elif nm in ALWAYS_SHUFFLED and not _has_seed(c):
            out.append(Finding("low", "no_random_state", loc(c.lineno),
                               f"{nm}() without random_state: it always shuffles, so folds differ on every run, so results are not reproducible"))
        if isinstance(f, ast.Name) and f.id in funcs:
            cls, call = f.id, f.id
        elif isinstance(f, ast.Attribute) and f.attr in FIT:
            b = f.value
            if isinstance(b, ast.Name):
                cls = ctor.get(b.id, "")
            elif isinstance(b, ast.Call):
                cls = _name(b)
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
            out.append(Finding(sev, "fit_on_eval_data", loc(c.lineno), f"{call}({args}) learns from evaluation data"))
        elif first is not None and c.lineno < first:
            if any(r in cls for r in RESAMPLE):
                kind = "resample_before_split"
            elif any(b in cls for b in BENIGN):
                kind = "encoder_before_split"
            elif any(u in cls for u in UNSURE):
                kind = "transformer_before_split"
            else:
                kind = "preprocess_before_split"
            out.append(Finding(_sev(cls), kind, loc(c.lineno), f"{call}() runs before the split at {loc(first)}"))
        elif tts_line is not None and c.lineno > tts_line and any(_names(a) & full for a in c.args):
            out.append(Finding("medium", "fit_on_full_data", loc(c.lineno),
                               f"{call}({args}) fits on the full dataset after the split at {loc(tts_line)}; "
                               "fine only for a final refit"))

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
        if not (isinstance(f, ast.Attribute) and f.attr in ("transform", "agg", "apply")):
            continue
        if f.attr in ("agg", "apply") and id(c) not in assigned_cols:
            continue
        if first is None or c.lineno >= first:
            continue
        sel, base = None, None
        if isinstance(f.value, ast.Subscript) and isinstance(f.value.value, ast.Call) \
                and _name(f.value.value) == "groupby":
            sel, base = f.value.slice, f.value.value
        elif isinstance(f.value, ast.Call) and _name(f.value) == "groupby":
            base = f.value
        if base is None:
            continue
        used = (_all_names(sel) | _strings(sel)) if sel is not None else set()
        keys = set()
        for a in base.args:
            keys |= _all_names(a) | _strings(a)
        if used & target or keys & target:
            out.append(Finding("high", "target_encode_before_split", loc(c.lineno),
                               f"{ast.unparse(c)} builds a feature from the target across all rows "
                               f"before the split at {loc(first)}"))
        else:
            out.append(Finding("medium", "group_stat_before_split", loc(c.lineno),
                               f"{ast.unparse(c)} uses group statistics from all rows before the "
                               f"split at {loc(first)}; held-out rows help build the feature"))
    # issue #77: hand-rolled scaling with full-data statistics before the split,
    # e.g. (X - X.mean()) / X.std() or X / X.max(). Statistics taken from a train
    # partition only occur after the split, so they are never reported.
    STATS = {"mean", "std", "var", "min", "max", "median", "quantile"}

    def stat_recv(n):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in STATS:
            if n.args and isinstance(n.func.value, ast.Name) and n.func.value.id in ("np", "numpy"):
                return set().union(*(_all_names(a) for a in n.args))
            return _all_names(n.func.value)
        return None

    if first is not None:
        inner = {id(k) for b in ast.walk(tree) if isinstance(b, ast.BinOp)
                 for k in (b.left, b.right) if isinstance(k, ast.BinOp)}
        binops = [b for b in ast.walk(tree) if isinstance(b, ast.BinOp)]
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
            outside = {m.id for m in ast.walk(b) if isinstance(m, ast.Name) and id(m) not in skip}
            if outside & recv:
                out.append(Finding("high", "manual_scale_before_split", loc(b.lineno),
                                   f"{ast.unparse(b)} scales with statistics from all rows "
                                   f"before the split at {loc(first)}"))

    # fillna with a statistic computed on all rows before the split (e.g. df.fillna(df.mean())).
    if first is not None:
        for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
            if _name(c) != "fillna" or c.lineno >= first:
                continue
            vals = list(c.args) + [kw.value for kw in c.keywords]
            uses_stat = False
            for v in vals:
                for n in ast.walk(v):
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                            and n.func.attr in ("mean", "median", "mode"):
                        uses_stat = True
                        break
                if uses_stat:
                    break
            if not uses_stat:
                continue
            out.append(Finding("medium", "fillna_before_split", loc(c.lineno),
                               f"{ast.unparse(c)} imputes with a statistic from all rows "
                               f"before the split at {loc(first)}"))
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
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
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
    if not any(f.kind == "parse_error" for f in out):
        return out
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
    return out + [Finding("low", "parse_error", str(p), "; ".join(skipped))]


def scan_path(path) -> list[Finding]:
    p = Path(path)
    if p.is_file():
        files = [p]
    else:
        files = [f for f in sorted(p.rglob("*")) if f.suffix in (".py", ".ipynb") and not (SKIP & set(f.parts))]
    out = []
    for f in files:
        out += scan_file(f)
    return out