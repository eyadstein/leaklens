from __future__ import annotations

import ast
import warnings

from .leakage import Finding
from .scanner import BENIGN, FUNCS, HINTS, RESAMPLE, _all_names, _name, _sev, _tnames

CV_FUNCS = {"cross_val_score", "cross_validate", "cross_val_predict", "learning_curve", "validation_curve",
            "permutation_test_score"}
SEARCH = ("GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "BayesSearchCV")


def _fit_cls(c, ctor, funcs):
    f = c.func
    if isinstance(f, ast.Name) and f.id in funcs:
        return f.id
    if isinstance(f, ast.Attribute) and f.attr in ("fit_transform", "fit_resample"):
        b = f.value
        if isinstance(b, ast.Name):
            cls = ctor.get(b.id, "")
        elif isinstance(b, ast.Call):
            cls = _name(b)
        else:
            cls = ""
        if any(h in cls for h in HINTS):
            return cls
    return None


def _on_partition(names) -> bool:
    return any("train" in n.lower() or n.lower().startswith("x_tr") for n in names)


def scan_cv(src: str, label: str = "<src>", cells=None) -> list[Finding]:
    """Preprocessing or resampling done on all rows before cross-validation / hyperparameter search."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError:
        return []
    ctor, assigns, calls, funcs = {}, [], [], set()
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
    assigns.sort(key=lambda a: a.lineno)

    # Estimators fitted via a plain .fit() / .fit_transform() / .fit_resample() call keep the
    # class and the data they learned from, so a later ".transform(X)" inherits the same taint
    # as "fit_transform" would have (issue: fit-then-transform before CV evaded the tracker).
    fitted = {}
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        if not (isinstance(f, ast.Attribute) and f.attr in ("fit", "fit_transform", "fit_resample")):
            continue
        b = f.value
        if not isinstance(b, ast.Name):
            continue
        cls = ctor.get(b.id, "")
        if not any(h in cls for h in HINTS):
            continue
        argn = set()
        for x in c.args:
            argn |= _all_names(x)
        fitted.setdefault(b.id, []).append((c.lineno, cls, argn))

    tr = {}
    for a in assigns:
        if isinstance(a.value, ast.Call) and _name(a.value) == "train_test_split":
            continue
        hit = None
        for c in ast.walk(a.value):
            if isinstance(c, ast.Call):
                if (isinstance(c.func, ast.Attribute) and c.func.attr == "transform"
                        and isinstance(c.func.value, ast.Name)):
                    events = [e for e in fitted.get(c.func.value.id, []) if e[0] <= a.lineno]
                    if events:
                        line, cls, argn = events[-1]
                        if not any(b in cls for b in BENIGN):
                            hit = (line, cls, argn)
                            break
                cls = _fit_cls(c, ctor, funcs)
                if cls and not any(b in cls for b in BENIGN):
                    argn = set()
                    for x in c.args:
                        argn |= _all_names(x)
                    hit = (a.lineno, cls, argn)
                    break
        if hit is None:
            srcs = sorted(_all_names(a.value) & set(tr))
            if srcs:
                hit = tr[srcs[0]]
        if hit:
            for t in a.targets:
                for nm in _tnames(t):
                    tr[nm] = hit

    split_lines = [c.lineno for c in calls if _name(c) == "train_test_split"]

    def loc(line):
        if cells:
            h = [(s, i) for s, i in cells if s <= line]
            if h:
                s, i = h[-1]
                return f"{label}[cell {i}]:{line - s + 1}"
        return f"{label}:{line}"

    groups: dict = {}
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        fname = _name(c)
        xs, who = None, ""
        if fname in CV_FUNCS:
            xs = c.args[1:2] + [kw.value for kw in c.keywords if kw.arg == "X"]
            who = fname
        elif isinstance(f, ast.Attribute) and f.attr == "fit":
            b = f.value
            cls = ctor.get(b.id, "") if isinstance(b, ast.Name) else (_name(b) if isinstance(b, ast.Call) else "")
            if any(s in cls for s in SEARCH):
                xs = c.args[:1] + [kw.value for kw in c.keywords if kw.arg == "X"]
                who = cls
        if not xs:
            continue
        hit = None
        for e in xs:
            for nm in sorted(_all_names(e)):
                if nm in tr and tr[nm][0] < c.lineno:
                    hit = tr[nm]
                    break
            if hit:
                break
        if not hit:
            continue
        line, cls, argn = hit
        resample = any(r in cls for r in RESAMPLE)
        kind = "resample_outside_cv" if resample else "preprocess_outside_cv"
        sev = _sev(cls)
        if not resample and _on_partition(argn) and any(s < line for s in split_lines):
            sev = "low"
        g = groups.setdefault((line, kind), {"first": c, "who": who, "n": 0, "cls": cls, "sev": sev})
        g["n"] += 1
    out = []
    for (line, kind), g in groups.items():
        more = g["n"] - 1
        extra = f" (and {more} more call{'s' if more > 1 else ''})" if more else ""
        out.append(Finding(g["sev"], kind, loc(g["first"].lineno),
                           f"{g['who']}(){extra} is given data transformed by {g['cls']} at {loc(line)}; the transform "
                           f"has seen every fold's validation rows. Put it inside a Pipeline passed to {g['who']}"))
    return out