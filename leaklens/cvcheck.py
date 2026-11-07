from __future__ import annotations

import ast
import warnings

from .leakage import Finding
from .partition import any_test, infer_roles, name_role
from .scanner import (
    BENIGN,
    FUNCS,
    HINTS,
    RESAMPLE,
    _all_names,
    _finding_keys,
    _name,
    _sev,
    _tnames,
)

CV_FUNCS = {
    "cross_val_score",
    "cross_validate",
    "cross_val_predict",
    "learning_curve",
    "validation_curve",
    "permutation_test_score",
}
SEARCH = (
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "BayesSearchCV",
)
PARAM_GRID_FUNCS = {
    "ParameterGrid",
    "ParameterSampler",
    "product",
    "arange",
    "logspace",
    "linspace",
}


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


def _is_group_splitter(expr, group_vars) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in group_vars
    if isinstance(expr, ast.Call):
        name = _name(expr)
        return ("Group" in name or "group" in name) and (
            "Fold" in name or "Split" in name
        )
    return False


def _full_data_names(assigns, tts, first_split):
    """Names that hold the full dataset feeding the first train_test_split."""
    full: set = set()
    if not tts:
        return full
    for arg in tts[0].args:
        full |= _all_names(arg)
    defs: dict = {}
    for a in assigns:
        if a.lineno < first_split and not (
            isinstance(a.value, ast.Call) and _name(a.value) == "train_test_split"
        ):
            for t in a.targets:
                for nm in _tnames(t):
                    defs.setdefault(nm, []).append(a.value)
    work = list(full)
    while work:
        nm = work.pop()
        for value in defs.get(nm, []):
            for name in _all_names(value):
                if name in defs and name not in full:
                    full.add(name)
                    work.append(name)
    for a in assigns:
        if isinstance(a.value, ast.Call) and _name(a.value) == "train_test_split":
            continue
        if _all_names(a.value) & full:
            for t in a.targets:
                full |= _tnames(t)
    return full


def scan_cv(src: str, label: str = "<src>", cells=None) -> list[Finding]:
    """Tuning, cross-validation, and preprocessing misused around validation folds."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError:
        return []
    keys = _finding_keys(tree)
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
        elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith(
            "sklearn.preprocessing"
        ):
            for al in n.names:
                if al.name in FUNCS:
                    funcs.add(al.asname or al.name)
    assigns.sort(key=lambda a: a.lineno)

    roles = infer_roles(assigns)
    tts = sorted(
        (c for c in calls if _name(c) == "train_test_split"), key=lambda c: c.lineno
    )
    first_split = tts[0].lineno if tts else None
    full = _full_data_names(assigns, tts, first_split)
    split_inputs = _all_names(tts[0].args[0]) if tts and tts[0].args else set()
    # A split of X_train into train/validation is a nested holdout, not a
    # top-level test split; tuning on X_train there is not a test-set leak.
    nested_split = bool(split_inputs) and all(name_role(n) for n in split_inputs)
    group_vars = {
        name: cls
        for name, cls in ctor.items()
        if ("Group" in cls and ("Fold" in cls or "Split" in cls))
    }

    def is_test(expr) -> bool:
        return any_test(_all_names(expr), roles)

    def loc(line):
        if cells:
            h = [(s, i) for s, i in cells if s <= line]
            if h:
                s, i = h[-1]
                return f"{label}[cell {i}]:{line - s + 1}"
        return f"{label}:{line}"

    # Estimators fitted via a plain .fit() / .fit_transform() / .fit_resample() call keep the
    # class and the data they learned from, so a later ".transform(X)" inherits the same taint
    # as "fit_transform" would have (issue: fit-then-transform before CV evaded the tracker).
    fitted = {}
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        if not (
            isinstance(f, ast.Attribute)
            and f.attr in ("fit", "fit_transform", "fit_resample")
        ):
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
                if (
                    isinstance(c.func, ast.Attribute)
                    and c.func.attr == "transform"
                    and isinstance(c.func.value, ast.Name)
                ):
                    events = [
                        e for e in fitted.get(c.func.value.id, []) if e[0] <= a.lineno
                    ]
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

    split_lines = [c.lineno for c in tts]

    groups: dict = {}
    special: list[Finding] = []
    for c in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
        f = c.func
        fname = _name(c)
        xs, who, kind_of_call = None, "", ""
        if fname in CV_FUNCS:
            xs = c.args[1:2] + [kw.value for kw in c.keywords if kw.arg == "X"]
            who = fname
            kind_of_call = "cv"
        elif isinstance(f, ast.Attribute) and f.attr == "fit":
            b = f.value
            cls = (
                ctor.get(b.id, "")
                if isinstance(b, ast.Name)
                else (_name(b) if isinstance(b, ast.Call) else "")
            )
            if any(s in cls for s in SEARCH):
                xs = c.args[:1] + [kw.value for kw in c.keywords if kw.arg == "X"]
                who = cls
                kind_of_call = "search"
        if not xs:
            continue
        arg_names = (
            set().union(*(_all_names(e) for e in xs if e is not None)) if xs else set()
        )
        on_test = any(is_test(e) for e in xs)

        # Hyperparameter search fitted/scored on the test partition itself.
        if kind_of_call == "search" and on_test:
            special.append(
                Finding(
                    "high",
                    "tuning_on_test_data",
                    loc(c.lineno),
                    f"{who}.fit() uses test data for hyperparameter tuning; the test "
                    f"set must only be used once for final evaluation. Tune on training "
                    f"data with cross-validation instead.",
                    stable_key=keys[id(c)],
                )
            )
            continue
        # Cross-validation over the test partition: legitimate evaluation happens
        # once, not through repeated folds that overlap each other's test rows.
        if kind_of_call == "cv" and on_test:
            special.append(
                Finding(
                    "medium",
                    "cv_on_test_data",
                    loc(c.lineno),
                    f"{who}() is run on the held-out test partition; cross-validation "
                    f"there reuses test rows as both training and validation data. "
                    f"Cross-validate on X_train and evaluate on X_test once.",
                    stable_key=keys[id(c)],
                )
            )
            continue
        # Hyperparameter choice made before the explicit split, using the whole
        # dataset that the split later divides. A manually built X_train slice is
        # not the full frame, so role hints on the arguments take precedence.
        manual_subset = any(name_role(n) in ("train", "validation") for n in arg_names)
        if (
            first_split is not None
            and c.lineno < first_split
            and arg_names & full
            and not manual_subset
        ):
            special.append(
                Finding(
                    "medium",
                    "tuning_before_split",
                    loc(c.lineno),
                    f"{who}() tunes or cross-validates on the full dataset before the "
                    f"split at {loc(first_split)}; the held-out rows help choose the "
                    f"model/hyperparameters. Split first, then tune on X_train.",
                    stable_key=keys[id(c)],
                )
            )
            continue
        # A search fitted on the original full frame after the split exists.
        if (
            kind_of_call == "search"
            and first_split is not None
            and c.lineno > first_split
            and arg_names & full
            and not nested_split
        ):
            special.append(
                Finding(
                    "medium",
                    "tuning_on_full_data",
                    loc(c.lineno),
                    f"{who}.fit() uses data that also feeds the earlier split at "
                    f"{loc(first_split)}; tune on the training partition only.",
                    stable_key=keys[id(c)],
                )
            )

        hit = None
        for e in xs:
            if e is None:
                continue
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
        g = groups.setdefault(
            (line, kind), {"first": c, "who": who, "n": 0, "cls": cls, "sev": sev}
        )
        g["n"] += 1

    # Custom hyperparameter loops: for C in [...]: model.fit(X_test, y_test).
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        grid = isinstance(node.iter, (ast.List, ast.Tuple, ast.Set)) or (
            isinstance(node.iter, ast.Call) and _name(node.iter) in PARAM_GRID_FUNCS
        )
        if not grid:
            continue
        for inner in ast.walk(node):
            if not (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "fit"
            ):
                continue
            names = (
                set().union(*(_all_names(a) for a in inner.args))
                if inner.args
                else set()
            )
            if any_test(names, roles):
                special.append(
                    Finding(
                        "high",
                        "tuning_on_test_data",
                        loc(inner.lineno),
                        "a loop over hyperparameter values fits the model on the test "
                        "partition; selecting the best loop iteration on test data "
                        "leaks the test set. Use GridSearchCV(cv=...) on X_train.",
                        stable_key=keys[id(inner)],
                    )
                )
                break

    # Group-aware splitters used without the entity column.
    for c in calls:
        if not (isinstance(c.func, ast.Attribute) and c.func.attr == "split"):
            continue
        receiver = c.func.value
        receiver_cls = ""
        if isinstance(receiver, ast.Name):
            receiver_cls = ctor.get(receiver.id, "")
        elif isinstance(receiver, ast.Call):
            receiver_cls = _name(receiver)
        if "Group" not in receiver_cls:
            continue
        if any(kw.arg == "groups" for kw in c.keywords) or len(c.args) >= 3:
            continue
        special.append(
            Finding(
                "medium",
                "group_cv_missing_groups",
                loc(c.lineno),
                f"{receiver_cls}.split() is called without groups=; the same entity "
                f"can appear in a training and a validation fold. Pass the entity "
                f"column (groups=...) or use GroupKFold correctly.",
                stable_key=keys[id(c)],
            )
        )
    for c in calls:
        if _name(c) not in CV_FUNCS:
            continue
        if any(kw.arg == "groups" for kw in c.keywords):
            continue
        cv = next((kw.value for kw in c.keywords if kw.arg == "cv"), None)
        if cv is None or not _is_group_splitter(cv, group_vars):
            continue
        special.append(
            Finding(
                "medium",
                "group_cv_missing_groups",
                loc(c.lineno),
                f"{_name(c)}() uses a group-aware splitter without groups=; folds can "
                f"share entities. Pass groups=entity_ids.",
                stable_key=keys[id(c)],
            )
        )
    # GridSearchCV(cv=GroupKFold(...)).fit(X, y) without groups=.
    for a in assigns:
        if not (
            isinstance(a.value, ast.Call) and any(s in _name(a.value) for s in SEARCH)
        ):
            continue
        cv = next((kw.value for kw in a.value.keywords if kw.arg == "cv"), None)
        if cv is None or not _is_group_splitter(cv, group_vars):
            continue
        names = set()
        for t in a.targets:
            names |= _tnames(t)
        if not names:
            continue
        for c in calls:
            if not (
                isinstance(c.func, ast.Attribute)
                and c.func.attr == "fit"
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id in names
                and c.lineno > a.lineno
            ):
                continue
            if any(kw.arg == "groups" for kw in c.keywords):
                continue
            special.append(
                Finding(
                    "medium",
                    "group_cv_missing_groups",
                    loc(c.lineno),
                    f"{_name(a.value)} was built with a group-aware cv splitter, but "
                    f"fit() receives no groups=; folds can share entities.",
                    stable_key=keys[id(c)],
                )
            )

    # Repeated scoring of the same held-out partition suggests model selection on
    # the test set. Low severity: some notebooks legitimately log several metrics.
    scored = [
        c
        for c in calls
        if isinstance(c.func, ast.Attribute)
        and c.func.attr == "score"
        and c.args
        and is_test(c.args[0])
    ]
    if len(scored) >= 3:
        special.append(
            Finding(
                "low",
                "test_set_reuse",
                loc(scored[0].lineno),
                f"the held-out partition is scored {len(scored)} times; choosing between "
                f"models or hyperparameters after looking at test scores leaks the test "
                f"set. Select on a validation split and touch the test set once.",
                stable_key=keys[id(scored[0])],
            )
        )

    out = []
    for (line, kind), g in groups.items():
        more = g["n"] - 1
        extra = f" (and {more} more call{'s' if more > 1 else ''})" if more else ""
        out.append(
            Finding(
                g["sev"],
                kind,
                loc(g["first"].lineno),
                f"{g['who']}(){extra} is given data transformed by {g['cls']} at {loc(line)}; the transform "
                f"has seen every fold's validation rows. Put it inside a Pipeline passed to {g['who']}",
                stable_key=keys[id(g["first"])],
            )
        )
    return out + special
