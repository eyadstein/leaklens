from leaklens.flow import merge, scan_flow
from leaklens.leakage import Finding
from leaklens.scanner import scan_source

H = (
    "from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder\n"
    "from sklearn.compose import ColumnTransformer\n"
    "from sklearn.model_selection import StratifiedKFold, train_test_split\n"
)


def kinds(body):
    return [(f.kind, f.severity) for f in scan_source(H + body)]


def test_scaler_then_manual_kfold_loop():
    k = kinds(
        "X = MinMaxScaler().fit_transform(x)\ncv = StratifiedKFold(5)\nfor tr, te in cv.split(X, y):\n    pass\n"
    )
    assert ("preprocess_before_split", "high") in k


def test_second_split_after_full_data_scaler():
    k = kinds(
        "a = train_test_split(X, y, random_state=0)\nX = MinMaxScaler().fit_transform(x)\nb = train_test_split(X, y, random_state=0)\n"
    )
    assert ("preprocess_before_split", "high") in k


def test_per_fold_scaler_is_clean():
    body = (
        "cv = StratifiedKFold(5)\nfor tr, te in cv.split(X, y):\n    sc = MinMaxScaler()\n"
        "    Xtr = sc.fit_transform(X[tr])\n    Xte = sc.transform(X[te])\n"
    )
    assert not [f for f in scan_source(H + body) if f.severity in ("high", "medium")]


def test_column_transformer_inner_classes():
    one = kinds(
        "ct = ColumnTransformer([('o', OneHotEncoder(), [0])])\nX = ct.fit_transform(X)\na = train_test_split(X, y, random_state=0)\n"
    )
    assert ("encoder_before_split", "low") in one
    sc = kinds(
        "ct = ColumnTransformer([('s', StandardScaler(), [0])])\nX = ct.fit_transform(X)\na = train_test_split(X, y, random_state=0)\n"
    )
    assert ("preprocess_before_split", "high") in sc


def test_taint_is_killed_by_reassignment():
    src = (
        H
        + "X = MinMaxScaler().fit_transform(x)\nX = load_fresh()\na = train_test_split(X, y, random_state=0)\n"
    )
    assert scan_flow(src) == []


def test_merge_replaces_in_place_and_dedupes():
    core = [
        Finding("low", "no_random_state", "f:1", "d"),
        Finding("medium", "fit_on_full_data", "f:2", "d"),
        Finding("low", "no_random_state", "f:9", "d"),
    ]
    extra = [Finding("high", "preprocess_before_split", "f:2", "flow")]
    out = merge(core, extra)
    assert [f.kind for f in out] == [
        "no_random_state",
        "preprocess_before_split",
        "no_random_state",
    ]
    assert merge(core, []) == core


def test_repeated_cv_calls_are_grouped():
    src = (
        "from sklearn.preprocessing import MinMaxScaler\nfrom sklearn.model_selection import permutation_test_score\n"
        "Xs = MinMaxScaler().fit_transform(x)\n"
        + "".join(f"r{i} = permutation_test_score(m, Xs, y)\n" for i in range(5))
    )
    fs = [f for f in scan_source(src) if f.kind == "preprocess_outside_cv"]
    assert len(fs) == 1 and "4 more calls" in fs[0].detail
