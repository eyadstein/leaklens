"""Hyperparameter tuning and test-set reuse detectors."""

from leaklens.fixes import advice
from leaklens.scanner import scan_source


def findings(src):
    return scan_source(src)


def kinds(src):
    return {f.kind for f in findings(src)}


TUNING_HEAD = (
    "from sklearn.model_selection import GridSearchCV, train_test_split\n"
    "from sklearn.linear_model import LogisticRegression\n"
)


def test_holdout_named_partition_is_flagged():
    src = (
        TUNING_HEAD + "X_train, X_holdout, y_train, y_holdout = train_test_split("
        "X, y, random_state=0)\n"
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
        + "gs.fit(X_holdout, y_holdout)\n"
    )
    fs = [f for f in findings(src) if f.kind == "tuning_on_test_data"]
    assert fs and fs[0].severity == "high"
    assert "test set" in fs[0].detail


def test_positional_test_partition_is_flagged():
    src = (
        TUNING_HEAD
        + "a, b, c, d = train_test_split(X, y, random_state=0)\n"
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]})\n"
        + "gs.fit(b, d)\n"
    )
    assert "tuning_on_test_data" in kinds(src)


def test_search_on_train_is_clean():
    src = (
        TUNING_HEAD + "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
        + "gs.fit(X_train, y_train)\n"
        + "print(gs.score(X_test, y_test))\n"
    )
    assert "tuning_on_test_data" not in kinds(src)


def test_nested_cross_validation_is_clean():
    src = (
        "from sklearn.model_selection import GridSearchCV, cross_val_score, "
        "train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0, stratify=y)\n"
        "gs = GridSearchCV(model, grid, cv=5)\n"
        "scores = cross_val_score(gs, X_train, y_train, cv=5)\n"
        "print(gs.fit(X_train, y_train).score(X_test, y_test))\n"
    )
    assert not kinds(src)


def test_validation_split_tuning_is_clean():
    src = (
        TUNING_HEAD + "X_train, X_val, y_train, y_val = train_test_split("
        "X, y, random_state=0)\n"
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
        + "gs.fit(X_train, y_train)\n"
        + "print(gs.score(X_val, y_val))\n"
    )
    assert "tuning_on_test_data" not in kinds(src)
    assert "tuning_on_full_data" not in kinds(src)


def test_tuning_before_split_is_medium():
    src = (
        TUNING_HEAD
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
        + "gs.fit(X, y)\n"
        + "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
    )
    fs = [f for f in findings(src) if f.kind == "tuning_before_split"]
    assert fs and fs[0].severity == "medium"


def test_search_on_full_data_after_split_is_medium():
    src = (
        TUNING_HEAD + "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        + "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
        + "gs.fit(X, y)\n"
    )
    assert "tuning_on_full_data" in kinds(src)


def test_cross_validation_on_test_is_medium():
    src = (
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "scores = cross_val_score(model, X_test, y_test, cv=5)\n"
    )
    fs = [f for f in findings(src) if f.kind == "cv_on_test_data"]
    assert fs and fs[0].severity == "medium"


def test_cross_validation_on_train_is_clean():
    src = (
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
    )
    assert "cv_on_test_data" not in kinds(src)


def test_manual_train_slice_before_split_is_not_full_data_tuning():
    src = (
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_train = X.iloc[:80]\n"
        "y_train = y.iloc[:80]\n"
        "scores = cross_val_score(model, X_train, y_train, cv=5)\n"
        "X_fit, X_test, y_fit, y_test = train_test_split("
        "X, y, random_state=0)\n"
    )
    assert "tuning_before_split" not in kinds(src)


def test_custom_tuning_loop_on_test_is_high():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "for C in [0.1, 1.0, 10.0]:\n"
        "    model = SVC(C=C)\n"
        "    model.fit(X_test, y_test)\n"
    )
    fs = [f for f in findings(src) if f.kind == "tuning_on_test_data"]
    assert fs and fs[0].severity == "high"
    assert "loop" in fs[0].detail


def test_custom_loop_on_train_is_clean():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "for C in [0.1, 1.0]:\n"
        "    model = SVC(C=C)\n"
        "    model.fit(X_train, y_train)\n"
    )
    assert "tuning_on_test_data" not in kinds(src)


def test_kfold_loop_is_not_a_tuning_loop():
    src = (
        "from sklearn.model_selection import KFold\n"
        "kf = KFold(5)\n"
        "for tr, te in kf.split(X):\n"
        "    model.fit(X[tr], y[tr])\n"
    )
    assert not kinds(src)


def test_repeated_test_scores_are_reported():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "print(a.fit(X_train, y_train).score(X_test, y_test))\n"
        "print(b.fit(X_train, y_train).score(X_test, y_test))\n"
        "print(c.fit(X_train, y_train).score(X_test, y_test))\n"
    )
    fs = [f for f in findings(src) if f.kind == "test_set_reuse"]
    assert fs and fs[0].severity == "low"


def test_single_test_score_is_clean():
    src = (
        "from sklearn.model_selection import train_test_split\n"
        "X_train, X_test, y_train, y_test = train_test_split("
        "X, y, random_state=0)\n"
        "model.fit(X_train, y_train)\n"
        "print(model.score(X_test, y_test))\n"
    )
    assert "test_set_reuse" not in kinds(src)


def test_new_tuning_fix_hints_exist():
    for kind in (
        "tuning_before_split",
        "tuning_on_full_data",
        "cv_on_test_data",
        "test_set_reuse",
        "group_cv_missing_groups",
    ):
        assert advice(kind)
