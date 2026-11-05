"""Chronological splitting and group-aware cross-validation."""

from leaklens.scanner import scan_source


def kinds(src):
    return {f.kind for f in scan_source(src)}


SPLIT = "from sklearn.model_selection import train_test_split\n"


def test_time_sorted_default_shuffle_is_medium():
    src = (
        "import pandas as pd\n"
        + SPLIT
        + "ordered = df.sort_values('date')\n"
        + "X_train, X_test, y_train, y_test = train_test_split("
        "ordered[['x']], ordered['y'], random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "temporal_split_shuffle"]
    assert fs and fs[0].severity == "medium"


def test_time_sorted_explicit_shuffle_is_high():
    src = (
        "import pandas as pd\n"
        + SPLIT
        + "ordered = df.sort_values('timestamp')\n"
        + "X_train, X_test, y_train, y_test = train_test_split("
        "ordered[['x']], ordered['y'], shuffle=True, random_state=0)\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "temporal_split_shuffle"]
    assert fs and fs[0].severity == "high"


def test_time_sorted_chronological_split_is_clean():
    src = (
        "import pandas as pd\n"
        + SPLIT
        + "ordered = df.sort_values('date')\n"
        + "X_train, X_test, y_train, y_test = train_test_split("
        "ordered[['x']], ordered['y'], shuffle=False)\n"
    )
    assert "temporal_split_shuffle" not in kinds(src)


def test_sorting_an_unrelated_column_is_clean():
    src = (
        "import pandas as pd\n"
        + SPLIT
        + "ordered = df.sort_values('sales')\n"
        + "X_train, X_test, y_train, y_test = train_test_split("
        "ordered[['x']], ordered['y'], random_state=0)\n"
    )
    assert "temporal_split_shuffle" not in kinds(src)


def test_group_split_without_groups_is_medium():
    src = (
        "from sklearn.model_selection import GroupKFold\n"
        "kf = GroupKFold(n_splits=5)\n"
        "for tr, te in kf.split(X, y):\n"
        "    pass\n"
    )
    fs = [f for f in scan_source(src) if f.kind == "group_cv_missing_groups"]
    assert fs and fs[0].severity == "medium"


def test_group_split_with_groups_is_clean():
    src = (
        "from sklearn.model_selection import GroupKFold\n"
        "kf = GroupKFold(n_splits=5)\n"
        "for tr, te in kf.split(X, y, groups=patient_id):\n"
        "    pass\n"
    )
    assert "group_cv_missing_groups" not in kinds(src)


def test_group_cv_without_groups_is_flagged():
    src = (
        "from sklearn.model_selection import GroupKFold, cross_val_score\n"
        "kf = GroupKFold(n_splits=5)\n"
        "scores = cross_val_score(model, X, y, cv=kf)\n"
    )
    assert "group_cv_missing_groups" in kinds(src)


def test_group_cv_with_groups_is_clean():
    src = (
        "from sklearn.model_selection import GroupKFold, cross_val_score\n"
        "kf = GroupKFold(n_splits=5)\n"
        "scores = cross_val_score(model, X, y, cv=kf, groups=patient_id)\n"
    )
    assert "group_cv_missing_groups" not in kinds(src)


def test_gridsearch_group_cv_without_groups_is_flagged():
    src = (
        "from sklearn.model_selection import GroupKFold, GridSearchCV\n"
        "kf = GroupKFold(n_splits=5)\n"
        "gs = GridSearchCV(model, grid, cv=kf)\n"
        "gs.fit(X, y)\n"
    )
    assert "group_cv_missing_groups" in kinds(src)


def test_kfold_is_not_group_leakage():
    src = (
        "from sklearn.model_selection import KFold\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=0)\n"
        "for tr, te in kf.split(X, y):\n"
        "    pass\n"
    )
    assert "group_cv_missing_groups" not in kinds(src)
