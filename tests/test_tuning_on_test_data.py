"""Detector 2.4: hyperparameter tuning / model selection on the test set."""

from leaklens.scanner import scan_source


def kinds(src):
    return [f.kind for f in scan_source(src)]


LEAKY = (
    "from sklearn.model_selection import GridSearchCV, train_test_split\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]})\n"
    "gs.fit(X_test, y_test)\n"
)

CLEAN_CV = (
    "from sklearn.model_selection import GridSearchCV, train_test_split\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
    "gs = GridSearchCV(LogisticRegression(), {'C': [0.1, 1.0]}, cv=5)\n"
    "gs.fit(X_train, y_train)\n"
    "print(gs.score(X_test, y_test))\n"
)


def test_gridsearch_fit_on_test_is_high():
    fs = [f for f in scan_source(LEAKY) if f.kind == "tuning_on_test_data"]
    assert fs, scan_source(LEAKY)
    assert fs[0].severity == "high"
    assert "test" in fs[0].detail.lower()


def test_gridsearch_fit_on_train_is_clean():
    assert "tuning_on_test_data" not in kinds(CLEAN_CV)


def test_randomized_search_on_test_flagged():
    src = (
        "from sklearn.model_selection import RandomizedSearchCV\n"
        "rs = RandomizedSearchCV(model, dists)\n"
        "rs.fit(X_test, y_test)\n"
    )
    assert "tuning_on_test_data" in kinds(src)


def test_x_test_variable_name_flagged():
    src = (
        "from sklearn.model_selection import GridSearchCV\n"
        "gs = GridSearchCV(model, grid)\n"
        "gs.fit(X_test, y_test)\n"
    )
    assert "tuning_on_test_data" in kinds(src)


def test_normal_fit_on_test_not_tuned():
    # plain .fit() is not hyperparameter tuning; must not be flagged by 2.4
    src = (
        "from sklearn.linear_model import LogisticRegression\n"
        "m = LogisticRegression()\n"
        "m.fit(X_test, y_test)\n"
    )
    assert "tuning_on_test_data" not in kinds(src)


def test_score_on_test_is_clean():
    # evaluating (not tuning) on test is legitimate
    src = (
        "from sklearn.model_selection import GridSearchCV\n"
        "gs = GridSearchCV(model, grid, cv=5)\n"
        "best = gs.fit(X_train, y_train)\n"
        "s = best.score(X_test, y_test)\n"
    )
    assert "tuning_on_test_data" not in kinds(src)


def test_no_split_trace_no_false_positive():
    src = (
        "from sklearn.model_selection import GridSearchCV\n"
        "gs = GridSearchCV(model, grid, cv=5)\n"
        "gs.fit(X, y)\n"
    )
    assert "tuning_on_test_data" not in kinds(src)


def test_halving_search_on_test_flagged():
    src = (
        "from sklearn.model_selection import HalvingGridSearchCV\n"
        "gs = HalvingGridSearchCV(model, grid)\n"
        "gs.fit(X_test, y_test)\n"
    )
    assert "tuning_on_test_data" in kinds(src)
