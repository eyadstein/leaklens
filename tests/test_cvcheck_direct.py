"""Direct unit tests for leaklens/cvcheck.scan_cv (bypassing scanner.py)."""

from leaklens.cvcheck import scan_cv


def test_clean_code_no_findings():
    src = "from sklearn.model_selection import cross_val_score\ns = cross_val_score(m, X, y)\n"
    assert scan_cv(src) == []


def test_scaler_before_cross_val_score():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "s = cross_val_score(m, Xs, y)\n"
    )
    findings = scan_cv(src)
    assert len(findings) == 1
    f = findings[0]
    assert f.kind == "preprocess_outside_cv"
    assert f.severity == "high"
    assert "StandardScaler" in f.detail
    assert "cross_val_score" in f.detail


def test_syntax_error_returns_empty():
    assert scan_cv("def broken(:\n") == []


def test_smote_before_cv_is_resample():
    src = (
        "from imblearn.over_sampling import SMOTE\n"
        "from sklearn.model_selection import cross_val_score\n"
        "X2, y2 = SMOTE().fit_resample(X, y)\n"
        "s = cross_val_score(m, X2, y2)\n"
    )
    findings = scan_cv(src)
    assert len(findings) == 1
    assert findings[0].kind == "resample_outside_cv"
    assert findings[0].severity == "high"


def test_gridsearch_after_scaling():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import GridSearchCV\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "gs = GridSearchCV(m, params)\n"
        "gs.fit(Xs, y)\n"
    )
    findings = scan_cv(src)
    assert any(f.kind == "preprocess_outside_cv" for f in findings)


def test_pipeline_usage_is_clean():
    src = (
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "p = make_pipeline(StandardScaler(), m)\n"
        "s = cross_val_score(p, X, y)\n"
    )
    assert scan_cv(src) == []


def test_fit_on_train_only_downgrades_to_low():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score, train_test_split\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)\n"
        "sc = StandardScaler()\n"
        "sc.fit(X_tr)\n"
        "Xs = sc.transform(X_tr)\n"
        "s = cross_val_score(m, Xs, y_tr)\n"
    )
    findings = scan_cv(src)
    assert len(findings) == 1
    assert findings[0].severity == "low"


def test_fit_then_transform_before_cv_tracked():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "sc = StandardScaler()\n"
        "sc.fit(X)\n"
        "Xs = sc.transform(X)\n"
        "s = cross_val_score(m, Xs, y)\n"
    )
    findings = scan_cv(src)
    assert len(findings) == 1
    assert findings[0].kind == "preprocess_outside_cv"


def test_unfitted_transform_is_clean():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "sc = StandardScaler()\n"
        "Xs = sc.transform(X)\n"
        "s = cross_val_score(m, Xs, y)\n"
    )
    assert scan_cv(src) == []


def test_multiple_cv_calls_grouped():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score, cross_val_predict\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "a = cross_val_score(m, Xs, y)\n"
        "b = cross_val_predict(m, Xs, y)\n"
    )
    findings = scan_cv(src)
    # grouped by (line, kind) → single finding, mentions "and 1 more call"
    assert len(findings) == 1
    assert "1 more call" in findings[0].detail


def test_label_and_cells_used_in_location():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "s = cross_val_score(m, Xs, y)\n"
    )
    with_cells = scan_cv(src, label="nb", cells=[(0, 1), (5, 2)])
    assert "nb[cell" in with_cells[0].column


def test_label_only_location():
    src = (
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import cross_val_score\n"
        "Xs = StandardScaler().fit_transform(X)\n"
        "s = cross_val_score(m, Xs, y)\n"
    )
    f = scan_cv(src, label="file.py")[0]
    assert f.column.startswith("file.py:")
