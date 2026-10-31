from leaklens.scanner import scan_source

H = ("from sklearn.preprocessing import StandardScaler\n"
     "from sklearn.model_selection import cross_val_score, train_test_split\n")


def k(body):
    return [(f.kind, f.severity) for f in scan_source(H + body)]


def test_scaler_then_cv_is_high():
    assert ("preprocess_outside_cv", "high") in k("X = StandardScaler().fit_transform(X)\ns = cross_val_score(m, X, y)\n")


def test_pipeline_is_clean():
    assert not k("p = make_pipeline(StandardScaler(), m)\ns = cross_val_score(p, X, y)\n")


def test_after_split_is_downgraded_to_low():
    body = ("a = train_test_split(X, y, random_state=1)\n"
            "Xt = StandardScaler().fit_transform(X_train)\ns = cross_val_score(m, Xt, y_train)\n")
    assert ("preprocess_outside_cv", "low") in k(body)


def test_derived_name_is_tracked():
    assert ("preprocess_outside_cv", "high") in k("Xs = StandardScaler().fit_transform(X)\nX2 = Xs.copy()\ns = cross_val_score(m, X2, y)\n")


def test_smote_then_cv_is_high():
    src = ("from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import cross_val_score\n"
           "X, y = SMOTE().fit_resample(X, y)\ns = cross_val_score(m, X, y)\n")
    assert ("resample_outside_cv", "high") in [(f.kind, f.severity) for f in scan_source(src)]


def test_gridsearch_fit_on_scaled_x():
    src = ("from sklearn.model_selection import GridSearchCV\nfrom sklearn.preprocessing import StandardScaler\n"
           "Xs = StandardScaler().fit_transform(X)\ngs = GridSearchCV(m, g)\ngs.fit(Xs, y)\n")
    assert ("preprocess_outside_cv", "high") in [(f.kind, f.severity) for f in scan_source(src)]


def test_raw_x_is_clean():
    assert not k("s = cross_val_score(m, X, y)\n")

def test_fit_then_transform_before_cv_is_high():
    body = "sc = StandardScaler()\nsc.fit(X)\nXs = sc.transform(X)\ns = cross_val_score(m, Xs, y)\n"
    assert ("preprocess_outside_cv", "high") in k(body)


def test_fit_on_train_then_transform_before_cv_is_low():
    body = ("a = train_test_split(X, y, random_state=0)\nsc = StandardScaler()\n"
            "sc.fit(X_train)\nXs = sc.transform(X)\ns = cross_val_score(m, Xs, y)\n")
    assert ("preprocess_outside_cv", "low") in k(body)


def test_unfitted_transform_is_clean():
    body = "sc = StandardScaler()\nXs = sc.transform(X)\ns = cross_val_score(m, Xs, y)\n"
    assert not k(body)
