from leaklens.scanner import scan_source

H = ("from sklearn.preprocessing import StandardScaler\n"
     "from sklearn.model_selection import train_test_split\n")


def kinds(body):
    return [(f.kind, f.severity) for f in scan_source(H + body)]


def test_chained_before_split_is_high():
    assert ("preprocess_before_split", "high") in kinds("X = StandardScaler().fit_transform(X)\na = train_test_split(X, y)\n")


def test_chained_on_train_is_clean():
    assert not kinds("a = train_test_split(X, y, random_state=42)\nX_train = StandardScaler().fit_transform(X_train)\n")


def test_fit_on_full_after_split_is_medium():
    k = kinds("sc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nZ = sc.fit_transform(X)\n")
    assert ("fit_on_full_data", "medium") in k


def test_derived_copy_is_tracked():
    k = kinds("X2 = X.copy()\nsc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y)\nsc.fit(X2)\n")
    assert ("fit_on_full_data", "medium") in k


def test_train_partition_is_clean():
    assert not kinds("sc = StandardScaler()\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)\n"
                     "sc.fit(X_train)\nZ = sc.transform(X_test)\n")


def test_kfold_indexing_is_clean():
    src = ("from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import KFold\n"
           "kf = KFold(5)\nfor tr, te in kf.split(X):\n    sc = StandardScaler()\n    sc.fit(X[tr])\n")
    assert not scan_source(src)


def test_backward_closure_df():
    k = kinds("df = load()\nX = df.drop(columns=['t'])\ny = df['t']\nsc = StandardScaler()\n"
              "a = train_test_split(X, y)\nsc.fit(df)\n")
    assert ("fit_on_full_data", "medium") in k