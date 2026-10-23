from leaklens.scanner import scan_source

SCALE_FN = """
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
Xs = scale(X)
a = train_test_split(Xs, y)
"""

CT = """
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
ct = ColumnTransformer([])
X = ct.fit_transform(X)
a = train_test_split(X, y)
"""

OTHER_SCALE = """
from mylib import scale
from sklearn.model_selection import train_test_split
Xs = scale(X)
a = train_test_split(Xs, y, random_state=42)
"""


def test_function_style_scale_is_high():
    fs = scan_source(SCALE_FN)
    assert fs and fs[0].severity == "high" and fs[0].kind == "preprocess_before_split"


def test_column_transformer_is_medium():
    fs = scan_source(CT)
    assert fs and fs[0].severity == "medium" and fs[0].kind == "transformer_before_split"


def test_unrelated_scale_ignored():
    assert not scan_source(OTHER_SCALE)