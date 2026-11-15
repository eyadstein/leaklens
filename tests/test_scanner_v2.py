import pytest

from leaklens.cli import main
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
a = train_test_split(Xs, y, random_state=42, stratify=y)
"""


def test_function_style_scale_is_high():
    fs = scan_source(SCALE_FN)
    assert fs and fs[0].severity == "high" and fs[0].kind == "preprocess_before_split"


def test_column_transformer_is_medium():
    fs = scan_source(CT)
    assert (
        fs and fs[0].severity == "medium" and fs[0].kind == "transformer_before_split"
    )


def test_unrelated_scale_ignored():
    assert not scan_source(OTHER_SCALE)


# --- split/frame relationship: EDA on a frame the split does not divide ---

UNRELATED_EDA = """
import pandas as pd
from sklearn.model_selection import train_test_split
audit = pd.read_csv("audit.csv")
{eda}
frame = pd.read_csv("modelling.csv")
y = frame["target"]
X = frame.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
"""

UNRELATED_CASES = {
    "manual_scale_before_split": 'audit["z"] = (audit["x"] - audit["x"].mean()) / audit["x"].std()',
}


@pytest.mark.parametrize("kind,eda", sorted(UNRELATED_CASES.items()))
def test_unrelated_eda_frame_is_not_flagged(kind, eda):
    assert kind not in {f.kind for f in scan_source(UNRELATED_EDA.format(eda=eda))}


def test_cli_scan_unrelated_eda_exits_clean(tmp_path):
    f = tmp_path / "eda.py"
    f.write_text(
        UNRELATED_EDA.format(eda=UNRELATED_CASES["manual_scale_before_split"]),
        encoding="utf-8",
    )
    # today the unrelated frame's manual scaling is a high finding and exits 1
    assert main(["--scan", str(f)]) == 0


def test_unbound_frame_statistics_stay_flagged():
    src = """
from sklearn.model_selection import train_test_split
df["z"] = (df["x"] - df["x"].mean()) / df["x"].std()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
"""
    assert "manual_scale_before_split" in {f.kind for f in scan_source(src)}


def test_function_parameter_statistics_stay_flagged():
    src = """
from sklearn.model_selection import train_test_split
def prep(df):
    return (df - df.mean()) / df.std()
X_train, X_test, y_train, y_test = train_test_split(prep(X), y, random_state=0, stratify=y)
"""
    assert "manual_scale_before_split" in {f.kind for f in scan_source(src)}
