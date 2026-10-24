import json

from leaklens.cli import main
from leaklens.scanner import scan_file, scan_source

PRE = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""

GOOD = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

EVALFIT = """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Z = sc.fit_transform(X_test)
"""

SMOTE_BAD = """
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
sm = SMOTE()
X, y = sm.fit_resample(X, y)
a, b, c, d = train_test_split(X, y)
"""


def kinds(src):
    return {f.kind for f in scan_source(src)}


def test_preprocess_before_split():
    assert "preprocess_before_split" in kinds(PRE)


def test_good_pipeline_clean():
    assert not scan_source(GOOD)


def test_fit_on_eval():
    assert "fit_on_eval_data" in kinds(EVALFIT)


def test_resample_before_split():
    assert "resample_before_split" in kinds(SMOTE_BAD)


def test_notebook(tmp_path):
    cells = [{"cell_type": "code", "source": ["%matplotlib inline\n", "x = 1"]},
             {"cell_type": "code", "source": PRE.splitlines(keepends=True)}]
    f = tmp_path / "n.ipynb"
    f.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    fs = scan_file(f)
    assert any(x.kind == "preprocess_before_split" and "cell 1" in x.column for x in fs)


def test_cli_scan(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text(PRE, encoding="utf-8")
    assert main(["--scan", str(f)]) == 1
    g = tmp_path / "good.py"
    g.write_text(GOOD, encoding="utf-8")
    assert main(["--scan", str(g)]) == 0

TARGET_ENCODE = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["te"] = df.groupby("city")[y].transform("mean")
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""

TARGET_ENCODE_BY_NAME = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["te"] = df.groupby("city")["target"].transform("mean")
X_train, X_test, y_train, y_test = train_test_split(df.drop("target"), df["target"])
"""

GROUP_STAT = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["city_mean"] = df.groupby("city")["sales"].transform("mean")
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""

GROUP_STAT_AFTER_SPLIT = """
import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train["city_mean"] = X_train.groupby("city")["sales"].transform("mean")
"""


def test_target_encode_before_split_is_high():
    fs = [f for f in scan_source(TARGET_ENCODE) if f.kind == "target_encode_before_split"]
    assert fs and fs[0].severity == "high"
    assert "before the split" in fs[0].detail


def test_target_encode_by_column_name():
    assert "target_encode_before_split" in kinds(TARGET_ENCODE_BY_NAME)


def test_group_stat_before_split_is_medium():
    fs = [f for f in scan_source(GROUP_STAT) if f.kind == "group_stat_before_split"]
    assert fs and fs[0].severity == "medium"
    assert "target_encode_before_split" not in kinds(GROUP_STAT)


def test_group_stat_after_split_is_clean():
    assert not scan_source(GROUP_STAT_AFTER_SPLIT)


def test_fix_hints_exist_for_new_kinds():
    from leaklens.fixes import advice
    assert advice("target_encode_before_split")
    assert advice("group_stat_before_split")
