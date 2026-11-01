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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train["city_mean"] = X_train.groupby("city")["sales"].transform("mean")
"""

GROUP_STAT_ON_X_COLUMN = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["age_mean"] = df.groupby("city")["age"].transform("mean")
X_train, X_test, y_train, y_test = train_test_split(df[["age", "income"]], df["target"], random_state=42)
"""

AGG_INTO_COLUMN = """
import pandas as pd
from sklearn.model_selection import train_test_split
df["m"] = df.groupby("city")["sales"].agg("mean")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
"""

AGG_SUMMARY_ONLY = """
import pandas as pd
from sklearn.model_selection import train_test_split
summary = df.groupby("city")["sales"].agg("mean")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
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


def test_group_stat_on_x_column_stays_medium():
    fs = [f for f in scan_source(GROUP_STAT_ON_X_COLUMN)
          if f.kind in ("group_stat_before_split", "target_encode_before_split")]
    assert fs and fs[0].kind == "group_stat_before_split" and fs[0].severity == "medium"


def test_agg_into_column_is_group_stat():
    assert "group_stat_before_split" in kinds(AGG_INTO_COLUMN)


def test_agg_summary_is_not_flagged():
    assert not scan_source(AGG_SUMMARY_ONLY)


def test_fix_hints_exist_for_new_kinds():
    from leaklens.fixes import advice
    assert advice("target_encode_before_split")
    assert advice("group_stat_before_split")


MANUAL_SCALE = """
from sklearn.model_selection import train_test_split
X = (X - X.mean()) / X.std()
a = train_test_split(X, y, random_state=42)
"""

MANUAL_SCALE_TRAIN_ONLY = """
from sklearn.model_selection import train_test_split
a = train_test_split(X, y, random_state=42)
X_test = (X_test - X_train.mean()) / X_train.std()
"""


def test_manual_scale_before_split_is_high():
    fs = [f for f in scan_source(MANUAL_SCALE) if f.kind == "manual_scale_before_split"]
    assert fs and fs[0].severity == "high"
    assert "before the split" in fs[0].detail


def test_manual_scale_with_train_statistics_is_clean():
    assert not scan_source(MANUAL_SCALE_TRAIN_ONLY)


def test_manual_scale_fix_hint_exists():
    from leaklens.fixes import advice
    assert advice("manual_scale_before_split")


def test_notebook_one_bad_cell_does_not_hide_the_rest(tmp_path):
    cells = [{"cell_type": "code",
              "source": ["from sklearn.preprocessing import StandardScaler\n",
                         "sc = StandardScaler()\n", "X = sc.fit_transform(X)\n"]},
             {"cell_type": "code",
              "source": ["%%writefile output.txt\n",
                         "not python: 1 2 3 !!! ( )))\n", "unmatched '\n"]},
             {"cell_type": "code",
              "source": ["from sklearn.model_selection import train_test_split\n",
                         "a = train_test_split(X, y, random_state=0)\n"]}]
    f = tmp_path / "badcell.ipynb"
    f.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    fs = scan_file(f)
    kinds_ = [x.kind for x in fs]
    assert "preprocess_before_split" in kinds_, fs
    assert "parse_error" in kinds_ and any("cell 1 skipped" in x.detail for x in fs)
    assert any(x.kind == "preprocess_before_split" and "cell 0" in x.column for x in fs)

GET_DUMMIES_BEFORE_SPLIT = """
import pandas as pd
from sklearn.model_selection import train_test_split
X = pd.get_dummies(df)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""

GET_DUMMIES_AFTER_SPLIT = """
import pandas as pd
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
"""

def test_get_dummies_before_split_is_low_severity():
    findings = [
        f for f in scan_source(GET_DUMMIES_BEFORE_SPLIT)
        if f.kind == "get_dummies_before_split"
    ]
    assert findings
    assert findings[0].severity == "low"

def test_get_dummies_after_split_is_clean():
    findings = scan_source(GET_DUMMIES_AFTER_SPLIT)
    assert not any(f.kind == "get_dummies_before_split" for f in findings)

def test_unrelated_get_dummies_is_clean():
    src = """
import pandas as pd
from sklearn.model_selection import train_test_split
summary = pd.get_dummies(df)
a = train_test_split(X, y)
"""
    findings = scan_source(src)
    assert not any(f.kind == "get_dummies_before_split" for f in findings)

def test_get_dummies_import_alias_before_split():
    src = """
from pandas import get_dummies as dummies
from sklearn.model_selection import train_test_split
X = dummies(df)
a = train_test_split(X, y)
"""
    findings = scan_source(src)
    assert any(f.kind == "get_dummies_before_split" for f in findings)
