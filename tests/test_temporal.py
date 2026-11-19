from leaklens.scanner import scan_source


def kinds(source):
    return {finding.kind for finding in scan_source(source)}


def test_negative_target_shift_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = y.shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_centered_target_rolling_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
target_mean = y.rolling(5, center=True).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_trailing_target_rolling_before_split_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
target_mean = y.rolling(5).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_centered_rolling_is_reported_even_when_closed_left():
    source = """
from sklearn.model_selection import train_test_split
target_mean = y.rolling(5, center=True, closed="left").mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_left_closed_rolling_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
target_mean = y.rolling(5, closed="left").mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_temporal_feature_after_split_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
train_test_split(X, y)
future = y.shift(-1)
"""
    assert not ({"future_target_feature_before_split"} & kinds(source))


def test_rolling_unrelated_feature_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
feature_mean = sales.rolling(5).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_target_derived_alias_is_reported():
    source = """
from sklearn.model_selection import train_test_split
target_series = df["target"]
future = target_series.shift(periods=-2)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_target_column_negative_shift_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df["target"].shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_label_variable_column_negative_shift_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df["y"].shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_dot_access_target_shift_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df.y.shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_grouped_negative_shift_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df.groupby("entity")["y"].shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_grouped_negative_shift_with_periods_is_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df.groupby("entity")["y"].shift(periods=-2)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_grouped_negative_shift_on_feature_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
future = df.groupby("entity")["sales"].shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_positive_shift_with_periods_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
lag = y.shift(periods=2)
other = df["y"].shift(1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_grouped_trailing_rolling_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
smooth = df.groupby("entity")["y"].rolling(3).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_grouped_centered_rolling_is_reported():
    source = """
from sklearn.model_selection import train_test_split
smooth = df.groupby("entity")["y"].rolling(3, center=True).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_positive_shift_then_trailing_rolling_is_not_reported():
    source = """
from sklearn.model_selection import train_test_split
feature = y.shift(1).rolling(3).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" not in kinds(source)


def test_future_shift_inside_rolling_chain_is_reported():
    source = """
from sklearn.model_selection import train_test_split
feature = y.rolling(3).shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_assignment_of_future_feature_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
df["next_target"] = df["target"].shift(-1)
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


def test_one_finding_per_temporal_call():
    source = """
from sklearn.model_selection import train_test_split
future = y.shift(-1)
train_test_split(X, y)
"""
    found = [
        f for f in scan_source(source) if f.kind == "future_target_feature_before_split"
    ]
    assert len(found) == 1
