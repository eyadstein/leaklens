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


def test_target_rolling_before_split_is_reported():
    source = """
from sklearn.model_selection import train_test_split
target_mean = y.rolling(5).mean()
train_test_split(X, y)
"""
    assert "future_target_feature_before_split" in kinds(source)


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
    assert not ({"future_target_feature_before_split", "future_target_feature_before_split"} & kinds(source))


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
