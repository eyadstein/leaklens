import pandas as pd
import pytest

from leaklens.intelligence import DatasetImpact, analyze_dataset_impact


def test_measures_duplicate_test_rows():
    train = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    test = pd.DataFrame({"feature": [2, 4, 1], "target": [1, 0, 0]})

    impact = analyze_dataset_impact(train, test)

    assert impact.train_rows == 3
    assert impact.test_rows == 3
    assert impact.duplicate_test_rows == 2
    assert impact.duplicate_test_pct == pytest.approx(66.67)


def test_measures_shared_groups_and_affected_test_rows():
    train = pd.DataFrame({"patient": ["A", "B", "C"]})
    test = pd.DataFrame({"patient": ["B", "B", "D", None]})

    impact = analyze_dataset_impact(train, test, group_col="patient")

    assert impact.shared_group_count == 1
    assert impact.shared_group_test_rows == 2
    assert impact.shared_group_test_pct == 50.0


def test_measures_temporal_boundary_without_claiming_row_count():
    train = pd.DataFrame({"when": ["2025-01-03", "2025-01-05"]})
    test = pd.DataFrame({"when": ["2025-01-04", "2025-01-06"]})

    impact = analyze_dataset_impact(train, test, time_col="when")

    assert impact.temporal_overlap is True
    assert impact.train_max_time.startswith("2025-01-05")
    assert impact.test_min_time.startswith("2025-01-04")
    assert not hasattr(impact, "temporal_affected_rows")


def test_empty_test_data_does_not_divide_by_zero():
    train = pd.DataFrame({"x": [1, 2], "group": ["A", "B"], "when": ["2025-01-01", "2025-01-02"]})
    test = pd.DataFrame({"x": pd.Series(dtype=int), "group": pd.Series(dtype=str), "when": pd.Series(dtype=str)})

    impact = analyze_dataset_impact(
        train, test, group_col="group", time_col="when"
    )

    assert impact.test_rows == 0
    assert impact.duplicate_test_rows == 0
    assert impact.duplicate_test_pct == 0.0
    assert impact.shared_group_test_pct == 0.0
    assert impact.temporal_overlap is None


def test_invalid_optional_columns_are_rejected():
    train = pd.DataFrame({"x": [1]})
    test = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match="group column"):
        analyze_dataset_impact(train, test, group_col="missing")

    with pytest.raises(ValueError, match="time column"):
        analyze_dataset_impact(train, test, time_col="missing")


def test_input_types_are_validated():
    with pytest.raises(TypeError):
        analyze_dataset_impact([1, 2], pd.DataFrame({"x": [1]}))


def test_impact_serializes_to_dictionary():
    impact = DatasetImpact(
        train_rows=10,
        test_rows=5,
        duplicate_test_rows=1,
        duplicate_test_pct=20.0,
    )

    assert impact.to_dict()["duplicate_test_rows"] == 1
