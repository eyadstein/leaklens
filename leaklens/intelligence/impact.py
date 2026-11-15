"""Measured dataset and split impact metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ..leakage import temporal_extents


@dataclass(frozen=True, slots=True)
class DatasetImpact:
    """Observable dataset impact; metrics are measurements, not model estimates."""

    train_rows: int
    test_rows: int
    duplicate_test_rows: int
    duplicate_test_pct: float
    group_column: str | None = None
    shared_group_count: int = 0
    shared_group_test_rows: int = 0
    shared_group_test_pct: float = 0.0
    time_column: str | None = None
    temporal_overlap: bool | None = None
    train_max_time: str | None = None
    test_min_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_dataset_impact(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str | None = None,
    time_col: str | None = None,
) -> DatasetImpact:
    """Measure duplicate rows and optional group/time split overlap."""

    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames")

    train_rows = len(train)
    test_rows = len(test)

    common_columns = [column for column in train.columns if column in test.columns]
    duplicate_rows = 0

    if common_columns and test_rows and train_rows:
        train_hashes = set(
            pd.util.hash_pandas_object(train[common_columns], index=False)
        )
        duplicate_rows = int(
            pd.util.hash_pandas_object(test[common_columns], index=False)
            .isin(train_hashes)
            .sum()
        )

    duplicate_pct = duplicate_rows / test_rows * 100.0 if test_rows else 0.0

    shared_group_count = 0
    shared_group_test_rows = 0
    shared_group_pct = 0.0

    if group_col is not None:
        if group_col not in train.columns or group_col not in test.columns:
            raise ValueError(
                f"group column {group_col!r} must exist in both train and test"
            )

        train_groups = set(train[group_col].dropna())
        test_groups = test[group_col]
        shared_groups = train_groups & set(test_groups.dropna())
        shared_group_count = len(shared_groups)
        shared_group_test_rows = int(test_groups.isin(shared_groups).sum())
        shared_group_pct = (
            shared_group_test_rows / test_rows * 100.0 if test_rows else 0.0
        )

    temporal_overlap = None
    train_max_time = None
    test_min_time = None

    if time_col is not None:
        if time_col not in train.columns or time_col not in test.columns:
            raise ValueError(
                f"time column {time_col!r} must exist in both train and test"
            )

        train_max, test_min, _ = temporal_extents(train, test, time_col)
        if train_max is not None:
            train_max_time = train_max.isoformat()
        if test_min is not None:
            test_min_time = test_min.isoformat()
        if train_max is not None and test_min is not None:
            temporal_overlap = bool(train_max > test_min)

    return DatasetImpact(
        train_rows=train_rows,
        test_rows=test_rows,
        duplicate_test_rows=duplicate_rows,
        duplicate_test_pct=round(duplicate_pct, 2),
        group_column=group_col,
        shared_group_count=shared_group_count,
        shared_group_test_rows=shared_group_test_rows,
        shared_group_test_pct=round(shared_group_pct, 2),
        time_column=time_col,
        temporal_overlap=temporal_overlap,
        train_max_time=train_max_time,
        test_min_time=test_min_time,
    )
