"""Walk-forward split generation."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """One train/validation/test walk-forward split."""

    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_months: int = 48,
    validation_months: int = 12,
    test_months: int = 24,
    step_months: int = 12,
) -> list[WalkForwardSplit]:
    """Generate rolling train/validation/test windows from monthly dates."""
    _validate_split_inputs(
        dates=dates,
        train_months=train_months,
        validation_months=validation_months,
        test_months=test_months,
        step_months=step_months,
    )

    splits: list[WalkForwardSplit] = []
    start = 0
    split_id = 0
    total_months = train_months + validation_months + test_months
    while start + total_months <= len(dates):
        train_start = dates[start]
        train_end = dates[start + train_months - 1]
        validation_start = dates[start + train_months]
        validation_end = dates[start + train_months + validation_months - 1]
        test_start = dates[start + train_months + validation_months]
        test_end = dates[start + total_months - 1]
        splits.append(
            WalkForwardSplit(
                split_id=split_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            ),
        )
        split_id += 1
        start += step_months

    return splits


def _validate_split_inputs(
    dates: pd.DatetimeIndex,
    train_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
) -> None:
    """Validate walk-forward split inputs."""
    if dates.empty:
        raise ValueError("dates cannot be empty")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be sorted")
    if dates.has_duplicates:
        raise ValueError("dates cannot contain duplicates")
    if min(train_months, validation_months, test_months, step_months) < 1:
        raise ValueError("window lengths must be positive")
    if train_months + validation_months + test_months > len(dates):
        raise ValueError("not enough dates for one walk-forward split")
