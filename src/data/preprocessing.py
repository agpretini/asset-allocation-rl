"""Preprocessing helpers for monthly time-series data."""

import pandas as pd


def set_monthly_date_index(data: pd.DataFrame) -> pd.DataFrame:
    """Return data sorted by date with a monthly DatetimeIndex."""
    if "date" not in data.columns:
        raise ValueError("data must contain a date column")

    indexed = data.copy()
    indexed["date"] = pd.to_datetime(indexed["date"])
    indexed = indexed.sort_values("date").set_index("date")
    indexed.index.name = "date"
    validate_complete_monthly_index(indexed)
    return indexed


def validate_complete_monthly_index(data: pd.DataFrame) -> None:
    """Raise when data does not contain one row for each month-end date."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must use a DatetimeIndex")
    if data.index.has_duplicates:
        raise ValueError("monthly data contains duplicate dates")
    if data.empty:
        raise ValueError("monthly data cannot be empty")

    expected_index = pd.date_range(data.index.min(), data.index.max(), freq="ME")
    if not data.index.equals(expected_index):
        raise ValueError("monthly data must contain every month-end date")


def validate_no_missing_values(data: pd.DataFrame) -> None:
    """Raise when the dataset contains missing values."""
    if data.isna().any().any():
        raise ValueError("dataset contains missing values")


def impute_missing_prices(prices: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Impute missing prices with a past-only rolling average."""
    if window < 1:
        raise ValueError("window must be at least one")

    imputed = prices.copy()
    for column in imputed.columns:
        previous_average = (
            imputed[column]
            .shift(1)
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
        )
        imputed[column] = imputed[column].fillna(previous_average)

    return imputed.dropna()
