"""Return calculations for asset price data."""

import pandas as pd


def calculate_simple_returns(
    prices: pd.DataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    """Calculate simple monthly returns without using future prices.

    Parameters
    ----------
    prices
        Monthly asset prices indexed by date.
    column_map
        Mapping from raw price column names to desired return column names.
    """
    returns = prices[list(column_map)].pct_change()
    return returns.rename(columns=column_map)


def calculate_equal_weight_return(
    component_returns: pd.DataFrame,
    output_column: str,
) -> pd.Series:
    """Calculate equal-weight basket returns from component returns."""
    if component_returns.empty:
        raise ValueError("component_returns cannot be empty")
    return component_returns.mean(axis=1).rename(output_column)


def build_price_index(
    returns: pd.Series,
    output_column: str,
    initial_value: float = 100.0,
) -> pd.Series:
    """Build a price index from simple returns."""
    if initial_value <= 0.0:
        raise ValueError("initial_value must be positive")
    return (initial_value * (1.0 + returns).cumprod()).rename(output_column)
