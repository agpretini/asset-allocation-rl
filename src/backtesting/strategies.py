"""Rule-based portfolio strategies."""

import pandas as pd

ASSET_PREFIXES = ("arg", "ced", "sp500", "gold")


def equal_weight_strategy(history: pd.DataFrame) -> list[float]:
    """Allocate equally to the four asset classes."""
    del history
    return [0.25, 0.25, 0.25, 0.25]


def momentum_strategy(
    history: pd.DataFrame,
    lookback_months: int = 12,
) -> list[float]:
    """Allocate in proportion to positive trailing momentum."""
    _validate_history(history, lookback_months)
    momentum_values = []
    for prefix in ASSET_PREFIXES:
        prices = history[f"price_{prefix}"]
        momentum_values.append(
            float(prices.iloc[-1] / prices.iloc[-lookback_months - 1] - 1.0),
        )

    positive_momentum = [max(value, 0.0) for value in momentum_values]
    total_positive_momentum = sum(positive_momentum)
    if total_positive_momentum == 0.0:
        return equal_weight_strategy(history)
    return [value / total_positive_momentum for value in positive_momentum]


def volatility_weighted_strategy(
    history: pd.DataFrame,
    lookback_months: int = 12,
) -> list[float]:
    """Allocate using inverse trailing volatility."""
    _validate_history(history, lookback_months)
    inverse_volatility = []
    for prefix in ASSET_PREFIXES:
        volatility = float(history[f"r_{prefix}"].iloc[-lookback_months:].std())
        inverse_volatility.append(0.0 if volatility == 0.0 else 1.0 / volatility)

    total_inverse_volatility = sum(inverse_volatility)
    if total_inverse_volatility == 0.0:
        return equal_weight_strategy(history)
    return [value / total_inverse_volatility for value in inverse_volatility]


def _validate_history(history: pd.DataFrame, lookback_months: int) -> None:
    """Validate strategy history length and required columns."""
    if lookback_months < 1:
        raise ValueError("lookback_months must be positive")
    if len(history) <= lookback_months:
        raise ValueError("history must contain more rows than lookback_months")

    required_columns = {
        *(f"price_{prefix}" for prefix in ASSET_PREFIXES),
        *(f"r_{prefix}" for prefix in ASSET_PREFIXES),
    }
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"history is missing required columns: {missing}")
