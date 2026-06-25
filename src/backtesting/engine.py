"""Backtest fixed-weight portfolios on monthly return data."""

import pandas as pd


def run_fixed_weight_backtest(
    returns: pd.DataFrame,
    weights: dict[str, float],
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Run a no-cost fixed-weight benchmark backtest."""
    _validate_weights(weights)
    missing_columns = set(weights) - set(returns.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"returns are missing columns: {missing}")

    ordered_weights = pd.Series(weights, dtype=float)
    portfolio_returns = returns[ordered_weights.index].mul(ordered_weights).sum(axis=1)
    portfolio_values = initial_value * (1.0 + portfolio_returns).cumprod()
    drawdowns = portfolio_values / portfolio_values.cummax() - 1.0

    return pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "turnover": 0.0,
            "portfolio_value": portfolio_values,
            "drawdown": drawdowns,
        },
        index=returns.index,
    )


def _validate_weights(weights: dict[str, float]) -> None:
    """Validate fixed benchmark weights."""
    if not weights:
        raise ValueError("weights cannot be empty")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
