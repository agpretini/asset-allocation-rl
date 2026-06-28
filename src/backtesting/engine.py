"""Backtest fixed and rule-based portfolios on monthly return data."""

from collections.abc import Callable

import pandas as pd

from env.portfolio_env import calculate_turnover
from env.reward import calculate_reward

Strategy = Callable[[pd.DataFrame], list[float]]


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


def run_weight_strategy_backtest(
    dataset: pd.DataFrame,
    strategy: Strategy,
    initial_weights: list[float] | None = None,
    initial_value: float = 1.0,
    transaction_cost: float = 0.014,
    start_index: int = 12,
    return_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Backtest a strategy that selects target weights from historical data."""
    selected_return_columns = return_columns or ["r_arg", "r_ced", "r_sp500", "r_gold"]
    _validate_strategy_inputs(
        dataset=dataset,
        return_columns=selected_return_columns,
        initial_weights=initial_weights,
        initial_value=initial_value,
        transaction_cost=transaction_cost,
        start_index=start_index,
    )

    previous_weights = initial_weights or [0.25, 0.25, 0.25, 0.25]
    portfolio_value = initial_value
    peak_value = initial_value
    rows: list[dict] = []

    for current_index in range(start_index, len(dataset) - 1):
        history = dataset.iloc[: current_index + 1]
        weights = strategy(history)
        _validate_weight_vector(weights)

        next_row = dataset.iloc[current_index + 1]
        portfolio_return = float(
            sum(
                weight * next_row[column]
                for weight, column in zip(weights, selected_return_columns, strict=True)
            ),
        )
        turnover = calculate_turnover(weights, previous_weights)
        reward = calculate_reward(
            portfolio_return=portfolio_return,
            turnover=turnover,
            transaction_cost=transaction_cost,
        )
        portfolio_value *= 1.0 + reward
        peak_value = max(peak_value, portfolio_value)
        drawdown = portfolio_value / peak_value - 1.0

        rows.append(
            {
                "date": dataset.index[current_index + 1],
                "weights": weights,
                "portfolio_return": portfolio_return,
                "turnover": turnover,
                "rebalance_cost": transaction_cost * turnover,
                "reward": reward,
                "portfolio_value": portfolio_value,
                "drawdown": drawdown,
            },
        )
        previous_weights = weights

    return pd.DataFrame(rows).set_index("date")


def _validate_strategy_inputs(
    dataset: pd.DataFrame,
    return_columns: list[str],
    initial_weights: list[float] | None,
    initial_value: float,
    transaction_cost: float,
    start_index: int,
) -> None:
    """Validate inputs for a rule-based strategy backtest."""
    if dataset.empty:
        raise ValueError("dataset cannot be empty")
    missing_columns = set(return_columns) - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing return columns: {missing}")
    if initial_value <= 0.0:
        raise ValueError("initial_value must be positive")
    if transaction_cost < 0.0:
        raise ValueError("transaction_cost must be non-negative")
    if start_index < 0 or start_index >= len(dataset) - 1:
        raise ValueError("start_index must leave at least one next-period return")
    if initial_weights is not None:
        _validate_weight_vector(initial_weights)


def _validate_weight_vector(weights: list[float]) -> None:
    """Validate a strategy target-weight vector."""
    if len(weights) != 4:
        raise ValueError("weights must contain four values")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
