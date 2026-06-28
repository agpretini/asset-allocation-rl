"""Tests for rule-based baseline strategies."""

import pandas as pd
import pytest

from backtesting.engine import run_weight_strategy_backtest
from backtesting.strategies import (
    equal_weight_strategy,
    momentum_strategy,
    volatility_weighted_strategy,
)


@pytest.fixture
def strategy_dataset() -> pd.DataFrame:
    """Create deterministic monthly data for rule-based baselines."""
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    data: dict[str, list[float]] = {}
    price_paths = {
        "arg": [100 + 1 * month for month in range(15)],
        "ced": [100 + 3 * month for month in range(15)],
        "sp500": [100 + 2 * month for month in range(15)],
        "gold": [100 - 1 * month for month in range(15)],
    }

    for prefix, prices in price_paths.items():
        returns = [0.0]
        returns.extend(
            prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))
        )
        data[f"price_{prefix}"] = prices
        data[f"r_{prefix}"] = returns

    return pd.DataFrame(data, index=index)


def test_equal_weight_strategy_returns_25_percent_weights(
    strategy_dataset: pd.DataFrame,
) -> None:
    weights = equal_weight_strategy(strategy_dataset)

    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_momentum_strategy_allocates_to_positive_momentum(
    strategy_dataset: pd.DataFrame,
) -> None:
    weights = momentum_strategy(strategy_dataset.iloc[:13])

    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[2] > weights[0] > weights[3]
    assert weights[3] == pytest.approx(0.0)


def test_volatility_strategy_prefers_lower_volatility(
    strategy_dataset: pd.DataFrame,
) -> None:
    adjusted = strategy_dataset.copy()
    adjusted["r_arg"] = [0.0, 0.1, -0.1] * 5
    adjusted["r_gold"] = [0.0, 0.01, -0.01] * 5

    weights = volatility_weighted_strategy(adjusted.iloc[:13])

    assert sum(weights) == pytest.approx(1.0)
    assert weights[3] > weights[0]


def test_weight_strategy_backtest_uses_next_period_return(
    strategy_dataset: pd.DataFrame,
) -> None:
    result = run_weight_strategy_backtest(
        dataset=strategy_dataset,
        strategy=equal_weight_strategy,
        start_index=12,
    )
    next_returns = strategy_dataset.iloc[13][["r_arg", "r_ced", "r_sp500", "r_gold"]]

    assert len(result) == 2
    assert result["portfolio_return"].iloc[0] == pytest.approx(next_returns.mean())
    assert result["turnover"].iloc[0] == pytest.approx(0.0)
