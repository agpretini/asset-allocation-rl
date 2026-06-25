"""Tests for fixed benchmark backtests."""

import pandas as pd
import pytest

from backtesting.benchmarks import Benchmark, run_benchmarks
from backtesting.engine import run_fixed_weight_backtest


def test_fixed_weight_backtest_applies_weighted_returns() -> None:
    returns = pd.DataFrame(
        {
            "r_sp500": [0.10, -0.05],
            "r_gold": [0.02, 0.04],
        },
        index=pd.date_range("2021-01-31", periods=2, freq="ME"),
    )

    result = run_fixed_weight_backtest(
        returns=returns,
        weights={"r_sp500": 0.5, "r_gold": 0.5},
    )

    assert list(result["portfolio_return"]) == pytest.approx([0.06, -0.005])
    assert result["portfolio_value"].iloc[-1] == pytest.approx(1.06 * 0.995)
    assert result["turnover"].sum() == 0.0


def test_run_benchmarks_returns_one_result_per_benchmark() -> None:
    returns = pd.DataFrame(
        {
            "r_sp500": [0.01],
            "r_gold": [0.03],
        },
        index=pd.date_range("2021-01-31", periods=1, freq="ME"),
    )
    benchmarks = [
        Benchmark(key="sp500", name="100% S&P500", weights={"r_sp500": 1.0}),
        Benchmark(key="gold", name="100% Gold", weights={"r_gold": 1.0}),
    ]

    results = run_benchmarks(returns=returns, benchmarks=benchmarks)

    assert set(results) == {"sp500", "gold"}
    assert results["sp500"]["portfolio_return"].iloc[0] == pytest.approx(0.01)
