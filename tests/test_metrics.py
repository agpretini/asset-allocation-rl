"""Tests for benchmark evaluation metrics."""

import pandas as pd
import pytest

from evaluation.metrics import (
    calculate_annualized_volatility,
    calculate_cagr,
    calculate_maximum_drawdown,
    calculate_performance_metrics,
)


def test_cagr_uses_compounded_growth_over_years() -> None:
    assert calculate_cagr(initial_value=1.0, final_value=1.21, periods=12) == (
        pytest.approx(0.21)
    )


def test_annualized_volatility_scales_monthly_volatility() -> None:
    monthly_returns = pd.Series([0.01, -0.01, 0.01, -0.01])

    assert calculate_annualized_volatility(monthly_returns) == pytest.approx(
        monthly_returns.std(ddof=0) * 12**0.5,
    )


def test_maximum_drawdown_uses_running_peak() -> None:
    portfolio_values = pd.Series([1.0, 1.2, 0.9, 1.1])

    assert calculate_maximum_drawdown(portfolio_values) == pytest.approx(-0.25)


def test_performance_metrics_include_required_fields() -> None:
    monthly_returns = pd.Series([0.02, -0.01, 0.03])
    portfolio_values = (1.0 + monthly_returns).cumprod()
    turnover = pd.Series([0.0, 0.1, 0.0])

    metrics = calculate_performance_metrics(
        monthly_returns=monthly_returns,
        portfolio_values=portfolio_values,
        turnover=turnover,
    )

    assert set(metrics) == {
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "maximum_drawdown",
        "average_turnover",
    }
    assert metrics["average_turnover"] == pytest.approx(turnover.mean())
