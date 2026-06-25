"""Performance and risk metrics for monthly portfolio returns."""

from math import sqrt

import pandas as pd


def calculate_performance_metrics(
    monthly_returns: pd.Series,
    portfolio_values: pd.Series,
    turnover: pd.Series | None = None,
    initial_value: float = 1.0,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate the required benchmark evaluation metrics."""
    if monthly_returns.empty:
        raise ValueError("monthly_returns cannot be empty")
    if portfolio_values.empty:
        raise ValueError("portfolio_values cannot be empty")

    return {
        "cagr": calculate_cagr(
            initial_value=initial_value,
            final_value=float(portfolio_values.iloc[-1]),
            periods=len(monthly_returns),
        ),
        "volatility": calculate_annualized_volatility(monthly_returns),
        "sharpe": calculate_sharpe(monthly_returns, risk_free_rate=risk_free_rate),
        "sortino": calculate_sortino(monthly_returns),
        "maximum_drawdown": calculate_maximum_drawdown(portfolio_values),
        "average_turnover": float(turnover.mean()) if turnover is not None else 0.0,
    }


def calculate_cagr(
    initial_value: float,
    final_value: float,
    periods: int,
    periods_per_year: int = 12,
) -> float:
    """Calculate compound annual growth rate."""
    if initial_value <= 0.0:
        raise ValueError("initial_value must be positive")
    if periods <= 0:
        raise ValueError("periods must be positive")

    years = periods / periods_per_year
    return (final_value / initial_value) ** (1.0 / years) - 1.0


def calculate_annualized_volatility(
    monthly_returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized volatility from monthly returns."""
    return float(monthly_returns.std(ddof=0) * sqrt(periods_per_year))


def calculate_sharpe(
    monthly_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized Sharpe ratio."""
    volatility = calculate_annualized_volatility(monthly_returns, periods_per_year)
    if volatility == 0.0:
        return 0.0
    annualized_return = float(monthly_returns.mean() * periods_per_year)
    return (annualized_return - risk_free_rate) / volatility


def calculate_sortino(
    monthly_returns: pd.Series,
    periods_per_year: int = 12,
) -> float:
    """Calculate annualized Sortino ratio using downside volatility."""
    downside_returns = monthly_returns[monthly_returns < 0.0]
    downside_volatility = float(downside_returns.std(ddof=0) * sqrt(periods_per_year))
    if downside_volatility == 0.0:
        return 0.0
    annualized_return = float(monthly_returns.mean() * periods_per_year)
    return annualized_return / downside_volatility


def calculate_maximum_drawdown(portfolio_values: pd.Series) -> float:
    """Calculate the largest peak-to-trough drawdown."""
    drawdowns = portfolio_values / portfolio_values.cummax() - 1.0
    return float(drawdowns.min())
