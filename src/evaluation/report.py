"""Generate tabular benchmark performance reports."""

import pandas as pd

from evaluation.metrics import calculate_performance_metrics


def build_benchmark_report(
    benchmark_results: dict[str, pd.DataFrame],
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Build one metrics row per benchmark backtest result."""
    rows: dict[str, dict[str, float]] = {}
    for benchmark_key, result in benchmark_results.items():
        rows[benchmark_key] = calculate_performance_metrics(
            monthly_returns=result["portfolio_return"],
            portfolio_values=result["portfolio_value"],
            turnover=result["turnover"],
            initial_value=initial_value,
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def build_performance_report(
    results: dict[str, pd.DataFrame],
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Build a metrics report for net-return strategy backtest results."""
    rows: dict[str, dict[str, float]] = {}
    for key, result in results.items():
        return_column = "reward" if "reward" in result.columns else "portfolio_return"
        rows[key] = calculate_performance_metrics(
            monthly_returns=result[return_column],
            portfolio_values=result["portfolio_value"],
            turnover=result["turnover"],
            initial_value=initial_value,
        )
    return pd.DataFrame.from_dict(rows, orient="index")
