"""Run random and rule-based baseline backtests."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Generate baseline paths and a performance report."""
    from agents.random_agent import RandomAgent
    from backtesting.agents import run_agent_backtest
    from backtesting.benchmarks import run_benchmarks
    from backtesting.engine import run_weight_strategy_backtest
    from backtesting.strategies import (
        equal_weight_strategy,
        momentum_strategy,
        volatility_weighted_strategy,
    )
    from data.loaders import load_config
    from env.portfolio_env import PortfolioEnv
    from evaluation.report import build_performance_report
    from features.state_builder import StateBuilder

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    state_builder = StateBuilder(dataset)
    env = PortfolioEnv(dataset=dataset, state_builder=state_builder)
    random_agent = RandomAgent(n_actions=len(env.action_space), seed=42)

    baseline_results = {
        "random_agent": run_agent_backtest(env, random_agent),
        "equal_weight": run_weight_strategy_backtest(dataset, equal_weight_strategy),
        "momentum": run_weight_strategy_backtest(dataset, momentum_strategy),
        "volatility_weighted": run_weight_strategy_backtest(
            dataset,
            volatility_weighted_strategy,
        ),
    }
    first_backtest_date = max(
        result.index.min() for result in baseline_results.values()
    )
    baseline_results = {
        key: result.loc[first_backtest_date:]
        for key, result in baseline_results.items()
    }
    return_columns = [asset["return_column"] for asset in config["assets"].values()]
    benchmark_returns = dataset.loc[first_backtest_date:, return_columns]
    benchmark_results = run_benchmarks(benchmark_returns)
    baseline_results.update(benchmark_results)

    output_dir = Path("data/processed/baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_performance_report(baseline_results)
    report.to_csv(output_dir / "baseline_report.csv")

    paths = _build_paths_table(baseline_results)
    paths.to_csv(output_dir / "baseline_paths.csv")
    print(output_dir / "baseline_report.csv")


def _build_paths_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one combined portfolio-value table from backtest results."""
    path_columns = []
    for key, result in results.items():
        path_columns.append(result["portfolio_value"].rename(key))
    return pd.concat(path_columns, axis=1, sort=False)


if __name__ == "__main__":
    main()
