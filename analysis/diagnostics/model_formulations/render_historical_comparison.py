"""Compare A/B/C and baselines on the completed diagnostic holdout window."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIAGNOSTIC_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    """Write a cumulative-return SVG and metrics table without retraining."""
    from run import FORMULATIONS, _evaluate_formulation

    from agents.q_learning import QLearningAgent
    from backtesting.benchmarks import run_benchmarks
    from backtesting.engine import run_weight_strategy_backtest
    from backtesting.strategies import (
        equal_weight_strategy,
        momentum_strategy,
        volatility_weighted_strategy,
    )
    from data.loaders import load_config
    from evaluation.artifacts import (
        build_normalized_value_paths,
        write_cumulative_return_svg,
    )
    from evaluation.report import build_performance_report

    results_dir = DIAGNOSTIC_DIR / "results"
    configuration = pd.read_json(results_dir / "configuration.json", typ="series")
    train = _load_window(
        configuration["train_period"], load_config()["processed_dataset_path"]
    )
    evaluation = _load_window(
        configuration["evaluation_period"], load_config()["processed_dataset_path"]
    )
    formulations = {formulation.name: formulation for formulation in FORMULATIONS}
    model_results = {}
    for name in ("B_market_state_cost_at_decision", "C_market_state_no_cost"):
        agent = QLearningAgent.load(results_dir / "models" / f"{name}.json")
        model_results[name] = _evaluate_formulation(
            evaluation,
            train,
            _action_space(),
            formulations[name],
            agent,
            str(configuration["feature_mode"]),
        )

    model_dates = model_results["B_market_state_cost_at_decision"].index
    baseline_results = run_benchmarks(
        evaluation.loc[model_dates, ["r_arg", "r_ced", "r_sp500", "r_gold"]]
    )
    full_history = pd.concat([train, evaluation])
    for name, strategy in {
        "equal_weight": equal_weight_strategy,
        "momentum": momentum_strategy,
        "inverse_volatility": volatility_weighted_strategy,
    }.items():
        continuous = run_weight_strategy_backtest(full_history, strategy)
        baseline_results[name] = _rebase(continuous.loc[model_dates])

    selected_comparison_dir = (
        PROJECT_ROOT / "analysis/runs/selected_synthetic_trial_371/comparison"
    )
    a_paths = pd.read_csv(
        selected_comparison_dir / "test_split_1_cumulative_paths.csv",
        index_col="date",
        parse_dates=True,
    )
    a_metrics = pd.read_csv(
        selected_comparison_dir / "test_split_1_metrics.csv", index_col=0
    ).loc["q_learning_synthetic"]
    results = {
        "A_baseline": _a_path_result(a_paths.loc[model_dates, "q_learning_synthetic"]),
        **model_results,
        **baseline_results,
    }
    metrics = build_performance_report(results)
    metrics.loc["A_baseline"] = a_metrics[
        [
            "cagr",
            "volatility",
            "sharpe",
            "sortino",
            "maximum_drawdown",
            "average_turnover",
        ]
    ]
    metrics.insert(
        0,
        "cumulative_return",
        [
            float(result["portfolio_value"].iloc[-1] - 1.0)
            for result in results.values()
        ],
    )
    metrics.loc["A_baseline", "cumulative_return"] = float(
        a_metrics["final_portfolio_value"] - 1.0
    )
    metrics.index.name = "strategy"
    paths = build_normalized_value_paths(results)
    metrics.to_csv(results_dir / "historical_model_vs_baselines_metrics.csv")
    paths.to_csv(results_dir / "historical_model_vs_baselines_paths.csv")
    write_cumulative_return_svg(
        paths,
        results_dir / "historical_model_vs_baselines_cumulative_returns.svg",
        "Historical holdout: A/B/C versus baselines",
    )
    print(results_dir / "historical_model_vs_baselines_cumulative_returns.svg")


def _action_space() -> list[list[float]]:
    from env.action_space import generate_action_space

    return generate_action_space()


def _load_window(period: dict[str, str], dataset_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_parquet(dataset_path)
    return dataset.loc[pd.Timestamp(period["start"]) : pd.Timestamp(period["end"])]


def _rebase(result: pd.DataFrame) -> pd.DataFrame:
    """Rebase a continuous rule strategy to this holdout's first return."""
    rebased = result.copy()
    returns = rebased["net_portfolio_return"]
    rebased["portfolio_value"] = (1.0 + returns).cumprod()
    rebased["drawdown"] = (
        rebased["portfolio_value"] / rebased["portfolio_value"].cummax() - 1.0
    )
    return rebased


def _a_path_result(path: pd.Series) -> pd.DataFrame:
    """Adapt the retained normalized A path for comparison chart generation."""
    values = path.astype(float)
    returns = values.pct_change().fillna(0.0)
    return pd.DataFrame(
        {
            "net_portfolio_return": returns,
            "turnover": 0.0,
            "portfolio_value": values,
            "drawdown": values / values.cummax() - 1.0,
        },
        index=values.index,
    )


if __name__ == "__main__":
    main()
