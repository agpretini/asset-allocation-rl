"""Save the current Q-learning evaluation as a tracked experiment."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Create an experiment folder from current processed Q-learning outputs."""
    from data.loaders import load_config
    from experiments.tracking import ExperimentTracker, save_backtest_experiment

    config = load_config()
    result = pd.read_csv(
        "data/processed/q_learning/q_learning_paths.csv",
        index_col="date",
        parse_dates=True,
    )
    metrics = pd.read_csv(
        "data/processed/q_learning/q_learning_report.csv",
        index_col=0,
    )
    experiment_dir = save_backtest_experiment(
        tracker=ExperimentTracker(),
        config={
            "agent_type": "q_learning",
            "dataset": config["processed_dataset_path"],
            "reward": "portfolio_return - 0.014 * turnover",
            "feature_set": "market_momentum_volatility_macro_plus_weights",
        },
        metrics=metrics.loc["q_learning"].to_dict(),
        result=result,
        notes="Tracked Q-learning evaluation generated from processed outputs.\n",
        prefix="q_learning",
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()
