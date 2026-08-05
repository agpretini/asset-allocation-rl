"""Run a reproducible Q-learning hyperparameter search."""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Run a small grid search and save optimization artifacts."""
    from backtesting.splitter import generate_walk_forward_splits
    from data.loaders import load_config
    from experiments.optimizer import run_q_learning_grid_search
    from experiments.tracking import ExperimentTracker

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    splits = generate_walk_forward_splits(dataset.index)
    results, best_config = run_q_learning_grid_search(
        dataset=dataset,
        splits=splits,
        episodes=[15, 25],
        learning_rates=[0.05, 0.10, 0.30],
        learning_rate_schedules=["constant", "visit_decay"],
        epsilons=[0.50],
        epsilon_decays=[0.999954],
        state_precisions=[2],
        feature_modes=["compact_regime"],
        seed=42,
    )

    output_dir = Path("data/processed/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "hyperparameter_results.csv", index=False)
    (output_dir / "best_hyperparameters.yaml").write_text(
        yaml.safe_dump(best_config.__dict__, sort_keys=True),
        encoding="utf-8",
    )

    experiment_dir = ExperimentTracker().create_experiment(prefix="hpo")
    ExperimentTracker().save(
        experiment_dir=experiment_dir,
        config={
            "objective": "maximize mean walk-forward score",
            "dataset": config["processed_dataset_path"],
            "feature_mode": "compact_regime",
            "search_space": {
                "episodes": [15, 25],
                "learning_rates": [0.05, 0.10, 0.30],
                "learning_rate_schedules": ["constant", "visit_decay"],
                "visit_learning_rate_decay": 0.01,
                "epsilons": [0.50],
                "epsilon_decays": [0.999954],
                "state_precisions": [2],
            },
        },
        metrics=results.iloc[0].to_dict(),
        actions=pd.DataFrame(),
        portfolio_values=pd.DataFrame(),
        notes="Hyperparameter optimization for tabular Q-learning.\n",
    )
    print(output_dir / "hyperparameter_results.csv")


if __name__ == "__main__":
    main()
