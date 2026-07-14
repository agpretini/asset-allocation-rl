"""Compare raw and normalized-binned states with walk-forward evaluation."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Generate feature-mode comparison artifacts."""
    from agents.training import QLearningTrainingConfig
    from backtesting.feature_modes import (
        run_feature_mode_decision_comparison,
        run_feature_mode_walk_forward_comparison,
    )
    from backtesting.splitter import generate_walk_forward_splits
    from data.loaders import load_config

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    splits = generate_walk_forward_splits(dataset.index)
    report, paths = run_feature_mode_walk_forward_comparison(
        dataset=dataset,
        splits=splits,
        feature_modes=["raw", "normalized_binned"],
        training_config=QLearningTrainingConfig(episodes=25, seed=42),
    )
    decisions = run_feature_mode_decision_comparison(
        dataset=dataset,
        splits=splits,
        feature_modes=["raw", "normalized_binned"],
        training_config=QLearningTrainingConfig(episodes=25, seed=42),
    )

    output_dir = Path("data/processed/feature_modes")
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "feature_mode_report.csv", index=False)
    paths.to_csv(output_dir / "feature_mode_paths.csv")
    decisions.to_csv(output_dir / "feature_mode_decisions.csv", index=False)
    print(output_dir / "feature_mode_report.csv")


if __name__ == "__main__":
    main()
