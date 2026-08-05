"""Run walk-forward evaluation for the first RL agent."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Generate walk-forward Q-learning evaluation artifacts."""
    from agents.training import QLearningTrainingConfig
    from backtesting.splitter import generate_walk_forward_splits
    from backtesting.walk_forward import run_walk_forward_q_learning
    from data.loaders import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-mode",
        choices=["historical", "synthetic"],
        default="historical",
    )
    args = parser.parse_args()

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    splits = generate_walk_forward_splits(dataset.index)
    report, paths = run_walk_forward_q_learning(
        dataset=dataset,
        splits=splits,
        training_config=QLearningTrainingConfig(
            episodes=25,
            training_mode=args.training_mode,
            seed=42,
        ),
    )

    output_dir = Path("data/processed/walk_forward")
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "walk_forward_report.csv", index=False)
    paths.to_csv(output_dir / "walk_forward_paths.csv")
    print(output_dir / "walk_forward_report.csv")


if __name__ == "__main__":
    main()
