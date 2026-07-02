"""Train on all available data and predict the next monthly allocation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

if TYPE_CHECKING:
    from agents.training import QLearningTrainingConfig


def main() -> None:
    """Generate the next allocation prediction artifacts."""
    from data.loaders import load_config
    from prediction.next_allocation import predict_next_allocations

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    training_config = _load_best_training_config()
    current_weights = [0.25, 0.25, 0.25, 0.25]

    predictions, agent, training_history = predict_next_allocations(
        dataset=dataset,
        training_config=training_config,
        current_weights=current_weights,
    )
    infinite_horizon_prediction = [
        prediction
        for prediction in predictions
        if prediction.method == "infinite_horizon_q_learning"
    ][0]

    output_dir = (
        Path("data/processed/predictions") / infinite_horizon_prediction.target_period
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([prediction.to_dict() for prediction in predictions]).to_csv(
        output_dir / "allocation_predictions.csv",
        index=False,
    )
    pd.DataFrame([infinite_horizon_prediction.to_dict()]).to_csv(
        output_dir / "allocation_prediction.csv",
        index=False,
    )
    agent.save(output_dir / "trained_model.json")
    training_history.to_csv(output_dir / "training_history.csv", index=False)
    metadata = {
        "dataset_path": config["processed_dataset_path"],
        "dataset_start": pd.Timestamp(dataset.index.min()).date().isoformat(),
        "dataset_end": pd.Timestamp(dataset.index.max()).date().isoformat(),
        "current_weights_assumption": current_weights,
        "training_config": training_config.__dict__,
        "interpretation_note": (
            "This is a decision-support signal for the next monthly period, "
            "not an automatic trading instruction."
        ),
    }
    (output_dir / "prediction_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "allocation_prediction.csv")


def _load_best_training_config() -> QLearningTrainingConfig:
    """Load the best known Q-learning hyperparameters."""
    from agents.training import QLearningTrainingConfig

    payload = yaml.safe_load(
        Path("data/processed/optimization/best_hyperparameters.yaml").read_text(
            encoding="utf-8",
        ),
    )
    return QLearningTrainingConfig(
        episodes=int(payload["episodes"]),
        learning_rate=float(payload["learning_rate"]),
        discount_factor=float(payload["discount_factor"]),
        epsilon=float(payload["epsilon"]),
        min_epsilon=float(payload["min_epsilon"]),
        epsilon_decay=float(payload["epsilon_decay"]),
        state_precision=int(payload["state_precision"]),
        seed=int(payload["seed"]),
    )


if __name__ == "__main__":
    main()
