"""Train on all available data and predict the next monthly allocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from agents.training import QLearningTrainingConfig


def main() -> None:
    """Generate the next allocation prediction artifacts."""
    from data.loaders import load_config
    from features.transforms import FEATURE_MODES
    from prediction.next_allocation import predict_next_allocations

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="compact_regime",
        help="State feature representation to use for Q-learning.",
    )
    args = parser.parse_args()

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    training_config = _load_best_training_config()
    current_weights = [0.25, 0.25, 0.25, 0.25]

    predictions, agent, training_history = predict_next_allocations(
        dataset=dataset,
        training_config=training_config,
        current_weights=current_weights,
        feature_mode=args.feature_mode,
    )
    infinite_horizon_prediction = [
        prediction
        for prediction in predictions
        if prediction.method == "infinite_horizon_q_learning"
    ][0]

    output_dir = _prediction_output_dir(
        target_period=infinite_horizon_prediction.target_period,
        feature_mode=args.feature_mode,
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
        "feature_mode": args.feature_mode,
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


def _prediction_output_dir(target_period: str, feature_mode: str) -> Path:
    """Return the output directory for one prediction run."""
    base_dir = Path("analysis/forecasts/standard") / target_period
    if feature_mode == "compact_regime":
        return base_dir
    return base_dir / feature_mode


def _load_best_training_config() -> QLearningTrainingConfig:
    """Load the best known Q-learning hyperparameters."""
    from agents.training import QLearningTrainingConfig

    payload = yaml.safe_load(
        Path(
            "analysis/hyperparameter_search/current/best_hyperparameters.yaml"
        ).read_text(
            encoding="utf-8",
        ),
    )
    return QLearningTrainingConfig(
        episodes=int(payload["episodes"]),
        training_mode=payload.get("training_mode", "historical"),
        learning_rate=float(payload["learning_rate"]),
        learning_rate_schedule=payload.get("learning_rate_schedule", "constant"),
        visit_learning_rate_decay=float(payload.get("visit_learning_rate_decay", 0.01)),
        discount_factor=float(payload["discount_factor"]),
        epsilon=float(payload["epsilon"]),
        min_epsilon=float(payload["min_epsilon"]),
        epsilon_decay=float(payload["epsilon_decay"]),
        state_precision=int(payload["state_precision"]),
        seed=int(payload["seed"]),
    )


if __name__ == "__main__":
    main()
