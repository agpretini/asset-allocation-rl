"""Tests for filesystem experiment tracking."""

import json

import pandas as pd
import yaml

from experiments.tracking import ExperimentTracker, save_backtest_experiment


def test_experiment_tracker_writes_standard_artifacts(tmp_path) -> None:
    tracker = ExperimentTracker(root_dir=tmp_path)
    result = pd.DataFrame(
        {
            "action_id": [1, 2],
            "weights": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "reward": [0.01, 0.02],
            "portfolio_value": [1.01, 1.03],
        },
        index=pd.date_range("2022-01-31", periods=2, freq="ME"),
    )

    experiment_dir = save_backtest_experiment(
        tracker=tracker,
        config={"agent_type": "q_learning", "seed": 42},
        metrics={"cagr": 0.12},
        result=result,
        notes="Test experiment.\n",
    )

    assert (experiment_dir / "config.yaml").exists()
    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "actions.csv").exists()
    assert (experiment_dir / "portfolio_values.csv").exists()
    assert (experiment_dir / "notes.md").exists()

    config = yaml.safe_load((experiment_dir / "config.yaml").read_text())
    metrics = json.loads((experiment_dir / "metrics.json").read_text())
    assert config["agent_type"] == "q_learning"
    assert metrics["cagr"] == 0.12


def test_experiment_tracker_increments_ids(tmp_path) -> None:
    tracker = ExperimentTracker(root_dir=tmp_path)

    first = tracker.create_experiment(prefix="trial")
    second = tracker.create_experiment(prefix="trial")

    assert first.name == "trial_001"
    assert second.name == "trial_002"
