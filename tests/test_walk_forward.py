"""Tests for walk-forward split and evaluation utilities."""

import pandas as pd

from agents.training import QLearningTrainingConfig
from backtesting.feature_modes import run_feature_mode_walk_forward_comparison
from backtesting.splitter import generate_walk_forward_splits
from backtesting.walk_forward import (
    run_walk_forward_q_learning,
    run_walk_forward_train_validation_test,
)
from simulation.market_simulator import SimulationConfig


def test_generate_walk_forward_splits_rolls_forward() -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")

    splits = generate_walk_forward_splits(
        dates=dates,
        train_months=12,
        validation_months=6,
        test_months=6,
        step_months=6,
    )

    assert len(splits) == 3
    assert splits[0].train_start == dates[0]
    assert splits[1].train_start == dates[6]
    assert splits[0].test_end == dates[23]


def test_walk_forward_q_learning_returns_report_and_paths() -> None:
    dataset = _make_walk_forward_dataset()
    splits = generate_walk_forward_splits(
        dates=dataset.index,
        train_months=24,
        validation_months=6,
        test_months=18,
        step_months=6,
    )

    report, paths = run_walk_forward_q_learning(
        dataset=dataset,
        splits=splits,
        training_config=QLearningTrainingConfig(episodes=2, seed=7),
    )

    assert not report.empty
    assert not paths.empty
    assert {"split_id", "cagr", "maximum_drawdown", "q_table_states"}.issubset(
        report.columns,
    )


def test_feature_mode_comparison_runs_both_modes() -> None:
    dataset = _make_walk_forward_dataset()
    splits = generate_walk_forward_splits(
        dates=dataset.index,
        train_months=24,
        validation_months=6,
        test_months=18,
        step_months=6,
    )

    report, paths = run_feature_mode_walk_forward_comparison(
        dataset=dataset,
        splits=splits,
        feature_modes=["raw", "normalized_binned"],
        training_config=QLearningTrainingConfig(episodes=1, seed=7),
    )

    assert set(report["feature_mode"]) == {"raw", "normalized_binned"}
    assert not paths.empty
    assert any(column.startswith("raw_") for column in paths.columns)
    assert any(column.startswith("normalized_binned_") for column in paths.columns)


def test_synthetic_training_keeps_walk_forward_test_historical() -> None:
    dataset = _make_walk_forward_dataset()
    splits = generate_walk_forward_splits(
        dates=dataset.index,
        train_months=24,
        validation_months=6,
        test_months=18,
        step_months=18,
    )
    simulation_config = SimulationConfig(
        episode_count=2,
        episode_length=18,
        seed=13,
    )

    report, paths = run_walk_forward_q_learning(
        dataset=dataset,
        splits=splits,
        training_config=QLearningTrainingConfig(
            episodes=2,
            training_mode="synthetic",
            seed=13,
        ),
        simulation_config=simulation_config,
    )

    assert set(report["training_mode"]) == {"synthetic"}
    assert paths.index.isin(dataset.index).all()
    assert paths.index.min() >= splits[0].test_start
    assert paths.index.max() <= splits[0].test_end


def test_train_validation_test_protocol_has_both_historical_phases() -> None:
    dataset = _make_walk_forward_dataset()
    splits = generate_walk_forward_splits(
        dates=dataset.index,
        train_months=24,
        validation_months=6,
        test_months=18,
        step_months=18,
    )

    report, paths = run_walk_forward_train_validation_test(
        dataset=dataset,
        splits=splits,
        training_config=QLearningTrainingConfig(episodes=1, seed=7),
    )

    assert set(report["phase"]) == {"validation", "test"}
    assert set(paths) == {"validation_split_0", "test_split_0"}
    assert paths["validation_split_0"].index.max() <= splits[0].validation_end
    assert paths["test_split_0"].index.min() >= splits[0].test_start


def _make_walk_forward_dataset() -> pd.DataFrame:
    """Create deterministic monthly data for walk-forward tests."""
    index = pd.date_range("2018-01-31", periods=60, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(60)],
        "vix": [20.0 + 0.1 * month for month in range(60)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        prices = [100.0 + 2.0 * month for month in range(60)]
        returns = [0.0]
        returns.extend(
            prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))
        )
        data[f"price_{prefix}"] = prices
        data[f"r_{prefix}"] = returns
    return pd.DataFrame(data, index=index)
