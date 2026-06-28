"""Tests for walk-forward split and evaluation utilities."""

import pandas as pd

from agents.training import QLearningTrainingConfig
from backtesting.splitter import generate_walk_forward_splits
from backtesting.walk_forward import run_walk_forward_q_learning


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
        test_months=14,
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
