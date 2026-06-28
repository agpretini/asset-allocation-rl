"""Tests for hyperparameter optimization."""

import pandas as pd

from backtesting.splitter import generate_walk_forward_splits
from experiments.optimizer import run_q_learning_grid_search


def test_q_learning_grid_search_returns_ranked_results() -> None:
    dataset = _make_optimizer_dataset()
    splits = generate_walk_forward_splits(
        dates=dataset.index,
        train_months=24,
        validation_months=6,
        test_months=14,
        step_months=6,
    )

    results, best_config = run_q_learning_grid_search(
        dataset=dataset,
        splits=splits,
        episodes=[1],
        learning_rates=[0.05, 0.10],
        epsilons=[0.2],
        epsilon_decays=[0.9],
        state_precisions=[2],
        seed=5,
    )

    assert len(results) == 2
    assert results.iloc[0]["score"] >= results.iloc[1]["score"]
    assert best_config.learning_rate in {0.05, 0.10}


def _make_optimizer_dataset() -> pd.DataFrame:
    """Create deterministic monthly data for optimizer tests."""
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
