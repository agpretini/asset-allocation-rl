"""Tests for forward-looking allocation prediction."""

import pandas as pd

from agents.training import QLearningTrainingConfig
from prediction.next_allocation import predict_next_allocation, predict_next_allocations


def test_predict_next_allocation_returns_valid_weights() -> None:
    dataset = _make_prediction_dataset()

    prediction, agent, history = predict_next_allocation(
        dataset=dataset,
        training_config=QLearningTrainingConfig(episodes=2, seed=9),
        current_weights=[0.25, 0.25, 0.25, 0.25],
    )

    weights = [
        prediction.w_arg,
        prediction.w_ced,
        prediction.w_sp500,
        prediction.w_gold,
    ]
    assert prediction.target_period == "2021-04"
    assert 0 <= prediction.action_id < agent.n_actions
    assert sum(weights) == 1.0
    assert prediction.state_source in {
        "exact",
        "nearest_learned_state",
        "unseen_unlearned_state",
    }
    assert not history.empty


def test_predict_next_allocations_returns_myopic_and_infinite_horizon() -> None:
    dataset = _make_prediction_dataset()

    predictions, _, _ = predict_next_allocations(
        dataset=dataset,
        training_config=QLearningTrainingConfig(episodes=2, seed=9),
        current_weights=[0.25, 0.25, 0.25, 0.25],
    )

    methods = {prediction.method for prediction in predictions}
    assert methods == {
        "myopic_greedy_latest_return",
        "infinite_horizon_q_learning",
    }


def _make_prediction_dataset() -> pd.DataFrame:
    """Create deterministic monthly data for prediction tests."""
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(15)],
        "vix": [20.0 + month for month in range(15)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        prices = [100.0 + 3.0 * month for month in range(15)]
        returns = [0.0]
        returns.extend(
            prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))
        )
        data[f"price_{prefix}"] = prices
        data[f"r_{prefix}"] = returns
    return pd.DataFrame(data, index=index)
