"""Tests for forward-looking allocation prediction."""

import pandas as pd

from agents.q_learning import QLearningAgent
from agents.training import QLearningTrainingConfig
from prediction.next_allocation import (
    predict_allocations,
    predict_next_allocation,
    predict_next_allocations,
    select_modal_model_b_action,
    train_allocation_model,
)
from simulation.market_simulator import SimulationConfig


def test_modal_model_b_action_uses_lowest_id_for_modal_tie() -> None:
    action_id, share = select_modal_model_b_action(
        q_values=[1.0, 1.0],
        action_space=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        transaction_cost=0.014,
    )

    assert action_id == 0
    assert share == 0.5


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
    assert prediction.target_period == "2021-09"
    assert 0 <= prediction.action_id < agent.n_actions
    assert sum(weights) == 1.0
    assert prediction.state_source in {
        "exact",
        "nearest_learned_state",
        "unseen_unlearned_state",
    }
    assert not history.empty


def test_predict_next_allocations_returns_myopic_and_modal_model_b_policy() -> None:
    dataset = _make_prediction_dataset()

    predictions, _, _ = predict_next_allocations(
        dataset=dataset,
        training_config=QLearningTrainingConfig(episodes=2, seed=9),
        current_weights=[0.25, 0.25, 0.25, 0.25],
    )

    methods = {prediction.method for prediction in predictions}
    assert methods == {
        "myopic_greedy_latest_return",
        "model_b_modal_market_policy",
    }


def test_predict_next_allocations_accepts_normalized_binned_features() -> None:
    dataset = _make_prediction_dataset()

    predictions, _, _ = predict_next_allocations(
        dataset=dataset,
        training_config=QLearningTrainingConfig(episodes=1, seed=9),
        current_weights=[0.25, 0.25, 0.25, 0.25],
        feature_mode="normalized_binned",
    )

    assert len(predictions) == 2


def test_inference_does_not_initialize_unseen_q_states() -> None:
    agent = QLearningAgent(n_actions=286)
    predictions = predict_allocations(_make_prediction_dataset(), agent)

    assert agent.q_table == {}
    assert predictions[-1].state_source == "unseen_unlearned_state"


def test_separate_training_supports_synthetic_model_b() -> None:
    result = train_allocation_model(
        _make_prediction_dataset(),
        QLearningTrainingConfig(episodes=2, training_mode="synthetic", seed=9),
        simulation_config=SimulationConfig(episode_length=16, seed=9),
    )

    assert len(result.history) == 2
    assert set(result.history["training_mode"]) == {"synthetic"}
    assert result.agent.q_table
    assert all(len(key.split("|")) == 4 for key in result.agent.q_table)


def _make_prediction_dataset() -> pd.DataFrame:
    """Create deterministic monthly data for prediction tests."""
    index = pd.date_range("2020-01-31", periods=20, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(20)],
        "vix": [20.0 + month for month in range(20)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        prices = [100.0 + 3.0 * month for month in range(20)]
        returns = [0.0]
        returns.extend(
            prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))
        )
        data[f"price_{prefix}"] = prices
        data[f"r_{prefix}"] = returns
    return pd.DataFrame(data, index=index)
