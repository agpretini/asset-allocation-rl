"""Tests for the tabular Q-learning agent."""

import pandas as pd
import pytest

from agents.q_learning import QLearningAgent
from agents.training import QLearningTrainingConfig, train_q_learning_agent
from env.portfolio_env import PortfolioEnv
from features.state_builder import StateBuilder


def test_q_learning_update_changes_q_value() -> None:
    agent = QLearningAgent(n_actions=2, learning_rate=0.5, epsilon=0.0, seed=1)
    state = [0.0, 1.0]
    next_state = [1.0, 0.0]

    agent.update(
        state=state,
        action_id=1,
        reward=0.10,
        next_state=next_state,
        done=True,
    )

    key = agent._state_key(state)
    assert agent.q_table[key][1] == pytest.approx(0.05)


def test_visit_decay_learning_rate_starts_at_point_three_and_decreases() -> None:
    agent = QLearningAgent(
        n_actions=1,
        learning_rate=0.3,
        learning_rate_schedule="visit_decay",
        visit_learning_rate_decay=0.01,
        epsilon=0.0,
    )
    state = [0.0]

    assert agent.learning_rate_for(state, action_id=0) == pytest.approx(0.3)
    agent.update(state, action_id=0, reward=1.0, next_state=state, done=True)
    assert agent.learning_rate_for(state, action_id=0) == pytest.approx(0.3 / 1.01)


def test_q_learning_model_round_trips(tmp_path) -> None:
    agent = QLearningAgent(n_actions=2, epsilon=0.0)
    agent.update([1.0], 0, 0.2, [2.0], done=True)

    path = agent.save(tmp_path / "model.json")
    loaded = QLearningAgent.load(path)

    assert loaded.n_actions == agent.n_actions
    assert loaded.q_table == agent.q_table


def test_train_q_learning_agent_populates_q_table() -> None:
    dataset = _make_environment_dataset()
    env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))

    result = train_q_learning_agent(
        env=env,
        config=QLearningTrainingConfig(episodes=3, seed=3),
    )

    assert len(result.history) == 3
    assert len(result.agent.q_table) > 0
    assert result.agent.training is False


def test_explicit_historical_mode_matches_existing_training() -> None:
    dataset = _make_environment_dataset()
    first_env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))
    second_env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))
    default_result = train_q_learning_agent(
        env=first_env,
        config=QLearningTrainingConfig(episodes=2, seed=11),
    )
    historical_result = train_q_learning_agent(
        env=second_env,
        config=QLearningTrainingConfig(
            episodes=2,
            training_mode="historical",
            seed=11,
        ),
    )

    assert historical_result.agent.q_table == default_result.agent.q_table
    assert historical_result.agent.visit_counts == default_result.agent.visit_counts


def _make_environment_dataset() -> pd.DataFrame:
    """Create a tiny environment-compatible dataset."""
    index = pd.date_range("2020-01-31", periods=16, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(16)],
        "vix": [20.0 + month for month in range(16)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        data[f"price_{prefix}"] = [100.0 + 5.0 * month for month in range(16)]
        data[f"r_{prefix}"] = [0.01] * 16
    return pd.DataFrame(data, index=index)
