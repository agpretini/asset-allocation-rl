"""Tests for the portfolio environment."""

import pandas as pd
import pytest

from env.portfolio_env import PortfolioEnv, calculate_turnover
from features.state_builder import StateBuilder


@pytest.fixture
def environment_dataset() -> pd.DataFrame:
    """Create a deterministic dataset for environment tests."""
    index = pd.date_range("2020-01-31", periods=14, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(14)],
        "vix": [20.0 + month for month in range(14)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        data[f"price_{prefix}"] = [100.0 + 10.0 * month for month in range(14)]

    data["r_arg"] = [0.0] * 14
    data["r_ced"] = [0.0] * 14
    data["r_sp500"] = [0.0] * 14
    data["r_gold"] = [0.0] * 14
    data["r_arg"][13] = 0.10
    data["r_ced"][13] = 0.02
    data["r_sp500"][13] = -0.01
    data["r_gold"][13] = 0.03

    return pd.DataFrame(data, index=index)


def test_calculate_turnover() -> None:
    turnover = calculate_turnover(
        current_weights=[1.0, 0.0, 0.0, 0.0],
        previous_weights=[0.25, 0.25, 0.25, 0.25],
    )

    assert turnover == pytest.approx(0.75)


def test_reset_returns_initial_state(environment_dataset: pd.DataFrame) -> None:
    state_builder = StateBuilder(environment_dataset)
    env = PortfolioEnv(dataset=environment_dataset, state_builder=state_builder)

    state = env.reset()

    assert len(state) == state_builder.state_size()
    assert state[-4:] == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_step_applies_next_period_returns_and_costs(
    environment_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(environment_dataset)
    env = PortfolioEnv(dataset=environment_dataset, state_builder=state_builder)
    env.reset()

    next_state, reward, done, info = env.step(action_id=0)

    assert info["weights"] == [1.0, 0.0, 0.0, 0.0]
    assert info["portfolio_return"] == pytest.approx(0.10)
    assert info["turnover"] == pytest.approx(0.75)
    assert info["rebalance_cost"] == pytest.approx(0.014 * 0.75)
    assert reward == pytest.approx(0.10 - 0.014 * 0.75)
    assert info["portfolio_value"] == pytest.approx(1.0 + reward)
    assert done is True
    assert next_state[-4:] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_step_rejects_invalid_action(environment_dataset: pd.DataFrame) -> None:
    state_builder = StateBuilder(environment_dataset)
    env = PortfolioEnv(dataset=environment_dataset, state_builder=state_builder)
    env.reset()

    with pytest.raises(ValueError, match="outside the action space"):
        env.step(action_id=999)
