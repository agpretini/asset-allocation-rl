"""Tests for environment-facing agent backtests."""

import pandas as pd

from agents.random_agent import RandomAgent
from backtesting.agents import run_agent_backtest
from env.portfolio_env import PortfolioEnv
from features.state_builder import StateBuilder


def test_random_agent_backtest_runs_to_completion() -> None:
    dataset = _make_environment_dataset()
    state_builder = StateBuilder(dataset)
    env = PortfolioEnv(dataset=dataset, state_builder=state_builder)
    agent = RandomAgent(n_actions=len(env.action_space), seed=1)

    result = run_agent_backtest(env, agent)

    assert len(result) == 2
    assert result.index.name == "date"
    assert {"action_id", "reward", "portfolio_value", "turnover"}.issubset(
        result.columns,
    )


def _make_environment_dataset() -> pd.DataFrame:
    """Create a tiny environment-compatible dataset."""
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(15)],
        "vix": [20.0 + month for month in range(15)],
    }
    for prefix in ["arg", "ced", "sp500", "gold"]:
        data[f"price_{prefix}"] = [100.0 + 5.0 * month for month in range(15)]
        data[f"r_{prefix}"] = [0.01] * 15
    return pd.DataFrame(data, index=index)
