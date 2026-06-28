"""Backtest environment-facing agents."""

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from env.portfolio_env import PortfolioEnv


class ActionAgent(Protocol):
    """Agent interface required by the environment backtest."""

    def select_action(self, state: Sequence[float]) -> int:
        """Select an action ID from the current state."""


def run_agent_backtest(env: PortfolioEnv, agent: ActionAgent) -> pd.DataFrame:
    """Run an agent through one full historical environment episode."""
    state = env.reset()
    rows: list[dict] = []
    done = False

    while not done:
        action_id = agent.select_action(state)
        state, _, done, info = env.step(action_id)
        rows.append(info)

    result = pd.DataFrame(rows)
    result = result.set_index("date")
    return result
