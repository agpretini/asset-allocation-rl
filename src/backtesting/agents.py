"""Backtest environment-facing agents."""

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from agents.q_learning import QLearningAgent
from env.portfolio_env import PortfolioEnv


class ActionAgent(Protocol):
    """Agent interface required by the environment backtest."""

    def select_action(self, state: Sequence[float]) -> int:
        """Select an action ID from the current state."""


class FrozenGreedyPolicy:
    """Read-only deterministic greedy view of a Q-learning agent.

    Evaluation must not create zero-valued Q-table entries for unseen states.
    Those states use action zero, which is also the deterministic tie-breaker.
    """

    def __init__(self, agent: QLearningAgent) -> None:
        self.agent = agent

    def select_action(self, state: Sequence[float]) -> int:
        """Return the lowest action ID with the best already-known value."""
        state_key = self.agent.state_key(state)
        q_values = self.agent.q_table.get(state_key)
        if q_values is None:
            return 0
        best_value = max(q_values)
        return min(
            action_id for action_id, value in enumerate(q_values) if value == best_value
        )


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
