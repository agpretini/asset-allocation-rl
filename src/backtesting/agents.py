"""Backtest environment-facing agents."""

from collections.abc import Sequence
from typing import Protocol

import pandas as pd

from agents.q_learning import QLearningAgent
from env.portfolio_env import PortfolioEnv, calculate_turnover


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


class CostAwareFrozenPolicy:
    """Read-only Model B policy: Q-value less immediate execution cost."""

    def __init__(
        self,
        agent: QLearningAgent,
        action_space: list[list[float]],
        transaction_cost: float = 0.014,
    ) -> None:
        self.agent = agent
        self.action_space = action_space
        self.transaction_cost = transaction_cost
        self.current_weights = [0.25, 0.25, 0.25, 0.25]

    def reset(self, weights: list[float] | None = None) -> None:
        self.current_weights = (weights or [0.25, 0.25, 0.25, 0.25]).copy()

    def select_action(self, state: Sequence[float]) -> int:
        q_values = self.agent.q_table.get(self.agent.state_key(state))
        if q_values is None:
            return 0
        scores = [
            value
            - self.transaction_cost * calculate_turnover(self.current_weights, target)
            for value, target in zip(q_values, self.action_space, strict=True)
        ]
        best = max(scores)
        return min(index for index, score in enumerate(scores) if score == best)

    def observe_action(self, action_id: int) -> None:
        self.current_weights = self.action_space[action_id].copy()


def run_agent_backtest(env: PortfolioEnv, agent: ActionAgent) -> pd.DataFrame:
    """Run an agent through one full historical environment episode."""
    state = env.reset()
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset(env.initial_weights)
    rows: list[dict] = []
    done = False

    while not done:
        action_id = agent.select_action(state)
        state, _, done, info = env.step(action_id)
        observe_action = getattr(agent, "observe_action", None)
        if callable(observe_action):
            observe_action(action_id)
        rows.append(info)

    result = pd.DataFrame(rows)
    result = result.set_index("date")
    return result
