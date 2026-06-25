"""Gym-like portfolio allocation environment."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from env.action_space import generate_action_space
from env.reward import calculate_reward
from features.state_builder import PortfolioContext, StateBuilder


@dataclass(frozen=True)
class StepResult:
    """Result returned by one environment step."""

    state: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class PortfolioEnv:
    """Monthly portfolio environment driven by historical return data."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        state_builder: StateBuilder,
        action_space: list[list[float]] | None = None,
        initial_weights: list[float] | None = None,
        initial_value: float = 1.0,
        transaction_cost: float = 0.014,
    ) -> None:
        """Create an environment for one historical dataset."""
        self.dataset = dataset
        self.state_builder = state_builder
        self.action_space = action_space or generate_action_space(step=0.10, n_assets=4)
        self.initial_weights = initial_weights or [0.25, 0.25, 0.25, 0.25]
        self.initial_value = initial_value
        self.transaction_cost = transaction_cost

        self._validate()
        self.current_index = self.state_builder.first_valid_index()
        self.current_weights = self.initial_weights.copy()
        self.portfolio_value = self.initial_value
        self.peak_value = self.initial_value

    def reset(self) -> list[float]:
        """Reset the environment and return the initial state."""
        self.current_index = self.state_builder.first_valid_index()
        self.current_weights = self.initial_weights.copy()
        self.portfolio_value = self.initial_value
        self.peak_value = self.initial_value
        return self._build_current_state()

    def step(self, action_id: int) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Apply an action and return next state, reward, done, and info."""
        if self.current_index >= len(self.dataset) - 1:
            raise RuntimeError("cannot step beyond the end of the dataset")
        if action_id < 0 or action_id >= len(self.action_space):
            raise ValueError("action_id is outside the action space")

        previous_weights = self.current_weights
        target_weights = self.action_space[action_id]
        next_index = self.current_index + 1
        next_row = self.dataset.iloc[next_index]

        portfolio_return = self._calculate_portfolio_return(next_row, target_weights)
        turnover = calculate_turnover(target_weights, previous_weights)
        reward = calculate_reward(
            portfolio_return=portfolio_return,
            turnover=turnover,
            transaction_cost=self.transaction_cost,
        )

        self.portfolio_value *= 1.0 + reward
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = self.portfolio_value / self.peak_value - 1.0
        self.current_index = next_index
        self.current_weights = target_weights.copy()

        done = self.current_index >= len(self.dataset) - 1
        info = {
            "date": self.dataset.index[self.current_index],
            "action_id": action_id,
            "weights": target_weights,
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "rebalance_cost": self.transaction_cost * turnover,
            "reward": reward,
            "portfolio_value": self.portfolio_value,
            "drawdown": drawdown,
        }
        return self._build_current_state(), reward, done, info

    def _calculate_portfolio_return(
        self,
        returns_row: pd.Series,
        weights: list[float],
    ) -> float:
        """Calculate weighted portfolio return for one row."""
        return float(
            sum(
                weight * returns_row[column]
                for weight, column in zip(
                    weights,
                    self.state_builder.return_columns,
                    strict=True,
                )
            ),
        )

    def _build_current_state(self) -> list[float]:
        """Build the state for the current time index."""
        return self.state_builder.build_state(
            self.current_index,
            PortfolioContext(weights=self.current_weights),
        )

    def _validate(self) -> None:
        """Validate environment inputs."""
        if self.dataset.empty:
            raise ValueError("dataset cannot be empty")
        if self.initial_value <= 0.0:
            raise ValueError("initial_value must be positive")
        if self.transaction_cost < 0.0:
            raise ValueError("transaction_cost must be non-negative")
        if len(self.initial_weights) != 4:
            raise ValueError("initial_weights must contain four weights")
        if abs(sum(self.initial_weights) - 1.0) > 1e-12:
            raise ValueError("initial_weights must sum to one")
        if self.state_builder.first_valid_index() >= len(self.dataset) - 1:
            raise ValueError("dataset must contain at least one tradable next period")


def calculate_turnover(
    current_weights: list[float],
    previous_weights: list[float],
) -> float:
    """Calculate allocation turnover between two weight vectors."""
    if len(current_weights) != len(previous_weights):
        raise ValueError("weight vectors must have the same length")
    return 0.5 * sum(
        abs(current - previous)
        for current, previous in zip(current_weights, previous_weights, strict=True)
    )
