"""Tabular Q-learning agent for the discrete portfolio action space."""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

LearningRateSchedule = Literal["constant", "visit_decay"]
CONSTANT_LEARNING_RATE: LearningRateSchedule = "constant"
VISIT_DECAY_LEARNING_RATE: LearningRateSchedule = "visit_decay"


class QLearningAgent:
    """A small tabular Q-learning agent with rounded-state discretization."""

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.10,
        learning_rate_schedule: LearningRateSchedule = CONSTANT_LEARNING_RATE,
        visit_learning_rate_decay: float = 0.01,
        discount_factor: float = 0.95,
        epsilon: float = 0.50,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.999954,
        state_precision: int = 2,
        seed: int | None = None,
    ) -> None:
        """Create a reproducible Q-learning agent."""
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if not 0.0 <= min_epsilon <= 1.0:
            raise ValueError("min_epsilon must be between zero and one")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if learning_rate_schedule not in {
            CONSTANT_LEARNING_RATE,
            VISIT_DECAY_LEARNING_RATE,
        }:
            raise ValueError("unsupported learning_rate_schedule")
        if visit_learning_rate_decay < 0.0:
            raise ValueError("visit_learning_rate_decay must be non-negative")
        if discount_factor < 0.0:
            raise ValueError("discount_factor must be non-negative")

        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        self.visit_learning_rate_decay = visit_learning_rate_decay
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.state_precision = state_precision
        self.training = True
        self.q_table: dict[str, list[float]] = {}
        self.visit_counts: dict[tuple[str, int], int] = {}
        self._rng = random.Random(seed)

    def select_action(self, state: Sequence[float]) -> int:
        """Select an epsilon-greedy action ID."""
        if self.training and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        return self._select_greedy_action(state)

    def update(
        self,
        state: Sequence[float],
        action_id: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        """Update one Q-value from a transition."""
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError("action_id is outside the action space")

        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)
        q_values = self._get_q_values(state_key)
        next_q_values = self._get_q_values(next_state_key)

        bootstrap_value = 0.0 if done else max(next_q_values)
        target = reward + self.discount_factor * bootstrap_value
        learning_rate = self._learning_rate_for(state_key, action_id)
        q_values[action_id] += learning_rate * (target - q_values[action_id])
        visit_key = (state_key, action_id)
        self.visit_counts[visit_key] = self.visit_counts.get(visit_key, 0) + 1

    def decay_epsilon(self) -> None:
        """Decay exploration after an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def set_training(self, training: bool) -> None:
        """Switch between training and greedy evaluation mode."""
        self.training = training

    def state_key(self, state: Sequence[float]) -> str:
        """Return the discretized key used by the Q-table for a state."""
        return self._state_key(state)

    def q_values_for_state(self, state: Sequence[float]) -> list[float]:
        """Return Q-values for a state, initializing unseen states."""
        return self._get_q_values(self._state_key(state))

    def learning_rate_for(self, state: Sequence[float], action_id: int) -> float:
        """Return the learning rate that the next update would use."""
        if action_id < 0 or action_id >= self.n_actions:
            raise ValueError("action_id is outside the action space")
        return self._learning_rate_for(self._state_key(state), action_id)

    def save(self, path: str | Path) -> Path:
        """Save the Q-table and hyperparameters as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "learning_rate_schedule": self.learning_rate_schedule,
            "visit_learning_rate_decay": self.visit_learning_rate_decay,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "state_precision": self.state_precision,
            "q_table": self.q_table,
            "visit_counts": [
                {"state_key": state_key, "action_id": action_id, "count": count}
                for (state_key, action_id), count in self.visit_counts.items()
            ],
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        """Load a Q-learning agent from JSON."""
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(
            n_actions=int(payload["n_actions"]),
            learning_rate=float(payload["learning_rate"]),
            learning_rate_schedule=payload.get(
                "learning_rate_schedule",
                CONSTANT_LEARNING_RATE,
            ),
            visit_learning_rate_decay=float(
                payload.get("visit_learning_rate_decay", 0.01),
            ),
            discount_factor=float(payload["discount_factor"]),
            epsilon=float(payload["epsilon"]),
            min_epsilon=float(payload["min_epsilon"]),
            epsilon_decay=float(payload["epsilon_decay"]),
            state_precision=int(payload["state_precision"]),
        )
        agent.q_table = {
            key: [float(value) for value in values]
            for key, values in payload["q_table"].items()
        }
        agent.visit_counts = {
            (str(item["state_key"]), int(item["action_id"])): int(item["count"])
            for item in payload.get("visit_counts", [])
        }
        return agent

    def _select_greedy_action(self, state: Sequence[float]) -> int:
        """Select an action with the highest known Q-value."""
        q_values = self._get_q_values(self._state_key(state))
        best_value = max(q_values)
        best_actions = [
            action_id for action_id, value in enumerate(q_values) if value == best_value
        ]
        return self._rng.choice(best_actions)

    def _state_key(self, state: Sequence[float]) -> str:
        """Discretize a state into a stable string key."""
        return "|".join(
            str(round(float(value), self.state_precision)) for value in state
        )

    def _get_q_values(self, state_key: str) -> list[float]:
        """Return the Q-values for a state, initializing unseen states."""
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        return self.q_table[state_key]

    def _learning_rate_for(self, state_key: str, action_id: int) -> float:
        """Return the rate for the next update of one state-action pair."""
        if self.learning_rate_schedule == CONSTANT_LEARNING_RATE:
            return self.learning_rate
        visits = self.visit_counts.get((state_key, action_id), 0)
        return self.learning_rate / (1.0 + self.visit_learning_rate_decay * visits)
