"""Random action baseline agent."""

import random
from collections.abc import Sequence


class RandomAgent:
    """Select a random action ID from the discrete action space."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        """Create a reproducible random agent."""
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        self.n_actions = n_actions
        self._rng = random.Random(seed)

    def select_action(self, state: Sequence[float]) -> int:
        """Return a random action ID.

        The state is accepted to match the environment-facing agent interface.
        """
        del state
        return self._rng.randrange(self.n_actions)
