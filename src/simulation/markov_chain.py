"""Finite-state Markov chain for monthly market regimes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulation.regimes import REGIME_ORDER, MarketRegime


@dataclass(frozen=True)
class RegimeMarkovChain:
    """Initial regime probabilities and monthly transition probabilities."""

    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray

    @classmethod
    def fit(
        cls,
        regimes: pd.Series,
        smoothing: float = 1.0,
    ) -> "RegimeMarkovChain":
        """Estimate probabilities from an ordered training regime series."""
        if regimes.empty:
            raise ValueError("regime series cannot be empty")
        if smoothing < 0.0:
            raise ValueError("transition smoothing must be non-negative")

        regime_to_index = {
            regime.value: index for index, regime in enumerate(REGIME_ORDER)
        }
        unknown = set(regimes.astype(str)) - set(regime_to_index)
        if unknown:
            raise ValueError(f"unknown market regimes: {sorted(unknown)}")

        encoded = np.array(
            [regime_to_index[value] for value in regimes.astype(str)],
            dtype=int,
        )
        initial_counts = np.bincount(encoded, minlength=len(REGIME_ORDER)).astype(float)
        initial_probabilities = initial_counts / initial_counts.sum()

        transition_counts = np.full(
            (len(REGIME_ORDER), len(REGIME_ORDER)),
            smoothing,
            dtype=float,
        )
        for current, following in zip(encoded[:-1], encoded[1:], strict=True):
            transition_counts[current, following] += 1.0
        row_totals = transition_counts.sum(axis=1, keepdims=True)
        zero_rows = row_totals[:, 0] == 0.0
        if zero_rows.any():
            transition_counts[zero_rows] = 1.0
            row_totals = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = transition_counts / row_totals
        return cls(
            initial_probabilities=initial_probabilities,
            transition_matrix=transition_matrix,
        )

    def sample_path(
        self,
        length: int,
        rng: np.random.Generator,
    ) -> list[MarketRegime]:
        """Sample one regime path of ``length`` months."""
        if length < 1:
            raise ValueError("regime path length must be positive")
        self._validate()
        current = int(rng.choice(len(REGIME_ORDER), p=self.initial_probabilities))
        path = [REGIME_ORDER[current]]
        for _ in range(1, length):
            current = int(
                rng.choice(len(REGIME_ORDER), p=self.transition_matrix[current])
            )
            path.append(REGIME_ORDER[current])
        return path

    def _validate(self) -> None:
        """Validate probability vector and matrix dimensions."""
        expected_size = len(REGIME_ORDER)
        if self.initial_probabilities.shape != (expected_size,):
            raise ValueError("invalid initial regime probability shape")
        if self.transition_matrix.shape != (expected_size, expected_size):
            raise ValueError("invalid transition matrix shape")
        if not np.isclose(self.initial_probabilities.sum(), 1.0):
            raise ValueError("initial regime probabilities must sum to one")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("transition matrix rows must sum to one")
        if (self.initial_probabilities < 0.0).any() or (
            self.transition_matrix < 0.0
        ).any():
            raise ValueError("Markov probabilities must be non-negative")
