"""Build environment states from market features and portfolio context."""

from dataclasses import dataclass

import pandas as pd

from features.market_features import build_market_features


@dataclass(frozen=True)
class PortfolioContext:
    """Portfolio variables included in the state."""

    weights: list[float]


class StateBuilder:
    """Construct state vectors without using future observations."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        asset_prefixes: list[str] | None = None,
        macro_columns: list[str] | None = None,
    ) -> None:
        """Prepare reusable market features for one dataset."""
        self.dataset = dataset
        self.asset_prefixes = asset_prefixes or ["arg", "ced", "sp500", "gold"]
        self.macro_columns = macro_columns or [
            column
            for column in ["usd_ars", "inflation", "vix"]
            if column in dataset.columns
        ]
        self.return_columns = [f"r_{prefix}" for prefix in self.asset_prefixes]
        self.market_features = build_market_features(
            dataset=dataset,
            asset_prefixes=self.asset_prefixes,
            macro_columns=self.macro_columns,
        )

    def build_state(
        self, row_index: int, portfolio_context: PortfolioContext
    ) -> list[float]:
        """Build the state vector for one row index and portfolio context."""
        self._validate_row_index(row_index)
        self._validate_portfolio_context(portfolio_context)

        feature_row = self.market_features.iloc[row_index]
        if feature_row.isna().any():
            raise ValueError("state contains missing features at this row")
        return [
            *[float(value) for value in feature_row.to_list()],
            *[float(weight) for weight in portfolio_context.weights],
        ]

    def first_valid_index(self) -> int:
        """Return the first row index where all market features are available."""
        valid_mask = ~self.market_features.isna().any(axis=1)
        if not valid_mask.any():
            raise ValueError("no rows contain a complete state")
        return int(valid_mask.to_numpy().argmax())

    def state_size(self) -> int:
        """Return the number of values in each state vector."""
        return len(self.market_features.columns) + len(self.asset_prefixes)

    def _validate_row_index(self, row_index: int) -> None:
        """Validate row index bounds."""
        if row_index < 0 or row_index >= len(self.dataset):
            raise ValueError("row_index is outside the dataset")

    def _validate_portfolio_context(self, portfolio_context: PortfolioContext) -> None:
        """Validate portfolio context weights."""
        if len(portfolio_context.weights) != len(self.asset_prefixes):
            raise ValueError("portfolio weights must match the number of assets")
        if any(weight < 0.0 for weight in portfolio_context.weights):
            raise ValueError("portfolio weights must be non-negative")
        if abs(sum(portfolio_context.weights) - 1.0) > 1e-12:
            raise ValueError("portfolio weights must sum to one")
