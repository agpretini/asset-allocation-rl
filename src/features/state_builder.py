"""Build environment states from market features and portfolio context."""

from dataclasses import dataclass

import pandas as pd

from features.market_features import (
    MACRO_MODE_CHANGE,
    MACRO_MODE_LEVEL,
    build_market_features,
)
from features.transforms import (
    COMPACT_REGIME_FEATURE_MODE,
    NORMALIZED_BINNED_FEATURE_MODE,
    TERCILE_BINNED_FEATURE_MODE,
    FeatureMode,
    transform_market_features,
)


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
        feature_mode: FeatureMode = COMPACT_REGIME_FEATURE_MODE,
        history_dataset: pd.DataFrame | None = None,
        include_portfolio_context: bool | None = None,
    ) -> None:
        """Prepare reusable market features for one dataset.

        ``history_dataset`` may contain observations strictly before
        ``dataset``. It seeds expanding transforms for an out-of-sample window
        without exposing any future observation.
        """
        self.dataset = dataset
        self.history_dataset = history_dataset
        self.asset_prefixes = asset_prefixes or ["arg", "ced", "sp500", "gold"]
        self.macro_columns = macro_columns or [
            column
            for column in ["usd_ars", "inflation", "vix"]
            if column in dataset.columns
        ]
        self.feature_mode = feature_mode
        self.include_portfolio_context = (
            feature_mode != COMPACT_REGIME_FEATURE_MODE
            if include_portfolio_context is None
            else include_portfolio_context
        )
        self.return_columns = [f"r_{prefix}" for prefix in self.asset_prefixes]
        macro_mode = (
            MACRO_MODE_CHANGE
            if feature_mode
            in {
                NORMALIZED_BINNED_FEATURE_MODE,
                TERCILE_BINNED_FEATURE_MODE,
                COMPACT_REGIME_FEATURE_MODE,
            }
            else MACRO_MODE_LEVEL
        )
        feature_dataset = self._build_feature_dataset()
        raw_market_features = build_market_features(
            dataset=feature_dataset,
            asset_prefixes=self.asset_prefixes,
            macro_columns=self.macro_columns,
            macro_mode=macro_mode,
        )
        transformed_market_features = transform_market_features(
            raw_market_features,
            feature_mode=feature_mode,
            asset_prefixes=self.asset_prefixes,
        )
        self.market_features = transformed_market_features.loc[dataset.index]

    def build_state(
        self, row_index: int, portfolio_context: PortfolioContext
    ) -> list[float]:
        """Build the state vector for one row index and portfolio context."""
        self._validate_row_index(row_index)
        self._validate_portfolio_context(portfolio_context)

        feature_row = self.market_features.iloc[row_index]
        if feature_row.isna().any():
            raise ValueError("state contains missing features at this row")
        market_state = [float(value) for value in feature_row.to_list()]
        if not self.include_portfolio_context:
            return market_state
        return [*market_state, *[float(weight) for weight in portfolio_context.weights]]

    def first_valid_index(self) -> int:
        """Return the first row index where all market features are available."""
        valid_mask = ~self.market_features.isna().any(axis=1)
        if not valid_mask.any():
            raise ValueError("no rows contain a complete state")
        return int(valid_mask.to_numpy().argmax())

    def state_size(self) -> int:
        """Return the number of values in each state vector."""
        return len(self.market_features.columns) + (
            len(self.asset_prefixes) if self.include_portfolio_context else 0
        )

    def _validate_row_index(self, row_index: int) -> None:
        """Validate row index bounds."""
        if row_index < 0 or row_index >= len(self.dataset):
            raise ValueError("row_index is outside the dataset")

    def _build_feature_dataset(self) -> pd.DataFrame:
        """Combine historical context and target rows for feature calculation."""
        if self.history_dataset is None or self.history_dataset.empty:
            return self.dataset
        if not self.history_dataset.index.is_monotonic_increasing:
            raise ValueError("history_dataset index must be sorted")
        if not self.dataset.index.is_monotonic_increasing:
            raise ValueError("dataset index must be sorted")
        if self.history_dataset.index.max() >= self.dataset.index.min():
            raise ValueError("history_dataset must end before dataset starts")
        missing_columns = set(self.dataset.columns) - set(self.history_dataset.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"history_dataset is missing columns: {missing}")
        return pd.concat([self.history_dataset[self.dataset.columns], self.dataset])

    def _validate_portfolio_context(self, portfolio_context: PortfolioContext) -> None:
        """Validate portfolio context weights."""
        if len(portfolio_context.weights) != len(self.asset_prefixes):
            raise ValueError("portfolio weights must match the number of assets")
        if any(weight < 0.0 for weight in portfolio_context.weights):
            raise ValueError("portfolio weights must be non-negative")
        if abs(sum(portfolio_context.weights) - 1.0) > 1e-12:
            raise ValueError("portfolio weights must sum to one")
