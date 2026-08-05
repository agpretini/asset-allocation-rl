"""Feature engineering and state construction."""

from features.market_features import build_market_features
from features.state_builder import PortfolioContext, StateBuilder
from features.transforms import (
    COMPACT_REGIME_FEATURE_MODE,
    FEATURE_MODES,
    NORMALIZED_BINNED_FEATURE_MODE,
    RAW_FEATURE_MODE,
    TERCILE_BINNED_FEATURE_MODE,
    FeatureMode,
)

__all__ = [
    "FEATURE_MODES",
    "COMPACT_REGIME_FEATURE_MODE",
    "NORMALIZED_BINNED_FEATURE_MODE",
    "RAW_FEATURE_MODE",
    "TERCILE_BINNED_FEATURE_MODE",
    "FeatureMode",
    "PortfolioContext",
    "StateBuilder",
    "build_market_features",
]
