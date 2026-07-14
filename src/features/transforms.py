"""State feature transformations."""

from typing import Literal

import pandas as pd

FeatureMode = Literal["raw", "normalized_binned"]

RAW_FEATURE_MODE: FeatureMode = "raw"
NORMALIZED_BINNED_FEATURE_MODE: FeatureMode = "normalized_binned"
FEATURE_MODES: tuple[FeatureMode, ...] = (
    RAW_FEATURE_MODE,
    NORMALIZED_BINNED_FEATURE_MODE,
)

Z_SCORE_MIN_PERIODS = 1
Z_SCORE_BINS = (-float("inf"), -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, float("inf"))
Z_SCORE_BIN_LABELS = (-3, -2, -1, 0, 1, 2, 3)


def transform_market_features(
    market_features: pd.DataFrame,
    feature_mode: FeatureMode,
) -> pd.DataFrame:
    """Apply the selected state feature transformation."""
    if feature_mode == RAW_FEATURE_MODE:
        return market_features
    if feature_mode == NORMALIZED_BINNED_FEATURE_MODE:
        return build_zscore_binned_features(market_features)
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def build_zscore_binned_features(market_features: pd.DataFrame) -> pd.DataFrame:
    """Convert market features to expanding z-score regime bins."""
    z_scores = build_expanding_zscores(market_features)
    binned_columns = {
        f"{column}_zbin": pd.cut(
            z_scores[column],
            bins=Z_SCORE_BINS,
            labels=Z_SCORE_BIN_LABELS,
        ).astype(float)
        for column in z_scores.columns
    }
    return pd.DataFrame(binned_columns, index=market_features.index)


def build_expanding_zscores(market_features: pd.DataFrame) -> pd.DataFrame:
    """Normalize each feature using only observations available up to time t."""
    expanding = market_features.expanding(min_periods=Z_SCORE_MIN_PERIODS)
    means = expanding.mean()
    standard_deviations = expanding.std(ddof=0).replace(0.0, 1.0)
    return (market_features - means) / standard_deviations
