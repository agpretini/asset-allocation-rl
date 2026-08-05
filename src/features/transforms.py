"""State feature transformations."""

from typing import Literal

import pandas as pd

FeatureMode = Literal[
    "raw",
    "normalized_binned",
    "tercile_binned",
    "compact_regime",
]

RAW_FEATURE_MODE: FeatureMode = "raw"
NORMALIZED_BINNED_FEATURE_MODE: FeatureMode = "normalized_binned"
TERCILE_BINNED_FEATURE_MODE: FeatureMode = "tercile_binned"
COMPACT_REGIME_FEATURE_MODE: FeatureMode = "compact_regime"
FEATURE_MODES: tuple[FeatureMode, ...] = (
    RAW_FEATURE_MODE,
    NORMALIZED_BINNED_FEATURE_MODE,
    TERCILE_BINNED_FEATURE_MODE,
    COMPACT_REGIME_FEATURE_MODE,
)

Z_SCORE_MIN_PERIODS = 1
Z_SCORE_BINS = (-float("inf"), -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, float("inf"))
Z_SCORE_BIN_LABELS = (-3, -2, -1, 0, 1, 2, 3)


def transform_market_features(
    market_features: pd.DataFrame,
    feature_mode: FeatureMode,
    asset_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the selected state feature transformation."""
    if feature_mode == RAW_FEATURE_MODE:
        return market_features
    if feature_mode == NORMALIZED_BINNED_FEATURE_MODE:
        return build_zscore_binned_features(market_features)
    if feature_mode == TERCILE_BINNED_FEATURE_MODE:
        return build_expanding_tercile_features(market_features)
    if feature_mode == COMPACT_REGIME_FEATURE_MODE:
        if asset_prefixes is None:
            raise ValueError("compact_regime requires asset_prefixes")
        return build_compact_regime_features(market_features, asset_prefixes)
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


def build_expanding_tercile_features(market_features: pd.DataFrame) -> pd.DataFrame:
    """Classify each feature as low, medium, or high using past-only terciles.

    Each row's thresholds use valid observations from the beginning of the
    series through the current row, so later observations cannot affect an
    earlier state. Values are encoded as 0 (low), 1 (medium), and 2 (high).
    """
    tercile_columns = {
        f"{column}_tercile": _build_expanding_terciles(market_features[column])
        for column in market_features.columns
    }
    return pd.DataFrame(tercile_columns, index=market_features.index)


def build_compact_regime_features(
    market_features: pd.DataFrame,
    asset_prefixes: list[str],
) -> pd.DataFrame:
    """Build a compact causal regime state for tabular Q-learning.

    Momentum is first discretized independently into expanding historical
    terciles. The average of the 1M and 3M momentum labels determines each
    asset's score. The best score determines the leader; ties use the supplied
    asset order. The leading score is then binned as low, medium, or high.
    USD/ARS change and VIX retain their individual expanding tercile regimes.
    """
    momentum_columns = [
        f"{prefix}_mom_{window}m" for prefix in asset_prefixes for window in (1, 3)
    ]
    required_columns = {*momentum_columns, "usd_ars_change", "vix"}
    missing_columns = required_columns - set(market_features.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"compact_regime is missing columns: {missing}")

    terciles = build_expanding_tercile_features(market_features)
    momentum_scores: dict[str, pd.Series] = {}
    for prefix in asset_prefixes:
        columns = [
            f"{prefix}_mom_1m_tercile",
            f"{prefix}_mom_3m_tercile",
        ]
        labels = terciles[columns]
        momentum_scores[prefix] = labels.mean(axis=1).where(labels.notna().all(axis=1))
    momentum_scores_frame = pd.DataFrame(
        momentum_scores,
        index=market_features.index,
    )
    all_momentum_available = momentum_scores_frame.notna().all(axis=1)
    available_scores = momentum_scores_frame.fillna(-float("inf"))
    leader_names = available_scores.idxmax(axis=1).where(all_momentum_available)
    leader_codes = leader_names.map(
        {prefix: index for index, prefix in enumerate(asset_prefixes)}
    )
    leading_score = available_scores.max(axis=1).where(all_momentum_available)
    strength = pd.cut(
        leading_score,
        bins=(-float("inf"), 2.0 / 3.0, 4.0 / 3.0, float("inf")),
        labels=(0, 1, 2),
    ).astype(float)

    return pd.DataFrame(
        {
            "momentum_leader": leader_codes.astype(float),
            "momentum_strength": strength,
            "usd_ars_regime": terciles["usd_ars_change_tercile"],
            "vix_regime": terciles["vix_tercile"],
        },
        index=market_features.index,
    )


def _build_expanding_terciles(feature: pd.Series) -> pd.Series:
    """Return one past-only tercile label per valid value in ``feature``."""
    lower_tercile = feature.expanding(min_periods=1).quantile(1.0 / 3.0)
    upper_tercile = feature.expanding(min_periods=1).quantile(2.0 / 3.0)
    valid = feature.notna()
    labels = pd.Series(float("nan"), index=feature.index, dtype=float)
    labels.loc[valid & feature.le(lower_tercile)] = 0.0
    labels.loc[valid & feature.gt(lower_tercile) & feature.le(upper_tercile)] = 1.0
    labels.loc[valid & feature.gt(upper_tercile)] = 2.0
    return labels
