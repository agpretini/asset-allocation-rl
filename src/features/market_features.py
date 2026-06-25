"""Market feature engineering for the portfolio state."""

import pandas as pd

MOMENTUM_WINDOWS = (1, 3, 6, 12)
VOLATILITY_WINDOWS = (3, 6, 12)


def build_market_features(
    dataset: pd.DataFrame,
    asset_prefixes: list[str],
    macro_columns: list[str],
) -> pd.DataFrame:
    """Build momentum, volatility, and macro features available at time t."""
    feature_frames: list[pd.Series] = []
    for prefix in asset_prefixes:
        price_column = f"price_{prefix}"
        return_column = f"r_{prefix}"
        _validate_columns(dataset, [price_column, return_column])

        for window in MOMENTUM_WINDOWS:
            feature_frames.append(
                (
                    dataset[price_column] / dataset[price_column].shift(window) - 1.0
                ).rename(
                    f"{prefix}_mom_{window}m",
                ),
            )
        for window in VOLATILITY_WINDOWS:
            feature_frames.append(
                dataset[return_column]
                .rolling(window=window)
                .std()
                .rename(
                    f"{prefix}_vol_{window}m",
                ),
            )

    _validate_columns(dataset, macro_columns)
    feature_frames.extend(dataset[column].rename(column) for column in macro_columns)
    return pd.concat(feature_frames, axis=1)


def _validate_columns(dataset: pd.DataFrame, columns: list[str]) -> None:
    """Raise when required columns are missing."""
    missing_columns = set(columns) - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required columns: {missing}")
