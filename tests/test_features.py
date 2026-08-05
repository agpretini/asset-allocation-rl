"""Tests for feature engineering and state construction."""

import pandas as pd
import pytest

from features.market_features import build_market_features
from features.state_builder import PortfolioContext, StateBuilder
from features.transforms import (
    COMPACT_REGIME_FEATURE_MODE,
    build_expanding_tercile_features,
    build_expanding_zscores,
)


@pytest.fixture
def feature_dataset() -> pd.DataFrame:
    """Create a deterministic dataset with enough history for 3M features."""
    index = pd.date_range("2020-01-31", periods=15, freq="ME")
    data: dict[str, list[float]] = {
        "usd_ars": [100.0 + month for month in range(15)],
        "vix": [20.0 + month for month in range(15)],
    }
    for offset, prefix in enumerate(["arg", "ced", "sp500", "gold"]):
        prices = [100.0 + offset + 10.0 * month for month in range(15)]
        returns = [0.0]
        returns.extend(
            prices[index] / prices[index - 1] - 1.0 for index in range(1, len(prices))
        )
        data[f"price_{prefix}"] = prices
        data[f"r_{prefix}"] = returns
    return pd.DataFrame(data, index=index)


def test_market_features_use_past_and_current_values_only(
    feature_dataset: pd.DataFrame,
) -> None:
    features = build_market_features(
        dataset=feature_dataset,
        asset_prefixes=["arg", "ced", "sp500", "gold"],
        macro_columns=["usd_ars", "vix"],
    )

    assert features.loc["2021-01-31", "arg_mom_3m"] == pytest.approx(
        220.0 / 190.0 - 1.0,
    )
    assert features.loc["2021-01-31", "arg_vol_3m"] == pytest.approx(
        feature_dataset["r_arg"].iloc[10:13].std(),
    )
    assert features.loc["2021-01-31", "usd_ars"] == pytest.approx(112.0)


def test_state_builder_appends_portfolio_weights(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset)
    state = state_builder.build_state(
        row_index=state_builder.first_valid_index(),
        portfolio_context=PortfolioContext(weights=[0.1, 0.2, 0.3, 0.4]),
    )

    assert len(state) == 8
    assert state[-4:] == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_state_builder_can_build_normalized_binned_state(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset, feature_mode="normalized_binned")
    first_valid_index = state_builder.first_valid_index()
    state = state_builder.build_state(
        row_index=first_valid_index,
        portfolio_context=PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
    )
    market_values = state[:-4]

    assert "usd_ars_change_zbin" in state_builder.market_features.columns
    assert "usd_ars_zbin" not in state_builder.market_features.columns
    assert len(state) == 26
    assert all(
        value in {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0} for value in market_values
    )


def test_state_builder_defaults_to_compact_regime_state(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset)
    first_valid_index = state_builder.first_valid_index()
    state = state_builder.build_state(
        row_index=first_valid_index,
        portfolio_context=PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
    )

    assert state_builder.feature_mode == COMPACT_REGIME_FEATURE_MODE
    assert list(state_builder.market_features.columns) == [
        "momentum_leader",
        "momentum_strength",
        "usd_ars_regime",
        "vix_regime",
    ]
    assert len(state) == 8
    assert state[0] in {0.0, 1.0, 2.0, 3.0}
    assert all(value in {0.0, 1.0, 2.0} for value in state[1:-4])


def test_state_builder_keeps_detailed_tercile_mode_available(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset, feature_mode="tercile_binned")

    assert "usd_ars_change_tercile" in state_builder.market_features.columns
    assert len(state_builder.market_features.columns) == 22


def test_compact_regime_does_not_use_future_observations(
    feature_dataset: pd.DataFrame,
) -> None:
    altered_future = feature_dataset.copy()
    altered_future.loc["2021-02-28":, "price_arg"] *= 100.0
    altered_future.loc["2021-02-28":, "vix"] *= 10.0

    original = StateBuilder(feature_dataset)
    altered = StateBuilder(altered_future)

    assert original.market_features.loc[:"2021-01-31"].equals(
        altered.market_features.loc[:"2021-01-31"],
    )


def test_market_features_can_use_usd_ars_change(
    feature_dataset: pd.DataFrame,
) -> None:
    features = build_market_features(
        dataset=feature_dataset,
        asset_prefixes=["arg", "ced", "sp500", "gold"],
        macro_columns=["usd_ars", "vix"],
        macro_mode="change",
    )

    assert "usd_ars_change" in features.columns
    assert "usd_ars" not in features.columns
    assert features.loc["2020-02-29", "usd_ars_change"] == pytest.approx(0.01)


def test_expanding_zscores_do_not_use_future_observations() -> None:
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 100.0]})

    z_scores = build_expanding_zscores(features)

    assert z_scores.loc[2, "feature"] == pytest.approx((3.0 - 2.0) / 0.8164965809)


def test_expanding_terciles_do_not_use_future_observations() -> None:
    features = pd.DataFrame({"feature": [3.0, 1.0, 2.0, 100.0]})

    terciles = build_expanding_tercile_features(features)

    assert list(terciles.loc[:2, "feature_tercile"]) == [0.0, 0.0, 1.0]


def test_state_builder_uses_prior_history_for_out_of_sample_rows(
    feature_dataset: pd.DataFrame,
) -> None:
    history = feature_dataset.iloc[:13]
    target = feature_dataset.iloc[13:]

    state_builder = StateBuilder(target, history_dataset=history)

    assert state_builder.first_valid_index() == 0
    assert state_builder.market_features.index.equals(target.index)


def test_state_builder_rejects_overlapping_history(
    feature_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="must end before"):
        StateBuilder(
            feature_dataset.iloc[12:],
            history_dataset=feature_dataset.iloc[:13],
        )


def test_state_builder_rejects_rows_with_incomplete_history(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset)

    with pytest.raises(ValueError, match="missing features"):
        state_builder.build_state(
            row_index=0,
            portfolio_context=PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
        )
