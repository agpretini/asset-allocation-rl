"""Tests for feature engineering and state construction."""

import pandas as pd
import pytest

from features.market_features import build_market_features
from features.state_builder import PortfolioContext, StateBuilder


@pytest.fixture
def feature_dataset() -> pd.DataFrame:
    """Create a deterministic dataset with enough history for 12M features."""
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

    assert features.loc["2021-01-31", "arg_mom_12m"] == pytest.approx(1.2)
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

    assert len(state) == 34
    assert state[-4:] == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_state_builder_rejects_rows_with_incomplete_history(
    feature_dataset: pd.DataFrame,
) -> None:
    state_builder = StateBuilder(feature_dataset)

    with pytest.raises(ValueError, match="missing features"):
        state_builder.build_state(
            row_index=0,
            portfolio_context=PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
        )
