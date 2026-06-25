"""Build the processed monthly dataset used by benchmarks and environments."""

from pathlib import Path

import pandas as pd

from data.loaders import CONFIG_PATH, get_configured_tickers, load_config
from data.preprocessing import (
    impute_missing_prices,
    validate_complete_monthly_index,
    validate_no_missing_values,
)
from data.returns import build_price_index, calculate_equal_weight_return
from data.yahoo import download_close_prices, resample_to_month_end


def build_monthly_dataset(
    config_path: Path = CONFIG_PATH,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Build a clean monthly dataset from Yahoo Finance data."""
    config = load_config(config_path)
    tickers = get_configured_tickers(config)
    daily_prices = download_close_prices(
        tickers=tickers,
        start_date=config["start_date"],
        end_date=end_date,
    )
    monthly_prices = impute_missing_prices(resample_to_month_end(daily_prices))
    component_returns = monthly_prices.pct_change()

    asset_returns = _build_asset_returns(component_returns, config)
    asset_prices = _build_asset_price_indexes(asset_returns, config)
    macro_data = _build_macro_data(monthly_prices, config)

    dataset = pd.concat([asset_prices, asset_returns, macro_data], axis=1)
    dataset = dataset.iloc[1:]

    validate_complete_monthly_index(dataset)
    validate_no_missing_values(dataset)
    return dataset


def write_monthly_dataset(
    config_path: Path = CONFIG_PATH,
    output_path: str | Path | None = None,
) -> Path:
    """Build and write the processed monthly dataset as parquet."""
    config = load_config(config_path)
    destination = Path(output_path or config["processed_dataset_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_monthly_dataset(config_path)
    dataset.to_parquet(destination)
    return destination


def _build_asset_returns(
    component_returns: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build configured asset-class returns from component returns."""
    returns = []
    for asset_key in config["asset_order"]:
        asset = config["assets"][asset_key]
        returns.append(
            calculate_equal_weight_return(
                component_returns[asset["tickers"]],
                output_column=asset["return_column"],
            ),
        )
    return pd.concat(returns, axis=1)


def _build_asset_price_indexes(
    asset_returns: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build price indexes for configured asset classes."""
    price_indexes = []
    for asset_key in config["asset_order"]:
        asset = config["assets"][asset_key]
        price_indexes.append(
            build_price_index(
                asset_returns[asset["return_column"]],
                output_column=asset["price_column"],
            ),
        )
    return pd.concat(price_indexes, axis=1)


def _build_macro_data(monthly_prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build configured macro columns from Yahoo Finance tickers."""
    macro_columns = {}
    for column, source in config.get("macro_sources", {}).items():
        macro_columns[column] = monthly_prices[source["ticker"]]
    return pd.DataFrame(macro_columns, index=monthly_prices.index)
