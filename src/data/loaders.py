"""Load configured raw market and macro data."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

CONFIG_PATH = Path("src/config/assets.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project asset, path, and benchmark configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("asset configuration must be a mapping")
    return config


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load raw monthly prices from CSV."""
    return pd.read_csv(path, parse_dates=["date"])


def load_macro(path: str | Path) -> pd.DataFrame:
    """Load raw monthly macro variables from CSV."""
    return pd.read_csv(path, parse_dates=["date"])


def get_configured_tickers(config: dict[str, Any]) -> list[str]:
    """Return every Yahoo ticker required by the configured dataset."""
    tickers: list[str] = []
    for asset_key in config["asset_order"]:
        tickers.extend(config["assets"][asset_key]["tickers"])

    macro_sources = config.get("macro_sources", {})
    tickers.extend(source["ticker"] for source in macro_sources.values())
    return list(dict.fromkeys(tickers))
