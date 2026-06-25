"""Tests for the monthly dataset pipeline."""

from pathlib import Path

import pandas as pd
import pytest

import data.dataset as dataset_module
from data.dataset import build_monthly_dataset, write_monthly_dataset


@pytest.fixture
def yahoo_prices(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Patch Yahoo downloads with deterministic daily close prices."""
    prices = pd.DataFrame(
        {
            "GGAL": [10.0, 11.0, 12.0, 13.0],
            "YPF": [20.0, 22.0, 24.0, 26.0],
            "CRESY": [30.0, 33.0, 36.0, 39.0],
            "LOMA": [40.0, 44.0, 48.0, 52.0],
            "TX": [50.0, 55.0, 60.0, 65.0],
            "GOOGL": [100.0, 105.0, 110.0, 115.0],
            "ASML": [100.0, 105.0, 110.0, 115.0],
            "MELI": [100.0, 105.0, 110.0, 115.0],
            "META": [100.0, 105.0, 110.0, 115.0],
            "MSFT": [100.0, 105.0, 110.0, 115.0],
            "NVDA": [100.0, 105.0, 110.0, 115.0],
            "KO": [100.0, 105.0, 110.0, 115.0],
            "SPY": [400.0, 420.0, 441.0, 463.05],
            "GLD": [180.0, 171.0, 179.55, 188.5275],
            "USDARS=X": [800.0, 820.0, 840.0, 860.0],
            "^VIX": [20.0, 21.0, 19.0, 18.0],
        },
        index=pd.to_datetime(
            ["2016-01-29", "2016-02-29", "2016-03-31", "2016-04-29"],
        ),
    )
    prices.index.name = "date"

    def fake_download_close_prices(
        tickers: list[str],
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        del start_date, end_date
        return prices[tickers]

    monkeypatch.setattr(
        dataset_module,
        "download_close_prices",
        fake_download_close_prices,
    )
    return prices


def test_build_monthly_dataset_has_expected_columns(yahoo_prices: pd.DataFrame) -> None:
    del yahoo_prices
    dataset = build_monthly_dataset()

    assert list(dataset.columns) == [
        "price_arg",
        "price_ced",
        "price_sp500",
        "price_gold",
        "r_arg",
        "r_ced",
        "r_sp500",
        "r_gold",
        "usd_ars",
        "vix",
    ]


def test_monthly_dataset_has_no_missing_dates_or_values(
    yahoo_prices: pd.DataFrame,
) -> None:
    del yahoo_prices
    dataset = build_monthly_dataset()
    expected_index = pd.date_range(dataset.index.min(), dataset.index.max(), freq="ME")

    assert dataset.index.equals(expected_index)
    assert not dataset.isna().any().any()


def test_monthly_returns_use_current_and_prior_prices_only(
    yahoo_prices: pd.DataFrame,
) -> None:
    del yahoo_prices
    dataset = build_monthly_dataset()

    assert dataset.loc[pd.Timestamp("2016-02-29"), "r_arg"] == pytest.approx(0.10)
    assert dataset.loc[pd.Timestamp("2016-02-29"), "r_sp500"] == pytest.approx(0.05)
    assert dataset.loc[pd.Timestamp("2016-02-29"), "r_gold"] == pytest.approx(-0.05)


def test_incomplete_current_month_is_excluded(yahoo_prices: pd.DataFrame) -> None:
    del yahoo_prices
    dataset = build_monthly_dataset()

    assert dataset.index.max() == pd.Timestamp("2016-03-31")


def test_monthly_dataset_can_be_written_as_parquet(
    tmp_path: Path,
    yahoo_prices: pd.DataFrame,
) -> None:
    del yahoo_prices
    output_path = write_monthly_dataset(
        output_path=tmp_path / "monthly_dataset.parquet",
    )

    saved = pd.read_parquet(output_path)
    assert len(saved) == len(build_monthly_dataset())
