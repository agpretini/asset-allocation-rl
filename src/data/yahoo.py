"""Yahoo Finance data access."""

from collections.abc import Sequence

import pandas as pd
import yfinance as yf


def download_close_prices(
    tickers: Sequence[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily Yahoo Finance close prices for the requested tickers."""
    if not tickers:
        raise ValueError("tickers cannot be empty")

    downloaded = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if downloaded.empty:
        raise ValueError("Yahoo Finance returned no data")

    close_prices = _extract_close_prices(downloaded, tickers)
    close_prices.index = pd.to_datetime(close_prices.index)
    close_prices.index.name = "date"
    return close_prices.sort_index()


def resample_to_month_end(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily close prices to completed month-end close prices."""
    if not isinstance(daily_prices.index, pd.DatetimeIndex):
        raise ValueError("daily_prices must use a DatetimeIndex")

    monthly_prices = daily_prices.resample("ME").last()
    last_observation_date = daily_prices.index.max().normalize()
    last_month_end = last_observation_date + pd.offsets.MonthEnd(0)
    if last_observation_date < last_month_end:
        monthly_prices = monthly_prices.loc[monthly_prices.index < last_month_end]

    return monthly_prices


def _extract_close_prices(
    downloaded: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Extract Yahoo's unadjusted close prices from yfinance output."""
    if isinstance(downloaded.columns, pd.MultiIndex):
        close_prices = downloaded["Close"]
    else:
        close_prices = downloaded[["Close"]]
        close_prices.columns = [tickers[0]]

    return close_prices.reindex(columns=list(tickers))
