# Data Layer

This module builds the Milestone 1 dataset from Yahoo Finance.

The pipeline downloads daily close prices, resamples them to month-end,
imputes missing prices from prior observations only, computes simple returns
from prices at `t-1` to `t`, and writes
`data/processed/monthly_dataset.parquet`.

Returns are calculated only from current and prior prices, so the processed
dataset does not require future observations.

Argentine equities and CEDEARs are represented as equal-weight baskets defined
in `docs/DATA_DECISIONS.md`.
