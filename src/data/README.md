# Data Layer

This module builds the Milestone 1 dataset from Yahoo Finance.

The pipeline downloads daily close prices, resamples them to month-end,
imputes missing prices from prior observations only, computes simple returns
from prices at `t-1` to `t`, and writes
`data/processed/dataset/monthly_dataset.parquet`.

Returns are calculated only from current and prior prices, so the processed
dataset does not require future observations.

Argentine equities and CEDEARs are represented as equal-weight baskets defined
in `docs/DATA_DECISIONS.md`.

Current generated outputs:

- `data/processed/dataset/monthly_dataset.parquet`
- `data/processed/benchmarks/benchmark_report.csv`
- `data/processed/baselines/baseline_report.csv`
- `data/processed/baselines/baseline_paths.csv`

The real Yahoo Finance dataset currently includes `usd_ars` and `vix` macro
columns. Inflation remains unimplemented until a concrete source is configured.
