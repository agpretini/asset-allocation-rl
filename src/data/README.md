# Data Library

This package downloads configured Yahoo Finance prices, creates causal monthly
returns and market levels, validates the result, and writes the canonical
dataset. Use `uv run python src/data/build_dataset.py` to rebuild
`data/processed/dataset/monthly_dataset.parquet`.

`build_monthly_dataset(end_date=...)` also builds an in-memory dataset for the
API without replacing the canonical research input. The exclusive end date
limits downloads and selects completed calendar months. Without one, the
current month is excluded using the Buenos Aires date. A completed month is
kept when its final trading day precedes a weekend or holiday at month-end.
