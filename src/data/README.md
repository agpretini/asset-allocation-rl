# Data Library

This package downloads configured Yahoo Finance prices, creates causal monthly
returns and market levels, validates the result, and writes the canonical
dataset. Use `uv run python src/data/build_dataset.py` to rebuild
`data/processed/dataset/monthly_dataset.parquet`.
