# Data

`data/processed/dataset/monthly_dataset.parquet` is the canonical monthly
dataset. Build it with `uv run python src/data/build_dataset.py`.

This directory deliberately does not store reports, trained models, forecasts,
or experiment results. Those reproducible analytical artifacts are organized
under `analysis/`.
