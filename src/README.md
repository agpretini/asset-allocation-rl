# Production Source

This directory contains reusable implementation code. Entry points for the
canonical dataset, default agent training, and standard prediction are located
in `data/build_dataset.py`, `agents/train.py`, and
`prediction/run_prediction.py`, respectively. Analytical runners do not live
here; they belong under `analysis/`. Execute them directly as
`uv run python src/<package>/<entry_point>.py`.
