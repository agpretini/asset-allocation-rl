# Production Source

This directory contains reusable implementation code. Entry points for the
canonical dataset, default agent training, and standard prediction are located
in `data/build_dataset.py`, `agents/train.py`, and
`prediction/run_prediction.py`, respectively. Analytical runners do not live
here; they belong under `analysis/`. Execute them directly as
`uv run python src/<package>/<entry_point>.py`.

`api/` provides the local FastAPI entry point. Its HTTP layer delegates to a
small service that calls the existing data and model functions and persists
local snapshots under `artifacts/`. See `api/README.md` for local and Docker
commands.
