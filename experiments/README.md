# Experiment Runners

This directory contains runners, not analytical results.

- `hpo/run.py`: compact-state hyperparameter search.
- `mode_comparison/run.py`: resumable Model B synthetic-only HPO using the
  prior mode-comparison grid as reference.
- `tracking/run.py`: snapshot the default model run.

Run outputs are stored in `analysis/hyperparameter_search/` or in the relevant
model run under `analysis/runs/`. For example:

```bash
uv run python experiments/hpo/run.py
uv run python experiments/mode_comparison/run.py
```
