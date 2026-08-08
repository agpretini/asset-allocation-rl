# Analysis

This directory contains every reproducible analytical artifact and its runner.
It is separate from `data/processed`, which contains only the canonical monthly
dataset. Run scripts from the repository root.

## Layout

- `baselines/`: passive and rule-based reference strategies.
- `walk_forward/`: standard Q-learning walk-forward evaluation.
- `feature_modes/`: state-representation comparison.
- `portfolio/`: allocation, turnover, drawdown, and regime analysis.
- `comparison/`: standard model-versus-baseline reports.
- `convergence/`: synthetic-training convergence studies.
- `hyperparameter_search/`: HPO results and immutable snapshots.
- `runs/`: complete model-run artifacts, including the selected synthetic run.
- `forecasts/`: standard forward-looking allocation outputs.

The selected model is fully self-contained in
`runs/selected_synthetic_trial_371/`; do not scatter new artifacts for that run
across topic folders.
