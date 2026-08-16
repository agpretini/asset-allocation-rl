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
- `runs/`: complete model-run artifacts, including the selected Model B run.
- `forecasts/`: standard forward-looking allocation outputs.
- `diagnostics/`: focused controlled investigations of policy behaviour.

The selected product model is fully self-contained in
`runs/model_b_synthetic_trial_169/`; do not scatter its artifacts across topic
folders. Trial 371 remains a legacy research run.
