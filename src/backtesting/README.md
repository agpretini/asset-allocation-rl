# Backtesting Layer

This module contains benchmark and baseline backtesting utilities.

Implemented pieces:

- `benchmarks.py`: configured passive benchmarks, including 100% S&P500 and
  50% S&P500 / 50% Gold
- `agents.py`: runs environment-facing agents through `PortfolioEnv`
- `strategies.py`: rule-based baselines, including equal weight, momentum, and
  inverse-volatility allocation
- `engine.py`: fixed-weight and rule-based backtest engines
- `splitter.py`: rolling train/validation/test walk-forward split generation
- `walk_forward.py`: walk-forward Q-learning evaluation
- `feature_modes.py`: side-by-side walk-forward comparison of state feature
  representations

Backtest outputs include gross monthly returns, net portfolio returns where
transaction costs apply, CRRA rewards (`γ = 0.5`), turnover, rebalancing costs,
portfolio value, and drawdown. Performance metrics use net portfolio returns,
not utility rewards.

Walk-forward test states retain train-plus-validation history for feature
calculation. Expanding tercile thresholds therefore use all information known
before the test window and update only as each test month becomes available.

With `training_mode="synthetic"`, every split fits its simulator only on
`train_dataset`. The test environment still uses real `test_dataset` rows and
real train-plus-validation history, so reported metrics never use synthetic
observations.

The `scripts/run_backtest.py` command generates:

- `data/processed/baselines/baseline_report.csv`
- `data/processed/baselines/baseline_paths.csv`

The `scripts/run_walk_forward.py` command generates:

- `data/processed/walk_forward/walk_forward_report.csv`
- `data/processed/walk_forward/walk_forward_paths.csv`

The `scripts/compare_feature_modes.py` command generates:

- `data/processed/feature_modes/feature_mode_report.csv`
- `data/processed/feature_modes/feature_mode_paths.csv`

Walk-forward reports are also used by the hyperparameter optimizer to compare
Q-learning configurations reproducibly.
