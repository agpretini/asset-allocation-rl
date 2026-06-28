# Scripts

This directory contains command-line entry points for rebuilding datasets,
running backtests, training agents, evaluating results, and generating reports.

Run scripts from the repository root with `uv run python`.

## Recommended Order

```bash
uv run python scripts/build_dataset.py
uv run python scripts/run_benchmarks.py
uv run python scripts/run_backtest.py
uv run python scripts/train_agent.py
uv run python scripts/run_walk_forward.py
uv run python scripts/optimize_hyperparameters.py
uv run python scripts/evaluate_results.py
uv run python scripts/analyze_portfolio.py
uv run python scripts/compare_results.py
```

## Script Reference

### `build_dataset.py`

Downloads Yahoo Finance data and builds the canonical monthly dataset.

Output:

- `data/processed/dataset/monthly_dataset.parquet`

### `run_benchmarks.py`

Runs passive benchmark portfolios.

Output:

- `data/processed/benchmarks/benchmark_report.csv`

### `run_backtest.py`

Runs random and rule-based baseline strategies.

Outputs:

- `data/processed/baselines/baseline_report.csv`
- `data/processed/baselines/baseline_paths.csv`

### `train_agent.py`

Trains and evaluates the first RL agent, currently tabular Q-learning.

Outputs:

- `data/processed/q_learning/q_learning_model.json`
- `data/processed/q_learning/q_learning_training_history.csv`
- `data/processed/q_learning/q_learning_report.csv`
- `data/processed/q_learning/q_learning_paths.csv`

### `run_walk_forward.py`

Runs walk-forward Q-learning evaluation.

Outputs:

- `data/processed/walk_forward/walk_forward_report.csv`
- `data/processed/walk_forward/walk_forward_paths.csv`

### `optimize_hyperparameters.py`

Runs Q-learning hyperparameter optimization over walk-forward splits.

Outputs:

- `data/processed/optimization/hyperparameter_results.csv`
- `data/processed/optimization/best_hyperparameters.yaml`
- `experiments/hpo_###/`

### `evaluate_results.py`

Stores the current Q-learning evaluation as a tracked experiment.

Output:

- `experiments/q_learning_###/`

### `analyze_portfolio.py`

Analyzes Q-learning portfolio behavior.

Outputs:

- `data/processed/analysis/portfolio_analysis.md`
- `data/processed/analysis/portfolio_allocations.csv`
- `data/processed/analysis/allocation_summary.csv`
- `data/processed/analysis/regime_summary.csv`
- `data/processed/analysis/turnover_summary.csv`
- `data/processed/analysis/drawdown_summary.csv`

### `compare_results.py`

Builds one consolidated model-vs-baselines report.

Outputs:

- `data/processed/comparison/model_vs_baselines.md`
- `data/processed/comparison/model_vs_baselines_report.csv`
- `data/processed/comparison/model_vs_baselines_deltas.csv`
