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
uv run python scripts/run_full_mode_hpo_comparison.py
uv run python scripts/compare_feature_modes.py
uv run python scripts/optimize_hyperparameters.py
uv run python scripts/evaluate_results.py
uv run python scripts/analyze_portfolio.py
uv run python scripts/compare_results.py
uv run python scripts/predict_next_allocation.py
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

Use `--training-mode synthetic` to fit the simulator on the available training
data and consume its episode stream. The default is `historical`.

Outputs:

- `data/processed/q_learning/q_learning_model.json`
- `data/processed/q_learning/q_learning_training_history.csv`
- `data/processed/q_learning/q_learning_report.csv`
- `data/processed/q_learning/q_learning_paths.csv`

### `run_walk_forward.py`

Runs walk-forward Q-learning evaluation.

Use `--training-mode synthetic` to fit a separate simulator on every training
window. Test windows and all reported metrics remain historical.

Outputs:

- `data/processed/walk_forward/walk_forward_report.csv`
- `data/processed/walk_forward/walk_forward_paths.csv`

### `compare_feature_modes.py`

Runs the same walk-forward Q-learning evaluation for:

- `raw`: original unnormalized market features.
- `normalized_binned`: USD/ARS change, expanding z-score normalization, and
  discrete financial regime bins.
- `tercile_binned`: USD/ARS monthly change plus expanding historical terciles
  for all market variables: low (`0`), medium (`1`), and high (`2`). This is
  the default state mode.

Outputs:

- `data/processed/feature_modes/feature_mode_report.csv`
- `data/processed/feature_modes/feature_mode_paths.csv`
- `data/processed/feature_modes/feature_mode_decisions.csv`

### `optimize_hyperparameters.py`

Runs Q-learning hyperparameter optimization over walk-forward splits.

Outputs:

- `data/processed/optimization/hyperparameter_results.csv`
- `data/processed/optimization/best_hyperparameters.yaml`
- `experiments/hpo_###/`

### `run_full_mode_hpo_comparison.py`

Extends the `hpo_004` grid with 18,000 and 30,000 episodes, `epsilon=0.50`,
and `epsilon_decay=0.999954`, then evaluates every cross-product combination
in both `historical` and `synthetic` modes. It writes a resumable tracked
experiment under `experiments/hpo_mode_comparison_###/`; `results.csv` is
updated after every completed trial.

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

### `predict_next_allocation.py`

Trains Q-learning on all available data using the best known hyperparameters
and writes a forward-looking allocation signal for the next month.

By default it uses the compact-regime tabular state. To generate the
normalized-binned version without overwriting the default prediction:

```bash
uv run python scripts/predict_next_allocation.py --feature-mode normalized_binned
```

Outputs:

- `data/processed/predictions/YYYY-MM/allocation_predictions.csv`
- `data/processed/predictions/YYYY-MM/allocation_prediction.csv`
- `data/processed/predictions/YYYY-MM/prediction_metadata.json`
- `data/processed/predictions/YYYY-MM/trained_model.json`
- `data/processed/predictions/YYYY-MM/training_history.csv`
- `data/processed/predictions/YYYY-MM/<feature_mode>/` when using a
  non-default feature mode
