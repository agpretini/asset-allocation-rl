# Experiment Layer

This module implements Milestone 10 and Milestone 11.

Implemented pieces:

- `tracking.py`: creates numbered experiment folders and writes standard
  artifacts: `config.yaml`, `metrics.json`, `actions.csv`,
  `portfolio_values.csv`, and `notes.md`
- `optimizer.py`: runs a reproducible tabular Q-learning hyperparameter grid
  over walk-forward splits

Scripts:

- `scripts/evaluate_results.py`: saves the current Q-learning evaluation as a
  tracked experiment
- `scripts/optimize_hyperparameters.py`: runs a small Q-learning grid search

The grid compares constant learning rates with the visit-decay schedule. Its
requested candidate starts at `0.30` and uses
`learning_rate / (1 + 0.01 * visits(state, action))`.
It evaluates the default `compact_regime` state representation: momentum
leader, momentum strength, USD/ARS regime, VIX regime, and portfolio weights.
The detailed `tercile_binned` representation remains available for comparison
and the planned DQN work.
  and saves optimization artifacts

Generated outputs include:

- `data/processed/optimization/hyperparameter_results.csv`
- `data/processed/optimization/best_hyperparameters.yaml`
- `experiments/q_learning_*/`
- `experiments/hpo_*/`
