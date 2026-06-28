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
  and saves optimization artifacts

Generated outputs include:

- `data/processed/optimization/hyperparameter_results.csv`
- `data/processed/optimization/best_hyperparameters.yaml`
- `experiments/q_learning_*/`
- `experiments/hpo_*/`
