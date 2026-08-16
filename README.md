# Portfolio MDP

Research repository for monthly four-asset allocation with tabular Q-learning,
historical backtesting, and a regime-based synthetic market simulator.

## Repository Contract

- `src/` contains reusable production code, including dataset generation,
  training, and prediction entry points.
- `data/processed/` contains only the canonical monthly dataset.
- `analysis/` contains all generated analytical outputs and their topic runners.
- `experiments/` contains HPO and tracking runners; their results are written
  under `analysis/hyperparameter_search/`.
- `tests/` contains the pytest suite.

Start with [analysis/README.md](analysis/README.md) for results and with the
README in each `src` package for the production pipeline.

## Main Commands

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python src/data/build_dataset.py
uv run python src/agents/train.py
uv run python analysis/baselines/run_benchmarks.py
uv run python analysis/walk_forward/run.py
uv run python experiments/hpo/run.py
```

The selected operational Model B configuration is documented and executed from:

```bash
caffeinate -i uv run python analysis/runs/run.py
```

It derives its destination from `run_label` in the selected-model YAML and
writes validation, test, policy, and forecast artifacts together. The current
selected run is `analysis/runs/model_b_synthetic_trial_076/`. Its August 2026
strategic Model B signal, from the July 2026 close, is 100% Gold; see its dated
forecast directory for the full decision record and assumptions.

## Model Summary

The default tabular state is Model B `compact_regime`: momentum leader and
strength, USD/ARS regime, and VIX regime. It has 108 market states and a
286-allocation action grid. The current allocation is excluded from the learned
state; at execution, each target is scored as Q-value minus immediate
rebalancing cost, and the published target is the deterministic modal choice
across all permitted origins. Rewards use CRRA utility with relative risk
aversion `gamma = 0.5`; performance reports use financial net returns rather
than utility.

Synthetic training fits a three-regime VIX model only on the available training
data, simulates primitive monthly variables lazily, and reconstructs features
through the same causal pipeline. Validation and test remain historical.

## Methodological Note

Trial 76 is the selected Model B configuration for product analysis. Its
initial HPO ranking was calculated before the cost-aware Model B evaluator was
corrected, so the committed production run must be interpreted as an honest
configuration evaluation rather than a claim of an untouched final holdout.
Future selection should use validation-only metrics and retain a final holdout.
