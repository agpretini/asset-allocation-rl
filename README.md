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

The selected operational model is documented and executed from:

```bash
uv run python analysis/runs/selected_synthetic_trial_371/run.py
```

It writes all validation, test, policy, and forecast artifacts beside that
runner. The current August 2026 infinite-horizon signal is 100% S&P 500 from
the July 2026 close; see its dated forecast directory for assumptions.

## Model Summary

The default tabular state is `compact_regime`: momentum leader and strength,
USD/ARS regime, VIX regime, and current allocation. It has a 286-allocation
action grid and a 30,888 action-grid state table. Rewards use CRRA utility with
relative risk aversion `gamma = 0.5`; performance reports use financial net
returns rather than utility.

Synthetic training fits a three-regime VIX model only on the available training
data, simulates primitive monthly variables lazily, and reconstructs features
through the same causal pipeline. Validation and test remain historical.

## Methodological Note

Trial 371 was selected from HPO walk-forward test metrics. It is useful for
operational analysis, but future research should reserve a final untouched
holdout or select configurations from validation-only metrics.
