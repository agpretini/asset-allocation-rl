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

## How the Model Works

The selected formulation is **Model B**, a tabular Q-learning allocation model.
At each month it observes a compact market regime, selects one of 286 discrete
four-asset target allocations, receives a CRRA utility reward, and updates its
Q-table from the observed transition.

The learned state is intentionally market-only (`compact_regime`):

- momentum leader: Argentine equities, CEDEARs, S&P 500, or Gold;
- momentum strength: low, medium, or high;
- USD/ARS monthly-change regime: low, medium, or high; and
- VIX regime: low, medium, or high.

This produces 108 possible market states (`4 × 3 × 3 × 3`) rather than a
separate learned state for every starting portfolio. The 286 target portfolios
are allocations on a 10% weight grid that sum to 100%.

The training reward is CRRA utility of the portfolio's **net** monthly return,
with relative risk aversion `gamma = 0.5`; rebalancing cost is charged through
turnover. Reporting is deliberately separate: CAGR, Sharpe, drawdown, and
other evaluation metrics use financial net returns, not utility.

At historical evaluation time, Model B keeps the actual portfolio outside the
learned state solely to price execution. It selects the target maximizing
`Q(market_state, target) − rebalance_cost(current, target)`. For a published
single allocation signal, the same decision is evaluated from all 286 permitted
origins and the deterministic modal target is returned. This makes the policy
interpretable while retaining rebalancing awareness.

## Synthetic Training

Synthetic episodes expand the experience available to Q-learning without
turning validation or test into simulations. For every training window, the
simulator is fitted only on the observations available in that window:

1. Historical VIX terciles define low-, medium-, and high-volatility regimes.
2. A smoothed Markov chain estimates how regimes transition month to month.
3. Each regime gets a regularized multivariate Student-t distribution (`nu=8`
   by default) over primitive monthly variables: log returns for the four
   assets, log change in USD/ARS, and log change in VIX.
4. A lazy, seeded generator samples one regime path and one return vector per
   month, reconstructs prices/USD-ARS/VIX, and passes them through the same
   causal feature and `StateBuilder` pipeline used by historical data.

The agent updates its Q-table from these streaming synthetic episodes, so tens
of thousands of episodes do not need to live in memory. No future historical
observations are used to fit the simulator or calculate state thresholds.

Validation remains a frozen-policy backtest on real validation observations;
test retrains from scratch on train plus validation and is evaluated only on
real test observations. The full-data forecast follows the same rule: it fits
the simulator only on data available at the signal date.

