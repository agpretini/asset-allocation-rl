# Portfolio MDP

Research project for dynamic allocation across four asset classes using a
monthly Markov Decision Process.

## Development

Install the project and development tools:

```bash
uv sync
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Build the processed monthly dataset:

```bash
uv run python scripts/build_dataset.py
```

Generate the passive benchmark report:

```bash
uv run python scripts/run_benchmarks.py
```

Run random and rule-based baseline backtests:

```bash
uv run python scripts/run_backtest.py
```

Train the first RL agent:

```bash
uv run python scripts/train_agent.py
```

Run walk-forward evaluation:

```bash
uv run python scripts/run_walk_forward.py
```

Track the current Q-learning evaluation as an experiment:

```bash
uv run python scripts/evaluate_results.py
```

Run Q-learning hyperparameter optimization:

```bash
uv run python scripts/optimize_hyperparameters.py
```

Analyze portfolio behavior:

```bash
uv run python scripts/analyze_portfolio.py
```

Compare the model against all baselines:

```bash
uv run python scripts/compare_results.py
```

The data layer downloads real Yahoo Finance close prices for the configured
asset universe, then builds monthly USD return series for the Argentine equity
basket, CEDEAR basket, SPY, and GLD.

Generated artifacts:

- `data/processed/dataset/monthly_dataset.parquet`
- `data/processed/benchmarks/benchmark_report.csv`
- `data/processed/baselines/baseline_report.csv`
- `data/processed/q_learning/q_learning_report.csv`
- `data/processed/walk_forward/walk_forward_report.csv`
- `data/processed/optimization/hyperparameter_results.csv`
- `data/processed/analysis/portfolio_analysis.md`
- `data/processed/comparison/model_vs_baselines.md`

See `data/README.md` for the full output directory map.

## How To Evaluate The Model

The model should not be judged only by its standalone return. It should be
evaluated against passive and rule-based alternatives over the same dates.

The main comparison report is:

```text
data/processed/comparison/model_vs_baselines.md
```

Regenerate it with:

```bash
uv run python scripts/compare_results.py
```

Use these questions to decide whether the model is useful:

- Does it beat `sp500` and `sp500_gold_50_50` on CAGR?
- Does it beat simple rule-based baselines such as `equal_weight`,
  `momentum`, and `volatility_weighted`?
- Is its Sharpe ratio competitive, not just its raw return?
- Is maximum drawdown acceptable relative to the benchmarks?
- Is turnover low enough to be realistic after transaction costs?
- Does walk-forward performance remain stable across splits?

Important reports:

- `data/processed/comparison/model_vs_baselines.md`: model vs all baselines.
- `data/processed/walk_forward/walk_forward_report.csv`: robustness across
  rolling historical windows.
- `data/processed/analysis/portfolio_analysis.md`: allocation behavior,
  turnover, drawdowns, and regime behavior.
- `data/processed/optimization/hyperparameter_results.csv`: whether better
  hyperparameters improved walk-forward metrics.

A model is more convincing when it improves risk-adjusted metrics and drawdown,
not only CAGR. A high-CAGR model with poor Sharpe, large drawdowns, or very high
turnover is probably not robust enough for real allocation decisions.

## How To Use It For Future Decisions

The intended practical use is monthly asset-allocation support, not automatic
trading.

At the end of each month:

1. Rebuild the dataset with the latest available Yahoo Finance data.

   ```bash
   uv run python scripts/build_dataset.py
   ```

2. Retrain or reload the latest selected agent.

   ```bash
   uv run python scripts/train_agent.py
   ```

3. Inspect the latest action selected by the model in:

   ```text
   data/processed/q_learning/q_learning_paths.csv
   ```

   The relevant fields are:

   - `date`
   - `action_id`
   - `weights`
   - `turnover`
   - `rebalance_cost`
   - `portfolio_value`
   - `drawdown`

4. Interpret `weights` as the model's suggested target allocation for the next
   monthly period across:

   ```text
   Argentine Equities, CEDEARs, S&P500, Gold
   ```

5. Compare that suggested allocation with the baseline reports and portfolio
   analysis before acting.

The model output should be treated as a decision-support signal. In practice,
you would still check:

- whether the suggested turnover is too high;
- whether transaction costs make the rebalance unattractive;
- whether the allocation is too concentrated for your risk tolerance;
- whether the model is behaving reasonably in the current regime;
- whether recent performance is consistent with walk-forward results.

For real use, a conservative workflow is to rebalance only when the new target
allocation differs meaningfully from the current allocation, and only if the
expected improvement justifies transaction costs.

## Current Interpretation

The current Q-learning agent is a first research baseline, not a production
allocator.

In the latest generated comparison, it beats some passive benchmarks on CAGR,
but it does not dominate the stronger rule-based baselines on risk-adjusted
metrics. Its turnover is also high, which is a warning sign because frequent
rebalancing can make live performance worse after real-world frictions.

That means the current model is useful for validating the pipeline and studying
behavior, but it should not yet be treated as a final allocation strategy.

Implemented roadmap status:

- Milestone 0: project setup
- Milestone 1: real monthly data layer
- Milestone 2: passive benchmark framework
- Milestone 3: discrete action space
- Milestone 4: portfolio environment
- Milestone 5: feature/state builder
- Milestone 6: random agent baseline
- Milestone 7: rule-based baselines
- Milestone 8: first RL agent
- Milestone 9: walk-forward evaluation
- Milestone 10: experiment tracking
- Milestone 11: hyperparameter optimization
- Milestone 12: portfolio analysis

The environment follows the project Gym-like interface:

```python
state = env.reset()
next_state, reward, done, info = env.step(action_id)
```

Current caveat: inflation is part of the product-spec target state, but no
concrete Yahoo Finance source is configured for it yet. The current real-data
state uses `usd_ars` and `vix` as macro features.
