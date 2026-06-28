# Portfolio MDP - Roadmap

## Purpose

This document defines the implementation roadmap of the project.

The roadmap is organized into incremental milestones.

Each milestone should produce a working and testable artifact.

Progression to the next milestone should only happen after the current milestone is stable and validated.

---

# Guiding Principle

The project follows the philosophy:

```text
Data → Benchmarks → Environment → Agent → Backtesting → Optimization
```

A simple working system is preferred over an incomplete sophisticated system.

---

# Milestone 0 - Project Setup

## Goal

Create the project foundation and development environment.

## Deliverables

- Repository created.
- uv configured.
- Project structure created.
- Documentation initialized.
- pytest configured.
- ruff configured.
- Git repository initialized.

## Success Criteria

The following commands run successfully:

```bash
uv sync
pytest
ruff check .
ruff format .
```

---

# Milestone 1 - Data Layer

## Goal

Build a clean monthly dataset.

## Deliverables

### Data Sources

Select representative instruments for:

- Argentine Equities
- CEDEARs
- S&P500
- Gold

### Data Pipeline

Implement:

```text
src/data/
```

modules.

### Dataset

Generate:

```text
data/processed/dataset/monthly_dataset.parquet
```

containing:

- Monthly prices
- Monthly returns
- Macro variables

## Success Criteria

Dataset can be rebuilt from raw sources.

No missing dates.

No look-ahead bias.

---

# Milestone 2 - Benchmark Framework

## Goal

Implement benchmark portfolios.

## Deliverables

### Benchmark 1

```text
100% S&P500
```

### Benchmark 2

```text
50% S&P500
50% Gold
```

### Backtest Engine

Run benchmark portfolios through historical data.

### Metrics

Implement:

- CAGR
- Volatility
- Sharpe
- Sortino
- Maximum Drawdown

## Success Criteria

Benchmark performance report can be generated.

---

# Milestone 3 - Action Space

## Goal

Implement portfolio allocation space.

## Deliverables

### Action Generator

Generate all valid allocations.

Constraints:

```text
4 assets
10% increments
weights sum to 100%
```

Expected:

```text
286 actions
```

### Tests

Validate:

- Sum of weights.
- Number of actions.
- Weight discretization.

## Success Criteria

Action space generator passes all tests.

---

# Milestone 4 - Environment

## Goal

Implement the portfolio MDP environment.

## Deliverables

### PortfolioEnv

Implement:

```python
env.reset()
env.step(action_id)
```

### Reward Function

Implement:

```text
reward =
portfolio_return
-
rebalancing_cost
```

### Portfolio Accounting

Track:

- Portfolio value
- Turnover
- Costs
- Returns

## Success Criteria

Environment can simulate one full historical period.

---

# Milestone 5 - Feature Engineering

## Goal

Create the state representation.

## Deliverables

### Market Features

Initial candidates:

- 1M momentum
- 3M momentum
- 6M momentum
- 12M momentum
- Volatility
- USD/ARS
- Inflation
- VIX

### Portfolio Features

- Current allocation
- Drawdown
- Portfolio value

### State Builder

Implement:

```python
state_builder.build_state(...)
```

## Success Criteria

State vector is generated without leakage.

---

# Milestone 6 - Random Agent Baseline

## Goal

Validate the environment.

## Deliverables

Implement:

```text
Random Agent
```

that selects a random portfolio every month.

## Evaluation

Compare against:

- Benchmark 1
- Benchmark 2

## Success Criteria

Full training and evaluation pipeline executes end-to-end.

---

# Milestone 7 - Rule-Based Baselines

## Goal

Create stronger baselines before RL.

## Deliverables

### Equal Weight

```text
25 / 25 / 25 / 25
```

### Momentum Portfolio

Simple momentum-based allocation.

### Volatility-Based Allocation

Simple risk-adjusted allocation.

## Success Criteria

Rule-based strategies are fully backtested.

---

# Milestone 8 - First RL Agent

## Goal

Train the first RL model.

## Candidate

### DQN

Primary candidate.

Alternative:

### Q-Learning

If state discretization is feasible.

## Deliverables

- Training pipeline.
- Saved model.
- Evaluation report.

## Success Criteria

Agent completes training successfully.

---

# Milestone 9 - Walk-Forward Evaluation

## Goal

Evaluate robustness.

## Deliverables

### Rolling Windows

Example:

```text
Train
Validate
Test

Shift

Train
Validate
Test
```

### Multiple Regimes

Evaluate across:

- Bull markets
- Bear markets
- Sideways markets

## Success Criteria

Agent performance is stable across multiple windows.

---

# Milestone 10 - Experiment Tracking

## Goal

Create a reproducible experimentation framework.

## Deliverables

Store:

- Configurations
- Metrics
- Portfolio paths
- Actions selected

Example:

```text
experiments/
├── experiment_001/
├── experiment_002/
└── ...
```

## Success Criteria

Every result can be reproduced.

---

# Milestone 11 - Hyperparameter Optimization

## Goal

Improve agent performance.

## Candidate Areas

- Learning rate
- Network size
- Exploration schedule
- Feature selection

## Success Criteria

Performance improvement is measurable and reproducible.

---

# Milestone 12 - Portfolio Analysis

## Goal

Understand agent behavior.

## Deliverables

Analyze:

- Asset allocations over time.
- Portfolio turnover.
- Drawdowns.
- Regime behavior.

Questions:

- When does the agent increase Argentine exposure?
- When does the agent prefer gold?
- How stable are allocations?

## Success Criteria

Agent decisions become explainable.

---

# Future Ideas

Potential future extensions:

- Add Cash as an asset.
- Add Bonds.
- Add Bitcoin.
- Move from 10% to 5% allocation increments.
- Continuous action spaces.
- PPO.
- SAC.
- Dynamic transaction costs.
- Inflation-adjusted rewards.
- Multi-objective optimization.

---

# Definition of Project Success

The project is considered successful if:

1. The RL agent consistently outperforms passive benchmarks.
2. Performance is validated out-of-sample.
3. Results are reproducible.
4. Transaction costs are realistically modeled.
5. The agent remains interpretable.
6. The system can be extended without major redesign.
