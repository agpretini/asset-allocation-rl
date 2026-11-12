# Portfolio MDP - Architecture

## Purpose

This document describes the technical architecture of the project.

While `docs/PRODUCT_SPEC.md` defines the modeling decisions, this document
defines how the system should be implemented. It is the source of truth for
module ownership, not for product-policy parameters.

“Model B” is the selected reduced-state formulation. Its definition and the
historical A/C comparators are introduced in `docs/PRODUCT_SPEC.md`; this
document describes only how B is partitioned into modules.

If an implementation detail conflicts with `PRODUCT_SPEC.md`, the product specification takes precedence.

---

# High-Level Architecture

The local HTTP adapter lives in `src/api/main.py`. Its service coordinates
existing dataset and model functions without implementing model logic.
`prediction/next_allocation.py` exposes separate training and inference
operations; research wrappers compose them. Named operational configurations
are loaded by `config/models.py` from `src/config/`.

The API keeps immutable model/data/metadata snapshots under `artifacts/model/`
and publishes a snapshot through an atomic `current.json` pointer replacement.
Predictions are saved under `artifacts/predictions/`. Runtime files are ignored
by Git and persisted through a Docker bind mount. One Uvicorn worker and an
in-process lock serialize operations; retraining is synchronous. See
`src/api/README.md` for the HTTP contract and commands.

The project is organized into the following layers:

```text
Data Layer
│
├── Raw Data Loaders
├── Data Cleaning
├── Return Calculation
│
Feature Layer
│
├── Market Features
├── Portfolio Features
├── State Builder
│
Simulation Layer
│
├── VIX Regime Model
├── Markov Chain
├── Regime Student-t Models
├── Lazy Episode Generator
│
Environment Layer
│
├── Action Space Generator
├── Portfolio Environment
├── Reward Calculator
│
Agent Layer
│
├── Q-Learning Agent
├── DQN Agent
├── Double DQN Agent
│
Backtesting Layer
│
├── Walk-Forward Splitter
├── Backtest Engine
├── Benchmark Engine
│
Evaluation Layer
│
├── Performance Metrics
├── Risk Metrics
├── Turnover Metrics
│
Experiment Layer
│
├── Training Scripts
├── Evaluation Scripts
├── Experiment Logs
```

---

# Recommended Repository Structure

```text
portfolio-mdp/
│
├── README.md
│
├── requirements.txt
├── pyproject.toml
│
├── docs/
│   ├── PRODUCT_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   └── EXPERIMENTS.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_backtests.ipynb
│   └── 04_agent_training.ipynb
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── env/
│   ├── agents/
│   ├── backtesting/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── build_dataset.py
│   ├── train_agent.py
│   ├── run_backtest.py
│   └── evaluate_results.py
│
└── tests/
    ├── test_action_space.py
    ├── test_reward.py
    ├── test_environment.py
    ├── test_metrics.py
    └── test_backtesting.py
```

---

# Core Design Principles

1. Keep the environment independent from the agent.
2. Keep reward calculation isolated and testable.
3. Keep feature engineering separate from backtesting logic.
4. Avoid data leakage by construction.
5. Every experiment must be reproducible.
6. Every trained agent must be evaluated against benchmarks.
7. Avoid mixing notebook logic with production code.
8. Prefer simple, inspectable modules before complex abstractions.

---

# Data Layer

## Responsibilities

The data layer is responsible for:

- Loading historical prices.
- Loading macroeconomic variables.
- Cleaning missing values.
- Aligning data to monthly frequency.
- Computing asset returns.
- Saving processed datasets.

## Suggested Modules

```text
src/data/
├── loaders.py
├── preprocessing.py
├── returns.py
└── dataset.py
```

## Expected Outputs

The main output of the data layer should be a clean monthly dataset with:

- Date index.
- Asset returns.
- Asset prices.
- Optional macro variables.
- No look-ahead bias.

Example:

| date | r_arg | r_ced | r_sp500 | r_gold | usd_ars | inflation | vix |
|------|------|------|------|------|------|------|------|
| 2020-01-31 | ... | ... | ... | ... | ... | ... | ... |

## Data Leakage Rules

Feature values at time `t` must only use information available at or before time `t`.

Returns used to compute the reward must correspond to the following period, usually `t+1`.

---

# Feature Layer

## Responsibilities

The feature layer builds the state representation used by the RL agent.

It should transform raw market data into the learned representation and expose
portfolio context separately to the execution layer.

The target Model B tabular implementation is `compact_regime`: a causal momentum
leader, momentum-strength, USD/ARS-change regime, and VIX regime. Momentum,
USD/ARS, and VIX inputs use expanding historical terciles through the current
row only; leader ties use the fixed asset order. This creates 108 learned
market states. `PortfolioEnv` still owns current weights for accounting and
turnover, while the B policy adapter applies the decision-time cost adjustment
outside the learned state.

The repository is migrating to this target architecture. Until the migration
items in `ROADMAP.md` are implemented, A-compatible source paths remain
historical reference behavior rather than the product contract.

The detailed `tercile_binned` representation is retained as a non-default mode
for comparison and for the planned DQN implementation.

When building states for an out-of-sample window, `StateBuilder` accepts a
strictly earlier history dataset. A validation window can use train history and
walk-forward test evaluation uses train-plus-validation history, then slices
the resulting features back to the target window.

## Suggested Modules

```text
src/features/
├── market_features.py
├── portfolio_features.py
└── state_builder.py
```

## Candidate Market Features

- 1-month return.
- 3-month momentum.
- 3-month volatility.
- 6-month volatility.
- 12-month volatility.
- USD/ARS change.
- Inflation.
- VIX level.
- VIX change.

## Portfolio Context

Current portfolio weights remain environment context for turnover, costs, and
execution diagnostics. They must not be appended to the Model B tabular state.
Portfolio value, drawdown, and prior actions are non-product research features.

## State Builder

The `StateBuilder` should expose a simple interface:

```python
state = state_builder.build_state(t, portfolio_context)
```

where:

- `t` is the current time index.
- `portfolio_context` supplies current weights for environment accounting; the
  Model B state output must ignore it. Portfolio value and drawdown are not
  learned Model B features.

---

# Action Space

## Responsibilities

The action space module generates all valid portfolio allocations.

Weights are discretized in 10% increments.

The action space must contain all allocations satisfying:

```text
w_arg + w_ced + w_sp500 + w_gold = 1.0
```

with:

```text
w_i ∈ {0.0, 0.1, 0.2, ..., 1.0}
```

## Expected Number of Actions

The expected number of actions is:

```text
286
```

## Suggested Module

```text
src/env/action_space.py
```

## Suggested Interface

```python
actions = generate_action_space(step=0.10, n_assets=4)
```

Expected output:

```python
[
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    ...
]
```

## Tests

The action space must satisfy:

- All weights are non-negative.
- All weights sum to 1.
- All weights are multiples of 0.10.
- Total number of actions equals 286.

---

# Simulation Layer

## Responsibilities

The optional simulation layer is fitted only on an active training dataset. It
classifies VIX into three fixed training-tercile regimes, estimates a smoothed
monthly Markov chain, and fits one regularized multivariate Student-t model per
regime.

Only primitive monthly variables are sampled: four asset log returns, USD/ARS
log change, and VIX log change. The simulator reconstructs the canonical ten
dataset columns before `StateBuilder` derives any feature.

## Modules

```text
src/simulation/
├── regimes.py
├── markov_chain.py
├── multivariate_t.py
└── market_simulator.py
```

`MarketSimulator.stream_episodes()` returns a lazy iterator. Each yielded data
frame is converted to the existing `PortfolioEnv`; previous episodes need not
remain in memory.

In walk-forward mode the simulator receives `train_dataset` only. Test always
uses real rows and a frozen Q-table.

---

# Environment Layer

## Responsibilities

The environment implements the MDP dynamics.

At each monthly step:

1. The agent selects an action ID.
2. The action ID is mapped to portfolio weights.
3. Turnover is calculated versus previous weights.
4. Rebalancing cost is calculated.
5. Next-period asset returns are applied.
6. Portfolio value is updated.
7. Reward is calculated.
8. Next state is returned.

## Suggested Modules

```text
src/env/
├── portfolio_env.py
├── action_space.py
└── reward.py
```

## PortfolioEnv Interface

The environment should follow a Gym-like interface:

```python
state = env.reset()

next_state, reward, done, info = env.step(action_id)
```

## Step Logic

At time `t`, the agent observes state `s_t`.

The agent selects an action:

```text
a_t
```

The environment maps the action to weights:

```text
w_t
```

The portfolio return is calculated using next-period returns:

```text
R_portfolio,t+1 =
w_arg,t * R_arg,t+1 +
w_ced,t * R_ced,t+1 +
w_sp500,t * R_sp500,t+1 +
w_gold,t * R_gold,t+1
```

Turnover is calculated as:

```text
turnover_t =
0.5 * sum(abs(w_i,t - w_i,t-1))
```

Rebalancing cost is:

```text
rebalance_cost_t =
0.014 * turnover_t
```

Reward is:

```text
R_net,t = R_portfolio,t+1 - rebalance_cost_t
reward_t = 2 * (sqrt(1 + R_net,t) - 1)
```

Portfolio value is updated as:

```text
portfolio_value_t+1 =
portfolio_value_t * (1 + R_net,t)
```

## Info Dictionary

The environment should return an `info` dictionary containing:

```python
{
    "date": date,
    "action_id": action_id,
    "weights": weights,
    "portfolio_return": portfolio_return,
    "net_portfolio_return": net_portfolio_return,
    "turnover": turnover,
    "rebalance_cost": rebalance_cost,
    "reward": reward,
    "portfolio_value": portfolio_value,
    "drawdown": drawdown
}
```

---

# Reward Module

## Responsibilities

The reward module must calculate the reward independently from the environment.

This allows testing reward logic without running full simulations.

## Suggested Module

```text
src/env/reward.py
```

## Suggested Interface

```python
reward = calculate_reward(
    portfolio_return=portfolio_return,
    turnover=turnover,
    transaction_cost=0.014,
    risk_aversion=0.5,
)
```

## Required Formula

```text
net_return = portfolio_return - transaction_cost * turnover
reward = 2 * (sqrt(1 + net_return) - 1)
```

The reward is CRRA utility with `γ = 0.5`; portfolio accounting uses
`net_return`, rather than utility. A separate volatility penalty is not used.

---

# Agent Layer

## Responsibilities

The agent layer contains RL algorithms.

Agents must interact with the environment only through:

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

## Suggested Modules

```text
src/agents/
├── base_agent.py
├── q_learning.py
├── dqn.py
└── double_dqn.py
```

## Initial Agents

### Q-Learning

The current Model B product agent over the 108 compact market states and 286
target allocations.

### Model B Policy Adapter

The adapter owns inference rather than the Q-table itself. It receives market
state `m`, current weights `p`, and the Q-values, then scores each action as
`Q(m, a) - 0.014 × turnover(p, a)`. The product publisher evaluates that rule
over all 286 origin portfolios and emits the deterministic modal target plus
its modal-share diagnostic. This keeps learned state, execution costs, and
published strategic policy as separate responsibilities.

### DQN

Deferred candidate for the detailed feature representation.

### Double DQN

Improvement over DQN to reduce overestimation bias.

## Out of Scope Initially

The following algorithms are out of scope for the initial implementation:

- PPO
- DDPG
- SAC
- TD3

These may be considered later if the project moves to continuous actions.

---

# Backtesting Layer

## Responsibilities

The backtesting layer evaluates trained agents on historical out-of-sample periods.

It must support walk-forward evaluation.

## Suggested Modules

```text
src/backtesting/
├── splitter.py
├── engine.py
└── benchmarks.py
```

## Walk-Forward Backtesting

A typical walk-forward setup:

```text
Train period:      2010-01 to 2017-12
Validation period: 2018-01 to 2019-12
Test period:       2020-01 to 2021-12
```

Then roll forward:

```text
Train period:      2012-01 to 2019-12
Validation period: 2020-01 to 2021-12
Test period:       2022-01 to 2023-12
```

Exact windows will be defined during experimentation.

## Backtest Engine

The backtest engine should produce a time series with:

- Dates.
- Selected actions.
- Portfolio weights.
- Portfolio returns.
- Rebalancing costs.
- Portfolio value.
- Drawdowns.

---

# Benchmark Engine

## Responsibilities

The benchmark engine calculates passive benchmark performance.

Required benchmarks:

1. 100% S&P500
2. 50% S&P500 / 50% Gold

## Suggested Module

```text
src/backtesting/benchmarks.py
```

## Benchmark Rules

Benchmarks should:

- Use the same return data as the agent.
- Use the same evaluation period.
- Be evaluated using the same metrics.
- Be clearly separated from training.

---

# Evaluation Layer

## Responsibilities

The evaluation layer calculates performance and risk metrics.

## Suggested Modules

```text
src/evaluation/
├── metrics.py
└── report.py
```

## Required Metrics

### CAGR

```text
CAGR = (final_value / initial_value) ** (1 / years) - 1
```

### Annualized Volatility

```text
annualized_volatility = std(monthly_returns) * sqrt(12)
```

### Sharpe Ratio

```text
sharpe = (mean(monthly_returns) * 12 - risk_free_rate) / annualized_volatility
```

### Sortino Ratio

```text
sortino = annualized_return / annualized_downside_volatility
```

### Maximum Drawdown

```text
drawdown_t = portfolio_value_t / running_max_t - 1
maximum_drawdown = min(drawdown_t)
```

### Average Turnover

```text
average_turnover = mean(turnover_t)
```

## Reports

Evaluation reports should compare:

- RL agent.
- 100% S&P500 benchmark.
- 50% S&P500 / 50% Gold benchmark.

---

# Experiment Layer

## Responsibilities

The experiment layer orchestrates training, backtesting, and evaluation.

## Suggested Files

```text
scripts/
├── build_dataset.py
├── train_agent.py
├── run_backtest.py
└── evaluate_results.py
```

## Experiment Tracking

Each experiment should store:

- Experiment ID.
- Dataset version.
- Feature set.
- Agent type.
- Hyperparameters.
- Training period.
- Validation period.
- Test period.
- Reward definition.
- Evaluation metrics.
- Notes.

Experiment metadata may be stored in:

```text
experiments/
├── experiment_001/
│   ├── config.yaml
│   ├── metrics.json
│   ├── actions.csv
│   ├── portfolio_values.csv
│   └── notes.md
```

---

# Configuration

Project configuration should be centralized.

Suggested files:

```text
src/config/
├── assets.yaml
├── training.yaml
├── backtesting.yaml
└── paths.yaml
```

## assets.yaml

Should define:

- Asset names.
- Tickers or data sources.
- Asset order.

## training.yaml

Should define:

- Agent type.
- Hyperparameters.
- Random seed.
- Number of episodes.

## backtesting.yaml

Should define:

- Train period.
- Validation period.
- Test period.
- Walk-forward settings.

---

# Testing Strategy

Tests should be created before or alongside core implementation.

## Required Tests

### Action Space Tests

- All actions sum to 1.
- All weights are non-negative.
- Number of actions is 286.
- All weights are multiples of 0.10.

### Reward Tests

- Reward equals CRRA utility of return net of rebalancing cost.
- Zero turnover produces no rebalancing cost.
- Full turnover produces expected cost.

### Environment Tests

- `reset()` returns a valid state.
- `step()` returns next state, reward, done, and info.
- Portfolio value updates correctly.
- Done becomes true at the end of the dataset.
- No future data is used in the state.

### Metric Tests

- CAGR is calculated correctly.
- Maximum drawdown is calculated correctly.
- Volatility is annualized correctly.
- Turnover is calculated correctly.

---

# Implementation Order

Recommended implementation sequence:

1. Create action space generator.
2. Implement reward calculator.
3. Build data loading and monthly returns.
4. Implement basic feature builder.
5. Implement `PortfolioEnv`.
6. Implement passive benchmarks.
7. Implement evaluation metrics.
8. Implement simple baseline strategy.
9. Implement Q-Learning or DQN.
10. Add walk-forward backtesting.
11. Add experiment tracking.
12. Refactor and expand features.

---

# Non-Functional Requirements

## Reproducibility

All training scripts must accept a random seed.

## Modularity

Environment, agent, features, and evaluation should be separate modules.

## Interpretability

It should be possible to inspect:

- Selected portfolio weights.
- Portfolio turnover.
- Rebalancing costs.
- Monthly rewards.
- Portfolio value evolution.

## Simplicity

Avoid unnecessary abstractions in the first implementation.

The first working version should prioritize correctness and clarity over algorithmic complexity.

---

# Open Decisions

The following decisions remain open and should be resolved during implementation:

1. Representative ticker for Argentine Equities.
2. Representative ticker for CEDEARs.
3. Whether returns should be measured in USD, ARS, or inflation-adjusted ARS.
4. Exact feature set for the first model.
5. Exact train / validation / test periods.
6. Whether benchmarks should include transaction costs.
7. Whether portfolio value should be tracked in nominal or real terms.
8. Whether cash should be added as a fifth asset later.
