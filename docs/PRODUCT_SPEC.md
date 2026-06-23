# Portfolio MDP - Product Specification

## Purpose

This document is the source of truth for the project.

All implementations must follow the modeling decisions defined here.

If implementation details conflict with this specification, this specification takes precedence.

---

# Objective

Develop a Reinforcement Learning agent capable of dynamically selecting a capital allocation across multiple asset classes in order to maximize long-term portfolio growth.

The project aims to answer the following question:

> Given the current market environment, what capital allocation across the available asset classes maximizes future portfolio performance net of rebalancing costs?

This project does **not** attempt to perform stock picking. The problem is formulated as a dynamic asset allocation problem.

---

# Asset Universe

The initial asset classes are:

1. Argentine Equities
2. CEDEARs
3. S&P500
4. Gold

The concrete implementation of each asset class is defined in `DATA_DECISIONS.md`.

---

# MDP Formulation

The problem is modeled as a Markov Decision Process (MDP).

## State

The state represents the current market environment and the current portfolio situation.

The Version 1 state definition is fixed and consists of the following variables.

### Market Variables

For each asset class:

- Argentine Equities
- CEDEARs
- S&P500
- Gold

the following features are included:

#### Momentum Features

- 1-month momentum
- 3-month momentum
- 6-month momentum
- 12-month momentum

#### Volatility Features

- 3-month rolling volatility
- 6-month rolling volatility
- 12-month rolling volatility

This produces:

```text
4 assets × (4 momentum features + 3 volatility features)
= 28 features
```

### Macroeconomic Variables

The following macroeconomic variables are included:

- USD/ARS exchange rate
- Monthly inflation rate
- VIX level

This produces:

```text
3 features
```

### Portfolio Variables

The portfolio state includes:

- Current weight in Argentine Equities
- Current weight in CEDEARs
- Current weight in S&P500
- Current weight in Gold
- Current portfolio drawdown

This produces:

```text
5 features
```

### State Dimension

The complete state vector contains:

```text
28 market features
+ 3 macro features
+ 5 portfolio features

= 36 total features
```

### State Vector

Conceptually:

```text
State_t = [

    ARG_mom_1m,
    ARG_mom_3m,
    ARG_mom_6m,
    ARG_mom_12m,

    ARG_vol_3m,
    ARG_vol_6m,
    ARG_vol_12m,

    CED_mom_1m,
    CED_mom_3m,
    CED_mom_6m,
    CED_mom_12m,

    CED_vol_3m,
    CED_vol_6m,
    CED_vol_12m,

    SPY_mom_1m,
    SPY_mom_3m,
    SPY_mom_6m,
    SPY_mom_12m,

    SPY_vol_3m,
    SPY_vol_6m,
    SPY_vol_12m,

    GLD_mom_1m,
    GLD_mom_3m,
    GLD_mom_6m,
    GLD_mom_12m,

    GLD_vol_3m,
    GLD_vol_6m,
    GLD_vol_12m,

    usd_ars,
    inflation,
    vix,

    w_arg,
    w_ced,
    w_sp500,
    w_gold,

    current_drawdown

]
```

### Data Leakage Constraint

Every feature in the state must be computed exclusively using information available at time t.

No future observations may be used when constructing the state.

This requirement is mandatory and must be enforced throughout the project.
---

## Action Space

An action consists of selecting a target portfolio allocation.

Portfolio weights are discretized in 10% increments.

Let:

- w_arg
- w_ced
- w_sp500
- w_gold

Subject to:

```text
w_arg + w_ced + w_sp500 + w_gold = 100%
```

and

```text
w_i ∈ {0%, 10%, 20%, ..., 100%}
```

This produces:

```text
C(13,3) = 286
```

possible portfolios.

Each portfolio will be represented internally by an integer Action ID.

Example:

| Action ID | ARG | CEDEAR | SP500 | GOLD |
|------------|------------|------------|------------|------------|
| 0 | 100 | 0 | 0 | 0 |
| 1 | 90 | 10 | 0 | 0 |
| ... | ... | ... | ... | ... |
| 285 | ... | ... | ... | ... |

---

## Decision Frequency

The environment operates on a fixed decision frequency.

The specific frequency used by the implementation is defined in `DATA_DECISIONS.md`.

---

## Transition Dynamics

After selecting a portfolio:

1. The market evolves during the next period.
2. Asset returns are observed.
3. Portfolio return is calculated.
4. Portfolio value is updated.
5. A new state is generated.

Transition dynamics will be obtained directly from historical data.

The project does not attempt to explicitly estimate a transition matrix.

---

# Reward Function

The reward function aims to maximize portfolio growth net of transaction costs.

## Portfolio Return

Let:

```text
R_portfolio,t
```

be the portfolio return for a single environment step.

The exact return calculation methodology is defined in `DATA_DECISIONS.md`.

Conceptually:

```text
R_portfolio,t =
Σ (weight_i × asset_return_i)
```

---

## Rebalancing Costs

Assumptions:

- 0.7% transaction cost when selling
- 0.7% transaction cost when buying

Total cost:

```text
1.4% of the capital effectively reallocated
```

Define:

```text
turnover_t =
0.5 * Σ |w_i,t - w_i,t-1|
```

Then:

```text
rebalance_cost_t =
0.014 * turnover_t
```

---

## Reward

The training reward is defined as:

```text
reward_t =
R_portfolio,t - rebalance_cost_t
```

### Important

Volatility is intentionally excluded from the reward function.

Risk management will be handled through evaluation metrics during backtesting rather than during training.

---

# Candidate Algorithms

Because the action space is discrete (286 portfolios), the project will initially focus on discrete-action RL algorithms.

Candidate approaches:

- Q-Learning
- Deep Q Network (DQN)
- Double DQN

Continuous-action algorithms such as:

- PPO
- DDPG
- SAC
- TD3

are considered out of scope for the initial implementation.

---

# Backtesting Framework

Evaluation will be performed using walk-forward out-of-sample backtesting.

The primary objective is to evaluate the agent's ability to adapt to changing market regimes.

---

# Evaluation Metrics

The reward function must not be confused with evaluation metrics.

The agent is trained to maximize reward.

Performance will be evaluated using:

## CAGR

Compound Annual Growth Rate.

Measures long-term compounded growth.

## Annualized Volatility

Measures overall portfolio risk.

## Sharpe Ratio

Risk-adjusted return based on total volatility.

## Sortino Ratio

Risk-adjusted return based only on downside volatility.

## Maximum Drawdown

Largest peak-to-trough decline.

This is considered a critical robustness metric.

## Average Turnover

Measures portfolio rotation over time.

Helps identify excessively active strategies that may be impractical in real-world conditions.

---

# Benchmarks

All experiments must be compared against simple passive benchmarks.

The initial benchmark set consists of:

## Benchmark 1

100% S&P500

## Benchmark 2

50% S&P500 / 50% Gold

The exact implementation of these benchmarks is defined in `DATA_DECISIONS.md`.

The agent will only be considered successful if it consistently outperforms these benchmarks on relevant return and risk metrics.

---

# Design Principles

1. Simplicity before complexity.
2. Avoid overfitting.
3. Maintain interpretability.
4. Incorporate realistic transaction costs.
5. Always compare against passive benchmarks.
6. Validate out-of-sample before drawing conclusions.
7. Treat the project as an asset allocation system, not a price prediction system.
8. Prioritize robustness over short-term performance.

---

# Explicit Non Goals

This project does NOT attempt to:

- Predict future prices.
- Forecast macroeconomic variables.
- Perform stock picking.
- Trade intraday.
- Optimize tax treatment.
- Perform high-frequency market timing.

The sole objective of this project is dynamic asset allocation across a predefined set of asset classes.