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

The detailed feature mode consists of the following variables. It is retained
for comparison and is the planned input to a future DQN; the compact regime
state defined below is the default for tabular Q-learning.

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

Momentum is defined as cumulative simple return over the corresponding lookback window.

Examples:

```text
mom_1m = P_t / P_t-1 - 1

mom_3m = P_t / P_t-3 - 1

```


#### Volatility Features

- 3-month rolling volatility
- 6-month rolling volatility
- 12-month rolling volatility

This produces:

```text
4 assets × (2 momentum features + 3 volatility features)
= 20 features
```

Volatility is defined as the rolling standard deviation of monthly returns.

Examples:

```text
vol_3m = std(R_t, R_t-1, R_t-2)

vol_6m = rolling std over last 6 monthly returns

vol_12m = rolling std over last 12 monthly returns
```

### Macroeconomic Variables

The following macroeconomic variables are included:

- Monthly USD/ARS change
- VIX level

The current implementation produces:

```text
2 features
```

### Portfolio Variables

The portfolio state includes:

- Current weight in Argentine Equities
- Current weight in CEDEARs
- Current weight in S&P500
- Current weight in Gold

This produces:

```text
4 features
```

### State Dimension

The complete state vector contains:

```text
20 market features
+ 2 macro features
+ 4 portfolio features

= 26 total features
```

### State Vector

Conceptually:

```text
State_t = [

    ARG_mom_1m,
    ARG_mom_3m,
    ARG_vol_3m,
    ARG_vol_6m,
    ARG_vol_12m,

    CED_mom_1m,
    CED_mom_3m,
    CED_vol_3m,
    CED_vol_6m,
    CED_vol_12m,

    SPY_mom_1m,
    SPY_mom_3m,
    SPY_vol_3m,
    SPY_vol_6m,
    SPY_vol_12m,

    GLD_mom_1m,
    GLD_mom_3m,
    GLD_vol_3m,
    GLD_vol_6m,
    GLD_vol_12m,

    usd_ars_change,
    vix,

    w_arg,
    w_ced,
    w_sp500,
    w_gold

]
```

### Data Leakage Constraint

Every feature in the state must be computed exclusively using information available at time t.

No future observations may be used when constructing the state.

This requirement is mandatory and must be enforced throughout the project.

### State Discretization

The default tabular representation is the compact regime state below. It uses
expanding terciles through time `t` only, then aggregates the detailed inputs:

```text
momentum_leader:   0=ARG, 1=CEDEARs, 2=S&P500, 3=Gold
momentum_strength: 0=low, 1=medium, 2=high
usd_ars_regime:    0=low, 1=medium, 2=high
vix_regime:        0=low, 1=medium, 2=high
w_arg, w_ced, w_sp500, w_gold
```

For each asset, its 1M and 3M momentum labels are averaged. The highest score
is the leader; ties follow the fixed asset order. The leading score is low when
`<= 2/3`, medium when `<= 4/3`, and high otherwise. USD/ARS uses monthly
percentage change before its expanding-tercile regime is calculated. Portfolio
weights are not binned: they retain their 10% action-space grid.

This representation has at most `4 × 3 × 3 × 3 × 286 = 30,888` states. The
previous detailed 22-feature `tercile_binned` state remains available for
comparison and is reserved as the intended input representation for the next
DQN experiment.

For walk-forward validation, the historical observations in the training window
seed the thresholds. For test, training plus validation observations seed them.
As each validation or test month is evaluated, its own observed features are
then included for the state at that month; later months are never included.
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

The reward function maximizes CRRA utility of portfolio growth net of
transaction costs. The fixed relative risk-aversion coefficient is `γ = 0.5`.

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

## Net Return And Reward

First calculate the net portfolio return:

```text
R_net,t = R_portfolio,t - rebalance_cost_t
```

The training reward is CRRA utility applied to this net return:

```text
reward_t = ((1 + R_net,t)^(1 - γ) - 1) / (1 - γ)
```

With the selected `γ = 0.5`, this becomes:

```text
reward_t = 2 * (sqrt(1 + R_net,t) - 1)
```

`1 + R_net,t` must be positive. Portfolio value and evaluation metrics remain
based on `R_net,t`, rather than on utility. CRRA makes the agent moderately
risk-averse; equivalent gains and losses in percentage terms do not receive
equal utility. Volatility is not added as a separate penalty because the
concave utility already accounts for risk preferences.

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

# Synthetic Training Model

Q-learning supports an optional synthetic training mode. For every walk-forward
split, its generator must be fitted exclusively on that split's training data.
Validation and test data must remain real and must never update the simulator or
Q-table.

Training VIX terciles define `LOW_VOL`, `MEDIUM_VOL`, and `HIGH_VOL`. A smoothed
Markov chain generates monthly regimes. Conditional on each regime, a separate
regularized multivariate Student-t distribution generates:

```text
log(1 + r_arg)
log(1 + r_ced)
log(1 + r_sp500)
log(1 + r_gold)
change(log(USD/ARS))
change(log(VIX))
```

Degrees of freedom are fixed and configurable, with `nu=8` as the initial
default. Prices and macro levels must be reconstructed as positive values, and
the standard feature pipeline must build all derived states. Episode generation
must be reproducible and lazy.

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
