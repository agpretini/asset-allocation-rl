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

Each asset class will be represented by a representative instrument or index to be defined during implementation.

---

# Data Sources

The initial implementation will use historical monthly data.

Candidate data sources include:

- Yahoo Finance
- Stooq
- FRED (macroeconomic indicators)

Representative instruments will be defined during implementation.

Potential examples:

| Asset Class | Representative Instrument |
|------------|------------|
| Argentine Equities | TBD |
| CEDEARs | TBD |
| S&P500 | SPY |
| Gold | GLD |

The exact mapping between asset classes and instruments is intentionally left open for future experimentation.

---

# MDP Formulation

The problem is modeled as a Markov Decision Process (MDP).

## State

The state represents the current market environment and portfolio situation.

The final state representation will be determined through experimentation, but may initially include:

### Market Variables

- Argentine Equities momentum
- CEDEAR momentum
- S&P500 momentum
- Gold momentum
- Recent volatility of each asset class
- Recent returns
- Inflation
- USD/ARS exchange rate
- VIX as a global risk indicator

### Portfolio Variables

- Current portfolio allocation
- Portfolio value
- Current drawdown

The final state design must avoid data leakage and only use information available at the time the decision is made.

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
w_i ∈ {0%,10%,20%,...,100%}
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

The decision frequency is monthly.

Each environment step represents one calendar month.

Daily frequency is intentionally avoided in order to reduce noise, overfitting, and excessive trading activity.

---

## Transition Dynamics

After selecting a portfolio:

1. The market evolves during the next month.
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

be the monthly portfolio return.

It is calculated as:

```text
R_portfolio,t =
w_arg * R_arg,t +
w_ced * R_ced,t +
w_sp500 * R_sp500,t +
w_gold * R_gold,t
```

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

Continuous-action algorithms such as PPO, DDPG, SAC, or TD3 are considered out of scope for the initial implementation.

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

Fixed allocation:

```text
SP500 = 100%
```

## Benchmark 2

50% S&P500
50% Gold

Fixed allocation:

```text
SP500 = 50%
Gold = 50%
```

The agent will only be considered successful if it consistently outperforms these benchmarks on relevant return and risk metrics.

---

# Design Principles

1. Simplicity before complexity.
2. Avoid overfitting.
3. Maintain interpretability.
4. Use monthly decision frequency.
5. Incorporate realistic transaction costs.
6. Always compare against passive benchmarks.
7. Validate out-of-sample before drawing conclusions.
8. Treat the project as an asset allocation system, not a price prediction system.

---

# Explicit Non Goals

This project does NOT attempt to:

- Predict future prices.
- Forecast macroeconomic variables.
- Perform stock picking.
- Trade intraday.
- Optimize tax treatment.
- Perform market timing at high frequency.

The sole objective of this project is dynamic asset allocation across a predefined set of asset classes.