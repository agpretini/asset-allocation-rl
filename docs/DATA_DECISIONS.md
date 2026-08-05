# Portfolio MDP - Data Decisions

## Purpose

This document records all data-related decisions that have been finalized for the project.

These decisions are considered part of the project's baseline assumptions.

Any future modifications should be explicitly documented and justified.

---

# Base Currency

## Decision

All portfolio returns, performance metrics, and portfolio values will be measured in:

```text
USD
```

## Rationale

Using USD as the base currency provides a consistent framework across all asset classes.

Advantages:

- S&P500 is naturally denominated in USD.
- Gold is naturally denominated in USD.
- International equities are primarily evaluated in USD.
- Reduces distortions caused by local inflation.
- Simplifies cross-asset comparisons.

## Implications

All portfolio returns should be calculated using USD-denominated prices.

Future extensions may include:

- ARS-denominated analysis.
- Inflation-adjusted ARS analysis.

These are currently out of scope.

---

# Asset Class Representation

The project models four asset classes.

---

## Argentine Equities

### Decision

Argentine Equities will be represented by an equally weighted portfolio of:

- GGAL
- YPF
- CRESY
- LOMA
- TX

### Portfolio Construction

Equal-weight allocation:

```text
20% GGAL
20% YPF
20% CRESY
20% LOMA
20% TX
```

### Return Calculation

The monthly return of the Argentine Equities asset class is:

```text
R_arg,t =
0.20 * R_GGAL,t +
0.20 * R_YPF,t +
0.20 * R_CRESY,t +
0.20 * R_LOMA,t +
0.20 * R_TX,t
```

### Rationale

This approach reduces company-specific risk and provides broader exposure to the Argentine equity market than using a single stock.

---

## CEDEAR Portfolio

### Decision

The CEDEAR asset class will be represented by an equally weighted portfolio of:

- GOOGL
- ASML
- MELI
- META
- MSFT
- NVDA
- KO

### Portfolio Construction

Equal-weight allocation:

```text
14.2857% GOOGL
14.2857% ASML
14.2857% MELI
14.2857% META
14.2857% MSFT
14.2857% NVDA
14.2857% KO
```

### Return Calculation

The monthly return of the CEDEAR asset class is:

```text
R_ced,t =
(
R_GOOGL,t +
R_ASML,t +
R_MELI,t +
R_META,t +
R_MSFT,t +
R_NVDA,t +
R_KO,t
) / 7
```

### Rationale

The basket provides exposure to:

- Technology
- Semiconductors
- E-commerce
- Social Media
- Software
- Consumer Staples

while avoiding concentration in a single company.

---

## S&P500

### Decision

The S&P500 asset class will be represented by:

```text
SPY
```

### Rationale

SPY is one of the most liquid and widely used ETFs tracking the S&P500 index.

---

## Gold

### Decision

Gold will be represented by:

```text
GLD
```

### Rationale

GLD is one of the most liquid and widely used ETFs providing exposure to gold.

---

# Data Frequency

## Decision

The project will use:

```text
Monthly Frequency
```

for:

- State generation
- Actions
- Rewards
- Portfolio rebalancing
- Backtesting

### Rationale

Monthly frequency:

- Reduces market noise.
- Reduces overfitting.
- Better aligns with long-term investing.
- Reduces transaction costs.
- Simplifies experimentation.

---

# Return Calculation

## Decision

Returns will be calculated using simple returns.

Formula:

```text
R_t = (P_t / P_t-1) - 1
```

### Rationale

Simple returns are easier to interpret and directly compatible with portfolio weighting calculations.

---

# Portfolio Rebalancing

## Decision

All portfolio allocations are assumed to be rebalanced at the beginning of each monthly period.

The selected action determines the target allocation for the following month.

### Implications

Portfolio returns for month t+1 are generated using the allocation selected at month t.

This prevents look-ahead bias.

---

# Training Reward

## Decision

The agent is trained with constant-relative-risk-aversion (CRRA) utility with
relative risk aversion `γ = 0.5`.

For each monthly decision, the net portfolio return is calculated first:

```text
R_net,t = R_portfolio,t - 0.014 * turnover_t
```

The reward supplied to the RL agent is:

```text
reward_t = 2 * (sqrt(1 + R_net,t) - 1)
```

Portfolio accounting remains based on `R_net,t`; CRRA utility is only the
learning signal. The net gross return must remain positive.

## Rationale

`γ = 0.5` introduces moderate aversion to risk while retaining meaningful
preference for growth. It penalizes a loss slightly more than it rewards an
equal percentage gain, unlike a raw-return reward.

---

# State Discretization

## Decision

All market-state variables are discretized using expanding historical terciles.
Momentum, volatility, USD/ARS monthly change, and VIX are encoded as low (`0`),
medium (`1`), or high (`2`). USD/ARS is represented by its percentage change
from the previous month rather than its absolute level.

For each feature and period, the lower and upper cutoffs are its 33.33% and
66.67% quantiles over valid observations available up to and including that
period. Portfolio allocation weights are kept as their original 10% grid.

In walk-forward evaluation, the training window initializes validation
thresholds. The combined training and validation history initializes test
thresholds. This history is valid information at the start of each subsequent
window and is expanded only with observations as they become available.

## Rationale

This substantially reduces the tabular state space while preserving relative
market regimes. Expanding thresholds prevent any use of future information.

---

# Benchmark Definitions

## Benchmark 1

100% S&P500

```text
SPY = 100%
```

## Benchmark 2

50% S&P500 + 50% Gold

```text
SPY = 50%
GLD = 50%
```

---

# Historical Coverage

## Decision

The historical dataset will start on:

```text
2016-01-01
```

### Rationale

This period includes multiple market regimes:

- Bull markets
- Bear markets
- COVID crash and recovery
- High inflation environments
- Rising and falling interest rate cycles
- Different Argentine political and economic regimes

This should provide sufficient diversity for training and evaluation.

---

# Data Source

## Decision

The project will use:

```text
Yahoo Finance
```

as the official data source for the initial implementation.

### Rationale

Advantages:

- Free access.
- Broad asset coverage.
- Easy integration with Python.
- Sufficient quality for research and experimentation.

### Implications

All assets and benchmarks must be sourced from Yahoo Finance unless explicitly documented otherwise.

---

# Missing Data Treatment

## Decision

Missing observations will be imputed using a rolling 3-month moving average of the affected asset.

### Formula

```text
P_t =
(P_t-1 + P_t-2 + P_t-3) / 3
```

when sufficient history exists.

### Fallback Rule

If fewer than 3 previous observations are available:

1. Use all available historical observations.
2. If no historical observations exist, drop the affected period.

### Rationale

This approach:

- Preserves continuity of the time series.
- Avoids introducing future information.
- Avoids look-ahead bias.
- Keeps the implementation simple.

---

# Dividend Treatment

## Decision

Dividends will be ignored in the initial version of the project.

### Rationale

The primary objective is to validate the MDP and RL framework.

Dividend-adjusted returns can be incorporated in a future version if necessary.

### Implications

Returns will be calculated using price appreciation only.

---

# Benchmark Transaction Costs

## Decision

Benchmark portfolios will NOT include transaction costs.

### Rationale

The benchmarks represent idealized passive investment strategies.

This provides a cleaner reference point for comparison against the RL agent.

### Implications

Benchmark performance will be calculated directly from asset returns without applying rebalancing costs.

---

# Synthetic Training Data

## Decision

Synthetic observations may be used only for Q-learning training. Each simulator
is fitted from the active training window and cannot inspect validation or test.
Validation, test, portfolio reports, and model comparisons remain historical.

The simulator models six primitive monthly variables: four asset log returns,
USD/ARS log change, and VIX log change. It does not simulate momentum,
volatility, tercile labels, portfolio weights, rewards, or final states.

Training VIX terciles define three regimes. Regime transitions use a smoothed
Markov matrix, and each regime has a separate regularized multivariate Student-t
distribution with configurable fixed degrees of freedom (`nu=8` by default).
Market levels are reconstructed before the normal feature pipeline runs.

---

# Compact Tabular State

## Decision

Tabular Q-learning uses `compact_regime` as its default state representation.
The state aggregates detailed causal market inputs into momentum leader,
momentum strength, USD/ARS regime, and VIX regime, then appends the current
portfolio weights. This constrains the theoretical tabular state space to
30,888 states including portfolio allocations.

## Rationale

The detailed `tercile_binned` state has 22 market features. Its combinatorial
space made exact state revisits rare even with synthetic episodes, preventing
tabular Q-learning from accumulating reliable state-action estimates. The
compact representation preserves directional and risk-regime information while
making repeated visits feasible.

## Future Direction

The detailed state is retained unchanged as the intended DQN input. A DQN can
generalize across related states and is therefore the appropriate way to return
to a richer state representation after the compact tabular baseline is
evaluated.

---

# Data Decisions Status

All data-related decisions required for Version 1 have been finalized.

No open data decisions remain.

Future modifications should be documented through a formal update to this file.
