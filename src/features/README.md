# Feature Layer

`StateBuilder` builds causal market states and appends the current portfolio
weights. The default is a compact regime representation designed for tabular
Q-learning; it deliberately limits the reachable state space so state-action
pairs can be revisited during synthetic training.

## Default: `compact_regime`

The market part of the state has four discrete values:

| Feature | Values | Construction |
| --- | --- | --- |
| `momentum_leader` | `0` ARG, `1` CEDEARs, `2` S&P 500, `3` Gold | Asset with the highest mean of its 1M and 3M expanding-tercile momentum labels. Ties follow this asset order. |
| `momentum_strength` | `0` low, `1` medium, `2` high | Tercile-like bin of the winning asset's mean momentum label: `<= 2/3`, `<= 4/3`, `> 4/3`. |
| `usd_ars_regime` | `0`, `1`, `2` | Expanding historical tercile of monthly USD/ARS change. |
| `vix_regime` | `0`, `1`, `2` | Expanding historical tercile of VIX. |

Each momentum input is first classified independently with expanding terciles
using observations available through the current month. Therefore no feature,
leader selection, or strength label uses future data. Out-of-sample windows use
strictly earlier history as their initial context and then expand month by
month.

The four portfolio weights are appended unchanged on the 10% action grid. With
286 feasible allocations, the theoretical maximum is:

```text
4 leaders × 3 strengths × 3 USD/ARS regimes × 3 VIX regimes × 286 portfolios
= 30,888 states
```

The actual state vector has 8 values: four market-regime values plus four
weights.

## Other feature modes

- `raw`: original continuous market inputs, rounded only when the Q-table key
  is created.
- `normalized_binned`: expanding z-score bins for every detailed feature.
- `tercile_binned`: the previous detailed state: 1M/3M momentum, 3M/6M/12M
  volatility, USD/ARS change, and VIX per asset or macro series. It has 22
  market values and is retained for comparison and for the planned DQN input.

The detailed features remain calculated by `market_features.py`; compact mode
uses only the signals needed to build its four regimes.

## Planned next step: DQN

The next representation experiment should restore `tercile_binned` (or a
carefully selected continuous version) as the input to a Deep Q-Network. A DQN
approximates values across similar states, avoiding the exact-state revisit
requirement that makes detailed states unsuitable for a tabular Q-table.
