# Backtesting Library

This package implements passive benchmarks, rule-based strategies, agent
backtests, split generation, feature-mode comparisons, and walk-forward
evaluation. It contains no output directories; analytical runners in
`analysis/` write the resulting reports. Synthetic walk-forward training fits
only the permitted pre-evaluation history, while validation and test paths are
always historical.

The default product evaluation uses `CostAwareFrozenPolicy`: it reads Model B's
market-only Q-values and selects `Q(m, a) - 0.014 × turnover(current, a)` while
keeping the Q-table immutable. Portfolio accounting still deducts the same
rebalance cost in the environment.
