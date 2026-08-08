# Backtesting Library

This package implements passive benchmarks, rule-based strategies, agent
backtests, split generation, feature-mode comparisons, and walk-forward
evaluation. It contains no output directories; analytical runners in
`analysis/` write the resulting reports. Synthetic walk-forward training fits
only the permitted pre-evaluation history, while validation and test paths are
always historical.
