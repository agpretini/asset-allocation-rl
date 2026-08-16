# Evaluation Layer

This module calculates performance metrics and tabular reports:

- CAGR
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Average turnover

`report.py` can build reports for passive benchmarks and for strategy backtests.
When present, it uses `net_portfolio_return` rather than the CRRA training
reward, so performance metrics remain expressed in financial returns. Reports
are plain data frames so they can be saved as CSV, displayed in notebooks, or
compared in tests.

`artifacts.py` additionally writes dependency-free SVG cumulative-value charts.
The selected Model B workflow writes one chart and one metrics table per test
split under `analysis/runs/<run_label>/comparison/` (currently
`analysis/runs/model_b_synthetic_trial_076/comparison/`).
Test windows overlap, so they are intentionally not stitched into a single
time series or pooled cumulative return.
