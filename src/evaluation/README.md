# Evaluation Layer

This module calculates performance metrics and tabular reports:

- CAGR
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Average turnover

`report.py` can build reports for passive benchmarks and for net-return
strategy backtests. Reports are plain data frames so they can be saved as CSV,
displayed in notebooks, or compared in tests.
