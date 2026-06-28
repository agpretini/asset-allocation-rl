# Analysis Layer

This module implements Milestone 12 portfolio analysis.

Implemented pieces:

- allocation parsing from backtest outputs
- per-asset allocation summaries
- turnover and drawdown summaries
- simple market-regime classification from S&P500 returns and VIX
- allocation/performance summaries by regime
- Markdown report generation

`scripts/analyze_portfolio.py` reads the Q-learning path output and writes:

- `data/processed/analysis/portfolio_allocations.csv`
- `data/processed/analysis/allocation_summary.csv`
- `data/processed/analysis/regime_summary.csv`
- `data/processed/analysis/turnover_summary.csv`
- `data/processed/analysis/drawdown_summary.csv`
- `data/processed/analysis/portfolio_analysis.md`
