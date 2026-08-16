# Selected Model B Results

This is the committed production-style evaluation of the selected Model B
trial 76 configuration. It uses the corrected cost-aware frozen policy for
historical validation and test decisions.

| Historical test split | CAGR | Sharpe | Maximum drawdown | Average turnover |
| --- | ---: | ---: | ---: | ---: |
| 0 (2022-12 to 2024-11) | 23.63% | 2.14 | -6.52% | 25.65% |
| 1 (2023-12 to 2025-11) | 14.77% | 1.31 | -7.55% | 38.91% |

The model has a 108-state market-only Q-table. Its first validation split was
negative (CAGR -24.27%, Sharpe -0.96), and the committed baseline tables show
that it does not consistently outperform all passive baselines. The selected
status means this is the current product configuration under evaluation; it is
not evidence of a final investment recommendation or benchmark dominance.

For August 2026, the full-data Model B policy selects 100% Gold. Its modal
target share is 100% across the 286 permitted origin allocations; moving from
the equal-weight execution reference implies 75% turnover. See
`forecast/2026-08/` for the reproducible decision record.
