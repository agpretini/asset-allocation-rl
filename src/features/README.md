# Feature Layer

This module builds the Milestone 5 state representation.

Market features are computed from information available at time `t` only:

- 1, 3, 6, and 12 month momentum from asset price indexes
- 3, 6, and 12 month rolling volatility from monthly returns
- configured macro columns available in the processed dataset

The state builder appends the current portfolio weights to those market
features. It does not use future returns.
