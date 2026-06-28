# Environment Layer

This module contains the Milestone 3 and Milestone 4 environment components.

Implemented pieces:

- `action_space.py`: generates all valid long-only portfolio allocations for a
  configurable number of assets and weight increment
- `reward.py`: calculates reward as portfolio return minus rebalancing cost
- `portfolio_env.py`: implements the Gym-like monthly portfolio environment

`PortfolioEnv` exposes:

```python
state = env.reset()
next_state, reward, done, info = env.step(action_id)
```

Each step maps an action ID to target weights, applies next-period returns,
calculates turnover and transaction costs, updates portfolio value, and returns
an info dictionary with portfolio accounting details.
