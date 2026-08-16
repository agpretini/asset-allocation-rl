# Model B Policy and Coverage

`compact_policy_all_states.csv` contains the deployed modal target for all 108
market-only compact-regime states. For each state, Model B scores all actions
from all 286 possible origin allocations using Q-value less immediate
rebalancing cost, then publishes the deterministic modal target.

`state_visit_summary.csv` reports training coverage at state and state-action
level. `compact_policy_balanced_reference.*` visualizes all regimes, while
`current_market_policy_by_starting_allocation.*` fixes the latest market state
and shows how the execution-level cost-aware choice varies by origin.
