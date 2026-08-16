# Policy Artifacts

`compact_policy_all_states.csv` describes the legacy Model A greedy action for
every compact action-grid state. `state_visit_summary.csv` aggregates full-data
training visits by state. The SVG renders target-weight heatmaps for 108 market
regimes.

`current_market_policy_by_starting_allocation.csv` and its SVG fix the latest
observed market regime, then show every action-grid starting portfolio plus the
actual 25/25/25/25 prediction assumption. Each row records its target portfolio
and the total number of training visits to that exact state.
