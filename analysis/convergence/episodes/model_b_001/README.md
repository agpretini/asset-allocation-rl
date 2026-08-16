# Model B Episode Convergence Run

This directory is reserved for the 70,000-episode convergence study of the
selected Model B formulation. It is separate from the legacy
`../episode_convergence.*` artifacts, which used the prior policy definition.

Model B uses the four-variable `compact_regime` market state, CRRA reward on
net return, and a read-only cost-aware validation policy:

```text
argmax_a [Q(market_state, a) - 0.014 × turnover(current_portfolio, a)]
```

The runner writes checkpoint data every 2,000 episodes. `episode_convergence`
in this folder always refers to this B configuration; inspect its metadata JSON
for the exact split, simulation parameters, and epsilon schedule.
