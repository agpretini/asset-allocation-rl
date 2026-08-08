# Synthetic Episode Convergence Analysis

This experiment measures whether tabular Q-learning continues to improve over
70,000 synthetic episodes or reaches a stable greedy policy.

## Design

- The latest walk-forward split is used by default.
- The simulator is fitted only on the historical training window.
- Synthetic episodes are generated lazily, one at a time.
- Evaluation uses the real historical validation window only.
- The test window is not used for fitting, training, checkpoint selection, or
  this curve.
- The policy is evaluated before training and every 2,000 episodes thereafter.
- Every checkpoint uses a deterministic greedy policy with `epsilon=0`.
- The evaluator never calls `update`, never inserts Q-table states, and never
  changes training epsilon.
- Greedy ties are resolved by the lowest action ID, so checkpoint results are
  deterministic for a fixed Q-table.

The plotted metric is cumulative CRRA reward on validation:

```text
validation_cumulative_reward = sum(validation.reward)
```

The CSV also records final value, CAGR, Sharpe, drawdown, turnover, Q-table
size, state-action pair count, and training epsilon.

## Default configuration

```text
training episodes:       70,000
evaluation interval:      2,000
synthetic episode length: simulation.yaml (120 months)
state representation:     compact_regime
learning rate:             0.30
learning-rate schedule:    visit_decay
visit decay:               0.01
discount factor:           0.95
initial epsilon:           0.50
minimum epsilon:           0.02
epsilon decay:             0.999954 per episode
seed:                      simulation.yaml (42)
```

With these values, `epsilon = max(0.02, 0.50 × 0.999954^episode)` reaches its
floor approximately at episode 70,000. Exploration therefore remains material
throughout this experiment instead of collapsing during its first few dozen
episodes.

`compact_regime` has at most 30,888 reachable state combinations including the
current portfolio allocation. This makes repeated visits practical for tabular
Q-learning and keeps the dense 286-action Q-table tractable. The former
detailed `tercile_binned` state is retained for comparison and the planned DQN
work, but is not appropriate for this 70,000-episode tabular experiment.

## Run

From the repository root:

```bash
uv run python analysis/convergence/episodes/run.py
```

Short smoke test:

```bash
uv run python analysis/convergence/episodes/run.py \
  --episodes 20 \
  --evaluation-interval 5 \
  --episode-length 24 \
  --output-prefix smoke
```

## Outputs

- `episode_convergence.csv`: checkpoint metrics.
- `episode_convergence.svg`: line chart of validation cumulative reward.
- `episode_convergence_metadata.json`: configuration and split dates.

CSV and SVG are updated after every checkpoint, so partial results remain
available if a run is stopped. The SVG is dependency-free and opens in any web
browser.

## Interpretation

A rising curve that later becomes approximately flat suggests convergence. A
noisy curve can indicate an unstable policy or a short validation period; a
sustained decline can indicate overfitting to the simulator.

Read the reward curve together with `q_table_states` and
`state_action_pairs`. A compact state makes a flat curve much more meaningful
than before, because states can be revisited. After selecting a checkpoint on
validation, perform exactly one final historical evaluation on test.

## Planned next step: detailed state with DQN

Once this compact tabular baseline is evaluated, restore the current detailed
state definition (`tercile_binned`) and train a DQN. A neural value function
can generalize across related detailed states, whereas a tabular Q-table needs
exact state repetition.
