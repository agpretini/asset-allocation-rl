# Agent Layer

This module currently contains Milestone 6 and Milestone 8 agent logic.

Implemented pieces:

- `random_agent.py`: selects random discrete action IDs from the environment
  action space, with optional seeding for reproducibility
- `q_learning.py`: tabular Q-learning agent for the discrete action space
- `training.py`: reproducible Q-learning training loop

The random agent is a baseline for validating that `PortfolioEnv` can run a
complete episode end to end. The first RL agent is intentionally tabular and
simple; DQN remains a future upgrade once the evaluation loop is stable.

Q-learning supports a constant learning rate and `visit_decay`. The latter
uses `learning_rate / (1 + 0.01 * visits(state, action))`; the hyperparameter
search evaluates the requested `0.30` initial rate alongside fixed-rate runs.

The same loop supports `historical` and `synthetic` training. Historical mode
resets one real environment for every episode. Synthetic mode consumes a lazy
stream of generated `PortfolioEnv` instances. Agent updates and epsilon decay
are otherwise identical.

`scripts/train_agent.py` saves:

- `data/processed/q_learning/q_learning_model.json`
- `data/processed/q_learning/q_learning_training_history.csv`
- `data/processed/q_learning/q_learning_report.csv`
- `data/processed/q_learning/q_learning_paths.csv`

Planned future agents from the roadmap include DQN and Double DQN.

Hyperparameter search for the tabular agent is implemented in
`src/experiments/optimizer.py`.

Portfolio behavior analysis for trained agent outputs is implemented in
`src/analysis/portfolio.py`.
