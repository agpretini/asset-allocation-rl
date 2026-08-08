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
uses `learning_rate / (1 + visit_learning_rate_decay * visits(state, action))`.
The selected trial uses an initial rate of `0.30` and a decay coefficient of
`0.01`; the hyperparameter search also evaluates fixed-rate alternatives.

The same loop supports `historical` and `synthetic` training. Historical mode
resets one real environment for every episode. Synthetic mode consumes a lazy
stream of generated `PortfolioEnv` instances. Agent updates and epsilon decay
are otherwise identical.

`uv run python src/agents/train.py` saves:

- `analysis/runs/q_learning_default/training/q_learning_model.json`
- `analysis/runs/q_learning_default/training/q_learning_training_history.csv`
- `analysis/runs/q_learning_default/training/q_learning_report.csv`
- `analysis/runs/q_learning_default/training/q_learning_paths.csv`

Planned future agents from the roadmap include DQN and Double DQN.

Hyperparameter search for the tabular agent is implemented in
`src/experiments/optimizer.py`.

Portfolio behavior analysis for trained agent outputs is implemented in
`src/analysis/portfolio.py`.
