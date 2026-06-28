"""Training loops for baseline RL agents."""

from dataclasses import dataclass

import pandas as pd

from agents.q_learning import QLearningAgent
from env.portfolio_env import PortfolioEnv


@dataclass(frozen=True)
class QLearningTrainingConfig:
    """Configuration for tabular Q-learning training."""

    episodes: int = 25
    learning_rate: float = 0.10
    discount_factor: float = 0.95
    epsilon: float = 0.30
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.95
    state_precision: int = 2
    seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
    """Q-learning training output."""

    agent: QLearningAgent
    history: pd.DataFrame


def train_q_learning_agent(
    env: PortfolioEnv,
    config: QLearningTrainingConfig | None = None,
) -> TrainingResult:
    """Train a tabular Q-learning agent on one environment."""
    training_config = config or QLearningTrainingConfig()
    if training_config.episodes < 1:
        raise ValueError("episodes must be positive")

    agent = QLearningAgent(
        n_actions=len(env.action_space),
        learning_rate=training_config.learning_rate,
        discount_factor=training_config.discount_factor,
        epsilon=training_config.epsilon,
        min_epsilon=training_config.min_epsilon,
        epsilon_decay=training_config.epsilon_decay,
        state_precision=training_config.state_precision,
        seed=training_config.seed,
    )
    rows: list[dict[str, float | int]] = []

    for episode in range(training_config.episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action_id = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_id)
            agent.update(
                state=state,
                action_id=action_id,
                reward=reward,
                next_state=next_state,
                done=done,
            )
            state = next_state
            episode_reward += reward
            steps += 1

        rows.append(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "q_table_states": len(agent.q_table),
            },
        )
        agent.decay_epsilon()

    agent.set_training(False)
    return TrainingResult(agent=agent, history=pd.DataFrame(rows))
