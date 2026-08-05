"""Training loops for baseline RL agents."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Literal

import pandas as pd

from agents.q_learning import (
    CONSTANT_LEARNING_RATE,
    LearningRateSchedule,
    QLearningAgent,
)
from env.portfolio_env import PortfolioEnv

TrainingMode = Literal["historical", "synthetic"]
HISTORICAL_TRAINING_MODE: TrainingMode = "historical"
SYNTHETIC_TRAINING_MODE: TrainingMode = "synthetic"


@dataclass(frozen=True)
class QLearningTrainingConfig:
    """Configuration for tabular Q-learning training."""

    episodes: int = 25
    training_mode: TrainingMode = HISTORICAL_TRAINING_MODE
    learning_rate: float = 0.10
    learning_rate_schedule: LearningRateSchedule = CONSTANT_LEARNING_RATE
    visit_learning_rate_decay: float = 0.01
    discount_factor: float = 0.95
    epsilon: float = 0.50
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.999954
    state_precision: int = 2
    seed: int = 42


@dataclass(frozen=True)
class TrainingResult:
    """Q-learning training output."""

    agent: QLearningAgent
    history: pd.DataFrame


def train_q_learning_agent(
    env: PortfolioEnv | None,
    config: QLearningTrainingConfig | None = None,
    synthetic_episode_envs: Iterable[PortfolioEnv] | None = None,
) -> TrainingResult:
    """Train on repeated historical data or a stream of synthetic environments."""
    training_config = config or QLearningTrainingConfig()
    if training_config.episodes < 1:
        raise ValueError("episodes must be positive")
    episode_envs = _build_episode_iterator(
        env=env,
        config=training_config,
        synthetic_episode_envs=synthetic_episode_envs,
    )
    try:
        first_env = next(episode_envs)
    except StopIteration as error:
        raise ValueError("training environment stream cannot be empty") from error

    agent = QLearningAgent(
        n_actions=len(first_env.action_space),
        learning_rate=training_config.learning_rate,
        learning_rate_schedule=training_config.learning_rate_schedule,
        visit_learning_rate_decay=training_config.visit_learning_rate_decay,
        discount_factor=training_config.discount_factor,
        epsilon=training_config.epsilon,
        min_epsilon=training_config.min_epsilon,
        epsilon_decay=training_config.epsilon_decay,
        state_precision=training_config.state_precision,
        seed=training_config.seed,
    )
    rows: list[dict[str, object]] = []

    for episode, episode_env in enumerate(
        _prepend_environment(first_env, episode_envs),
    ):
        state = episode_env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action_id = agent.select_action(state)
            next_state, reward, done, _ = episode_env.step(action_id)
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
                "training_mode": training_config.training_mode,
                "episode_reward": episode_reward,
                "steps": steps,
                "epsilon": agent.epsilon,
                "learning_rate_schedule": agent.learning_rate_schedule,
                "q_table_states": len(agent.q_table),
            },
        )
        agent.decay_epsilon()

    if len(rows) != training_config.episodes:
        raise ValueError(
            "training environment stream ended before the configured episodes",
        )

    agent.set_training(False)
    return TrainingResult(agent=agent, history=pd.DataFrame(rows))


def _build_episode_iterator(
    env: PortfolioEnv | None,
    config: QLearningTrainingConfig,
    synthetic_episode_envs: Iterable[PortfolioEnv] | None,
) -> Iterator[PortfolioEnv]:
    """Return a bounded lazy iterator for the selected training mode."""
    if config.training_mode == HISTORICAL_TRAINING_MODE:
        if env is None:
            raise ValueError("historical training requires env")
        if synthetic_episode_envs is not None:
            raise ValueError("historical training cannot use synthetic_episode_envs")
        return (env for _ in range(config.episodes))
    if config.training_mode == SYNTHETIC_TRAINING_MODE:
        if synthetic_episode_envs is None:
            raise ValueError("synthetic training requires synthetic_episode_envs")
        return iter(islice(synthetic_episode_envs, config.episodes))
    raise ValueError(f"unsupported training_mode: {config.training_mode}")


def _prepend_environment(
    first: PortfolioEnv,
    remaining: Iterator[PortfolioEnv],
) -> Iterator[PortfolioEnv]:
    """Put a consumed first environment back in front of its iterator."""
    yield first
    yield from remaining
