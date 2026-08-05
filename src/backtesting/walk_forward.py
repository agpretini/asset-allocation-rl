"""Walk-forward evaluation for the first RL agent."""

from dataclasses import replace

import pandas as pd

from agents.training import (
    SYNTHETIC_TRAINING_MODE,
    QLearningTrainingConfig,
    TrainingResult,
    train_q_learning_agent,
)
from backtesting.agents import run_agent_backtest
from backtesting.splitter import WalkForwardSplit
from env.portfolio_env import PortfolioEnv
from evaluation.report import build_performance_report
from features.state_builder import StateBuilder
from features.transforms import COMPACT_REGIME_FEATURE_MODE, FeatureMode
from simulation.market_simulator import (
    MarketSimulator,
    SimulationConfig,
    load_simulation_config,
)


def run_walk_forward_q_learning(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    training_config: QLearningTrainingConfig | None = None,
    feature_mode: FeatureMode = COMPACT_REGIME_FEATURE_MODE,
    simulation_config: SimulationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate Q-learning across walk-forward splits."""
    rows: list[dict] = []
    paths: list[pd.Series] = []

    for split in splits:
        train_dataset = dataset.loc[split.train_start : split.train_end]
        test_dataset = dataset.loc[split.test_start : split.test_end]
        test_history = dataset.loc[: split.validation_end]

        selected_training_config = training_config or QLearningTrainingConfig()
        training_result = _train_for_split(
            train_dataset=train_dataset,
            training_config=selected_training_config,
            feature_mode=feature_mode,
            simulation_config=simulation_config,
        )

        test_env = PortfolioEnv(
            dataset=test_dataset,
            state_builder=StateBuilder(
                test_dataset,
                feature_mode=feature_mode,
                history_dataset=test_history,
            ),
        )
        test_result = run_agent_backtest(test_env, training_result.agent)
        report = build_performance_report({"q_learning": test_result})
        metrics = report.loc["q_learning"].to_dict()
        rows.append(
            {
                "feature_mode": feature_mode,
                "split_id": split.split_id,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "q_table_states": len(training_result.agent.q_table),
                "training_mode": selected_training_config.training_mode,
                **metrics,
            },
        )
        paths.append(
            test_result["portfolio_value"].rename(f"split_{split.split_id}"),
        )

    return pd.DataFrame(rows), pd.concat(paths, axis=1, sort=False)


def _train_for_split(
    train_dataset: pd.DataFrame,
    training_config: QLearningTrainingConfig,
    feature_mode: FeatureMode,
    simulation_config: SimulationConfig | None,
) -> TrainingResult:
    """Train one split historically or from a lazy synthetic episode stream."""
    if training_config.training_mode == "historical":
        train_env = PortfolioEnv(
            dataset=train_dataset,
            state_builder=StateBuilder(train_dataset, feature_mode=feature_mode),
        )
        return train_q_learning_agent(env=train_env, config=training_config)

    if training_config.training_mode != SYNTHETIC_TRAINING_MODE:
        raise ValueError(f"unsupported training_mode: {training_config.training_mode}")
    selected_simulation_config = simulation_config or load_simulation_config()
    selected_simulation_config = replace(
        selected_simulation_config,
        episode_count=training_config.episodes,
    )
    simulator = MarketSimulator.fit(train_dataset, selected_simulation_config)
    episode_envs = (
        PortfolioEnv(
            dataset=episode,
            state_builder=StateBuilder(
                episode,
                feature_mode=feature_mode,
                history_dataset=train_dataset,
            ),
        )
        for episode in simulator.stream_episodes()
    )
    return train_q_learning_agent(
        env=None,
        config=training_config,
        synthetic_episode_envs=episode_envs,
    )
