"""Walk-forward evaluation for the first RL agent."""

from collections.abc import Callable
from dataclasses import replace

import pandas as pd

from agents.q_learning import QLearningAgent
from agents.training import (
    SYNTHETIC_TRAINING_MODE,
    QLearningTrainingConfig,
    TrainingResult,
    train_q_learning_agent,
)
from backtesting.agents import CostAwareFrozenPolicy, run_agent_backtest
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

WalkForwardProgressCallback = Callable[[str, int, str], None]


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

        test_result = _run_historical_evaluation(
            evaluation_dataset=test_dataset,
            history_dataset=test_history,
            agent=training_result.agent,
            feature_mode=feature_mode,
        )
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


def run_walk_forward_train_validation_test(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    training_config: QLearningTrainingConfig,
    feature_mode: FeatureMode = COMPACT_REGIME_FEATURE_MODE,
    simulation_config: SimulationConfig | None = None,
    progress_callback: WalkForwardProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate validation and test without allowing test data into training.

    The supplied configuration is treated as fixed.  For every split, the
    validation policy is trained only on ``train``.  The final test policy is
    a fresh agent trained on ``train + validation``; synthetic simulator
    parameters are therefore fitted only on that combined pre-test history.
    """
    rows: list[dict] = []
    paths: dict[str, pd.DataFrame] = {}
    for split in splits:
        _emit_progress(
            progress_callback, "split_started", split.split_id, "Preparing data"
        )
        train = dataset.loc[split.train_start : split.train_end]
        validation = dataset.loc[split.validation_start : split.validation_end]
        train_validation = dataset.loc[split.train_start : split.validation_end]
        test = dataset.loc[split.test_start : split.test_end]

        _emit_progress(
            progress_callback,
            "validation_training",
            split.split_id,
            "Training on train window for validation",
        )
        validation_training = _train_for_split(
            train,
            training_config,
            feature_mode,
            simulation_config,
        )
        _emit_progress(
            progress_callback,
            "validation_evaluation",
            split.split_id,
            "Evaluating frozen policy on historical validation",
        )
        validation_result = _run_historical_evaluation(
            validation,
            train,
            validation_training.agent,
            feature_mode,
        )
        _emit_progress(
            progress_callback,
            "test_training",
            split.split_id,
            "Retraining on train plus validation for test",
        )
        final_training = _train_for_split(
            train_validation,
            training_config,
            feature_mode,
            simulation_config,
        )
        _emit_progress(
            progress_callback,
            "test_evaluation",
            split.split_id,
            "Evaluating frozen policy on historical test",
        )
        test_result = _run_historical_evaluation(
            test,
            train_validation,
            final_training.agent,
            feature_mode,
        )
        for phase, result, training_result in (
            ("validation", validation_result, validation_training),
            ("test", test_result, final_training),
        ):
            metrics = (
                build_performance_report({"q_learning": result})
                .loc["q_learning"]
                .to_dict()
            )
            rows.append(
                {
                    "feature_mode": feature_mode,
                    "split_id": split.split_id,
                    "phase": phase,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "training_mode": training_config.training_mode,
                    "q_table_states": len(training_result.agent.q_table),
                    "final_portfolio_value": float(result["portfolio_value"].iloc[-1]),
                    **metrics,
                },
            )
            paths[f"{phase}_split_{split.split_id}"] = result
        _emit_progress(
            progress_callback,
            "split_completed",
            split.split_id,
            "Validation and test metrics completed",
        )
    return pd.DataFrame(rows), paths


def _emit_progress(
    callback: WalkForwardProgressCallback | None,
    stage: str,
    split_id: int,
    message: str,
) -> None:
    """Notify an optional caller without coupling evaluation to file output."""
    if callback is not None:
        callback(stage, split_id, message)


def _run_historical_evaluation(
    evaluation_dataset: pd.DataFrame,
    history_dataset: pd.DataFrame,
    agent: QLearningAgent,
    feature_mode: FeatureMode,
) -> pd.DataFrame:
    """Backtest an immutable greedy policy on real observations."""
    q_table_size = len(agent.q_table)
    visit_count = sum(agent.visit_counts.values())
    env = PortfolioEnv(
        dataset=evaluation_dataset,
        state_builder=StateBuilder(
            evaluation_dataset,
            feature_mode=feature_mode,
            history_dataset=history_dataset,
        ),
    )
    result = run_agent_backtest(
        env, CostAwareFrozenPolicy(agent, env.action_space, env.transaction_cost)
    )
    if (
        len(agent.q_table) != q_table_size
        or sum(agent.visit_counts.values()) != visit_count
    ):
        raise RuntimeError("greedy evaluation modified the trained Q-learning agent")
    return result


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
