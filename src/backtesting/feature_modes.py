"""Compare Q-learning walk-forward performance across feature modes."""

import pandas as pd

from agents.training import QLearningTrainingConfig, train_q_learning_agent
from backtesting.agents import run_agent_backtest
from backtesting.splitter import WalkForwardSplit
from backtesting.walk_forward import run_walk_forward_q_learning
from env.portfolio_env import PortfolioEnv
from features.state_builder import StateBuilder
from features.transforms import FeatureMode


def run_feature_mode_walk_forward_comparison(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_modes: list[FeatureMode],
    training_config: QLearningTrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run identical walk-forward evaluations for each feature mode."""
    reports: list[pd.DataFrame] = []
    paths: list[pd.DataFrame] = []

    for feature_mode in feature_modes:
        report, path = run_walk_forward_q_learning(
            dataset=dataset,
            splits=splits,
            training_config=training_config,
            feature_mode=feature_mode,
        )
        reports.append(report)
        paths.append(path.add_prefix(f"{feature_mode}_"))

    return (
        pd.concat(reports, ignore_index=True),
        pd.concat(paths, axis=1, sort=False),
    )


def run_feature_mode_decision_comparison(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_modes: list[FeatureMode],
    training_config: QLearningTrainingConfig | None = None,
) -> pd.DataFrame:
    """Return walk-forward action paths for each feature mode."""
    decision_paths: list[pd.DataFrame] = []

    for feature_mode in feature_modes:
        for split in splits:
            train_dataset = dataset.loc[split.train_start : split.train_end]
            test_dataset = dataset.loc[split.test_start : split.test_end]
            test_history = dataset.loc[: split.validation_end]

            train_env = PortfolioEnv(
                dataset=train_dataset,
                state_builder=StateBuilder(train_dataset, feature_mode=feature_mode),
            )
            training_result = train_q_learning_agent(
                env=train_env,
                config=training_config,
            )

            test_env = PortfolioEnv(
                dataset=test_dataset,
                state_builder=StateBuilder(
                    test_dataset,
                    feature_mode=feature_mode,
                    history_dataset=test_history,
                ),
            )
            path = run_agent_backtest(test_env, training_result.agent).reset_index()
            weights = pd.DataFrame(
                path["weights"].to_list(),
                columns=["w_arg", "w_ced", "w_sp500", "w_gold"],
                index=path.index,
            )
            path = pd.concat([path.drop(columns=["weights"]), weights], axis=1)
            path.insert(0, "split_id", split.split_id)
            path.insert(0, "feature_mode", feature_mode)
            decision_paths.append(path)

    return pd.concat(decision_paths, ignore_index=True)
