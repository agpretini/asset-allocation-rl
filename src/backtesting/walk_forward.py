"""Walk-forward evaluation for the first RL agent."""

import pandas as pd

from agents.training import QLearningTrainingConfig, train_q_learning_agent
from backtesting.agents import run_agent_backtest
from backtesting.splitter import WalkForwardSplit
from env.portfolio_env import PortfolioEnv
from evaluation.report import build_performance_report
from features.state_builder import StateBuilder


def run_walk_forward_q_learning(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    training_config: QLearningTrainingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate Q-learning across walk-forward splits."""
    rows: list[dict] = []
    paths: list[pd.Series] = []

    for split in splits:
        train_dataset = dataset.loc[split.train_start : split.train_end]
        test_dataset = dataset.loc[split.test_start : split.test_end]

        train_env = PortfolioEnv(
            dataset=train_dataset,
            state_builder=StateBuilder(train_dataset),
        )
        training_result = train_q_learning_agent(
            env=train_env,
            config=training_config,
        )

        test_env = PortfolioEnv(
            dataset=test_dataset,
            state_builder=StateBuilder(test_dataset),
        )
        test_result = run_agent_backtest(test_env, training_result.agent)
        report = build_performance_report({"q_learning": test_result})
        metrics = report.loc["q_learning"].to_dict()
        rows.append(
            {
                "split_id": split.split_id,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "validation_start": split.validation_start,
                "validation_end": split.validation_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "q_table_states": len(training_result.agent.q_table),
                **metrics,
            },
        )
        paths.append(
            test_result["portfolio_value"].rename(f"split_{split.split_id}"),
        )

    return pd.DataFrame(rows), pd.concat(paths, axis=1, sort=False)
