"""Hyperparameter optimization for the tabular Q-learning agent."""

from dataclasses import asdict, dataclass
from itertools import product

import pandas as pd

from agents.q_learning import (
    CONSTANT_LEARNING_RATE,
    VISIT_DECAY_LEARNING_RATE,
    LearningRateSchedule,
)
from agents.training import QLearningTrainingConfig
from backtesting.splitter import WalkForwardSplit
from backtesting.walk_forward import run_walk_forward_q_learning
from features.transforms import COMPACT_REGIME_FEATURE_MODE, FeatureMode


@dataclass(frozen=True)
class HyperparameterTrial:
    """One hyperparameter optimization result."""

    trial_id: int
    feature_mode: FeatureMode
    config: QLearningTrainingConfig
    mean_cagr: float
    mean_sharpe: float
    mean_maximum_drawdown: float
    mean_average_turnover: float
    score: float


def run_q_learning_grid_search(
    dataset: pd.DataFrame,
    splits: list[WalkForwardSplit],
    episodes: list[int] | None = None,
    learning_rates: list[float] | None = None,
    learning_rate_schedules: list[LearningRateSchedule] | None = None,
    epsilons: list[float] | None = None,
    epsilon_decays: list[float] | None = None,
    state_precisions: list[int] | None = None,
    feature_modes: list[FeatureMode] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, QLearningTrainingConfig]:
    """Evaluate a small Q-learning hyperparameter grid."""
    selected_feature_modes = feature_modes or [COMPACT_REGIME_FEATURE_MODE]
    trial_configs = _build_trial_configs(
        episodes=episodes or [15, 25],
        learning_rates=learning_rates or [0.05, 0.10],
        learning_rate_schedules=learning_rate_schedules
        or [CONSTANT_LEARNING_RATE, VISIT_DECAY_LEARNING_RATE],
        epsilons=epsilons or [0.50],
        epsilon_decays=epsilon_decays or [0.999954],
        state_precisions=state_precisions or [2],
        seed=seed,
    )

    rows: list[dict] = []
    trial_id = 0
    for feature_mode in selected_feature_modes:
        for config in trial_configs:
            report, _ = run_walk_forward_q_learning(
                dataset=dataset,
                splits=splits,
                training_config=config,
                feature_mode=feature_mode,
            )
            summary = _summarize_trial(
                trial_id=trial_id,
                feature_mode=feature_mode,
                config=config,
                report=report,
            )
            rows.append(
                {
                    "trial_id": summary.trial_id,
                    "feature_mode": summary.feature_mode,
                    **asdict(summary.config),
                    "mean_cagr": summary.mean_cagr,
                    "mean_sharpe": summary.mean_sharpe,
                    "mean_maximum_drawdown": summary.mean_maximum_drawdown,
                    "mean_average_turnover": summary.mean_average_turnover,
                    "score": summary.score,
                },
            )
            trial_id += 1

    results = pd.DataFrame(rows).sort_values(
        ["score", "mean_cagr"],
        ascending=False,
    )
    best_row = results.iloc[0]
    best_config = QLearningTrainingConfig(
        episodes=int(best_row["episodes"]),
        training_mode=str(best_row["training_mode"]),
        learning_rate=float(best_row["learning_rate"]),
        learning_rate_schedule=str(best_row["learning_rate_schedule"]),
        visit_learning_rate_decay=float(best_row["visit_learning_rate_decay"]),
        discount_factor=float(best_row["discount_factor"]),
        epsilon=float(best_row["epsilon"]),
        min_epsilon=float(best_row["min_epsilon"]),
        epsilon_decay=float(best_row["epsilon_decay"]),
        state_precision=int(best_row["state_precision"]),
        seed=int(best_row["seed"]),
    )
    return results, best_config


def _build_trial_configs(
    episodes: list[int],
    learning_rates: list[float],
    learning_rate_schedules: list[LearningRateSchedule],
    epsilons: list[float],
    epsilon_decays: list[float],
    state_precisions: list[int],
    seed: int,
) -> list[QLearningTrainingConfig]:
    """Expand hyperparameter values into training configs."""
    return [
        QLearningTrainingConfig(
            episodes=episode_count,
            learning_rate=learning_rate,
            learning_rate_schedule=learning_rate_schedule,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            state_precision=state_precision,
            seed=seed,
        )
        for (
            episode_count,
            learning_rate,
            learning_rate_schedule,
            epsilon,
            epsilon_decay,
            state_precision,
        ) in product(
            episodes,
            learning_rates,
            learning_rate_schedules,
            epsilons,
            epsilon_decays,
            state_precisions,
        )
    ]


def _summarize_trial(
    trial_id: int,
    feature_mode: FeatureMode,
    config: QLearningTrainingConfig,
    report: pd.DataFrame,
) -> HyperparameterTrial:
    """Summarize one walk-forward report as an optimization trial."""
    mean_cagr = float(report["cagr"].mean())
    mean_sharpe = float(report["sharpe"].mean())
    mean_maximum_drawdown = float(report["maximum_drawdown"].mean())
    mean_average_turnover = float(report["average_turnover"].mean())
    score = mean_sharpe + mean_cagr + mean_maximum_drawdown
    return HyperparameterTrial(
        trial_id=trial_id,
        feature_mode=feature_mode,
        config=config,
        mean_cagr=mean_cagr,
        mean_sharpe=mean_sharpe,
        mean_maximum_drawdown=mean_maximum_drawdown,
        mean_average_turnover=mean_average_turnover,
        score=score,
    )
