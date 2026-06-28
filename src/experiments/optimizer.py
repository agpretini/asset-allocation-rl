"""Hyperparameter optimization for the tabular Q-learning agent."""

from dataclasses import asdict, dataclass
from itertools import product

import pandas as pd

from agents.training import QLearningTrainingConfig
from backtesting.splitter import WalkForwardSplit
from backtesting.walk_forward import run_walk_forward_q_learning


@dataclass(frozen=True)
class HyperparameterTrial:
    """One hyperparameter optimization result."""

    trial_id: int
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
    epsilons: list[float] | None = None,
    epsilon_decays: list[float] | None = None,
    state_precisions: list[int] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, QLearningTrainingConfig]:
    """Evaluate a small Q-learning hyperparameter grid."""
    trial_configs = _build_trial_configs(
        episodes=episodes or [15, 25],
        learning_rates=learning_rates or [0.05, 0.10],
        epsilons=epsilons or [0.20, 0.30],
        epsilon_decays=epsilon_decays or [0.90, 0.95],
        state_precisions=state_precisions or [2],
        seed=seed,
    )

    rows: list[dict] = []
    for trial_id, config in enumerate(trial_configs):
        report, _ = run_walk_forward_q_learning(
            dataset=dataset,
            splits=splits,
            training_config=config,
        )
        summary = _summarize_trial(trial_id=trial_id, config=config, report=report)
        rows.append(
            {
                "trial_id": summary.trial_id,
                **asdict(summary.config),
                "mean_cagr": summary.mean_cagr,
                "mean_sharpe": summary.mean_sharpe,
                "mean_maximum_drawdown": summary.mean_maximum_drawdown,
                "mean_average_turnover": summary.mean_average_turnover,
                "score": summary.score,
            },
        )

    results = pd.DataFrame(rows).sort_values(
        ["score", "mean_cagr"],
        ascending=False,
    )
    best_row = results.iloc[0]
    best_config = QLearningTrainingConfig(
        episodes=int(best_row["episodes"]),
        learning_rate=float(best_row["learning_rate"]),
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
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            state_precision=state_precision,
            seed=seed,
        )
        for (
            episode_count,
            learning_rate,
            epsilon,
            epsilon_decay,
            state_precision,
        ) in product(
            episodes,
            learning_rates,
            epsilons,
            epsilon_decays,
            state_precisions,
        )
    ]


def _summarize_trial(
    trial_id: int,
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
        config=config,
        mean_cagr=mean_cagr,
        mean_sharpe=mean_sharpe,
        mean_maximum_drawdown=mean_maximum_drawdown,
        mean_average_turnover=mean_average_turnover,
        score=score,
    )
