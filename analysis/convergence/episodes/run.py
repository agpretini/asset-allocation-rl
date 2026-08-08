"""Train synthetically and plot greedy historical-validation checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if TYPE_CHECKING:
    from collections.abc import Sequence

    from agents.q_learning import QLearningAgent
    from env.portfolio_env import PortfolioEnv


class FrozenGreedyPolicy:
    """Read-only deterministic greedy view of a Q-learning agent."""

    def __init__(self, agent: QLearningAgent) -> None:
        self.agent = agent

    def select_action(self, state: Sequence[float]) -> int:
        """Select the lowest action ID among maximal existing Q-values."""
        state_key = self.agent.state_key(state)
        q_values = self.agent.q_table.get(state_key)
        if q_values is None:
            return 0
        best_value = max(q_values)
        return min(
            action_id for action_id, value in enumerate(q_values) if value == best_value
        )


def main() -> None:
    """Run the configured synthetic-training convergence experiment."""
    from agents.q_learning import QLearningAgent
    from backtesting.splitter import generate_walk_forward_splits
    from data.loaders import load_config
    from env.portfolio_env import PortfolioEnv
    from features.state_builder import StateBuilder
    from simulation.market_simulator import MarketSimulator, load_simulation_config

    args = _parse_args()
    _validate_args(args)
    project_config = load_config()
    dataset = pd.read_parquet(project_config["processed_dataset_path"])
    split = generate_walk_forward_splits(dataset.index)[args.split_id]
    train_dataset = dataset.loc[split.train_start : split.train_end]
    validation_dataset = dataset.loc[split.validation_start : split.validation_end]

    base_simulation_config = load_simulation_config()
    simulation_config = replace(
        base_simulation_config,
        episode_count=args.episodes,
        episode_length=args.episode_length
        if args.episode_length is not None
        else base_simulation_config.episode_length,
    )
    simulator = MarketSimulator.fit(train_dataset, simulation_config)
    agent = QLearningAgent(
        n_actions=286,
        learning_rate=args.learning_rate,
        learning_rate_schedule="visit_decay",
        visit_learning_rate_decay=args.visit_learning_rate_decay,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        state_precision=2,
        seed=simulation_config.seed,
    )

    output_prefix = Path(__file__).resolve().parent / args.output_prefix
    rows = [
        _evaluate_checkpoint(
            episode=0,
            agent=agent,
            validation_dataset=validation_dataset,
            validation_history=train_dataset,
        ),
    ]
    _write_outputs(rows, output_prefix)
    print(_format_progress(rows[-1]), flush=True)

    for episode_number, synthetic_dataset in enumerate(
        simulator.stream_episodes(),
        start=1,
    ):
        training_env = PortfolioEnv(
            dataset=synthetic_dataset,
            state_builder=StateBuilder(
                synthetic_dataset,
                history_dataset=train_dataset,
            ),
        )
        _train_one_episode(agent, training_env)
        agent.decay_epsilon()

        if episode_number % args.evaluation_interval == 0:
            rows.append(
                _evaluate_checkpoint(
                    episode=episode_number,
                    agent=agent,
                    validation_dataset=validation_dataset,
                    validation_history=train_dataset,
                ),
            )
            _write_outputs(rows, output_prefix)
            print(_format_progress(rows[-1]), flush=True)

    metadata = {
        "dataset_path": project_config["processed_dataset_path"],
        "split_id": split.split_id,
        "train_start": split.train_start.date().isoformat(),
        "train_end": split.train_end.date().isoformat(),
        "validation_start": split.validation_start.date().isoformat(),
        "validation_end": split.validation_end.date().isoformat(),
        "test_start_reserved": split.test_start.date().isoformat(),
        "test_end_reserved": split.test_end.date().isoformat(),
        "episodes": args.episodes,
        "evaluation_interval": args.evaluation_interval,
        "learning_rate": args.learning_rate,
        "learning_rate_schedule": "visit_decay",
        "visit_learning_rate_decay": args.visit_learning_rate_decay,
        "discount_factor": args.discount_factor,
        "epsilon": args.epsilon,
        "min_epsilon": args.min_epsilon,
        "epsilon_decay": args.epsilon_decay,
        "simulation_config": asdict(simulation_config),
    }
    output_prefix.with_name(f"{output_prefix.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """Parse experiment parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=70_000)
    parser.add_argument("--evaluation-interval", type=int, default=2_000)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--split-id", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=0.30)
    parser.add_argument("--visit-learning-rate-decay", type=float, default=0.01)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.50)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999954)
    parser.add_argument("--output-prefix", default="episode_convergence")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate checkpoint experiment arguments."""
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive")
    if args.episodes % args.evaluation_interval != 0:
        raise ValueError("episodes must be divisible by evaluation_interval")


def _train_one_episode(agent: QLearningAgent, env: PortfolioEnv) -> None:
    """Apply Q-learning updates for one environment episode."""
    state = env.reset()
    done = False
    while not done:
        action_id = agent.select_action(state)
        next_state, reward, done, _ = env.step(action_id)
        agent.update(state, action_id, reward, next_state, done)
        state = next_state


def _evaluate_checkpoint(
    episode: int,
    agent: QLearningAgent,
    validation_dataset: pd.DataFrame,
    validation_history: pd.DataFrame,
) -> dict[str, float | int]:
    """Evaluate a read-only epsilon-zero policy on historical validation."""
    from backtesting.agents import run_agent_backtest
    from env.portfolio_env import PortfolioEnv
    from evaluation.report import build_performance_report
    from features.state_builder import StateBuilder

    q_table_size_before = len(agent.q_table)
    visit_count_before = sum(agent.visit_counts.values())
    validation_env = PortfolioEnv(
        dataset=validation_dataset,
        state_builder=StateBuilder(
            validation_dataset,
            history_dataset=validation_history,
        ),
    )
    result = run_agent_backtest(validation_env, FrozenGreedyPolicy(agent))
    metrics = build_performance_report({"validation": result}).loc["validation"]
    if len(agent.q_table) != q_table_size_before:
        raise RuntimeError("greedy evaluation modified the Q-table")
    if sum(agent.visit_counts.values()) != visit_count_before:
        raise RuntimeError("greedy evaluation modified state-action visits")
    return {
        "episode": episode,
        "validation_cumulative_reward": float(result["reward"].sum()),
        "validation_final_value": float(result["portfolio_value"].iloc[-1]),
        "validation_cagr": float(metrics["cagr"]),
        "validation_sharpe": float(metrics["sharpe"]),
        "validation_maximum_drawdown": float(metrics["maximum_drawdown"]),
        "validation_average_turnover": float(metrics["average_turnover"]),
        "q_table_states": len(agent.q_table),
        "state_action_pairs": len(agent.visit_counts),
        "training_epsilon": float(agent.epsilon),
    }


def _write_outputs(
    rows: list[dict[str, float | int]],
    output_prefix: Path,
) -> None:
    """Persist checkpoint data and its dependency-free SVG line chart."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_prefix.with_suffix(".csv"), index=False)
    _write_reward_svg(results, output_prefix.with_suffix(".svg"))


def _write_reward_svg(results: pd.DataFrame, destination: Path) -> None:
    """Write a line chart as SVG without an external plot dependency."""
    width, height = 1000, 600
    left, right, top, bottom = 100, 40, 60, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = results["episode"].astype(float)
    y_values = results["validation_cumulative_reward"].astype(float)
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        padding = max(abs(y_min) * 0.1, 0.01)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.10
        y_min -= padding
        y_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    points = " ".join(
        f"{x_position(x):.2f},{y_position(y):.2f}"
        for x, y in zip(x_values, y_values, strict=True)
    )
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="500" y="32" text-anchor="middle" '
            'font-family="sans-serif" font-size="22">'
            "Greedy validation reward during synthetic training</text>"
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height - bottom}" stroke="#222"/>'
        ),
        (
            f'<line x1="{left}" y1="{height - bottom}" '
            f'x2="{width - right}" y2="{height - bottom}" stroke="#222"/>'
        ),
    ]
    for tick in range(6):
        ratio = tick / 5
        y_value = y_max - ratio * (y_max - y_min)
        y = top + ratio * plot_height
        elements.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
                    f'y2="{y:.2f}" stroke="#e5e7eb"/>'
                ),
                (
                    f'<text x="{left - 12}" y="{y + 5:.2f}" '
                    'text-anchor="end" font-family="sans-serif" font-size="13">'
                    f"{y_value:.4f}</text>"
                ),
            ],
        )
    for tick in range(8):
        ratio = tick / 7
        x_value = x_min + ratio * (x_max - x_min)
        x = left + ratio * plot_width
        elements.append(
            (
                f'<text x="{x:.2f}" y="{height - bottom + 28}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                f"{int(round(x_value)):,}</text>"
            ),
        )
    elements.extend(
        [
            (
                f'<polyline points="{points}" fill="none" stroke="#2563eb" '
                'stroke-width="3" stroke-linejoin="round"/>'
            ),
            *[
                (
                    f'<circle cx="{x_position(x):.2f}" '
                    f'cy="{y_position(y):.2f}" r="4" fill="#1d4ed8"/>'
                )
                for x, y in zip(x_values, y_values, strict=True)
            ],
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 25}" '
                'text-anchor="middle" font-family="sans-serif" font-size="15">'
                "Synthetic training episodes</text>"
            ),
            (
                f'<text x="24" y="{top + plot_height / 2:.2f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="15" '
                f'transform="rotate(-90 24 {top + plot_height / 2:.2f})">'
                "Cumulative CRRA reward on historical validation</text>"
            ),
            "</svg>",
        ],
    )
    destination.write_text("\n".join(elements), encoding="utf-8")


def _format_progress(row: dict[str, float | int]) -> str:
    """Return one concise progress line."""
    return (
        f"episode={int(row['episode'])} "
        f"validation_reward={row['validation_cumulative_reward']:.6f} "
        f"q_states={int(row['q_table_states'])} "
        f"state_action_pairs={int(row['state_action_pairs'])}"
    )


if __name__ == "__main__":
    main()
