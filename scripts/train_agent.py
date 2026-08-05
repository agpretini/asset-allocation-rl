"""Train the first RL agent and save its outputs."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Train a tabular Q-learning agent on the processed dataset."""
    from agents.training import QLearningTrainingConfig, train_q_learning_agent
    from backtesting.agents import run_agent_backtest
    from data.loaders import load_config
    from env.portfolio_env import PortfolioEnv
    from evaluation.report import build_performance_report
    from features.state_builder import StateBuilder
    from simulation.market_simulator import MarketSimulator, load_simulation_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-mode",
        choices=["historical", "synthetic"],
        default="historical",
    )
    args = parser.parse_args()

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    training_config = QLearningTrainingConfig(
        episodes=40,
        training_mode=args.training_mode,
        seed=42,
    )
    if args.training_mode == "historical":
        env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))
        training_result = train_q_learning_agent(
            env=env,
            config=training_config,
        )
    else:
        simulation_config = load_simulation_config()
        simulator = MarketSimulator.fit(dataset, simulation_config)
        episode_envs = (
            PortfolioEnv(
                dataset=episode,
                state_builder=StateBuilder(
                    episode,
                    history_dataset=dataset,
                ),
            )
            for episode in simulator.stream_episodes()
        )
        training_result = train_q_learning_agent(
            env=None,
            config=replace(
                training_config,
                episodes=simulation_config.episode_count,
            ),
            synthetic_episode_envs=episode_envs,
        )

    output_dir = Path("data/processed/q_learning")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_result.agent.save(output_dir / "q_learning_model.json")
    training_result.history.to_csv(
        output_dir / "q_learning_training_history.csv", index=False
    )

    evaluation_env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))
    evaluation_result = run_agent_backtest(evaluation_env, training_result.agent)
    evaluation_result.to_csv(output_dir / "q_learning_paths.csv")
    build_performance_report({"q_learning": evaluation_result}).to_csv(
        output_dir / "q_learning_report.csv",
    )
    print(output_dir / "q_learning_model.json")


if __name__ == "__main__":
    main()
