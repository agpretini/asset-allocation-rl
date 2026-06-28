"""Train the first RL agent and save its outputs."""

import sys
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

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    env = PortfolioEnv(dataset=dataset, state_builder=StateBuilder(dataset))
    training_result = train_q_learning_agent(
        env=env,
        config=QLearningTrainingConfig(episodes=40, seed=42),
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
