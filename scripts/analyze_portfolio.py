"""Analyze Q-learning portfolio behavior."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Generate Milestone 12 portfolio analysis artifacts."""
    from analysis.portfolio import analyze_portfolio_behavior
    from data.loaders import load_config

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    q_learning_result = pd.read_csv(
        "data/processed/q_learning/q_learning_paths.csv",
        index_col="date",
        parse_dates=True,
    )

    analysis = analyze_portfolio_behavior(
        backtest_result=q_learning_result,
        dataset=dataset,
    )

    output_dir = Path("data/processed/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis["allocations"].to_csv(output_dir / "portfolio_allocations.csv")
    analysis["allocation_summary"].to_csv(output_dir / "allocation_summary.csv")
    analysis["regime_summary"].to_csv(output_dir / "regime_summary.csv")
    analysis["turnover_summary"].to_csv(
        output_dir / "turnover_summary.csv",
        index=False,
    )
    analysis["drawdown_summary"].to_csv(
        output_dir / "drawdown_summary.csv",
        index=False,
    )
    (output_dir / "portfolio_analysis.md").write_text(
        str(analysis["text_report"]),
        encoding="utf-8",
    )
    print(output_dir / "portfolio_analysis.md")


if __name__ == "__main__":
    main()
