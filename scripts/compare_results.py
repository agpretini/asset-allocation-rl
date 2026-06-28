"""Generate a model-vs-baselines comparison report."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Write consolidated comparison reports."""
    from evaluation.comparison import (
        build_model_comparison,
        build_model_comparison_markdown,
    )

    model_report = pd.read_csv(
        "data/processed/q_learning/q_learning_report.csv",
        index_col=0,
    )
    baseline_report = pd.read_csv(
        "data/processed/baselines/baseline_report.csv",
        index_col=0,
    )

    combined, deltas = build_model_comparison(
        model_report=model_report,
        baseline_report=baseline_report,
    )
    markdown = build_model_comparison_markdown(combined=combined, deltas=deltas)

    output_dir = Path("data/processed/comparison")
    combined.to_csv(output_dir / "model_vs_baselines_report.csv")
    deltas.to_csv(output_dir / "model_vs_baselines_deltas.csv")
    (output_dir / "model_vs_baselines.md").write_text(markdown, encoding="utf-8")
    print(output_dir / "model_vs_baselines.md")


if __name__ == "__main__":
    main()
