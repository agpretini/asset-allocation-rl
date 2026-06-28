"""Generate a benchmark performance report from the processed dataset."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    """Run configured benchmarks and save a CSV report."""
    import pandas as pd

    from backtesting.benchmarks import run_benchmarks
    from data.loaders import load_config
    from evaluation.report import build_benchmark_report

    config = load_config()
    dataset = pd.read_parquet(config["processed_dataset_path"])
    return_columns = [asset["return_column"] for asset in config["assets"].values()]

    results = run_benchmarks(dataset[return_columns])
    report = build_benchmark_report(results)

    output_path = Path("data/processed/benchmarks/benchmark_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
