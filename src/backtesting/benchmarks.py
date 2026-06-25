"""Configured passive benchmark portfolios."""

from dataclasses import dataclass

import pandas as pd

from backtesting.engine import run_fixed_weight_backtest
from data.loaders import CONFIG_PATH, load_config


@dataclass(frozen=True)
class Benchmark:
    """A passive benchmark allocation."""

    key: str
    name: str
    weights: dict[str, float]


def load_configured_benchmarks() -> list[Benchmark]:
    """Load benchmark definitions from project configuration."""
    config = load_config(CONFIG_PATH)
    return [
        Benchmark(
            key=key,
            name=definition["name"],
            weights={
                column: float(weight)
                for column, weight in definition["weights"].items()
            },
        )
        for key, definition in config["benchmarks"].items()
    ]


def run_benchmarks(
    returns: pd.DataFrame,
    benchmarks: list[Benchmark] | None = None,
    initial_value: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Run every configured benchmark on the same return data."""
    selected_benchmarks = benchmarks or load_configured_benchmarks()
    return {
        benchmark.key: run_fixed_weight_backtest(
            returns=returns,
            weights=benchmark.weights,
            initial_value=initial_value,
        )
        for benchmark in selected_benchmarks
    }
