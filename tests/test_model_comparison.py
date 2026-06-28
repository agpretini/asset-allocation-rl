"""Tests for model-vs-baseline comparison reports."""

import pandas as pd
import pytest

from evaluation.comparison import (
    build_model_comparison,
    build_model_comparison_markdown,
)


def test_build_model_comparison_combines_reports_and_deltas() -> None:
    model_report = pd.DataFrame(
        {
            "cagr": [0.10],
            "volatility": [0.20],
            "sharpe": [0.50],
            "sortino": [0.60],
            "maximum_drawdown": [-0.30],
            "average_turnover": [0.40],
        },
        index=["q_learning"],
    )
    baseline_report = pd.DataFrame(
        {
            "cagr": [0.08],
            "volatility": [0.15],
            "sharpe": [0.70],
            "sortino": [0.80],
            "maximum_drawdown": [-0.20],
            "average_turnover": [0.00],
        },
        index=["sp500"],
    )

    combined, deltas = build_model_comparison(model_report, baseline_report)

    assert list(combined.index) == ["q_learning", "sp500"]
    assert combined.loc["q_learning", "kind"] == "model"
    assert deltas.loc["sp500", "model_minus_cagr"] == pytest.approx(0.02)
    assert deltas.loc["sp500", "model_minus_sharpe"] == pytest.approx(-0.20)


def test_model_comparison_markdown_contains_rank_section() -> None:
    model_report = pd.DataFrame(
        {
            "cagr": [0.10],
            "volatility": [0.20],
            "sharpe": [0.50],
            "sortino": [0.60],
            "maximum_drawdown": [-0.30],
            "average_turnover": [0.40],
        },
        index=["q_learning"],
    )
    baseline_report = pd.DataFrame(
        {
            "cagr": [0.08],
            "volatility": [0.15],
            "sharpe": [0.70],
            "sortino": [0.80],
            "maximum_drawdown": [-0.20],
            "average_turnover": [0.00],
        },
        index=["sp500"],
    )
    combined, deltas = build_model_comparison(model_report, baseline_report)

    markdown = build_model_comparison_markdown(combined, deltas)

    assert "Model Vs Baselines" in markdown
    assert "Model Rank" in markdown
