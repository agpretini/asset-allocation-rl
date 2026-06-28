"""Tests for portfolio behavior analysis."""

import pandas as pd
import pytest

from analysis.portfolio import (
    analyze_portfolio_behavior,
    build_allocation_frame,
    classify_market_regimes,
    summarize_allocations,
)


def test_build_allocation_frame_parses_weight_strings() -> None:
    result = pd.DataFrame(
        {"weights": ["[0.7, 0.2, 0.0, 0.1]", "[0.0, 0.0, 0.5, 0.5]"]},
        index=pd.date_range("2022-01-31", periods=2, freq="ME"),
    )

    allocations = build_allocation_frame(result)

    assert list(allocations.columns) == ["w_arg", "w_ced", "w_sp500", "w_gold"]
    assert allocations.iloc[0].to_list() == pytest.approx([0.7, 0.2, 0.0, 0.1])


def test_summarize_allocations_reports_stability() -> None:
    allocations = pd.DataFrame(
        {
            "w_arg": [0.7, 0.6, 0.0],
            "w_ced": [0.2, 0.2, 0.2],
            "w_sp500": [0.0, 0.1, 0.4],
            "w_gold": [0.1, 0.1, 0.4],
        },
    )

    summary = summarize_allocations(allocations)

    assert summary.loc["arg", "months_above_50pct"] == 2
    assert summary.loc["ced", "weight_std"] == pytest.approx(0.0)


def test_classify_market_regimes_uses_current_month_data() -> None:
    dataset = pd.DataFrame(
        {
            "r_sp500": [0.04, -0.04, -0.09, 0.01],
            "vix": [18.0, 20.0, 24.0, 35.0],
        },
        index=pd.date_range("2022-01-31", periods=4, freq="ME"),
    )

    regimes = classify_market_regimes(dataset)

    assert regimes.to_list() == ["bull", "bear", "stress", "stress"]


def test_analyze_portfolio_behavior_returns_all_artifacts() -> None:
    dates = pd.date_range("2022-01-31", periods=3, freq="ME")
    result = pd.DataFrame(
        {
            "weights": [
                "[0.7, 0.2, 0.0, 0.1]",
                "[0.0, 0.0, 0.5, 0.5]",
                "[0.1, 0.1, 0.1, 0.7]",
            ],
            "reward": [0.01, -0.02, 0.03],
            "turnover": [0.4, 0.7, 0.3],
            "drawdown": [0.0, -0.02, 0.0],
        },
        index=dates,
    )
    dataset = pd.DataFrame(
        {
            "r_sp500": [0.04, -0.04, 0.01],
            "vix": [18.0, 20.0, 35.0],
        },
        index=dates,
    )

    analysis = analyze_portfolio_behavior(result, dataset)

    assert set(analysis) == {
        "allocations",
        "allocation_summary",
        "regime_summary",
        "drawdown_summary",
        "turnover_summary",
        "text_report",
    }
    assert "Portfolio Analysis" in str(analysis["text_report"])
