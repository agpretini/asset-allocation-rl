"""Analyze portfolio allocations, turnover, drawdowns, and regimes."""

import ast

import pandas as pd

ASSET_LABELS = ("arg", "ced", "sp500", "gold")


def analyze_portfolio_behavior(
    backtest_result: pd.DataFrame,
    dataset: pd.DataFrame,
) -> dict[str, pd.DataFrame | str]:
    """Build all Milestone 12 portfolio analysis artifacts."""
    allocations = build_allocation_frame(backtest_result)
    allocation_summary = summarize_allocations(allocations)
    regimes = classify_market_regimes(dataset).reindex(allocations.index)
    regime_summary = build_regime_summary(backtest_result, allocations, regimes)
    drawdown_summary = build_drawdown_summary(backtest_result)
    turnover_summary = build_turnover_summary(backtest_result)
    text_report = build_text_report(
        allocation_summary=allocation_summary,
        turnover_summary=turnover_summary,
        drawdown_summary=drawdown_summary,
        regime_summary=regime_summary,
    )

    return {
        "allocations": allocations,
        "allocation_summary": allocation_summary,
        "regime_summary": regime_summary,
        "drawdown_summary": drawdown_summary,
        "turnover_summary": turnover_summary,
        "text_report": text_report,
    }


def build_allocation_frame(backtest_result: pd.DataFrame) -> pd.DataFrame:
    """Parse the backtest weights column into one allocation column per asset."""
    if "weights" not in backtest_result.columns:
        raise ValueError("backtest_result must contain a weights column")

    allocations = [_parse_weights(value) for value in backtest_result["weights"]]
    allocation_frame = pd.DataFrame(
        allocations,
        columns=[f"w_{asset}" for asset in ASSET_LABELS],
        index=backtest_result.index,
    )
    if allocation_frame.isna().any().any():
        raise ValueError("allocations contain missing values")
    return allocation_frame


def summarize_allocations(allocations: pd.DataFrame) -> pd.DataFrame:
    """Summarize allocation level and stability for each asset."""
    rows = []
    for column in allocations.columns:
        rows.append(
            {
                "asset": column.removeprefix("w_"),
                "mean_weight": float(allocations[column].mean()),
                "min_weight": float(allocations[column].min()),
                "max_weight": float(allocations[column].max()),
                "weight_std": float(allocations[column].std(ddof=0)),
                "months_above_50pct": int((allocations[column] > 0.50).sum()),
            },
        )
    return pd.DataFrame(rows).set_index("asset")


def classify_market_regimes(dataset: pd.DataFrame) -> pd.Series:
    """Classify simple market regimes from current-month SP500 return and VIX."""
    _validate_regime_columns(dataset)
    regimes = []
    for _, row in dataset.iterrows():
        if row["vix"] >= 30.0 or row["r_sp500"] <= -0.08:
            regimes.append("stress")
        elif row["r_sp500"] >= 0.03:
            regimes.append("bull")
        elif row["r_sp500"] <= -0.03:
            regimes.append("bear")
        else:
            regimes.append("sideways")
    return pd.Series(regimes, index=dataset.index, name="regime")


def build_regime_summary(
    backtest_result: pd.DataFrame,
    allocations: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Summarize performance and allocations by regime."""
    analysis_data = pd.concat(
        [
            backtest_result[["reward", "turnover", "drawdown"]],
            allocations,
            regimes,
        ],
        axis=1,
    ).dropna()
    grouped = analysis_data.groupby("regime")
    return grouped.agg(
        months=("reward", "count"),
        mean_reward=("reward", "mean"),
        mean_turnover=("turnover", "mean"),
        worst_drawdown=("drawdown", "min"),
        mean_arg_weight=("w_arg", "mean"),
        mean_ced_weight=("w_ced", "mean"),
        mean_sp500_weight=("w_sp500", "mean"),
        mean_gold_weight=("w_gold", "mean"),
    )


def build_turnover_summary(backtest_result: pd.DataFrame) -> pd.DataFrame:
    """Summarize portfolio turnover."""
    if "turnover" not in backtest_result.columns:
        raise ValueError("backtest_result must contain a turnover column")
    turnover = backtest_result["turnover"]
    return pd.DataFrame(
        {
            "average_turnover": [float(turnover.mean())],
            "max_turnover": [float(turnover.max())],
            "zero_turnover_months": [int((turnover == 0.0).sum())],
        },
    )


def build_drawdown_summary(backtest_result: pd.DataFrame) -> pd.DataFrame:
    """Summarize drawdown behavior."""
    if "drawdown" not in backtest_result.columns:
        raise ValueError("backtest_result must contain a drawdown column")
    drawdown = backtest_result["drawdown"]
    return pd.DataFrame(
        {
            "maximum_drawdown": [float(drawdown.min())],
            "average_drawdown": [float(drawdown.mean())],
            "months_in_drawdown": [int((drawdown < 0.0).sum())],
        },
    )


def build_text_report(
    allocation_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    drawdown_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> str:
    """Build a concise Markdown portfolio analysis report."""
    highest_mean_asset = allocation_summary["mean_weight"].idxmax()
    highest_stability_asset = allocation_summary["weight_std"].idxmin()
    arg_summary = allocation_summary.loc["arg"]
    gold_summary = allocation_summary.loc["gold"]

    lines = [
        "# Portfolio Analysis",
        "",
        "## Allocation Behavior",
        f"- Highest average exposure: {highest_mean_asset}.",
        f"- Most stable allocation: {highest_stability_asset}.",
        (
            "- Argentine exposure above 50%: "
            f"{int(arg_summary['months_above_50pct'])} months."
        ),
        f"- Gold exposure above 50%: {int(gold_summary['months_above_50pct'])} months.",
        "",
        "## Risk And Trading",
        f"- Average turnover: {turnover_summary['average_turnover'].iloc[0]:.4f}.",
        f"- Maximum turnover: {turnover_summary['max_turnover'].iloc[0]:.4f}.",
        f"- Maximum drawdown: {drawdown_summary['maximum_drawdown'].iloc[0]:.4f}.",
        "",
        "## Regime Behavior",
        _to_markdown_table(regime_summary),
        "",
    ]
    return "\n".join(lines)


def _to_markdown_table(data: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional deps."""
    table = data.reset_index()
    columns = [str(column) for column in table.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        values = [_format_markdown_value(row[column]) for column in table.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_markdown_value(value: object) -> str:
    """Format a value for a compact Markdown table."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_weights(value: object) -> list[float]:
    """Parse list-like weights stored by pandas CSV output."""
    if isinstance(value, list):
        weights = value
    elif isinstance(value, str):
        weights = ast.literal_eval(value)
    else:
        raise ValueError("weights must be a list or stringified list")

    if len(weights) != len(ASSET_LABELS):
        raise ValueError("weights must contain four values")
    weights = [float(weight) for weight in weights]
    if any(weight < 0.0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one")
    return weights


def _validate_regime_columns(dataset: pd.DataFrame) -> None:
    """Validate the columns required for regime classification."""
    required_columns = {"r_sp500", "vix"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required regime columns: {missing}")
