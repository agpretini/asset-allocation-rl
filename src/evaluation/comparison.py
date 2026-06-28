"""Compare model metrics against baseline metrics."""

import pandas as pd

METRIC_DIRECTIONS = {
    "cagr": "higher",
    "volatility": "lower",
    "sharpe": "higher",
    "sortino": "higher",
    "maximum_drawdown": "higher",
    "average_turnover": "lower",
}


def build_model_comparison(
    model_report: pd.DataFrame,
    baseline_report: pd.DataFrame,
    model_name: str = "q_learning",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build combined metrics and model-minus-baseline deltas."""
    if model_name not in model_report.index:
        raise ValueError(f"model_report must contain {model_name}")

    combined = pd.concat([model_report, baseline_report])
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    combined.insert(
        0,
        "kind",
        ["model" if index == model_name else "baseline" for index in combined.index],
    )

    model_metrics = combined.loc[model_name, list(METRIC_DIRECTIONS)]
    deltas = baseline_report[list(METRIC_DIRECTIONS)].copy()
    for metric in METRIC_DIRECTIONS:
        deltas[metric] = model_metrics[metric] - baseline_report[metric]
    deltas.index.name = "baseline"
    deltas.columns = [f"model_minus_{column}" for column in deltas.columns]
    return combined, deltas


def build_model_comparison_markdown(
    combined: pd.DataFrame,
    deltas: pd.DataFrame,
    model_name: str = "q_learning",
) -> str:
    """Build a concise Markdown comparison report."""
    model_rank = _build_model_rank_summary(combined, model_name)
    lines = [
        "# Model Vs Baselines",
        "",
        "## Metric Table",
        _to_markdown_table(combined),
        "",
        "## Model Minus Baseline",
        "Positive deltas are good for CAGR, Sharpe, Sortino, and maximum drawdown.",
        "Negative deltas are good for volatility and turnover.",
        "",
        _to_markdown_table(deltas),
        "",
        "## Model Rank",
        _to_markdown_table(model_rank),
        "",
    ]
    return "\n".join(lines)


def _build_model_rank_summary(
    combined: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Rank the model on each metric against all compared strategies."""
    rows = []
    metric_data = combined.drop(columns=["kind"])
    for metric, direction in METRIC_DIRECTIONS.items():
        ascending = direction == "lower"
        ranks = metric_data[metric].rank(ascending=ascending, method="min")
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "model_value": float(metric_data.loc[model_name, metric]),
                "model_rank": int(ranks.loc[model_name]),
                "strategies_compared": len(metric_data),
                "best_strategy": str(ranks.idxmin()),
            },
        )
    return pd.DataFrame(rows)


def _to_markdown_table(data: pd.DataFrame) -> str:
    """Render a dataframe as a small Markdown table without optional deps."""
    table = data.reset_index()
    columns = [str(column) for column in table.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        values = [_format_value(row[column]) for column in table.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_value(value: object) -> str:
    """Format values for Markdown output."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
