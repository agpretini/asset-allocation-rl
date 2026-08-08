"""Portable CSV and SVG artifacts for strategy comparisons."""

# ruff: noqa: E501

from pathlib import Path

import pandas as pd


def build_normalized_value_paths(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align portfolio-value paths, rebased to one at their first observation."""
    paths = {
        name: result["portfolio_value"] / float(result["portfolio_value"].iloc[0])
        for name, result in results.items()
    }
    return pd.DataFrame(paths)


def write_cumulative_return_svg(
    paths: pd.DataFrame,
    destination: Path,
    title: str,
) -> None:
    """Write a dependency-free cumulative-return line chart."""
    width, height = 1200, 650
    left, right, top, bottom = 85, 40, 65, 125
    plot_width, plot_height = width - left - right, height - top - bottom
    values = paths.astype(float)
    minimum, maximum = float(values.min().min()), float(values.max().max())
    if minimum == maximum:
        minimum -= 0.05
        maximum += 0.05
    else:
        padding = (maximum - minimum) * 0.08
        minimum -= padding
        maximum += padding

    def x(position: int) -> float:
        return left + position / max(len(paths) - 1, 1) * plot_width

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * plot_height

    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ca8a04",
        "#7c3aed",
        "#0891b2",
        "#4b5563",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="600" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111"/>',
    ]
    for tick in range(6):
        ratio = tick / 5
        value = maximum - ratio * (maximum - minimum)
        position = top + ratio * plot_height
        parts.extend(
            [
                f'<line x1="{left}" y1="{position:.1f}" x2="{width - right}" y2="{position:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{position + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}x</text>',
            ]
        )
    for series_index, column in enumerate(paths.columns):
        color = palette[series_index % len(palette)]
        points = " ".join(
            f"{x(index):.1f},{y(float(value)):.1f}"
            for index, value in enumerate(paths[column])
            if pd.notna(value)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        legend_x = 70 + (series_index % 4) * 270
        legend_y = height - 80 + (series_index // 4) * 25
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 27}" y="{legend_y + 5}" font-family="sans-serif" font-size="13">{column}</text>',
            ]
        )
    start, end = (
        pd.Timestamp(paths.index.min()).strftime("%Y-%m"),
        pd.Timestamp(paths.index.max()).strftime("%Y-%m"),
    )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="14">Historical test months: {start} to {end}; values rebased to 1.0</text>',
            f'<text x="23" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 23 {top + plot_height / 2:.1f})">Cumulative portfolio value</text>',
            "</svg>",
        ]
    )
    destination.write_text("\n".join(parts), encoding="utf-8")
