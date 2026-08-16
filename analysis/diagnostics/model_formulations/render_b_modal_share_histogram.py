"""Render the distribution of Model B's modal-target share by market regime."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import pandas as pd

DIAGNOSTIC_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Write a dependency-free SVG histogram from the full market-policy CSV."""
    results_dir = DIAGNOSTIC_DIR / "results"
    policy = pd.read_csv(
        results_dir / "full_market_policy_B_market_state_cost_at_decision.csv"
    )
    shares = policy["modal_target_share"].astype(float)
    bucket_ids = (shares.clip(upper=0.999999) * 10).astype(int)
    counts = bucket_ids.value_counts().reindex(range(10), fill_value=0).sort_index()
    destination = results_dir / "B_modal_target_share_histogram.svg"
    _write_histogram(counts, shares, destination)
    print(destination)


def _write_histogram(counts: pd.Series, shares: pd.Series, destination: Path) -> None:
    """Draw one SVG bar per ten-percentage-point modal-share bucket."""
    width, height = 1250, 720
    left, right, top, bottom = 95, 45, 120, 115
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = max(int(counts.max()), 1)
    bar_width = plot_width / len(counts) * 0.72
    gap = plot_width / len(counts) * 0.28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="625" y="35" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="bold">Model B: consistency of the modal target</text>',
        '<text x="625" y="61" text-anchor="middle" font-family="sans-serif" font-size="21" font-weight="bold">across starting portfolios</text>',
        '<text x="625" y="88" text-anchor="middle" font-family="sans-serif" font-size="14">Each market regime contributes one observation: the fraction of 286 starting portfolios selecting its most frequent cost-aware target.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111"/>',
    ]
    for tick in range(6):
        value = round(maximum * tick / 5)
        y = height - bottom - value / maximum * plot_height
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value}</text>',
            ]
        )
    for index, count in enumerate(counts.to_list()):
        x = left + index * (bar_width + gap) + gap / 2
        bar_height = int(count) / maximum * plot_height
        y = height - bottom - bar_height
        label = f"{index * 10}–{(index + 1) * 10}%"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">{int(count)}</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{height - bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 32}" text-anchor="middle" font-family="sans-serif" font-size="14">Modal-target share across the 286 starting portfolios</text>',
            f'<text x="25" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 25 {top + plot_height / 2:.1f})">Number of market regimes</text>',
            f'<text x="{width - right}" y="{height - 72}" text-anchor="end" font-family="sans-serif" font-size="13">Median: {shares.median():.1%} | Min: {shares.min():.1%} | Max: {shares.max():.1%}</text>',
            "</svg>",
        ]
    )
    destination.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
