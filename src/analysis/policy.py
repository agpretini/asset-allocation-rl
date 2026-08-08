"""Inspectable artifacts for a compact-regime tabular Q-learning policy."""

# ruff: noqa: E501

from collections import defaultdict
from itertools import product
from pathlib import Path

import pandas as pd

from agents.q_learning import QLearningAgent
from env.action_space import generate_action_space
from env.portfolio_env import calculate_turnover

ASSET_COLUMNS = ("w_arg", "w_ced", "w_sp500", "w_gold")
ASSET_LABELS = ("Argentine equities", "CEDEARs", "S&P 500", "Gold")
LEADER_LABELS = ("ARG", "CEDEARs", "S&P 500", "Gold")
REGIME_LABELS = ("Low", "Medium", "High")


def build_compact_policy_table(agent: QLearningAgent) -> pd.DataFrame:
    """Enumerate the policy over all compact market states and allocations."""
    action_space = generate_action_space()
    visit_summary = build_state_visit_summary(agent).set_index("state_key")
    rows: list[dict[str, object]] = []
    for leader, strength, usd, vix in product(range(4), range(3), range(3), range(3)):
        for current_action_id, current_weights in enumerate(action_space):
            state = [leader, strength, usd, vix, *current_weights]
            state_key = agent.state_key(state)
            q_values = agent.q_table.get(state_key)
            known_state = q_values is not None
            q_values = q_values or [0.0] * agent.n_actions
            best_value = max(q_values)
            action_id = min(
                index for index, value in enumerate(q_values) if value == best_value
            )
            target_weights = action_space[action_id]
            visits = (
                visit_summary.loc[state_key]
                if state_key in visit_summary.index
                else None
            )
            rows.append(
                {
                    "momentum_leader": leader,
                    "momentum_strength": strength,
                    "usd_ars_regime": usd,
                    "vix_regime": vix,
                    "current_action_id": current_action_id,
                    "state_key": state_key,
                    "known_state": known_state,
                    "state_total_visits": int(visits["total_visits"])
                    if visits is not None
                    else 0,
                    "visited_action_count": int(visits["visited_action_count"])
                    if visits is not None
                    else 0,
                    "selected_action_id": action_id,
                    "selected_q_value": float(best_value),
                    "expected_turnover": calculate_turnover(
                        target_weights, current_weights
                    ),
                    "w_arg": target_weights[0],
                    "w_ced": target_weights[1],
                    "w_sp500": target_weights[2],
                    "w_gold": target_weights[3],
                },
            )
    return pd.DataFrame(rows)


def build_state_visit_summary(agent: QLearningAgent) -> pd.DataFrame:
    """Aggregate training visits by full state without changing the Q-table."""
    visits_by_state: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (state_key, action_id), count in agent.visit_counts.items():
        visits_by_state[state_key].append((action_id, count))
    rows: list[dict[str, object]] = []
    for state_key in agent.q_table:
        values = [float(value) for value in state_key.split("|")]
        if len(values) != 8:
            raise ValueError("compact-regime state key must contain eight values")
        action_visits = visits_by_state.get(state_key, [])
        total_visits = sum(count for _, count in action_visits)
        most_visited_action, max_action_visits = (
            min(action_visits, key=lambda item: (-item[1], item[0]))
            if action_visits
            else (-1, 0)
        )
        rows.append(
            {
                "state_key": state_key,
                "momentum_leader": int(values[0]),
                "momentum_strength": int(values[1]),
                "usd_ars_regime": int(values[2]),
                "vix_regime": int(values[3]),
                "current_w_arg": values[4],
                "current_w_ced": values[5],
                "current_w_sp500": values[6],
                "current_w_gold": values[7],
                "total_visits": total_visits,
                "visited_action_count": len(action_visits),
                "most_visited_action_id": most_visited_action,
                "most_visited_action_count": max_action_visits,
            },
        )
    columns = [
        "state_key",
        "momentum_leader",
        "momentum_strength",
        "usd_ars_regime",
        "vix_regime",
        "current_w_arg",
        "current_w_ced",
        "current_w_sp500",
        "current_w_gold",
        "total_visits",
        "visited_action_count",
        "most_visited_action_id",
        "most_visited_action_count",
    ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["total_visits", "state_key"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_reference_regime_policy(policy_table: pd.DataFrame) -> pd.DataFrame:
    """Return the 108 market-regime decisions from a balanced discrete portfolio."""
    action_space = generate_action_space()
    reference_action_id = action_space.index([0.3, 0.3, 0.2, 0.2])
    reference = policy_table.loc[
        policy_table["current_action_id"] == reference_action_id
    ].copy()
    return reference.sort_values(
        ["vix_regime", "usd_ars_regime", "momentum_leader", "momentum_strength"],
    ).reset_index(drop=True)


def write_reference_policy_svg(
    reference_policy: pd.DataFrame,
    destination: Path,
) -> None:
    """Write four readable heatmaps covering all 108 reference market states."""
    width, height = 1440, 1460
    panel_width, panel_height = 650, 650
    panel_positions = ((55, 95), (735, 95), (55, 790), (735, 790))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="720" y="35" text-anchor="middle" font-family="sans-serif" font-size="24">Compact-regime greedy policy: target weights by full market regime</text>',
        '<text x="720" y="60" text-anchor="middle" font-family="sans-serif" font-size="14">Current allocation: [30% ARG, 30% CEDEARs, 20% S&amp;P 500, 20% Gold]. Rows: VIX / USD-ARS / momentum leader; columns: momentum strength.</text>',
    ]
    for asset_index, (column, label, position) in enumerate(
        zip(ASSET_COLUMNS, ASSET_LABELS, panel_positions, strict=True)
    ):
        parts.extend(
            _build_asset_heatmap(
                reference_policy=reference_policy,
                weight_column=column,
                title=label,
                asset_index=asset_index,
                x_offset=position[0],
                y_offset=position[1],
                panel_width=panel_width,
                panel_height=panel_height,
            )
        )
    parts.append("</svg>")
    destination.write_text("\n".join(parts), encoding="utf-8")


def _build_asset_heatmap(
    reference_policy: pd.DataFrame,
    weight_column: str,
    title: str,
    asset_index: int,
    x_offset: int,
    y_offset: int,
    panel_width: int,
    panel_height: int,
) -> list[str]:
    """Render one 36-by-3 regime heatmap for an asset's target weight."""
    color_endpoints = ("#2563eb", "#dc2626", "#16a34a", "#ca8a04")
    left, top, right, bottom = 175, 62, 12, 30
    cell_width = (panel_width - left - right) / 3
    cell_height = (panel_height - top - bottom) / 36
    parts = [
        f'<rect x="{x_offset}" y="{y_offset}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#9ca3af"/>',
        f'<text x="{x_offset + panel_width / 2:.1f}" y="{y_offset + 30}" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">{title}</text>',
    ]
    for strength, label in enumerate(REGIME_LABELS):
        x = x_offset + left + (strength + 0.5) * cell_width
        parts.append(
            f'<text x="{x:.1f}" y="{y_offset + 52}" text-anchor="middle" font-family="sans-serif" font-size="12">Strength: {label}</text>'
        )
    row = 0
    for vix, usd, leader in product(range(3), range(3), range(4)):
        row_y = y_offset + top + row * cell_height
        row_label = f"VIX {REGIME_LABELS[vix][0]} · USD {REGIME_LABELS[usd][0]} · {LEADER_LABELS[leader]}"
        parts.append(
            f'<text x="{x_offset + left - 7}" y="{row_y + cell_height * 0.72:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{row_label}</text>'
        )
        for strength in range(3):
            match = reference_policy.loc[
                (reference_policy["vix_regime"] == vix)
                & (reference_policy["usd_ars_regime"] == usd)
                & (reference_policy["momentum_leader"] == leader)
                & (reference_policy["momentum_strength"] == strength),
                weight_column,
            ]
            if len(match) != 1:
                raise ValueError(
                    "reference policy must contain exactly one row per regime"
                )
            value = float(match.iloc[0])
            x = x_offset + left + strength * cell_width
            fill = _weight_color(color_endpoints[asset_index], value)
            text_color = "white" if value >= 0.55 else "#111827"
            parts.extend(
                [
                    f'<rect x="{x:.1f}" y="{row_y:.1f}" width="{cell_width:.1f}" height="{cell_height:.1f}" fill="{fill}" stroke="white"/>',
                    f'<text x="{x + cell_width / 2:.1f}" y="{row_y + cell_height * 0.72:.1f}" text-anchor="middle" fill="{text_color}" font-family="sans-serif" font-size="10">{value:.1f}</text>',
                ]
            )
        row += 1
    return parts


def _weight_color(endpoint: str, weight: float) -> str:
    """Mix white into an asset color according to its selected weight."""
    red, green, blue = (int(endpoint[index : index + 2], 16) for index in (1, 3, 5))
    mix = 1.0 - weight * 0.88
    return f"#{int(255 * mix + red * (1 - mix)):02x}{int(255 * mix + green * (1 - mix)):02x}{int(255 * mix + blue * (1 - mix)):02x}"
