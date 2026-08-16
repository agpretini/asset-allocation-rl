"""Inspectable artifacts for a compact-regime tabular Q-learning policy."""

# ruff: noqa: E501

from collections import defaultdict
from html import escape
from itertools import product
from math import log1p
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
    """Enumerate the deployable policy over every compact market regime.

    Legacy Model A agents include the current allocation in their learned state,
    producing one row per market regime and origin allocation. Model B agents
    have market-only states: the product rule reports the modal cost-aware
    target across all possible origins for each market regime.
    """
    if _uses_market_only_state(agent):
        return _build_model_b_policy_table(agent)

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
    """Aggregate training visits by state without changing the Q-table."""
    visits_by_state: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (state_key, action_id), count in agent.visit_counts.items():
        visits_by_state[state_key].append((action_id, count))
    rows: list[dict[str, object]] = []
    state_widths = {len(state_key.split("|")) for state_key in agent.q_table}
    if len(state_widths) > 1:
        raise ValueError("Q-table contains mixed state representations")
    state_width = next(iter(state_widths), 4)
    if state_width not in {4, 8}:
        raise ValueError("compact-regime state key must contain four or eight values")
    for state_key in agent.q_table:
        values = [float(value) for value in state_key.split("|")]
        action_visits = visits_by_state.get(state_key, [])
        total_visits = sum(count for _, count in action_visits)
        most_visited_action, max_action_visits = (
            min(action_visits, key=lambda item: (-item[1], item[0]))
            if action_visits
            else (-1, 0)
        )
        row = {
            "state_key": state_key,
            "momentum_leader": int(values[0]),
            "momentum_strength": int(values[1]),
            "usd_ars_regime": int(values[2]),
            "vix_regime": int(values[3]),
            "total_visits": total_visits,
            "visited_action_count": len(action_visits),
            "most_visited_action_id": most_visited_action,
            "most_visited_action_count": max_action_visits,
        }
        if state_width == 8:
            row.update(
                {
                    "current_w_arg": values[4],
                    "current_w_ced": values[5],
                    "current_w_sp500": values[6],
                    "current_w_gold": values[7],
                }
            )
        rows.append(row)
    columns = [
        "state_key",
        "momentum_leader",
        "momentum_strength",
        "usd_ars_regime",
        "vix_regime",
        "total_visits",
        "visited_action_count",
        "most_visited_action_id",
        "most_visited_action_count",
    ]
    if state_width == 8:
        columns[5:5] = [
            "current_w_arg",
            "current_w_ced",
            "current_w_sp500",
            "current_w_gold",
        ]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["total_visits", "state_key"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_reference_regime_policy(policy_table: pd.DataFrame) -> pd.DataFrame:
    """Return one deployable decision for each of the 108 market regimes."""
    if "current_action_id" not in policy_table.columns:
        return policy_table.sort_values(
            ["vix_regime", "usd_ars_regime", "momentum_leader", "momentum_strength"],
        ).reset_index(drop=True)
    action_space = generate_action_space()
    reference_action_id = action_space.index([0.3, 0.3, 0.2, 0.2])
    reference = policy_table.loc[
        policy_table["current_action_id"] == reference_action_id
    ].copy()
    return reference.sort_values(
        ["vix_regime", "usd_ars_regime", "momentum_leader", "momentum_strength"],
    ).reset_index(drop=True)


def build_fixed_market_policy_table(
    agent: QLearningAgent,
    market_state: list[float],
    actual_starting_weights: list[float],
) -> pd.DataFrame:
    """Show greedy targets for every starting allocation in one market regime."""
    if len(market_state) != 4 or len(actual_starting_weights) != 4:
        raise ValueError(
            "market_state and actual_starting_weights must have four values"
        )
    action_space = generate_action_space()
    if _uses_market_only_state(agent):
        return _build_model_b_fixed_market_policy(
            agent=agent,
            market_state=market_state,
            actual_starting_weights=actual_starting_weights,
            action_space=action_space,
        )
    starts = [("prediction_assumption", -1, actual_starting_weights)] + [
        ("action_grid", action_id, weights)
        for action_id, weights in enumerate(action_space)
        if weights != actual_starting_weights
    ]
    visit_summary = build_state_visit_summary(agent).set_index("state_key")
    rows: list[dict[str, object]] = []
    for source, starting_action_id, weights in starts:
        state = [*market_state, *weights]
        state_key = agent.state_key(state)
        q_values = agent.q_table.get(state_key)
        known_state = q_values is not None
        q_values = q_values or [0.0] * agent.n_actions
        best_value = max(q_values)
        target_action_id = min(
            action_id for action_id, value in enumerate(q_values) if value == best_value
        )
        target_weights = action_space[target_action_id]
        visits = (
            visit_summary.loc[state_key] if state_key in visit_summary.index else None
        )
        rows.append(
            {
                "starting_portfolio_type": source,
                "starting_action_id": starting_action_id,
                "state_key": state_key,
                "known_state": known_state,
                "state_total_visits": int(visits["total_visits"])
                if visits is not None
                else 0,
                "visited_action_count": int(visits["visited_action_count"])
                if visits is not None
                else 0,
                "selected_action_id": target_action_id,
                "selected_q_value": float(best_value),
                "expected_turnover": calculate_turnover(target_weights, weights),
                **{
                    f"start_{asset}": weight
                    for asset, weight in zip(
                        ("arg", "ced", "sp500", "gold"), weights, strict=True
                    )
                },
                **{
                    f"target_{asset}": weight
                    for asset, weight in zip(
                        ("arg", "ced", "sp500", "gold"), target_weights, strict=True
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def _uses_market_only_state(agent: QLearningAgent) -> bool:
    """Return whether an agent was trained with the four-field Model B state."""
    if not agent.q_table:
        return False
    widths = {len(state_key.split("|")) for state_key in agent.q_table}
    if len(widths) != 1:
        raise ValueError("Q-table contains mixed state representations")
    return widths == {4}


def _build_model_b_policy_table(agent: QLearningAgent) -> pd.DataFrame:
    """Build the market-only Model B policy using its modal target rule."""
    action_space = generate_action_space()
    visit_summary = build_state_visit_summary(agent).set_index("state_key")
    rows: list[dict[str, object]] = []
    for leader, strength, usd, vix in product(range(4), range(3), range(3), range(3)):
        state = [leader, strength, usd, vix]
        state_key = agent.state_key(state)
        q_values = agent.q_table.get(state_key)
        known_state = q_values is not None
        q_values = q_values or [0.0] * agent.n_actions
        target_action_id, modal_share = _modal_cost_aware_action(q_values, action_space)
        visits = (
            visit_summary.loc[state_key] if state_key in visit_summary.index else None
        )
        target_weights = action_space[target_action_id]
        rows.append(
            {
                "momentum_leader": leader,
                "momentum_strength": strength,
                "usd_ars_regime": usd,
                "vix_regime": vix,
                "state_key": state_key,
                "known_state": known_state,
                "state_total_visits": int(visits["total_visits"])
                if visits is not None
                else 0,
                "visited_action_count": int(visits["visited_action_count"])
                if visits is not None
                else 0,
                "selected_action_id": target_action_id,
                "selected_q_value": float(q_values[target_action_id]),
                "modal_target_share": modal_share,
                "w_arg": target_weights[0],
                "w_ced": target_weights[1],
                "w_sp500": target_weights[2],
                "w_gold": target_weights[3],
            }
        )
    return pd.DataFrame(rows)


def _build_model_b_fixed_market_policy(
    agent: QLearningAgent,
    market_state: list[float],
    actual_starting_weights: list[float],
    action_space: list[list[float]],
) -> pd.DataFrame:
    """Show Model B's cost-aware target for every possible origin allocation."""
    starts = [("prediction_assumption", -1, actual_starting_weights)] + [
        ("action_grid", action_id, weights)
        for action_id, weights in enumerate(action_space)
        if weights != actual_starting_weights
    ]
    state_key = agent.state_key(market_state)
    q_values = agent.q_table.get(state_key)
    known_state = q_values is not None
    q_values = q_values or [0.0] * agent.n_actions
    visits = build_state_visit_summary(agent).set_index("state_key")
    state_visits = visits.loc[state_key] if state_key in visits.index else None
    rows: list[dict[str, object]] = []
    for source, starting_action_id, weights in starts:
        scores = [
            value - 0.014 * calculate_turnover(weights, target)
            for value, target in zip(q_values, action_space, strict=True)
        ]
        score = max(scores)
        target_action_id = min(
            action_id for action_id, value in enumerate(scores) if value == score
        )
        target_weights = action_space[target_action_id]
        rows.append(
            {
                "starting_portfolio_type": source,
                "starting_action_id": starting_action_id,
                "state_key": state_key,
                "known_state": known_state,
                "state_total_visits": int(state_visits["total_visits"])
                if state_visits is not None
                else 0,
                "visited_action_count": int(state_visits["visited_action_count"])
                if state_visits is not None
                else 0,
                "selected_action_id": target_action_id,
                "selected_q_value": float(q_values[target_action_id]),
                "selected_score": float(score),
                "expected_turnover": calculate_turnover(target_weights, weights),
                **{
                    f"start_{asset}": weight
                    for asset, weight in zip(
                        ("arg", "ced", "sp500", "gold"), weights, strict=True
                    )
                },
                **{
                    f"target_{asset}": weight
                    for asset, weight in zip(
                        ("arg", "ced", "sp500", "gold"), target_weights, strict=True
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def _modal_cost_aware_action(
    q_values: list[float], action_space: list[list[float]]
) -> tuple[int, float]:
    """Choose the deterministic modal cost-aware target over all origins."""
    targets: list[int] = []
    for starting_weights in action_space:
        scores = [
            value - 0.014 * calculate_turnover(starting_weights, target)
            for value, target in zip(q_values, action_space, strict=True)
        ]
        best = max(scores)
        targets.append(
            min(index for index, value in enumerate(scores) if value == best)
        )
    target_counts = pd.Series(targets).value_counts()
    max_count = int(target_counts.max())
    action_id = min(
        int(target_id)
        for target_id, count in target_counts.items()
        if int(count) == max_count
    )
    return action_id, max_count / len(action_space)


def write_fixed_market_policy_svg(
    policy: pd.DataFrame,
    market_state: list[float],
    signal_date: str,
    destination: Path,
    title: str = "August 2026 policy by starting allocation",
    decision_heading: str = "Greedy target portfolio",
    description: str | None = None,
) -> None:
    """Render all starting allocations, greedy targets, and visit counts."""
    row_height, header_height, width = 16, 130, 1330
    height = header_height + len(policy) * row_height + 45
    columns = [
        ("Start ARG", "start_arg", 210, "#2563eb"),
        ("Start CED", "start_ced", 275, "#dc2626"),
        ("Start S&P", "start_sp500", 340, "#16a34a"),
        ("Start Gold", "start_gold", 405, "#ca8a04"),
        ("Target ARG", "target_arg", 505, "#2563eb"),
        ("Target CED", "target_ced", 570, "#dc2626"),
        ("Target S&P", "target_sp500", 635, "#16a34a"),
        ("Target Gold", "target_gold", 700, "#ca8a04"),
    ]
    regime = (
        f"Momentum leader: {LEADER_LABELS[int(market_state[0])]} | strength: {REGIME_LABELS[int(market_state[1])]} | "
        f"USD/ARS: {REGIME_LABELS[int(market_state[2])]} | VIX: {REGIME_LABELS[int(market_state[3])]}"
    )
    max_visits = max(int(value) for value in policy["state_total_visits"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="30" y="32" font-family="sans-serif" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<text x="30" y="57" font-family="sans-serif" font-size="14">Observed market state on {escape(signal_date)}: {escape(regime)}</text>',
        f'<text x="30" y="82" font-family="sans-serif" font-size="13">{escape(description or "Each row fixes this market state. Colors and values show current versus greedy target weights; visits aggregate all training actions from that exact state.")}</text>',
        '<text x="30" y="116" font-family="sans-serif" font-size="12" font-weight="bold">Starting portfolio</text>',
        f'<text x="505" y="116" font-family="sans-serif" font-size="12" font-weight="bold">{escape(decision_heading)}</text>',
        '<text x="820" y="116" font-family="sans-serif" font-size="12" font-weight="bold">Training visits</text>',
    ]
    for label, _, x, _ in columns:
        parts.append(
            f'<text x="{x + 25}" y="128" text-anchor="middle" font-family="sans-serif" font-size="10">{escape(label)}</text>'
        )
    for row_index, (_, row) in enumerate(policy.iterrows()):
        y = header_height + row_index * row_height
        highlighted = row["starting_portfolio_type"] == "prediction_assumption"
        if highlighted:
            parts.append(
                f'<rect x="20" y="{y}" width="1280" height="{row_height}" fill="#fef3c7"/>'
            )
        label = (
            "Actual 25/25/25/25"
            if highlighted
            else f"Grid {int(row['starting_action_id']):03d}"
        )
        parts.append(
            f'<text x="25" y="{y + 12}" font-family="sans-serif" font-size="10" font-weight="bold">{label}</text>'
        )
        for _, column, x, color in columns:
            value = float(row[column])
            parts.extend(
                [
                    f'<rect x="{x}" y="{y + 1}" width="50" height="14" fill="{_weight_color(color, value)}" stroke="white"/>',
                    f'<text x="{x + 25}" y="{y + 12}" text-anchor="middle" font-family="sans-serif" font-size="9">{value:.1f}</text>',
                ]
            )
        visits = int(row["state_total_visits"])
        bar_width = 180 * log1p(visits) / log1p(max_visits) if max_visits else 0
        parts.extend(
            [
                f'<rect x="820" y="{y + 3}" width="180" height="10" fill="#f3f4f6"/>',
                f'<rect x="820" y="{y + 3}" width="{bar_width:.1f}" height="10" fill="#475569"/>',
                f'<text x="1010" y="{y + 12}" font-family="sans-serif" font-size="10">{visits:,}</text>',
                f'<text x="1090" y="{y + 12}" font-family="sans-serif" font-size="10">target action {int(row["selected_action_id"])}</text>',
            ]
        )
    parts.append("</svg>")
    destination.write_text("\n".join(parts), encoding="utf-8")


def write_reference_policy_svg(
    reference_policy: pd.DataFrame,
    destination: Path,
    title: str = "Compact-regime greedy policy: target weights by full market regime",
    subtitle: str = (
        "Current allocation: [30% ARG, 30% CEDEARs, 20% S&amp;P 500, 20% Gold]. "
        "Rows: VIX / USD-ARS / momentum leader; columns: momentum strength."
    ),
) -> None:
    """Write four readable heatmaps covering all 108 reference market states."""
    width, height = 1440, 1460
    panel_width, panel_height = 650, 650
    panel_positions = ((55, 95), (735, 95), (55, 790), (735, 790))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="720" y="35" text-anchor="middle" font-family="sans-serif" font-size="24">{escape(title)}</text>',
        f'<text x="720" y="60" text-anchor="middle" font-family="sans-serif" font-size="14">{subtitle}</text>',
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
        f'<text x="{x_offset + panel_width / 2:.1f}" y="{y_offset + 30}" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="bold">{escape(title)}</text>',
    ]
    for strength, label in enumerate(REGIME_LABELS):
        x = x_offset + left + (strength + 0.5) * cell_width
        parts.append(
            f'<text x="{x:.1f}" y="{y_offset + 52}" text-anchor="middle" font-family="sans-serif" font-size="12">Strength: {label}</text>'
        )
    row = 0
    for vix, usd, leader in product(range(3), range(3), range(4)):
        row_y = y_offset + top + row * cell_height
        row_label = escape(
            f"VIX {REGIME_LABELS[vix][0]} · USD {REGIME_LABELS[usd][0]} · "
            f"{LEADER_LABELS[leader]}"
        )
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
