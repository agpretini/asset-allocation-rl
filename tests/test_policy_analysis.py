"""Tests for compact-regime policy inspection artifacts."""

from itertools import product

import pandas as pd
import pytest

from agents.q_learning import QLearningAgent
from analysis.policy import (
    build_compact_policy_table,
    build_fixed_market_policy_table,
    build_policy_feature_effects,
    build_reference_regime_policy,
    build_state_visit_summary,
)


def test_compact_policy_table_covers_all_states_without_mutating_agent() -> None:
    agent = QLearningAgent(n_actions=286, seed=7)
    states_before = len(agent.q_table)

    policy = build_compact_policy_table(agent)
    reference = build_reference_regime_policy(policy)

    assert len(policy) == 4 * 3 * 3 * 3 * 286
    assert len(reference) == 4 * 3 * 3 * 3
    assert not policy["known_state"].any()
    assert len(agent.q_table) == states_before


def test_state_visit_summary_aggregates_state_action_counts() -> None:
    agent = QLearningAgent(n_actions=2, seed=7)
    state = [0.0, 1.0, 2.0, 0.0, 0.3, 0.3, 0.2, 0.2]
    state_key = agent.state_key(state)
    agent.q_table[state_key] = [0.0, 0.0]
    agent.visit_counts[(state_key, 0)] = 3
    agent.visit_counts[(state_key, 1)] = 5

    summary = build_state_visit_summary(agent)

    assert summary.loc[0, "total_visits"] == 8
    assert summary.loc[0, "visited_action_count"] == 2
    assert summary.loc[0, "most_visited_action_id"] == 1


def test_fixed_market_policy_includes_actual_and_grid_starting_allocations() -> None:
    agent = QLearningAgent(n_actions=286, seed=7)
    actual_state = [0.0, 1.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]
    agent.q_table[agent.state_key(actual_state)] = [0.0] * 286

    policy = build_fixed_market_policy_table(
        agent,
        market_state=actual_state[:4],
        actual_starting_weights=actual_state[4:],
    )

    assert len(policy) == 287
    assert policy.iloc[0]["starting_portfolio_type"] == "prediction_assumption"
    assert {"state_total_visits", "target_sp500"}.issubset(policy.columns)


def test_policy_feature_effects_match_other_contexts() -> None:
    rows = []
    for leader, strength, usd, vix in product(range(4), range(3), range(3), range(3)):
        rows.append(
            {
                "momentum_leader": leader,
                "momentum_strength": strength,
                "usd_ars_regime": usd,
                "vix_regime": vix,
                "w_arg": 0.1,
                "w_ced": 0.2,
                "w_sp500": 0.3,
                "w_gold": 0.4 + 0.1 * vix,
            }
        )

    _, effects = build_policy_feature_effects(pd.DataFrame(rows))

    vix_gold = effects.loc[
        (effects["feature"] == "vix_regime")
        & (effects["from_level"] == 0)
        & (effects["to_level"] == 2)
        & (effects["asset"] == "w_gold")
    ].iloc[0]
    assert vix_gold["context_count"] == 36
    assert vix_gold["mean_change_percentage_points"] == pytest.approx(20.0)
    assert vix_gold["positive_context_share"] == 1.0
