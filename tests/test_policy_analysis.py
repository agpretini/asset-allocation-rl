"""Tests for compact-regime policy inspection artifacts."""

from agents.q_learning import QLearningAgent
from analysis.policy import (
    build_compact_policy_table,
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
