"""Tests for the random baseline agent."""

from agents.random_agent import RandomAgent


def test_random_agent_selects_valid_action_ids() -> None:
    agent = RandomAgent(n_actions=3, seed=7)

    actions = [agent.select_action([]) for _ in range(20)]

    assert all(0 <= action < 3 for action in actions)


def test_random_agent_is_reproducible_with_seed() -> None:
    first_agent = RandomAgent(n_actions=10, seed=42)
    second_agent = RandomAgent(n_actions=10, seed=42)

    first_actions = [first_agent.select_action([]) for _ in range(5)]
    second_actions = [second_agent.select_action([]) for _ in range(5)]

    assert first_actions == second_actions
