"""Tests for Model B cost-aware policy selection."""

from agents.q_learning import QLearningAgent
from backtesting.agents import CostAwareFrozenPolicy


def test_cost_aware_policy_prefers_lower_turnover_when_q_advantage_is_small() -> None:
    action_space = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    agent = QLearningAgent(n_actions=2, seed=1)
    state = [0.0, 1.0, 1.0, 1.0]
    agent.q_table[agent.state_key(state)] = [1.00, 1.005]
    policy = CostAwareFrozenPolicy(agent, action_space, transaction_cost=0.014)
    policy.reset([1.0, 0.0, 0.0, 0.0])

    assert policy.select_action(state) == 0


def test_cost_aware_policy_tracks_executed_target_as_next_origin() -> None:
    action_space = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    agent = QLearningAgent(n_actions=2, seed=1)
    state = [0.0, 1.0, 1.0, 1.0]
    agent.q_table[agent.state_key(state)] = [1.0, 1.01]
    policy = CostAwareFrozenPolicy(agent, action_space, transaction_cost=0.014)
    policy.reset(action_space[0])
    policy.observe_action(1)

    assert policy.select_action(state) == 1
