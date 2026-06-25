"""Tests for portfolio reward calculation."""

import pytest

from env.reward import calculate_reward


def test_reward_subtracts_rebalancing_cost() -> None:
    reward = calculate_reward(portfolio_return=0.05, turnover=0.5)

    assert reward == pytest.approx(0.043)


def test_zero_turnover_has_no_rebalancing_cost() -> None:
    reward = calculate_reward(portfolio_return=0.02, turnover=0.0)

    assert reward == pytest.approx(0.02)


def test_full_turnover_uses_transaction_cost() -> None:
    reward = calculate_reward(
        portfolio_return=0.00,
        turnover=1.0,
        transaction_cost=0.014,
    )

    assert reward == pytest.approx(-0.014)
