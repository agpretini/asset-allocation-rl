"""Tests for portfolio reward calculation."""

from math import log

import pytest

from env.reward import calculate_reward


def test_reward_applies_crra_utility_to_net_return() -> None:
    reward = calculate_reward(portfolio_return=0.05, turnover=0.5)

    assert reward == pytest.approx(2.0 * ((1.0 + 0.043) ** 0.5 - 1.0))


def test_zero_turnover_has_no_rebalancing_cost() -> None:
    reward = calculate_reward(portfolio_return=0.02, turnover=0.0)

    assert reward == pytest.approx(2.0 * ((1.0 + 0.02) ** 0.5 - 1.0))


def test_full_turnover_uses_transaction_cost() -> None:
    reward = calculate_reward(
        portfolio_return=0.00,
        turnover=1.0,
        transaction_cost=0.014,
    )

    assert reward == pytest.approx(2.0 * ((1.0 - 0.014) ** 0.5 - 1.0))


def test_log_utility_is_used_when_risk_aversion_is_one() -> None:
    reward = calculate_reward(
        portfolio_return=0.10,
        turnover=0.0,
        risk_aversion=1.0,
    )

    assert reward == pytest.approx(log(1.10))


def test_reward_rejects_non_positive_net_growth_factor() -> None:
    with pytest.raises(ValueError, match="greater than negative one"):
        calculate_reward(portfolio_return=-1.0, turnover=0.0)
