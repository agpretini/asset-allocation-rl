"""Tests for the discrete portfolio action space."""

import pytest

from env.action_space import generate_action_space


@pytest.fixture(scope="module")
def action_space() -> list[list[float]]:
    """Generate the four-asset action space once for all tests."""
    return generate_action_space(step=0.10, n_assets=4)


def test_action_space_has_286_actions(action_space: list[list[float]]) -> None:
    assert len(action_space) == 286


def test_weights_sum_to_one(action_space: list[list[float]]) -> None:
    assert all(sum(weights) == pytest.approx(1.0) for weights in action_space)


def test_weights_are_non_negative(action_space: list[list[float]]) -> None:
    assert all(weight >= 0.0 for weights in action_space for weight in weights)


def test_weights_are_multiples_of_ten_percent(
    action_space: list[list[float]],
) -> None:
    assert all(
        weight * 10 == pytest.approx(round(weight * 10))
        for weights in action_space
        for weight in weights
    )
