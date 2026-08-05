"""Tests for regime-switching synthetic market simulation."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from agents.training import QLearningTrainingConfig, train_q_learning_agent
from env.portfolio_env import PortfolioEnv
from features.state_builder import StateBuilder
from simulation.market_simulator import MarketSimulator, SimulationConfig
from simulation.markov_chain import RegimeMarkovChain
from simulation.multivariate_t import MultivariateStudentT
from simulation.regimes import MarketRegime, VixRegimeModel


@pytest.fixture
def simulation_dataset() -> pd.DataFrame:
    """Create positive monthly levels and varying correlated returns."""
    rng = np.random.default_rng(17)
    index = pd.date_range("2015-01-31", periods=48, freq="ME", name="date")
    common = rng.normal(0.005, 0.025, size=48)
    returns = np.column_stack(
        [common + rng.normal(0.0, 0.01 + index * 0.002, size=48) for index in range(4)]
    )
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    data: dict[str, np.ndarray] = {}
    for column_index, asset in enumerate(("arg", "ced", "sp500", "gold")):
        data[f"price_{asset}"] = prices[:, column_index]
        data[f"r_{asset}"] = returns[:, column_index]
    data["usd_ars"] = 100.0 * np.exp(np.cumsum(rng.normal(0.015, 0.01, 48)))
    data["vix"] = np.tile([12.0, 18.0, 25.0, 35.0, 22.0, 15.0], 8)
    return pd.DataFrame(data, index=index)


def _small_config(seed: int = 9) -> SimulationConfig:
    return SimulationConfig(
        episode_count=3,
        episode_length=18,
        seed=seed,
        covariance_regularization=1e-5,
    )


def test_vix_regimes_use_training_terciles() -> None:
    vix = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    model = VixRegimeModel.fit(vix)

    regimes = model.classify(vix)

    assert list(regimes) == [
        MarketRegime.LOW_VOL.value,
        MarketRegime.LOW_VOL.value,
        MarketRegime.MEDIUM_VOL.value,
        MarketRegime.MEDIUM_VOL.value,
        MarketRegime.HIGH_VOL.value,
        MarketRegime.HIGH_VOL.value,
    ]


def test_markov_transition_rows_sum_to_one() -> None:
    regimes = pd.Series(
        ["LOW_VOL", "LOW_VOL", "MEDIUM_VOL", "HIGH_VOL", "MEDIUM_VOL"],
    )

    chain = RegimeMarkovChain.fit(regimes, smoothing=0.5)

    assert np.allclose(chain.transition_matrix.sum(axis=1), 1.0)
    assert chain.initial_probabilities.sum() == pytest.approx(1.0)


def test_student_t_scale_matrix_is_symmetric_positive_definite() -> None:
    observations = np.array(
        [[0.01, 0.02], [0.02, 0.04], [-0.01, -0.02]],
    )

    distribution = MultivariateStudentT.fit(
        observations,
        degrees_freedom=8.0,
        regularization=1e-5,
    )

    assert np.allclose(distribution.scale_matrix, distribution.scale_matrix.T)
    assert (np.linalg.eigvalsh(distribution.scale_matrix) > 0.0).all()


def test_simulation_is_reproducible_and_levels_stay_positive(
    simulation_dataset: pd.DataFrame,
) -> None:
    first = MarketSimulator.fit(simulation_dataset, _small_config(seed=21))
    second = MarketSimulator.fit(simulation_dataset, _small_config(seed=21))

    first_episode = next(first.stream_episodes())
    second_episode = next(second.stream_episodes())

    pd.testing.assert_frame_equal(first_episode, second_episode)
    level_columns = [
        "price_arg",
        "price_ced",
        "price_sp500",
        "price_gold",
        "usd_ars",
        "vix",
    ]
    assert (first_episode[level_columns] > 0.0).all().all()
    assert first_episode["vix"].between(8.0, 100.0).all()
    for distribution in first.regime_distributions.values():
        assert np.allclose(distribution.scale_matrix, distribution.scale_matrix.T)
        assert (np.linalg.eigvalsh(distribution.scale_matrix) > 0.0).all()


def test_simulator_fit_is_limited_to_supplied_training_rows(
    simulation_dataset: pd.DataFrame,
) -> None:
    train = simulation_dataset.iloc[:36]
    future = simulation_dataset.iloc[36:].copy()
    future["vix"] = 1000.0

    simulator = MarketSimulator.fit(train, _small_config())

    assert simulator.training_end == train.index.max()
    assert simulator.regime_model.upper_threshold == pytest.approx(
        train["vix"].quantile(2.0 / 3.0),
    )
    assert simulator.regime_model.upper_threshold != future["vix"].quantile(2.0 / 3.0)


def test_episode_generation_is_streaming(simulation_dataset: pd.DataFrame) -> None:
    simulator = MarketSimulator.fit(simulation_dataset, _small_config())

    episodes = simulator.stream_episodes()

    assert isinstance(episodes, Iterator)
    assert len(next(episodes)) == simulator.config.episode_length
    assert len(next(episodes)) == simulator.config.episode_length


def test_q_learning_trains_from_synthetic_episode_stream(
    simulation_dataset: pd.DataFrame,
) -> None:
    config = _small_config()
    simulator = MarketSimulator.fit(simulation_dataset, config)
    episode_envs = (
        PortfolioEnv(
            episode,
            StateBuilder(episode, history_dataset=simulation_dataset),
        )
        for episode in simulator.stream_episodes()
    )

    result = train_q_learning_agent(
        env=None,
        config=QLearningTrainingConfig(
            episodes=config.episode_count,
            training_mode="synthetic",
            seed=9,
        ),
        synthetic_episode_envs=episode_envs,
    )

    assert len(result.history) == config.episode_count
    assert set(result.history["training_mode"]) == {"synthetic"}
    assert len(result.agent.visit_counts) > 0
