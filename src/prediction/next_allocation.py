"""Train on all available data and predict the next target allocation."""

from dataclasses import asdict, dataclass, replace
from math import sqrt

import pandas as pd

from agents.q_learning import QLearningAgent
from agents.training import (
    SYNTHETIC_TRAINING_MODE,
    QLearningTrainingConfig,
    TrainingResult,
    train_q_learning_agent,
)
from env.portfolio_env import PortfolioEnv, calculate_turnover
from features.state_builder import PortfolioContext, StateBuilder
from features.transforms import COMPACT_REGIME_FEATURE_MODE, FeatureMode
from simulation.market_simulator import (
    MarketSimulator,
    SimulationConfig,
    load_simulation_config,
)


@dataclass(frozen=True)
class AllocationPrediction:
    """Forward-looking allocation prediction."""

    method: str
    signal_date: str
    target_period: str
    action_id: int
    w_arg: float
    w_ced: float
    w_sp500: float
    w_gold: float
    q_value: float
    score: float
    score_name: str
    state_source: str
    nearest_state_distance: float
    expected_turnover: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return a serializable representation."""
        return asdict(self)


def predict_next_allocation(
    dataset: pd.DataFrame,
    training_config: QLearningTrainingConfig,
    current_weights: list[float] | None = None,
    feature_mode: FeatureMode = COMPACT_REGIME_FEATURE_MODE,
    simulation_config: SimulationConfig | None = None,
) -> tuple[AllocationPrediction, QLearningAgent, pd.DataFrame]:
    """Train Q-learning and predict the infinite-horizon next allocation."""
    predictions, agent, history = predict_next_allocations(
        dataset=dataset,
        training_config=training_config,
        current_weights=current_weights,
        feature_mode=feature_mode,
        simulation_config=simulation_config,
    )
    infinite_horizon = [
        prediction
        for prediction in predictions
        if prediction.method == "infinite_horizon_q_learning"
    ][0]
    return infinite_horizon, agent, history


def predict_next_allocations(
    dataset: pd.DataFrame,
    training_config: QLearningTrainingConfig,
    current_weights: list[float] | None = None,
    feature_mode: FeatureMode = COMPACT_REGIME_FEATURE_MODE,
    simulation_config: SimulationConfig | None = None,
) -> tuple[list[AllocationPrediction], QLearningAgent, pd.DataFrame]:
    """Train on all data and produce myopic and infinite-horizon signals."""
    starting_weights = current_weights or [0.25, 0.25, 0.25, 0.25]
    state_builder = StateBuilder(dataset, feature_mode=feature_mode)
    env = PortfolioEnv(
        dataset=dataset,
        state_builder=state_builder,
        initial_weights=starting_weights,
    )
    training_result = _train_prediction_agent(
        dataset=dataset,
        env=env,
        training_config=training_config,
        feature_mode=feature_mode,
        simulation_config=simulation_config,
    )

    latest_index = len(dataset) - 1
    latest_state = state_builder.build_state(
        latest_index,
        PortfolioContext(weights=starting_weights),
    )
    q_values, state_source, nearest_distance = _resolve_q_values(
        agent=training_result.agent,
        state=latest_state,
    )
    signal_date = pd.Timestamp(dataset.index[latest_index])
    target_period = signal_date + pd.offsets.MonthEnd(1)
    myopic_prediction = _build_myopic_prediction(
        dataset=dataset,
        action_space=env.action_space,
        current_weights=starting_weights,
        signal_date=signal_date,
        target_period=target_period,
    )
    infinite_horizon_prediction = _build_infinite_horizon_prediction(
        q_values=q_values,
        action_space=env.action_space,
        current_weights=starting_weights,
        signal_date=signal_date,
        target_period=target_period,
        state_source=state_source,
        nearest_distance=nearest_distance,
    )
    return (
        [myopic_prediction, infinite_horizon_prediction],
        training_result.agent,
        training_result.history,
    )


def _train_prediction_agent(
    dataset: pd.DataFrame,
    env: PortfolioEnv,
    training_config: QLearningTrainingConfig,
    feature_mode: FeatureMode,
    simulation_config: SimulationConfig | None,
) -> TrainingResult:
    """Train on the full observed history in historical or synthetic mode."""
    if training_config.training_mode != SYNTHETIC_TRAINING_MODE:
        return train_q_learning_agent(env=env, config=training_config)
    selected_simulation = replace(
        simulation_config or load_simulation_config(),
        episode_count=training_config.episodes,
    )
    simulator = MarketSimulator.fit(dataset, selected_simulation)
    episode_envs = (
        PortfolioEnv(
            dataset=episode,
            state_builder=StateBuilder(
                episode,
                feature_mode=feature_mode,
                history_dataset=dataset,
            ),
            initial_weights=env.initial_weights,
        )
        for episode in simulator.stream_episodes()
    )
    return train_q_learning_agent(
        env=None,
        config=training_config,
        synthetic_episode_envs=episode_envs,
    )


def _build_infinite_horizon_prediction(
    q_values: list[float],
    action_space: list[list[float]],
    current_weights: list[float],
    signal_date: pd.Timestamp,
    target_period: pd.Timestamp,
    state_source: str,
    nearest_distance: float,
) -> AllocationPrediction:
    """Build the discounted infinite-horizon Q-learning signal."""
    action_id = _select_best_action(q_values)
    weights = action_space[action_id]
    return AllocationPrediction(
        method="infinite_horizon_q_learning",
        signal_date=signal_date.date().isoformat(),
        target_period=target_period.strftime("%Y-%m"),
        action_id=action_id,
        w_arg=weights[0],
        w_ced=weights[1],
        w_sp500=weights[2],
        w_gold=weights[3],
        q_value=float(q_values[action_id]),
        score=float(q_values[action_id]),
        score_name="discounted_q_value",
        state_source=state_source,
        nearest_state_distance=nearest_distance,
        expected_turnover=calculate_turnover(weights, current_weights),
    )


def _build_myopic_prediction(
    dataset: pd.DataFrame,
    action_space: list[list[float]],
    current_weights: list[float],
    signal_date: pd.Timestamp,
    target_period: pd.Timestamp,
) -> AllocationPrediction:
    """Build a one-step greedy signal using latest observed returns."""
    latest_returns = dataset.iloc[-1][["r_arg", "r_ced", "r_sp500", "r_gold"]]
    scores = [
        _calculate_latest_return_score(weights=weights, latest_returns=latest_returns)
        for weights in action_space
    ]
    action_id = _select_best_action(scores)
    weights = action_space[action_id]
    score = float(scores[action_id])
    return AllocationPrediction(
        method="myopic_greedy_latest_return",
        signal_date=signal_date.date().isoformat(),
        target_period=target_period.strftime("%Y-%m"),
        action_id=action_id,
        w_arg=weights[0],
        w_ced=weights[1],
        w_sp500=weights[2],
        w_gold=weights[3],
        q_value=0.0,
        score=score,
        score_name="latest_observed_return_proxy",
        state_source="latest_observed_returns",
        nearest_state_distance=0.0,
        expected_turnover=calculate_turnover(weights, current_weights),
    )


def _calculate_latest_return_score(
    weights: list[float],
    latest_returns: pd.Series,
) -> float:
    """Score an allocation by the latest observed weighted asset return."""
    return float(
        sum(
            weight * latest_returns[column]
            for weight, column in zip(
                weights,
                ["r_arg", "r_ced", "r_sp500", "r_gold"],
                strict=True,
            )
        ),
    )


def _resolve_q_values(
    agent: QLearningAgent,
    state: list[float],
) -> tuple[list[float], str, float]:
    """Return exact-state Q-values or nearest learned-state Q-values."""
    state_key = agent.state_key(state)
    exact_values = agent.q_table.get(state_key)
    if exact_values is not None and _has_learned_values(exact_values):
        return exact_values, "exact", 0.0

    nearest = _find_nearest_learned_state(agent=agent, state=state)
    if nearest is not None:
        nearest_values, distance = nearest
        return nearest_values, "nearest_learned_state", distance

    return agent.q_values_for_state(state), "unseen_unlearned_state", 0.0


def _find_nearest_learned_state(
    agent: QLearningAgent,
    state: list[float],
) -> tuple[list[float], float] | None:
    """Find the nearest state with non-zero Q-values."""
    best_values: list[float] | None = None
    best_distance: float | None = None
    for state_key, q_values in agent.q_table.items():
        if not _has_learned_values(q_values):
            continue
        candidate_state = [float(value) for value in state_key.split("|")]
        distance = _euclidean_distance(state, candidate_state)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_values = q_values

    if best_values is None or best_distance is None:
        return None
    return best_values, best_distance


def _select_best_action(q_values: list[float]) -> int:
    """Select the lowest action ID among tied best Q-values."""
    best_value = max(q_values)
    return min(
        action_id for action_id, value in enumerate(q_values) if value == best_value
    )


def _has_learned_values(q_values: list[float]) -> bool:
    """Return whether a Q-vector contains non-zero learned values."""
    return any(abs(value) > 1e-12 for value in q_values)


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    """Calculate Euclidean distance between two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("state vectors must have equal length")
    return sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left, right, strict=True)
        ),
    )
