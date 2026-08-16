"""Compare three Q-learning state/cost formulations under one fixed protocol.

The runner is intentionally isolated from the production training entry points.
It does not alter the selected model or its Q-table.  Model A can reuse the
selected full-sample checkpoint for the latest-market sensitivity panel, but a
fair historical metric comparison retrains A because no split-level checkpoint
or action path was retained for trial 371.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ASSET_NAMES = ("arg", "ced", "sp500", "gold")
TRANSACTION_COST = 0.014
CoverageKind = Literal["state", "state_action"]


@dataclass(frozen=True)
class Formulation:
    """The only dimensions allowed to differ among the three models."""

    name: str
    include_portfolio_in_state: bool
    training_uses_rebalance_cost: bool
    inference_uses_rebalance_cost: bool


FORMULATIONS = (
    Formulation("A_baseline", True, True, False),
    Formulation("B_market_state_cost_at_decision", False, True, True),
    Formulation("C_market_state_no_cost", False, False, False),
)


class MarketOnlyStateBuilder:
    """Adapter exposing StateBuilder market features without portfolio weights.

    It deliberately delegates feature construction to the existing
    ``StateBuilder``.  Thus B and C use byte-for-byte the same market feature
    pipeline as A; only the four current-allocation values are removed.
    """

    def __init__(self, base_builder: object) -> None:
        self._base = base_builder
        self.return_columns = base_builder.return_columns
        self.asset_prefixes = base_builder.asset_prefixes

    def build_state(self, row_index: int, portfolio_context: object) -> list[float]:
        del portfolio_context
        row = self._base.market_features.iloc[row_index]
        if row.isna().any():
            raise ValueError("state contains missing market features")
        return [float(value) for value in row.to_list()]

    def first_valid_index(self) -> int:
        return self._base.first_valid_index()


class FormulationPortfolioEnv:  # constructed dynamically to avoid production changes
    """PortfolioEnv-compatible environment with an explicitly selected reward.

    The environment's *wealth accounting* always deducts real trading costs.
    Only the training reward changes.  This makes C's reported performance a
    realistic executable net result even though C intentionally ignores costs
    when learning and selecting actions.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        state_builder: object,
        action_space: list[list[float]],
        training_uses_rebalance_cost: bool,
        initial_weights: list[float] | None = None,
    ) -> None:
        from env.portfolio_env import PortfolioEnv

        # Composition, instead of changing PortfolioEnv's production API.
        self._env = PortfolioEnv(
            dataset=dataset,
            state_builder=state_builder,
            action_space=action_space,
            initial_weights=initial_weights,
            transaction_cost=TRANSACTION_COST,
        )
        self.training_uses_rebalance_cost = training_uses_rebalance_cost

    @property
    def action_space(self) -> list[list[float]]:
        return self._env.action_space

    @property
    def initial_weights(self) -> list[float]:
        return self._env.initial_weights

    def reset(self) -> list[float]:
        return self._env.reset()

    def step(
        self, action_id: int
    ) -> tuple[list[float], float, bool, dict[str, object]]:
        from env.portfolio_env import calculate_turnover
        from env.reward import calculate_net_portfolio_return, calculate_reward

        env = self._env
        if env.current_index >= len(env.dataset) - 1:
            raise RuntimeError("cannot step beyond the end of the dataset")
        previous_weights = env.current_weights
        target_weights = env.action_space[action_id]
        next_index = env.current_index + 1
        next_row = env.dataset.iloc[next_index]
        portfolio_return = env._calculate_portfolio_return(next_row, target_weights)
        turnover = calculate_turnover(target_weights, previous_weights)
        net_return = calculate_net_portfolio_return(
            portfolio_return, turnover, TRANSACTION_COST
        )
        reward = calculate_reward(
            portfolio_return,
            turnover,
            TRANSACTION_COST if self.training_uses_rebalance_cost else 0.0,
            env.risk_aversion,
        )
        env.portfolio_value *= 1.0 + net_return
        env.peak_value = max(env.peak_value, env.portfolio_value)
        drawdown = env.portfolio_value / env.peak_value - 1.0
        env.current_index = next_index
        env.current_weights = target_weights.copy()
        done = env.current_index >= len(env.dataset) - 1
        info: dict[str, object] = {
            "date": env.dataset.index[env.current_index],
            "action_id": action_id,
            "weights": target_weights,
            "portfolio_return": portfolio_return,
            "net_portfolio_return": net_return,
            "turnover": turnover,
            "rebalance_cost": TRANSACTION_COST * turnover,
            "reward": reward,
            "portfolio_value": env.portfolio_value,
            "drawdown": drawdown,
        }
        return env._build_current_state(), reward, done, info


class DeterministicPolicy:
    """Read-only argmax policy, optionally netting immediate rebalance cost."""

    def __init__(
        self, agent: object, action_space: list[list[float]], cost_aware: bool
    ) -> None:
        self.agent = agent
        self.action_space = action_space
        self.cost_aware = cost_aware
        self.current_weights = [0.25, 0.25, 0.25, 0.25]

    def reset(self, weights: list[float] | None = None) -> None:
        self.current_weights = (weights or [0.25, 0.25, 0.25, 0.25]).copy()

    def select_action(self, state: Sequence[float]) -> int:
        from env.portfolio_env import calculate_turnover

        values = self.agent.q_table.get(self.agent.state_key(state))
        if values is None:
            return 0
        scores = [
            value
            - (
                TRANSACTION_COST * calculate_turnover(self.current_weights, weights)
                if self.cost_aware
                else 0.0
            )
            for value, weights in zip(values, self.action_space, strict=True)
        ]
        best = max(scores)
        return min(index for index, score in enumerate(scores) if score == best)

    def observe_action(self, action_id: int) -> None:
        self.current_weights = self.action_space[action_id].copy()


def main() -> None:
    """Train/evaluate the formulations and write comparison artifacts."""
    from agents.q_learning import QLearningAgent
    from data.loaders import load_config
    from env.action_space import generate_action_space

    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(load_config()["processed_dataset_path"])
    train = _window(dataset, args.train_start, args.train_end, "train")
    evaluation = _window(
        dataset, args.evaluation_start, args.evaluation_end, "evaluation"
    )
    if train.index.max() >= evaluation.index.min():
        raise ValueError("train period must end before evaluation period starts")
    action_space = generate_action_space()
    training_config = _training_config(args)
    models: dict[str, object] = {}
    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    visit_rows: list[dict[str, object]] = []

    for formulation in FORMULATIONS:
        print(
            f"Training {formulation.name} ({training_config.episodes} synthetic episodes)…"
        )
        training_result = _train_formulation(
            train, action_space, formulation, training_config, args.feature_mode
        )
        models[formulation.name] = training_result.agent
        result = _evaluate_formulation(
            evaluation,
            train,
            action_space,
            formulation,
            training_result.agent,
            args.feature_mode,
        )
        metric_rows.append(_metrics_row(formulation, result))
        coverage, distributions = _coverage_rows(
            formulation, training_result.agent, action_space, args.feature_mode
        )
        coverage_rows.append(coverage)
        visit_rows.extend(distributions)
        training_result.agent.save(output_dir / "models" / f"{formulation.name}.json")

    # Reuse the saved selected checkpoint only for the requested current-market
    # panel. It was trained through 2026-07, so it cannot be a fair model for
    # the historical holdout configured above.
    baseline_current_model = QLearningAgent.load(args.baseline_model)
    current_models = {"A_baseline": baseline_current_model}
    full_train = dataset
    for formulation in FORMULATIONS[1:]:
        print(f"Training full-data {formulation.name} for current-market sensitivity…")
        current_result = _train_formulation(
            full_train, action_space, formulation, training_config, args.feature_mode
        )
        current_models[formulation.name] = current_result.agent
        current_result.agent.save(
            output_dir / "models" / f"{formulation.name}_full_data.json"
        )
    sensitivity = _current_market_sensitivity(
        dataset, action_space, current_models, args.feature_mode
    )

    pd.DataFrame(metric_rows).to_csv(output_dir / "historical_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "coverage.csv", index=False)
    pd.DataFrame(visit_rows).to_csv(output_dir / "visit_distributions.csv", index=False)
    sensitivity.to_csv(output_dir / "current_market_sensitivity.csv", index=False)
    (output_dir / "configuration.json").write_text(
        json.dumps(
            {
                "train_period": _date_range(train),
                "evaluation_period": _date_range(evaluation),
                "training_config": asdict(training_config),
                "feature_mode": args.feature_mode,
                "action_count": len(action_space),
                "transaction_cost": TRANSACTION_COST,
                "baseline_current_model": str(Path(args.baseline_model).resolve()),
                "baseline_note": "A is retrained for complete historical metrics; the saved trial 371 full-data checkpoint is reused only in the current-market sensitivity analysis.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _build_summary(
            pd.DataFrame(metric_rows), pd.DataFrame(coverage_rows), sensitivity
        ),
        encoding="utf-8",
    )
    print(f"Wrote diagnostic artifacts to {output_dir}")


def _training_config(args: argparse.Namespace):
    from agents.training import QLearningTrainingConfig

    return QLearningTrainingConfig(
        episodes=args.episodes,
        training_mode="synthetic",
        learning_rate=args.learning_rate,
        learning_rate_schedule="visit_decay",
        visit_learning_rate_decay=args.visit_learning_rate_decay,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        epsilon_decay=args.epsilon_decay,
        state_precision=2,
        seed=args.seed,
    )


def _train_formulation(
    dataset, action_space, formulation, training_config, feature_mode
):
    from agents.training import train_q_learning_agent
    from simulation.market_simulator import MarketSimulator, load_simulation_config

    simulator_config = load_simulation_config()
    simulator_config = dataclass_replace(
        simulator_config,
        episode_count=training_config.episodes,
        seed=training_config.seed,
    )
    simulator = MarketSimulator.fit(dataset, simulator_config)
    episode_envs = (
        FormulationPortfolioEnv(
            episode,
            _state_builder(episode, dataset, formulation, feature_mode),
            action_space,
            formulation.training_uses_rebalance_cost,
        )
        for episode in simulator.stream_episodes()
    )
    return train_q_learning_agent(
        env=None, config=training_config, synthetic_episode_envs=episode_envs
    )


def dataclass_replace(instance, **changes):
    """Keep imports local and make the simulation override visually explicit."""
    from dataclasses import replace

    return replace(instance, **changes)


def _state_builder(dataset, history, formulation, feature_mode):
    from features.state_builder import StateBuilder

    builder = StateBuilder(dataset, feature_mode=feature_mode, history_dataset=history)
    return (
        builder
        if formulation.include_portfolio_in_state
        else MarketOnlyStateBuilder(builder)
    )


def _evaluate_formulation(
    evaluation, history, action_space, formulation, agent, feature_mode
):
    env = FormulationPortfolioEnv(
        evaluation,
        _state_builder(evaluation, history, formulation, feature_mode),
        action_space,
        formulation.training_uses_rebalance_cost,
    )
    policy = DeterministicPolicy(
        agent, action_space, formulation.inference_uses_rebalance_cost
    )
    state = env.reset()
    policy.reset()
    rows = []
    done = False
    while not done:
        action_id = policy.select_action(state)
        state, _, done, info = env.step(action_id)
        policy.observe_action(action_id)
        rows.append(info)
    return pd.DataFrame(rows).set_index("date")


def _metrics_row(formulation, result):
    from evaluation.metrics import calculate_performance_metrics

    metrics = calculate_performance_metrics(
        result["net_portfolio_return"], result["portfolio_value"], result["turnover"]
    )
    actions = result["action_id"]
    return {
        "model": formulation.name,
        "cumulative_return": float(result["portfolio_value"].iloc[-1] - 1.0),
        **metrics,
        "total_rebalance_cost": float(result["rebalance_cost"].sum()),
        "portfolio_changes": int(actions.ne(actions.shift()).iloc[1:].sum()),
        "distinct_targets": int(actions.nunique()),
        "evaluation_months": len(result),
    }


def _coverage_rows(formulation, agent, action_space, feature_mode):
    # compact_regime has four market variables: 4 leader identities × three
    # strength bins × three USD bins × three VIX bins = 108 market states.
    market_state_count = 108 if feature_mode == "compact_regime" else None
    if market_state_count is None:
        raise ValueError("the diagnostic currently supports compact_regime only")
    possible_states = market_state_count * (
        len(action_space) if formulation.include_portfolio_in_state else 1
    )
    possible_pairs = possible_states * len(action_space)
    visits_by_state: dict[str, int] = {}
    for (state_key, _), count in agent.visit_counts.items():
        visits_by_state[state_key] = visits_by_state.get(state_key, 0) + count
    coverage = {
        "model": formulation.name,
        "unique_q_table_states": len(agent.q_table),
        "possible_states": possible_states,
        "visited_states": len(visits_by_state),
        "state_coverage": len(visits_by_state) / possible_states,
        "possible_state_action_pairs": possible_pairs,
        "visited_state_action_pairs": len(agent.visit_counts),
        "state_action_coverage": len(agent.visit_counts) / possible_pairs,
    }
    rows = []
    for kind, values in (
        ("state", list(visits_by_state.values())),
        ("state_action", list(agent.visit_counts.values())),
    ):
        series = pd.Series(values, dtype=float)
        for percentile in (10, 25, 50, 75, 90):
            rows.append(
                {
                    "model": formulation.name,
                    "kind": kind,
                    "statistic": f"p{percentile}",
                    "visits": float(series.quantile(percentile / 100)),
                }
            )
        rows.append(
            {
                "model": formulation.name,
                "kind": kind,
                "statistic": "median",
                "visits": float(series.median()),
            }
        )
    return coverage, rows


def _current_market_sensitivity(dataset, action_space, models, feature_mode):
    from env.portfolio_env import calculate_turnover
    from features.state_builder import PortfolioContext, StateBuilder

    market_builder = StateBuilder(dataset, feature_mode=feature_mode)
    base_market_state = market_builder.build_state(
        len(dataset) - 1, PortfolioContext(weights=[0.25] * 4)
    )[: len(market_builder.market_features.columns)]
    rows = []
    for formulation in FORMULATIONS:
        agent = models[formulation.name]
        for starting_action_id, starting_weights in enumerate(action_space):
            state = (
                [*base_market_state, *starting_weights]
                if formulation.include_portfolio_in_state
                else base_market_state
            )
            values = agent.q_table.get(
                agent.state_key(state), [0.0] * len(action_space)
            )
            scores = [
                value
                - (
                    TRANSACTION_COST * calculate_turnover(starting_weights, target)
                    if formulation.inference_uses_rebalance_cost
                    else 0.0
                )
                for value, target in zip(values, action_space, strict=True)
            ]
            target_id = min(
                index for index, score in enumerate(scores) if score == max(scores)
            )
            target = action_space[target_id]
            rows.append(
                {
                    "model": formulation.name,
                    "signal_date": dataset.index[-1].date().isoformat(),
                    "starting_action_id": starting_action_id,
                    "target_action_id": target_id,
                    "state_key": agent.state_key(state),
                    "state_seen": agent.state_key(state) in agent.q_table,
                    "q_value": values[target_id],
                    "decision_score": scores[target_id],
                    "turnover": calculate_turnover(starting_weights, target),
                    "rebalance_cost": TRANSACTION_COST
                    * calculate_turnover(starting_weights, target),
                    **{
                        f"start_{asset}": weight
                        for asset, weight in zip(
                            ASSET_NAMES, starting_weights, strict=True
                        )
                    },
                    **{
                        f"target_{asset}": weight
                        for asset, weight in zip(ASSET_NAMES, target, strict=True)
                    },
                }
            )
    return pd.DataFrame(rows)


def _build_summary(metrics, coverage, sensitivity):
    sensitivity_counts = sensitivity.groupby("model")["target_action_id"].nunique()
    b = sensitivity.loc[sensitivity["model"] == "B_market_state_cost_at_decision"]
    c = sensitivity.loc[sensitivity["model"] == "C_market_state_no_cost"]
    b_c_same = (
        b["target_action_id"].to_numpy() == c["target_action_id"].to_numpy()
    ).mean()
    lines = [
        "# State and Cost Formulation Diagnostic",
        "",
        "## Scope",
        "",
        "All models use the same data, compact market features, 286 actions, synthetic simulator configuration, seed, episodes, learning-rate schedule, epsilon schedule, discount factor, and chronological train/evaluation windows. Historical wealth accounting always deducts the actual 1.4% turnover cost. The formulations differ only in state composition, training reward, and the decision-time cost adjustment documented below.",
        "",
        "## Historical evaluation",
        "",
        _markdown_table(metrics),
        "",
        "## Q-table coverage",
        "",
        _markdown_table(coverage),
        "",
        "## Fixed current-market sensitivity",
        "",
        f"- A distinct targets across 286 starting portfolios: {sensitivity_counts['A_baseline']}.",
        f"- B distinct targets across 286 starting portfolios: {sensitivity_counts['B_market_state_cost_at_decision']}.",
        f"- C distinct targets across 286 starting portfolios: {sensitivity_counts['C_market_state_no_cost']}.",
        f"- B and C choose the same target for {b_c_same:.1%} of starting portfolios. They share the reduced state and CRRA curvature, but B applies the cost in both reward and inference while C applies it in neither place; they are also separately trained Q-tables.",
        "",
        "## Interpretation guardrails",
        "",
        "- A historical metric row is retrained here. The saved trial-371 full-data model is reused only for A's latest-market sensitivity section: it includes data beyond the configured evaluation window and cannot provide an unbiased historical backtest.",
        "- All formulations use the same CRRA utility with gamma=0.5. A and B apply the rebalance cost inside that reward; C uses CRRA utility of gross return. This makes A-versus-B a state-reduction comparison and B-versus-C a full cost-treatment comparison.",
        "- C ignores costs for training and action selection, but its reported net return and total cost still charge the real cost of executing its selected rebalances. This isolates decision quality from an unrealistic free-trading performance assumption.",
        "- The sensitivity CSV has one row per starting allocation and model; use it to see exactly whether decision-time costs alter B's target.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without adding an optional dependency."""
    columns = list(frame.columns)
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            [
                f"{value:.4f}" if isinstance(value, float) else str(value)
                for value in values
            ]
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _window(dataset, start, end, label):
    selected = dataset.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    if selected.empty:
        raise ValueError(f"{label} period contains no dataset rows")
    return selected


def _date_range(dataset):
    return {
        "start": dataset.index.min().date().isoformat(),
        "end": dataset.index.max().date().isoformat(),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=18_000)
    parser.add_argument("--train-start", default="2018-12-31")
    parser.add_argument("--train-end", default="2023-11-30")
    parser.add_argument("--evaluation-start", default="2023-12-31")
    parser.add_argument("--evaluation-end", default="2025-11-30")
    parser.add_argument("--output-dir", default=str(RUN_DIR / "results"))
    parser.add_argument(
        "--feature-mode", default="compact_regime", choices=["compact_regime"]
    )
    parser.add_argument("--learning-rate", type=float, default=0.30)
    parser.add_argument("--visit-learning-rate-decay", type=float, default=0.01)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.999954)
    parser.add_argument(
        "--baseline-model",
        default=str(
            PROJECT_ROOT
            / "analysis/runs/selected_synthetic_trial_371/forecast/2026-08/selected_synthetic_trained_model.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
