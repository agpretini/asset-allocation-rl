"""Diagnose Q-policy sensitivity to starting allocations without training."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    """Write a read-only diagnosis for every action-grid starting allocation."""
    from agents.q_learning import QLearningAgent
    from data.loaders import load_config
    from env.action_space import generate_action_space
    from env.portfolio_env import calculate_turnover
    from env.reward import calculate_reward
    from features.state_builder import PortfolioContext, StateBuilder

    args = _parse_args()
    dataset = pd.read_parquet(load_config()["processed_dataset_path"])
    builder = StateBuilder(dataset, feature_mode="compact_regime")
    market_state = builder.build_state(
        len(dataset) - 1,
        PortfolioContext(weights=[0.25, 0.25, 0.25, 0.25]),
    )[:4]
    model_path = (
        PROJECT_ROOT
        / "analysis/runs/selected_synthetic_trial_371/forecast"
        / args.target_period
        / "selected_synthetic_trained_model.json"
    )
    agent = QLearningAgent.load(model_path)
    agent.set_training(False)
    action_space = generate_action_space()
    q_table_size = len(agent.q_table)
    visit_snapshot = dict(agent.visit_counts)
    latest_returns = dataset.iloc[-1][["r_arg", "r_ced", "r_sp500", "r_gold"]]
    state_visits: dict[str, int] = {}
    for (state_key, _), count in agent.visit_counts.items():
        state_visits[state_key] = state_visits.get(state_key, 0) + count
    rows: list[dict[str, object]] = []

    for starting_action_id, starting_weights in enumerate(action_space):
        state = [*market_state, *starting_weights]
        state_key = agent.state_key(state)
        q_values = agent.q_table.get(state_key)
        known_state = q_values is not None
        q_values = q_values or [0.0] * agent.n_actions
        ordered_values = sorted(q_values, reverse=True)
        best_value, second_value = ordered_values[0], ordered_values[1]
        best_actions = [
            index for index, value in enumerate(q_values) if value == best_value
        ]
        deterministic_action = min(best_actions)
        repeated_actions = [agent.select_action(state) for _ in range(args.repetitions)]
        q_target = action_space[deterministic_action]
        immediate_with_cost = _select_myopic_action(
            action_space,
            starting_weights,
            latest_returns,
            transaction_cost=0.014,
            calculate_reward=calculate_reward,
        )
        immediate_without_cost = _select_myopic_action(
            action_space,
            starting_weights,
            latest_returns,
            transaction_cost=0.0,
            calculate_reward=calculate_reward,
        )
        rows.append(
            {
                "starting_action_id": starting_action_id,
                "state_key": state_key,
                "market_momentum_leader": int(market_state[0]),
                "market_momentum_strength": int(market_state[1]),
                "market_usd_ars_regime": int(market_state[2]),
                "market_vix_regime": int(market_state[3]),
                "state_seen_in_training": known_state,
                "state_total_visits": state_visits.get(state_key, 0),
                "selected_action_id": deterministic_action,
                "selected_action_seen_from_state": (state_key, deterministic_action)
                in agent.visit_counts,
                "selected_action_visits": agent.visit_counts.get(
                    (state_key, deterministic_action), 0
                ),
                "selected_q_value": best_value,
                "second_q_value": second_value,
                "q_gap": best_value - second_value,
                "max_q_tie_count": len(best_actions),
                "practically_tied_actions_1e_4": sum(
                    best_value - value <= 1e-4 for value in q_values
                ),
                "repeated_inference_distinct_targets": len(set(repeated_actions)),
                "repeated_inference_targets": ";".join(
                    str(action) for action in sorted(set(repeated_actions))
                ),
                "turnover_to_q_target": calculate_turnover(q_target, starting_weights),
                "rebalance_cost_to_q_target": 0.014
                * calculate_turnover(q_target, starting_weights),
                "immediate_target_with_cost": immediate_with_cost,
                "immediate_target_without_cost": immediate_without_cost,
                "q_target_changes_when_cost_disabled_at_inference": False,
                **{
                    f"start_{name}": value
                    for name, value in zip(
                        ("arg", "ced", "sp500", "gold"), starting_weights, strict=True
                    )
                },
                **{
                    f"target_{name}": value
                    for name, value in zip(
                        ("arg", "ced", "sp500", "gold"), q_target, strict=True
                    )
                },
            }
        )

    if len(agent.q_table) != q_table_size or agent.visit_counts != visit_snapshot:
        raise RuntimeError("diagnostic modified the loaded Q-table or visit counts")
    results = pd.DataFrame(rows)
    output_dir = PROJECT_ROOT / "analysis/diagnostics/model_formulations/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "current_market_policy_sensitivity.csv", index=False)
    (output_dir / "current_market_policy_sensitivity.md").write_text(
        _build_summary(results, dataset.index[-1].date().isoformat(), args.repetitions),
        encoding="utf-8",
    )
    print(output_dir / "current_market_policy_sensitivity.md")


def _select_myopic_action(
    action_space, starting_weights, returns, transaction_cost, calculate_reward
):
    """Select a one-step CRRA action only for the transaction-cost counterfactual."""
    scores = []
    for weights in action_space:
        portfolio_return = float(
            sum(
                weight * returns[column]
                for weight, column in zip(weights, returns.index, strict=True)
            )
        )
        turnover = 0.5 * sum(
            abs(left - right)
            for left, right in zip(weights, starting_weights, strict=True)
        )
        scores.append(calculate_reward(portfolio_return, turnover, transaction_cost))
    best = max(scores)
    return min(index for index, score in enumerate(scores) if score == best)


def _build_summary(results: pd.DataFrame, signal_date: str, repetitions: int) -> str:
    """Create a concise, explicit interpretation of diagnostic limits."""
    sufficient = (results["state_total_visits"] >= 100) & (
        results["selected_action_visits"] >= 10
    )
    unstable = results["repeated_inference_distinct_targets"] > 1
    lines = [
        "# Current-Market Policy Sensitivity Diagnosis",
        "",
        f"Fixed market state observed on {signal_date}; evaluated {len(results)} action-grid starting portfolios.",
        "",
        "## Coverage",
        f"- States meeting the diagnostic coverage threshold (>=100 state visits and >=10 selected-action visits): {sufficient.mean():.1%}.",
        f"- Median state visits: {results['state_total_visits'].median():.0f}.",
        f"- Median selected-action visits: {results['selected_action_visits'].median():.0f}.",
        "",
        "## Greedy Stability",
        f"- Distinct Q-policy targets across 286 starting portfolios: {results['selected_action_id'].nunique()}.",
        f"- Exact maximum-Q ties: {(results['max_q_tie_count'] > 1).sum()} states.",
        f"- Practical ties (within 1e-4): {(results['practically_tied_actions_1e_4'] > 1).sum()} states.",
        f"- States whose QLearningAgent greedy target changed across {repetitions} repeated calls: {unstable.sum()}.",
        "",
        "## Cost Counterfactual",
        "- Q-policy targets cannot change when transaction cost is disabled only at inference: Q-values are fixed training outputs.",
        "- The CSV additionally reports a separate one-step CRRA diagnostic with cost on and off; it measures the mechanical cost effect, not a retrained Q-policy.",
        f"- Distinct one-step targets with cost: {results['immediate_target_with_cost'].nunique()}; without cost: {results['immediate_target_without_cost'].nunique()}.",
        f"- Starting portfolios whose one-step target changed when cost was removed: {(results['immediate_target_with_cost'] != results['immediate_target_without_cost']).sum()}.",
        "",
        "## Interpretation",
        "- Large target dispersion with no Q ties indicates different learned values, not random tie-breaking.",
        "- Coverage is sparse relative to the 286 actions available from each state; selected-action visits should be interpreted as repeated policy sampling, not independent evidence against all alternatives.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-period", default="2026-08")
    parser.add_argument("--repetitions", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()
