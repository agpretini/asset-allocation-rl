"""Render current-market starting-allocation policy charts for B and C.

This read-only renderer uses already-trained diagnostic checkpoints. It never
trains an agent or changes a Q-table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIAGNOSTIC_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ASSETS = ("arg", "ced", "sp500", "gold")
CURRENT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
TRANSACTION_COST = 0.014
MODELS = {
    "B_market_state_cost_at_decision": {
        "label": "Model B — market state + cost-aware decision",
        "heading": "Cost-aware target portfolio",
        "cost_aware": True,
    },
    "C_market_state_no_cost": {
        "label": "Model C — market state without costs",
        "heading": "Greedy target portfolio",
        "cost_aware": False,
    },
}


def main() -> None:
    """Write one highlighted current-market chart and CSV for B and C."""
    from agents.q_learning import QLearningAgent
    from analysis.policy import write_fixed_market_policy_svg
    from data.loaders import load_config
    from env.action_space import generate_action_space
    from env.portfolio_env import calculate_turnover
    from features.state_builder import PortfolioContext, StateBuilder

    results_dir = DIAGNOSTIC_DIR / "results"
    dataset = pd.read_parquet(load_config()["processed_dataset_path"])
    action_space = generate_action_space()
    builder = StateBuilder(dataset, feature_mode="compact_regime")
    market_state = builder.build_state(
        len(dataset) - 1, PortfolioContext(weights=CURRENT_WEIGHTS)
    )[: len(builder.market_features.columns)]
    sensitivity = pd.read_csv(results_dir / "current_market_sensitivity.csv")

    for model_name, specification in MODELS.items():
        agent = QLearningAgent.load(
            results_dir / "models" / f"{model_name}_full_data.json"
        )
        state_key = agent.state_key(market_state)
        state_visits = sum(
            count
            for (visited_state_key, _), count in agent.visit_counts.items()
            if visited_state_key == state_key
        )
        grid_policy = _grid_policy(
            sensitivity.loc[sensitivity["model"] == model_name], state_visits
        )
        actual_policy = _actual_policy(
            agent,
            action_space,
            state_key,
            state_visits,
            bool(specification["cost_aware"]),
            calculate_turnover,
        )
        policy = pd.concat([actual_policy, grid_policy], ignore_index=True)
        policy.to_csv(
            results_dir / f"current_market_policy_{model_name}.csv", index=False
        )
        write_fixed_market_policy_svg(
            policy=policy,
            market_state=market_state,
            signal_date=dataset.index[-1].date().isoformat(),
            destination=results_dir / f"current_market_policy_{model_name}.svg",
            title=(
                f"August 2026 policy by starting allocation — {specification['label']}"
            ),
            decision_heading=str(specification["heading"]),
            description=(
                "Each row fixes this market state. The highlighted row uses the "
                "actual 25/25/25/25 allocation; visits are training transitions "
                "from the shared market-only state."
            ),
        )
        print(results_dir / f"current_market_policy_{model_name}.svg")


def _grid_policy(rows: pd.DataFrame, state_visits: int) -> pd.DataFrame:
    """Adapt the existing 286-row sensitivity output to the SVG schema."""
    policy = rows.rename(columns={"target_action_id": "selected_action_id"}).copy()
    policy["starting_portfolio_type"] = "action_grid"
    policy["state_total_visits"] = state_visits
    return policy


def _actual_policy(
    agent,
    action_space,
    state_key: str,
    state_visits: int,
    cost_aware: bool,
    calculate_turnover,
) -> pd.DataFrame:
    """Add the non-grid 25/25/25/25 allocation as the highlighted first row."""
    q_values = agent.q_table.get(state_key)
    if q_values is None:
        raise ValueError("current reduced market state was not seen during training")
    scores = [
        value
        - (
            TRANSACTION_COST * calculate_turnover(CURRENT_WEIGHTS, target)
            if cost_aware
            else 0.0
        )
        for value, target in zip(q_values, action_space, strict=True)
    ]
    selected_action_id = min(
        index for index, score in enumerate(scores) if score == max(scores)
    )
    target = action_space[selected_action_id]
    return pd.DataFrame(
        [
            {
                "starting_portfolio_type": "prediction_assumption",
                "starting_action_id": -1,
                "state_key": state_key,
                "state_seen": True,
                "state_total_visits": state_visits,
                "selected_action_id": selected_action_id,
                "q_value": q_values[selected_action_id],
                "decision_score": scores[selected_action_id],
                "turnover": calculate_turnover(CURRENT_WEIGHTS, target),
                "rebalance_cost": TRANSACTION_COST
                * calculate_turnover(CURRENT_WEIGHTS, target),
                **{
                    f"start_{asset}": weight
                    for asset, weight in zip(ASSETS, CURRENT_WEIGHTS, strict=True)
                },
                **{
                    f"target_{asset}": weight
                    for asset, weight in zip(ASSETS, target, strict=True)
                },
            }
        ]
    )


if __name__ == "__main__":
    main()
