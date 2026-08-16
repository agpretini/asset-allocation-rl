"""Render one market-regime policy map for each reduced-state formulation."""

from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

import pandas as pd

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIAGNOSTIC_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ASSETS = ("arg", "ced", "sp500", "gold")
TRANSACTION_COST = 0.014
MODELS = {
    "B_market_state_cost_at_decision": True,
    "C_market_state_no_cost": False,
}


def main() -> None:
    """Write B modal-target and C greedy-target regime heatmaps without training."""
    from agents.q_learning import QLearningAgent
    from analysis.policy import write_reference_policy_svg
    from env.action_space import generate_action_space
    from env.portfolio_env import calculate_turnover

    results_dir = DIAGNOSTIC_DIR / "results"
    action_space = generate_action_space()
    for model_name, cost_aware in MODELS.items():
        agent = QLearningAgent.load(
            results_dir / "models" / f"{model_name}_full_data.json"
        )
        policy = _build_market_policy(
            agent, action_space, cost_aware, calculate_turnover
        )
        policy.to_csv(results_dir / f"full_market_policy_{model_name}.csv", index=False)
        title = (
            "Model B: modal cost-aware target by market regime"
            if cost_aware
            else "Model C: greedy target by market regime"
        )
        subtitle = (
            "Each cell is the target weight. B selects the most frequent "
            "cost-aware target across all 286 starting portfolios. "
            "Rows: VIX / USD-ARS / momentum leader; columns: momentum strength."
            if cost_aware
            else "Each cell is the target weight. C has one greedy target per "
            "market state. Rows: VIX / USD-ARS / momentum leader; columns: "
            "momentum strength."
        )
        destination = results_dir / f"full_market_policy_{model_name}.svg"
        write_reference_policy_svg(policy, destination, title, subtitle)
        print(destination)


def _build_market_policy(agent, action_space, cost_aware, calculate_turnover):
    """Summarize all 108 market policies into one target per market regime."""
    rows = []
    for leader, strength, usd, vix in product(range(4), range(3), range(3), range(3)):
        market_state = [leader, strength, usd, vix]
        target_ids = [
            _select_target(
                agent,
                market_state,
                starting_weights,
                action_space,
                cost_aware,
                calculate_turnover,
            )
            for starting_weights in action_space
        ]
        frequencies = Counter(target_ids)
        target_action_id, frequency = min(
            frequencies.items(), key=lambda item: (-item[1], item[0])
        )
        target = action_space[target_action_id]
        rows.append(
            {
                "momentum_leader": leader,
                "momentum_strength": strength,
                "usd_ars_regime": usd,
                "vix_regime": vix,
                "modal_target_action_id": target_action_id,
                "modal_target_count": frequency,
                "modal_target_share": frequency / len(action_space),
                "distinct_targets_across_starting_portfolios": len(frequencies),
                **{
                    f"w_{asset}": weight
                    for asset, weight in zip(ASSETS, target, strict=True)
                },
            }
        )
    return pd.DataFrame(rows)


def _select_target(
    agent,
    state,
    starting_weights,
    action_space,
    cost_aware,
    calculate_turnover,
) -> int:
    """Use the same deterministic B/C decision rule as the historical runner."""
    values = agent.q_table.get(agent.state_key(state), [0.0] * len(action_space))
    scores = [
        value
        - (
            TRANSACTION_COST * calculate_turnover(starting_weights, target)
            if cost_aware
            else 0.0
        )
        for value, target in zip(values, action_space, strict=True)
    ]
    return min(index for index, score in enumerate(scores) if score == max(scores))


if __name__ == "__main__":
    main()
