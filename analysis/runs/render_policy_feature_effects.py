"""Render finite-state partial-dependence-style artifacts for a selected run."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    """Write matched-context policy response tables, interpretation, and chart."""
    from analysis.policy import (
        build_policy_feature_effects,
        build_policy_feature_effects_markdown,
        write_policy_feature_effects_svg,
    )

    args = _parse_args()
    run_dir = Path(args.run_dir)
    policy_dir = run_dir / "policy"
    policy = pd.read_csv(policy_dir / "compact_policy_all_states.csv")
    level_means, matched_effects = build_policy_feature_effects(policy)
    level_means.to_csv(policy_dir / "policy_feature_level_means.csv", index=False)
    matched_effects.to_csv(
        policy_dir / "policy_feature_matched_effects.csv", index=False
    )
    write_policy_feature_effects_svg(
        level_means,
        policy_dir / "policy_feature_response_profiles.svg",
    )
    (policy_dir / "policy_feature_effects.md").write_text(
        build_policy_feature_effects_markdown(matched_effects), encoding="utf-8"
    )
    print(policy_dir / "policy_feature_effects.md")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="analysis/runs/model_b_synthetic_trial_169",
        help="Selected-model run containing policy/compact_policy_all_states.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
