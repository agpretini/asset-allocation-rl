"""Compare the latest HPO grid across historical and synthetic training."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from agents.training import QLearningTrainingConfig

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

LONG_EPISODE_COUNTS = (18_000, 30_000)
TRAINING_MODES = ("historical", "synthetic")
CURRENT_EPSILON = 0.50
CURRENT_EPSILON_DECAY = 0.999954


def main() -> None:
    """Run or resume the exhaustive historical-versus-synthetic HPO comparison."""
    from backtesting.splitter import generate_walk_forward_splits
    from backtesting.walk_forward import run_walk_forward_q_learning
    from data.loaders import load_config
    from experiments.tracking import ExperimentTracker

    args = _parse_args()
    project_config = load_config()
    dataset = pd.read_parquet(project_config["processed_dataset_path"])
    splits = generate_walk_forward_splits(dataset.index)
    source_experiment = Path(args.source_experiment)
    search_space = _load_search_space(source_experiment / "config.yaml")
    trial_configs = _build_trial_configs(search_space)
    if args.dry_run:
        print(f"source={source_experiment} trials={len(trial_configs)}")
        print(
            pd.DataFrame([asdict(config) for config in trial_configs])
            .groupby(["training_mode", "episodes"])
            .size()
            .to_string(),
        )
        return

    if args.resume_dir is None:
        tracker = ExperimentTracker()
        experiment_dir = tracker.create_experiment(prefix="hpo_mode_comparison")
        _write_experiment_config(
            experiment_dir=experiment_dir,
            source_experiment=source_experiment,
            dataset_path=project_config["processed_dataset_path"],
            split_count=len(splits),
            search_space=search_space,
            trial_count=len(trial_configs),
        )
    else:
        experiment_dir = Path(args.resume_dir)
        if not (experiment_dir / "config.yaml").is_file():
            raise ValueError("resume_dir must contain config.yaml")

    results_path = experiment_dir / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    completed_keys = set(results.get("trial_key", pd.Series(dtype=str)).astype(str))
    print(
        f"experiment={experiment_dir} trials={len(trial_configs)} "
        f"completed={len(completed_keys)}",
        flush=True,
    )

    for trial_id, training_config in enumerate(trial_configs):
        trial_key = _trial_key(training_config)
        if trial_key in completed_keys:
            continue
        report, _ = run_walk_forward_q_learning(
            dataset=dataset,
            splits=splits,
            training_config=training_config,
        )
        row = {
            "trial_id": trial_id,
            "trial_key": trial_key,
            **asdict(training_config),
            "feature_mode": "compact_regime",
            "walk_forward_splits": len(report),
            "mean_cagr": float(report["cagr"].mean()),
            "mean_sharpe": float(report["sharpe"].mean()),
            "mean_maximum_drawdown": float(report["maximum_drawdown"].mean()),
            "mean_average_turnover": float(report["average_turnover"].mean()),
            "score": float(
                report["sharpe"].mean()
                + report["cagr"].mean()
                + report["maximum_drawdown"].mean()
            ),
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.sort_values(["score", "mean_cagr"], ascending=False).to_csv(
            results_path,
            index=False,
        )
        completed_keys.add(trial_key)
        print(
            f"completed trial={trial_id + 1}/{len(trial_configs)} "
            f"mode={training_config.training_mode} "
            f"episodes={training_config.episodes} score={row['score']:.6f}",
            flush=True,
        )

    ordered_results = results.sort_values(["score", "mean_cagr"], ascending=False)
    ordered_results.to_csv(results_path, index=False)
    (experiment_dir / "summary.md").write_text(
        _build_summary(ordered_results),
        encoding="utf-8",
    )
    print(experiment_dir, flush=True)


def _parse_args() -> argparse.Namespace:
    """Parse runner arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-experiment", default="analysis/hyperparameter_search/hpo_004"
    )
    parser.add_argument("--resume-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_search_space(config_path: Path) -> dict[str, list[float | int | str]]:
    """Load the source HPO grid and append long episode counts."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_space = payload["search_space"]
    episodes = sorted({*source_space["episodes"], *LONG_EPISODE_COUNTS})
    return {
        "episodes": episodes,
        "learning_rates": list(source_space["learning_rates"]),
        "learning_rate_schedules": list(source_space["learning_rate_schedules"]),
        "epsilons": sorted({*source_space["epsilons"], CURRENT_EPSILON}),
        "epsilon_decays": sorted(
            {*source_space["epsilon_decays"], CURRENT_EPSILON_DECAY}
        ),
        "state_precisions": list(source_space["state_precisions"]),
        "visit_learning_rate_decay": float(source_space["visit_learning_rate_decay"]),
        "discount_factor": 0.95,
        "min_epsilon": 0.02,
        "seed": 42,
        "training_modes": list(TRAINING_MODES),
    }


def _build_trial_configs(
    search_space: dict[str, list[float | int | str]],
) -> list[QLearningTrainingConfig]:
    """Expand every requested HPO combination across both training modes."""
    from agents.training import QLearningTrainingConfig

    return [
        QLearningTrainingConfig(
            episodes=int(episode_count),
            training_mode=str(training_mode),
            learning_rate=float(learning_rate),
            learning_rate_schedule=str(learning_rate_schedule),
            visit_learning_rate_decay=float(search_space["visit_learning_rate_decay"]),
            discount_factor=float(search_space["discount_factor"]),
            epsilon=float(epsilon),
            min_epsilon=float(search_space["min_epsilon"]),
            epsilon_decay=float(epsilon_decay),
            state_precision=int(state_precision),
            seed=int(search_space["seed"]),
        )
        for (
            training_mode,
            episode_count,
            learning_rate,
            learning_rate_schedule,
            epsilon,
            epsilon_decay,
            state_precision,
        ) in product(
            search_space["training_modes"],
            search_space["episodes"],
            search_space["learning_rates"],
            search_space["learning_rate_schedules"],
            search_space["epsilons"],
            search_space["epsilon_decays"],
            search_space["state_precisions"],
        )
    ]


def _trial_key(config: QLearningTrainingConfig) -> str:
    """Return a stable key used to resume completed trials."""
    return "|".join(str(value) for value in asdict(config).values())


def _write_experiment_config(
    experiment_dir: Path,
    source_experiment: Path,
    dataset_path: str,
    split_count: int,
    search_space: dict[str, list[float | int | str]],
    trial_count: int,
) -> None:
    """Write immutable runner metadata before the first long trial starts."""
    payload = {
        "objective": "compare all hpo_004 configurations across training modes",
        "source_experiment": str(source_experiment),
        "dataset": dataset_path,
        "feature_mode": "compact_regime",
        "search_space": search_space,
        "walk_forward_splits": split_count,
        "trial_count": trial_count,
        "notes": (
            "The source grid is extended with 18000 and 30000 episodes. "
            "Both historical and synthetic training are run for every combination. "
            "Synthetic simulators fit only on their split's training window; "
            "walk-forward metrics remain historical test metrics."
        ),
    }
    (experiment_dir / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=True),
        encoding="utf-8",
    )
    (experiment_dir / "notes.md").write_text(f"{payload['notes']}\n", encoding="utf-8")


def _build_summary(results: pd.DataFrame) -> str:
    """Build a compact result handoff after all trials complete."""
    best = results.iloc[0]
    return "\n".join(
        [
            "# Full HPO Mode Comparison",
            "",
            f"Completed trials: {len(results)}",
            "",
            "## Best trial by score",
            "",
            f"- Training mode: `{best['training_mode']}`",
            f"- Episodes: {int(best['episodes'])}",
            f"- Learning rate: {best['learning_rate']}",
            f"- Learning-rate schedule: `{best['learning_rate_schedule']}`",
            f"- Epsilon: {best['epsilon']}",
            f"- Epsilon decay: {best['epsilon_decay']}",
            f"- Mean CAGR: {best['mean_cagr']:.6f}",
            f"- Mean Sharpe: {best['mean_sharpe']:.6f}",
            f"- Mean maximum drawdown: {best['mean_maximum_drawdown']:.6f}",
            f"- Score: {best['score']:.6f}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
