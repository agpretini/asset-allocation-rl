"""Filesystem experiment tracking."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class ExperimentTracker:
    """Create reproducible experiment folders with standard artifacts."""

    def __init__(self, root_dir: str | Path = "experiments") -> None:
        """Create a tracker rooted at ``root_dir``."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, prefix: str = "experiment") -> Path:
        """Create and return the next numbered experiment directory."""
        existing_ids = [
            _parse_experiment_id(path.name, prefix)
            for path in self.root_dir.iterdir()
            if path.is_dir()
        ]
        next_id = max(existing_ids, default=0) + 1
        experiment_dir = self.root_dir / f"{prefix}_{next_id:03d}"
        experiment_dir.mkdir(parents=False, exist_ok=False)
        return experiment_dir

    def save(
        self,
        experiment_dir: str | Path,
        config: dict[str, Any],
        metrics: dict[str, Any],
        actions: pd.DataFrame,
        portfolio_values: pd.DataFrame,
        notes: str,
    ) -> None:
        """Save standard experiment artifacts."""
        destination = Path(experiment_dir)
        destination.mkdir(parents=True, exist_ok=True)

        (destination / "config.yaml").write_text(
            yaml.safe_dump(_to_serializable(config), sort_keys=True),
            encoding="utf-8",
        )
        (destination / "metrics.json").write_text(
            json.dumps(_to_serializable(metrics), indent=2),
            encoding="utf-8",
        )
        actions.to_csv(destination / "actions.csv")
        portfolio_values.to_csv(destination / "portfolio_values.csv")
        (destination / "notes.md").write_text(notes, encoding="utf-8")


def save_backtest_experiment(
    tracker: ExperimentTracker,
    config: dict[str, Any],
    metrics: dict[str, Any],
    result: pd.DataFrame,
    notes: str,
    prefix: str = "experiment",
) -> Path:
    """Save one backtest result using the standard experiment layout."""
    experiment_dir = tracker.create_experiment(prefix=prefix)
    actions = _extract_actions(result)
    portfolio_values = result[["portfolio_value"]].copy()
    tracker.save(
        experiment_dir=experiment_dir,
        config=config,
        metrics=metrics,
        actions=actions,
        portfolio_values=portfolio_values,
        notes=notes,
    )
    return experiment_dir


def _extract_actions(result: pd.DataFrame) -> pd.DataFrame:
    """Extract action columns from a backtest result."""
    columns = [
        column
        for column in ["action_id", "weights", "turnover", "rebalance_cost", "reward"]
        if column in result.columns
    ]
    return result[columns].copy()


def _parse_experiment_id(name: str, prefix: str) -> int:
    """Parse an experiment directory suffix, returning zero on mismatch."""
    marker = f"{prefix}_"
    if not name.startswith(marker):
        return 0
    try:
        return int(name.removeprefix(marker))
    except ValueError:
        return 0


def _to_serializable(value: Any) -> Any:
    """Convert common project objects to JSON/YAML serializable structures."""
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
