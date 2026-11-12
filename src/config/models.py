"""Load named operational model configurations."""

import re
from pathlib import Path

import yaml

from agents.training import QLearningTrainingConfig
from config.paths import PROJECT_ROOT
from features.transforms import FEATURE_MODES, FeatureMode

DEFAULT_MODEL_CONFIG = "selected_synthetic_model"
CONFIG_DIR = PROJECT_ROOT / "src/config"


def load_model_config(
    name: str = DEFAULT_MODEL_CONFIG,
) -> tuple[QLearningTrainingConfig, FeatureMode]:
    """Load a fixed configuration, excluding research-only metadata."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("config must be a YAML name without a path or extension")
    path: Path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"unknown model config: {name}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "search_space" in payload:
        raise ValueError("config must describe one model, not a hyperparameter search")
    payload = payload.copy()
    payload.pop("run_label", None)
    payload.pop("walk_forward_splits", None)
    feature_mode = payload.pop("feature_mode", "compact_regime")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    if not {"episodes", "training_mode"} <= payload.keys():
        raise ValueError("config must include episodes and training_mode")
    try:
        return QLearningTrainingConfig(**payload), feature_mode
    except TypeError as error:
        raise ValueError(f"invalid model config: {error}") from error
