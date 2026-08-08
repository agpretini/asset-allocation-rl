"""Canonical filesystem locations for project inputs and analytical artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data/processed/dataset/monthly_dataset.parquet"
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
HPO_ROOT = ANALYSIS_ROOT / "hyperparameter_search"
RUNS_ROOT = ANALYSIS_ROOT / "runs"


def ensure_directory(path: Path) -> Path:
    """Create and return one analytical output directory."""
    path.mkdir(parents=True, exist_ok=True)
    return path
