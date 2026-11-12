"""Coordinate existing model functions and local persistent artifacts."""

import json
import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from agents.q_learning import QLearningAgent
from config.models import DEFAULT_MODEL_CONFIG, load_model_config
from config.paths import PROJECT_ROOT
from data.dataset import build_monthly_dataset
from data.preprocessing import (
    validate_complete_monthly_index,
    validate_no_missing_values,
)
from prediction.next_allocation import predict_allocations, train_allocation_model
from simulation.market_simulator import load_simulation_config

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
TIMEZONE = ZoneInfo("America/Argentina/Buenos_Aires")
logger = logging.getLogger("uvicorn.error")
_lock = Lock()


@contextmanager
def operation() -> Iterator[None]:
    """Allow one request at a time in the single server process."""
    if not _lock.acquire(blocking=False):
        raise RuntimeError("server busy; retry after the current operation finishes")
    try:
        yield
    finally:
        _lock.release()


def retrain(config: str = DEFAULT_MODEL_CONFIG) -> dict:
    """Refresh completed-month data and atomically publish a trained snapshot."""
    training_config, feature_mode = load_model_config(config)
    now = datetime.now(TIMEZONE)
    cutoff = pd.Timestamp(now.date()).to_period("M").start_time
    dataset = build_monthly_dataset(end_date=cutoff.date().isoformat())
    dataset = dataset.loc[dataset.index < cutoff]
    validate_complete_monthly_index(dataset)
    validate_no_missing_values(dataset)
    expected_end = cutoff - pd.offsets.MonthEnd(1)
    if dataset.index.max() != expected_end:
        raise ValueError(f"data must reach the completed month {expected_end:%Y-%m}")

    simulation_config = replace(
        load_simulation_config(),
        episode_count=training_config.episodes,
    )
    logger.info("Training %s through %s", config, expected_end.date())
    result = train_allocation_model(
        dataset,
        training_config,
        feature_mode,
        simulation_config,
    )
    metadata = {
        "config": config,
        "trained_at": datetime.now(TIMEZONE).isoformat(),
        "dataset_start": dataset.index.min().date().isoformat(),
        "dataset_end": expected_end.date().isoformat(),
        "feature_mode": feature_mode,
        "training_config": asdict(training_config),
        "simulation_config": asdict(simulation_config),
    }
    model_root = ARTIFACTS_ROOT / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    snapshot = Path(mkdtemp(prefix="run_", dir=model_root))
    pointer = model_root / "current.tmp.json"
    try:
        result.agent.save(snapshot / "trained_model.json")
        dataset.to_parquet(snapshot / "monthly_dataset.parquet")
        (snapshot / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        pointer.write_text(json.dumps(snapshot.name), encoding="utf-8")
        pointer.replace(model_root / "current.json")
    except Exception:
        pointer.unlink(missing_ok=True)
        shutil.rmtree(snapshot)
        raise
    logger.info("Model saved: %s", snapshot)
    return {**metadata, "model": str(snapshot.relative_to(PROJECT_ROOT))}


def predict(month: str) -> dict:
    """Load the current snapshot and save its next-month strategic allocation."""
    model_root = ARTIFACTS_ROOT / "model"
    pointer = model_root / "current.json"
    if not pointer.exists():
        raise FileNotFoundError("no trained model; call POST /retrain first")
    snapshot = model_root / json.loads(pointer.read_text(encoding="utf-8"))
    metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    expected_month = (
        pd.Timestamp(metadata["dataset_end"]) + pd.offsets.MonthEnd(1)
    ).strftime("%Y-%m")
    if month != expected_month:
        raise ValueError(f"this model can predict {expected_month}; requested {month}")
    dataset = pd.read_parquet(snapshot / "monthly_dataset.parquet")
    agent = QLearningAgent.load(snapshot / "trained_model.json")
    agent.set_training(False)
    prediction = next(
        item
        for item in predict_allocations(
            dataset,
            agent,
            feature_mode=metadata["feature_mode"],
        )
        if item.method == "model_b_modal_market_policy"
    )
    now = datetime.now(TIMEZONE)
    output_dir = ARTIFACTS_ROOT / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{month}_{now:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}.json"
    relative_path = str(path.relative_to(PROJECT_ROOT))
    message = f"se guardo el archivo {relative_path} con la prediccion"
    payload = {
        "month": month,
        "allocation": {
            "arg": prediction.w_arg,
            "ced": prediction.w_ced,
            "sp500": prediction.w_sp500,
            "gold": prediction.w_gold,
        },
        "prediction": prediction.to_dict(),
        "model": str(snapshot.relative_to(PROJECT_ROOT)),
        "model_metadata": metadata,
        "created_at": now.isoformat(),
        "file": relative_path,
        "message": message,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Prediccion para %s: %s", month, payload["allocation"])
    logger.info(message)
    return payload
