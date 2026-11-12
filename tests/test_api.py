"""HTTP and persistence checks with tiny local training and no downloads."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agents.q_learning import QLearningAgent
from api import service
from api.main import app
from config import models
from config.models import DEFAULT_MODEL_CONFIG, load_model_config


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Freeze time and supply a tiny named model and causal market history."""

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 31, 12, tzinfo=tz)

    index = pd.date_range("2025-02-28", periods=21, freq="ME")
    dataset = pd.DataFrame({"usd_ars": 100.0, "vix": 20.0}, index=index)
    for number, asset in enumerate(("arg", "ced", "sp500", "gold"), start=1):
        dataset[f"price_{asset}"] = [100 * (1 + number / 100) ** i for i in range(21)]
        dataset[f"r_{asset}"] = number / 100

    def download(end_date: str) -> pd.DataFrame:
        assert end_date == "2026-10-01"
        return dataset.copy()

    (tmp_path / "tiny.yaml").write_text(
        "episodes: 1\ntraining_mode: historical\nfeature_mode: compact_regime\n",
    )
    monkeypatch.setattr(models, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(service, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(service, "datetime", Clock)
    monkeypatch.setattr(service, "build_monthly_dataset", download)
    return TestClient(app)


def test_default_config_is_selected_synthetic_model() -> None:
    config, feature_mode = load_model_config()
    assert config.training_mode == "synthetic"
    assert config.episodes == 54000
    assert feature_mode == "compact_regime"


def test_empty_retrain_body_uses_default(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service, "retrain", lambda config: {"config": config})
    for kwargs in ({}, {"json": {}}):
        response = client.post("/retrain", **kwargs)
        assert response.status_code == 200
        assert response.json()["config"] == DEFAULT_MODEL_CONFIG


def test_retrain_then_predict_without_training(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = client.post("/retrain", json={"config": "tiny"})
    assert response.status_code == 200
    assert response.json()["dataset_end"] == "2026-09-30"
    snapshot = service.PROJECT_ROOT / response.json()["model"]
    model_before = (snapshot / "trained_model.json").read_bytes()

    def no_training(*args, **kwargs):
        pytest.fail("prediction must not train")

    monkeypatch.setattr(service, "train_allocation_model", no_training)
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = client.post("/predict", json={"month": "2026-10"})
    assert response.status_code == 200
    payload = response.json()
    saved = service.PROJECT_ROOT / payload["file"]
    assert json.loads(saved.read_text()) == payload
    assert sum(payload["allocation"].values()) == pytest.approx(1.0)
    assert payload["prediction"]["method"] == "model_b_modal_market_policy"
    expected_message = f"se guardo el archivo {payload['file']} con la prediccion"
    assert payload["message"] == expected_message
    assert payload["message"] in caplog.text
    assert "Prediccion para 2026-10" in caplog.text
    assert (snapshot / "trained_model.json").read_bytes() == model_before
    repeated = client.post("/predict", json={"month": "2026-10"})
    assert repeated.json()["file"] != payload["file"]


def test_failed_save_preserves_current_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert client.post("/retrain", json={"config": "tiny"}).status_code == 200
    pointer = service.ARTIFACTS_ROOT / "model/current.json"
    before = pointer.read_bytes()
    snapshots = list(pointer.parent.iterdir())

    def fail_save(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(QLearningAgent, "save", fail_save)
    with pytest.raises(OSError, match="disk failure"):
        client.post("/retrain", json={"config": "tiny"})
    assert pointer.read_bytes() == before
    assert list(pointer.parent.iterdir()) == snapshots
    assert client.post("/predict", json={"month": "2026-10"}).status_code == 200


def test_stale_dataset_is_rejected_before_training(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = service.build_monthly_dataset
    monkeypatch.setattr(
        service,
        "build_monthly_dataset",
        lambda **kw: download(**kw).iloc[:-2],
    )
    response = client.post("/retrain", json={"config": "tiny"})
    assert response.status_code == 422
    assert "2026-09" in response.json()["detail"]
    assert not service.ARTIFACTS_ROOT.exists()


def test_invalid_requests_and_missing_model(client: TestClient) -> None:
    assert client.post("/predict", json={"month": "2026-10"}).status_code == 409
    for month in ("2026-13", "2026-1", "yesterday"):
        assert client.post("/predict", json={"month": month}).status_code == 422
    for config in ("unknown", "../tiny"):
        assert client.post("/retrain", json={"config": config}).status_code == 422
    assert client.post("/retrain", json={"config": "tiny"}).status_code == 200
    for month in ("2026-09", "2026-11"):
        response = client.post("/predict", json={"month": month})
        assert response.status_code == 422
        assert "can predict 2026-10" in response.json()["detail"]


def test_busy_server_rejects_overlapping_requests(client: TestClient) -> None:
    with service.operation():
        assert client.post("/retrain").status_code == 409
        assert client.post("/predict", json={"month": "2026-10"}).status_code == 409
    assert client.post("/predict", json={"month": "2026-10"}).status_code == 409


def test_search_config_is_not_an_operational_model(
    client: TestClient,
    tmp_path: Path,
) -> None:
    (tmp_path / "search.yaml").write_text("search_space: {episodes: [1, 2]}\n")
    response = client.post("/retrain", json={"config": "search"})
    assert response.status_code == 422
    assert "hyperparameter search" in response.json()["detail"]
