"""Thin FastAPI endpoints; run with one Uvicorn worker."""

from collections.abc import Iterator

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import service
from config.models import DEFAULT_MODEL_CONFIG


def available_server() -> Iterator[None]:
    """Translate service errors into HTTP responses and serialize requests."""
    try:
        with service.operation():
            yield
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


app = FastAPI(title="Asset allocation", dependencies=[Depends(available_server)])


class RetrainRequest(BaseModel):
    """Optional named configuration."""

    config: str = DEFAULT_MODEL_CONFIG


class PredictRequest(BaseModel):
    """Requested allocation month."""

    month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")


@app.post("/retrain")
def retrain(request: RetrainRequest | None = None) -> dict:
    """Train through the last completed month."""
    return service.retrain(request.config if request else DEFAULT_MODEL_CONFIG)


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    """Predict with the saved model and persist the result."""
    return service.predict(request.month)
