# Local allocation API

`main.py` validates HTTP requests and delegates to `service.py`. The service
coordinates existing data, training and prediction functions; model logic stays
in `prediction/next_allocation.py` and `agents/`.

From the repository root:

```bash
uv sync
uv run uvicorn api.main:app --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/docs> to try the endpoints.

## Retrain

```bash
curl -X POST http://127.0.0.1:8000/retrain
curl -X POST http://127.0.0.1:8000/retrain \
  -H 'Content-Type: application/json' \
  -d '{"config":"selected_synthetic_model"}'
```

The optional body selects a YAML filename in `src/config/`, without its
extension. The default is the committed trial-169 configuration: 54,000
synthetic episodes and `compact_regime`. To expose another trial, save its
fixed parameters as another operational YAML; HPO search-space files are not
individual model configurations.

Retraining downloads Yahoo data through the last completed calendar month in
Buenos Aires, checks that the requested close exists, and trains from scratch.
It does not run HPO, backtests, or analytical reports. The request waits for
training to finish; the default training can take considerable time.

A successful run saves an immutable snapshot of the Q-table, dataset and
configuration, then atomically updates `artifacts/model/current.json`.
Training or save failures leave the previous model available. Old snapshots
remain local so saved predictions retain their model reference.

## Predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"month":"2026-10"}'
```

Use the month immediately after the saved model's `dataset_end`; the example
requires a model through September 2026. Other months are rejected. Prediction
loads the saved model and its data without downloading or retraining. The
published Model B target remains the deterministic mode across all 286 origin
portfolios, with the lowest action ID breaking ties.

The response includes `allocation` (fractions summing to one), prediction
diagnostics, model metadata, `file`, and `message`. A unique JSON file is saved
for each request under `artifacts/predictions/`. The server logs the allocation
and `se guardo el archivo {ruta/nombre} con la prediccion`.

HTTP 422 indicates an invalid configuration, month or dataset; HTTP 409 means
no model exists yet or another operation is running. Unexpected failures return
HTTP 500 and are recorded by Uvicorn. Run exactly one worker: the in-process
lock rejects overlapping requests instead of queuing expensive training.

## Docker

Install Docker with its engine running. A Docker account is not needed to build
and run this local image. From the repository root:

```bash
docker build -t asset-allocation-rl .
docker run --rm --name asset-allocation-api \
  -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  asset-allocation-rl
```

Use the same endpoints and documentation URL. The bind mount preserves models
and predictions in the repository when the container is removed. Downloads
require internet access during retraining; there is no external hosting or
database. Stop the server with Ctrl+C.
