"""The web API.

uvicorn creditlens.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from creditlens.config import DEFAULT_MODEL, STATIC_DIR
from creditlens.registry import ModelRegistry
from creditlens.schemas import PredictRequest, validate_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="CreditLens")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

registry = ModelRegistry()
registry.load()


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    """Liveness probe for Render."""
    return {"status": "ok", "datasets": sorted(registry.datasets)}


@app.get("/api/datasets")
def get_datasets():
    return registry.catalog()


@app.get("/api/metrics")
def get_metrics():
    return registry.metrics


@app.post("/api/predict")
def predict(request: PredictRequest):
    dataset = request.dataset
    if dataset not in registry.datasets:
        available = sorted(registry.datasets)
        return JSONResponse(
            status_code=404,
            content={"error": f"Unknown dataset '{dataset}'. Available: {available}"},
        )

    model_type = request.model_type or DEFAULT_MODEL
    models = registry.datasets[dataset]["models"]
    if model_type not in models:
        available = sorted(models)
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    f"Model '{model_type}' is not available for '{dataset}'. Available: {available}"
                )
            },
        )

    features = registry.datasets[dataset]["features"]
    values, errors = validate_features(dataset, features, request.model_extra or {})
    if errors:
        return JSONResponse(
            status_code=400,
            content={"error": "One or more inputs are invalid.", "fields": errors},
        )

    return registry.predict(dataset, model_type, values)


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    """Return JSON on a crash instead of leaking a traceback to the client."""
    log.exception("unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error."})
