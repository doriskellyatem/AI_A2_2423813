"""FastAPI service.

    uvicorn app:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import config
from src.inference import ArtifactsMissingError, load_bundle, predict_record
from src.schemas import FarmRecord


class PredictionResponse(BaseModel):
    verification_code: str
    yield_estimate_tons: float
    high_risk_prediction: bool
    high_risk_probability: float
    risk_band: str
    cluster: int
    cluster_profile: str
    human_review_required: bool
    notice: str


def create_app(root: Path = config.PROJECT_ROOT) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.bundle = load_bundle(root)
            app.state.load_error = None
        except ArtifactsMissingError as exc:
            app.state.bundle = None
            app.state.load_error = str(exc)
        yield

    app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(_: Request, exc: RequestValidationError):
        problems = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "(body)",
                     "problem": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "error": "invalid_input",
            "message": "The farm record was rejected. Fix the listed fields and retry.",
            "problems": problems})

    def _bundle(request: Request):
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(503, detail=request.app.state.load_error)
        return bundle

    @app.get("/health")
    def health(request: Request):
        bundle = request.app.state.bundle
        return {"status": "ok" if bundle else "models_not_loaded",
                "verification_code": bundle.metadata["verification_code"] if bundle else None}

    @app.get("/model-info")
    def model_info(request: Request):
        b = _bundle(request)
        sel_reg = next(c for c in b.regression_report["candidates"] if c["name"] == b.regression_report["selected_model"])
        sel_clf = next(c for c in b.classification_report["candidates"] if c["name"] == b.classification_report["selected_model"])
        return {
            "verification_code": b.metadata["verification_code"],
            "student_id": b.metadata["student_id"],
            "seed": b.metadata["seed"],
            "dataset_sha256": b.metadata["dataset_sha256"],
            "synthetic_data": True,
            "regression": {"model": b.regression_report["selected_model"], "test_metrics": sel_reg["test"]},
            "classification": {"model": b.classification_report["selected_model"], "test_metrics": sel_clf["test"]},
            "clustering": {"selected_k": b.clustering_report["selected_k"]},
            "input_ranges": config.VALUE_RANGES,
            "notice": config.SYNTHETIC_NOTICE,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(farm: FarmRecord, request: Request):
        return predict_record(_bundle(request), farm.model_dump())

    return app


app = create_app()
