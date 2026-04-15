from __future__ import annotations

import os
import time
from typing import Dict

from fastapi import FastAPI, Query
from pydantic import BaseModel

from ml_service.drift_detector import DriftDetector
from ml_service.metrics_store import fetch_recent_events, init_db, insert_event
from ml_service.model_registry import resolve_model_artifact_path
from ml_service.models.baseline_model import BaselineModel
from ml_service.models.trained_model import (
    TrainedDLModel,
    TrainedStatModel,
    dl_artifacts_exist,
    model_exists,
)

app = FastAPI(title="Predictive Twin Inference API", version="1.0.0")
drift_detector = DriftDetector()
MODEL_ARTIFACT_PATH = resolve_model_artifact_path(
    os.getenv("MODEL_ARTIFACT_PATH", "experiments/artifacts/trained_model.json")
)
DL_MODEL_PATH = os.getenv("DL_MODEL_PATH", "experiments/artifacts/dl_model.pt")
DL_METADATA_PATH = os.getenv("DL_METADATA_PATH", "experiments/artifacts/dl_model_metadata.json")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "trained").lower()
METRICS_DB_PATH = os.getenv("METRICS_DB_PATH", "experiments/metrics/metrics.db")

if MODEL_BACKEND == "heuristic":
    model = BaselineModel()
    ACTIVE_MODEL_BACKEND = "heuristic"
elif dl_artifacts_exist(DL_MODEL_PATH, DL_METADATA_PATH):
    try:
        model = TrainedDLModel.from_artifacts(DL_MODEL_PATH, DL_METADATA_PATH)
        ACTIVE_MODEL_BACKEND = "dl_trained"
    except Exception:
        if model_exists(MODEL_ARTIFACT_PATH):
            model = TrainedStatModel.from_artifact(MODEL_ARTIFACT_PATH)
            ACTIVE_MODEL_BACKEND = "trained_stat_fallback"
        else:
            model = BaselineModel()
            ACTIVE_MODEL_BACKEND = "heuristic_fallback"
elif model_exists(MODEL_ARTIFACT_PATH):
    model = TrainedStatModel.from_artifact(MODEL_ARTIFACT_PATH)
    ACTIVE_MODEL_BACKEND = "trained_stat"
else:
    model = BaselineModel()
    ACTIVE_MODEL_BACKEND = "heuristic_fallback"

METRICS: Dict[str, float] = {
    "requests_total": 0.0,
    "drift_warn_total": 0.0,
    "drift_critical_total": 0.0,
    "anomaly_score_latest": 0.0,
    "rul_latest": 0.0,
}
init_db(METRICS_DB_PATH)


class SensorEvent(BaseModel):
    timestamp: float
    asset_id: str
    temp: float
    vibration: float
    pressure: float
    health_index: float
    fault_active: float = 0.0


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_backend": ACTIVE_MODEL_BACKEND}


@app.post("/infer")
def infer(event: SensorEvent) -> Dict[str, object]:
    anomaly_score = model.anomaly_score(event.temp, event.vibration, event.pressure)
    rul = model.estimate_rul(event.health_index)
    drift = drift_detector.update(event.vibration)
    policy = drift_detector.retraining_policy(drift)

    METRICS["requests_total"] += 1
    METRICS["anomaly_score_latest"] = anomaly_score
    METRICS["rul_latest"] = rul
    if drift.severity == "warn":
        METRICS["drift_warn_total"] += 1
    if drift.severity == "critical":
        METRICS["drift_critical_total"] += 1

    insert_event(
        METRICS_DB_PATH,
        event_time=event.timestamp or time.time(),
        asset_id=event.asset_id,
        anomaly_score=anomaly_score,
        rul_hours=rul,
        drift_severity=drift.severity,
        drift_score=drift.score,
        model_backend=ACTIVE_MODEL_BACKEND,
    )

    return {
        "asset_id": event.asset_id,
        "anomaly_score": anomaly_score,
        "rul_hours": rul,
        "drift": {
            "detected": drift.drift_detected,
            "severity": drift.severity,
            "score": drift.score,
        },
        "retraining_policy": policy,
    }


@app.get("/metrics")
def metrics() -> Dict[str, float]:
    return {
        **METRICS,
        "model_backend_trained": 1.0 if "trained" in ACTIVE_MODEL_BACKEND else 0.0,
        "model_backend_dl": 1.0 if ACTIVE_MODEL_BACKEND == "dl_trained" else 0.0,
    }


@app.get("/metrics/history")
def metrics_history(limit: int = Query(default=200, ge=1, le=1000)) -> Dict[str, object]:
    events = fetch_recent_events(METRICS_DB_PATH, limit=limit)
    return {"events": events, "count": len(events)}
