from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_model_artifact_path(default_artifact_path: str) -> str:
    source = os.getenv("MODEL_SOURCE", "local").lower()
    if source != "mlflow":
        return default_artifact_path

    model_uri = os.getenv("MLFLOW_MODEL_URI", "")
    if not model_uri:
        return default_artifact_path

    try:
        import mlflow
    except Exception:
        return default_artifact_path

    target_dir = Path(os.getenv("MLFLOW_MODEL_CACHE_DIR", "experiments/artifacts/mlflow_cache"))
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(target_dir))
    except Exception:
        return default_artifact_path
    return downloaded


def resolve_registered_model_uri() -> Optional[str]:
    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    if not model_name:
        return None
    return f"models:/{model_name}/{model_stage}"
