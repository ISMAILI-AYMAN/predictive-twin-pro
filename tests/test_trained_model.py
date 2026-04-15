import json
from pathlib import Path

import pytest

from ml_service.models.trained_model import (
    LSTMAutoencoderNet,
    TrainedDLModel,
    TrainedStatModel,
    dl_artifacts_exist,
    model_exists,
)


def test_trained_model_can_load_and_predict(tmp_path: Path) -> None:
    artifact = {
        "feature_mean_temp": 80.0,
        "feature_mean_vibration": 0.03,
        "feature_mean_pressure": 98.0,
        "feature_std_temp": 1.0,
        "feature_std_vibration": 0.01,
        "feature_std_pressure": 2.0,
        "max_expected_wear_hours": 1000.0,
        "health_to_rul_scale": 1.0,
    }
    path = tmp_path / "trained_model.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert model_exists(str(path)) is True
    model = TrainedStatModel.from_artifact(str(path))
    assert model.anomaly_score(82.0, 0.05, 99.0) > 0.0
    assert model.estimate_rul(90.0) == 900.0


def test_dl_model_can_load_and_predict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = LSTMAutoencoderNet(input_dim=3, hidden_dim=8, latent_dim=4)
    model_path = tmp_path / "dl_model.pt"
    metadata_path = tmp_path / "dl_model_metadata.json"
    torch.save(model.state_dict(), model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "input_dim": 3,
                "hidden_dim": 8,
                "latent_dim": 4,
                "window_size": 8,
                "feature_mean": [80.0, 0.03, 98.0],
                "feature_std": [2.0, 0.01, 2.0],
                "max_expected_wear_hours": 1000.0,
            }
        ),
        encoding="utf-8",
    )

    assert dl_artifacts_exist(str(model_path), str(metadata_path))
    dl_model = TrainedDLModel.from_artifacts(str(model_path), str(metadata_path))
    assert dl_model.anomaly_score(82.0, 0.04, 99.0) >= 0.0
    assert dl_model.estimate_rul(90.0) == 900.0
