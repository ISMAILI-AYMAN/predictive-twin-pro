from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - allow non-DL fallback environments
    torch = None
    nn = None


class InferenceModel(Protocol):
    def anomaly_score(self, temp: float, vibration: float, pressure: float) -> float:
        ...

    def estimate_rul(self, health_index: float) -> float:
        ...


@dataclass
class TrainedStatModel:
    feature_mean_temp: float
    feature_mean_vibration: float
    feature_mean_pressure: float
    feature_std_temp: float
    feature_std_vibration: float
    feature_std_pressure: float
    max_expected_wear_hours: float
    health_to_rul_scale: float

    @classmethod
    def from_artifact(cls, artifact_path: str) -> "TrainedStatModel":
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        return cls(**payload)

    def anomaly_score(self, temp: float, vibration: float, pressure: float) -> float:
        temp_z = abs(temp - self.feature_mean_temp) / self.feature_std_temp
        vibration_z = abs(vibration - self.feature_mean_vibration) / self.feature_std_vibration
        pressure_z = abs(pressure - self.feature_mean_pressure) / self.feature_std_pressure
        return round((temp_z + vibration_z + pressure_z) / 3.0, 4)

    def estimate_rul(self, health_index: float) -> float:
        bounded_health = max(0.0, min(100.0, health_index))
        rul = (bounded_health / 100.0) * self.max_expected_wear_hours * self.health_to_rul_scale
        return round(rul, 2)


class LSTMAutoencoderNet(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, latent_dim: int = 8) -> None:
        if nn is None:
            raise RuntimeError("torch_not_available")
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_init = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, (h_n, _) = self.encoder(x)
        latent = self.latent(h_n[-1])
        decoder_h0 = self.decoder_init(latent).unsqueeze(0)
        decoder_c0 = torch.zeros_like(decoder_h0)
        decoder_input = torch.zeros_like(x)
        dec_out, _ = self.decoder(decoder_input, (decoder_h0, decoder_c0))
        reconstructed = self.output(dec_out)
        rul = self.rul_head(latent).squeeze(-1)
        return reconstructed, rul


@dataclass
class TrainedDLModel:
    model: LSTMAutoencoderNet
    window_size: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    max_expected_wear_hours: float

    @classmethod
    def from_artifacts(cls, model_path: str, metadata_path: str) -> "TrainedDLModel":
        if torch is None:
            raise RuntimeError("torch_not_available")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        model = LSTMAutoencoderNet(
            input_dim=metadata["input_dim"],
            hidden_dim=metadata["hidden_dim"],
            latent_dim=metadata["latent_dim"],
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return cls(
            model=model,
            window_size=metadata["window_size"],
            feature_mean=np.array(metadata["feature_mean"], dtype=np.float32),
            feature_std=np.array(metadata["feature_std"], dtype=np.float32),
            max_expected_wear_hours=float(metadata["max_expected_wear_hours"]),
        )

    def anomaly_score(self, temp: float, vibration: float, pressure: float) -> float:
        if torch is None:
            raise RuntimeError("torch_not_available")
        features = np.array([temp, vibration, pressure], dtype=np.float32)
        normalized = (features - self.feature_mean) / self.feature_std
        window = np.tile(normalized, (self.window_size, 1))
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            reconstructed, _ = self.model(tensor)
        reconstruction_error = torch.mean((reconstructed - tensor) ** 2).item()
        return round(float(reconstruction_error), 4)

    def estimate_rul(self, health_index: float) -> float:
        bounded_health = max(0.0, min(100.0, health_index))
        return round((bounded_health / 100.0) * self.max_expected_wear_hours, 2)


def model_exists(artifact_path: str) -> bool:
    return Path(artifact_path).exists()


def load_model_metadata(artifact_path: str) -> Dict[str, float]:
    return json.loads(Path(artifact_path).read_text(encoding="utf-8"))


def dl_artifacts_exist(model_path: str, metadata_path: str) -> bool:
    return Path(model_path).exists() and Path(metadata_path).exists()
