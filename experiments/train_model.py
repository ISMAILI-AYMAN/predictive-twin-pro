from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import mlflow
except Exception:  # pragma: no cover - optional runtime dependency in local setups
    mlflow = None

import torch
import torch.nn as nn
import torch.optim as optim


class LSTMAutoencoderNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 16, latent_dim: int = 8) -> None:
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
        _, (h_n, _) = self.encoder(x)
        latent = self.latent(h_n[-1])
        decoder_h0 = self.decoder_init(latent).unsqueeze(0)
        decoder_c0 = torch.zeros_like(decoder_h0)
        decoder_input = torch.zeros_like(x)
        dec_out, _ = self.decoder(decoder_input, (decoder_h0, decoder_c0))
        reconstructed = self.output(dec_out)
        rul = self.rul_head(latent).squeeze(-1)
        return reconstructed, rul


ARTIFACT_PATH = Path(
    os.getenv("TRAINED_MODEL_ARTIFACT", "experiments/artifacts/trained_model.json")
)
DL_MODEL_PATH = Path(os.getenv("DL_MODEL_PATH", "experiments/artifacts/dl_model.pt"))
DL_METADATA_PATH = Path(
    os.getenv("DL_METADATA_PATH", "experiments/artifacts/dl_model_metadata.json")
)


def generate_synthetic_dataset(size: int = 2000) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed=42)
    health = np.clip(rng.normal(loc=85.0, scale=18.0, size=size), 0.0, 100.0)
    wear = 100.0 - health

    temp = 75.0 + wear * 0.45 + rng.normal(0.0, 0.8, size=size)
    vibration = 0.02 + wear * 0.0048 + rng.normal(0.0, 0.003, size=size)
    pressure = 100.0 - wear * 0.03 + rng.normal(0.0, 1.0, size=size)

    return {
        "temp": temp,
        "vibration": vibration,
        "pressure": pressure,
        "health": health,
    }


def train_stat_model(dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    temp = dataset["temp"]
    vibration = dataset["vibration"]
    pressure = dataset["pressure"]
    health = dataset["health"]

    model_artifact = {
        "feature_mean_temp": float(np.mean(temp)),
        "feature_mean_vibration": float(np.mean(vibration)),
        "feature_mean_pressure": float(np.mean(pressure)),
        "feature_std_temp": float(np.std(temp) + 1e-6),
        "feature_std_vibration": float(np.std(vibration) + 1e-6),
        "feature_std_pressure": float(np.std(pressure) + 1e-6),
        "max_expected_wear_hours": 1000.0,
        "health_to_rul_scale": float(np.mean(health) / 100.0),
    }
    return model_artifact


def build_sequences(
    dataset: Dict[str, np.ndarray], window_size: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack([dataset["temp"], dataset["vibration"], dataset["pressure"]], axis=1)
    health = dataset["health"]

    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-6
    normalized = (features - feature_mean) / feature_std

    sequences = []
    rul_targets = []
    health_last = []
    for idx in range(window_size, len(normalized)):
        seq = normalized[idx - window_size : idx]
        sequences.append(seq)
        h_val = float(health[idx])
        health_last.append(h_val)
        rul_targets.append(h_val / 100.0)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(rul_targets, dtype=np.float32),
        feature_mean.astype(np.float32),
        feature_std.astype(np.float32),
    )


def train_dl_model(
    sequences: np.ndarray, rul_targets: np.ndarray, epochs: int = 12
) -> Tuple[LSTMAutoencoderNet, float]:
    model = LSTMAutoencoderNet(input_dim=3, hidden_dim=16, latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    recon_loss_fn = nn.MSELoss()
    rul_loss_fn = nn.MSELoss()

    tensor_x = torch.tensor(sequences, dtype=torch.float32)
    tensor_rul = torch.tensor(rul_targets, dtype=torch.float32)

    model.train()
    last_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, predicted_rul = model(tensor_x)
        recon_loss = recon_loss_fn(reconstructed, tensor_x)
        rul_loss = rul_loss_fn(predicted_rul, tensor_rul)
        loss = recon_loss + 0.3 * rul_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    model.eval()
    return model, last_loss


def save_artifact(artifact: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def save_dl_artifacts(
    model: LSTMAutoencoderNet,
    metadata: Dict[str, object],
    model_path: Path,
    metadata_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def log_to_mlflow(
    stat_artifact_path: Path,
    stat_artifact: Dict[str, float],
    dl_model_path: Path,
    dl_metadata_path: Path,
    final_loss: float,
) -> None:
    if mlflow is None:
        print("mlflow_not_available skipping_mlflow_logging")
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./experiments/mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive-twin-pro")
    register_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="train_lstm_autoencoder"):
        mlflow.log_params(
            {
                "dataset_size": 2000,
                "artifact_type": "dl-lstm-autoencoder",
                "window_size": 16,
                "hidden_dim": 16,
                "latent_dim": 8,
            }
        )
        mlflow.log_metrics(
            {
                "final_loss": final_loss,
                "feature_mean_temp": stat_artifact["feature_mean_temp"],
                "feature_mean_vibration": stat_artifact["feature_mean_vibration"],
            }
        )
        mlflow.log_artifact(str(stat_artifact_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(dl_model_path), artifact_path="model_artifacts")
        mlflow.log_artifact(str(dl_metadata_path), artifact_path="model_artifacts")

        if register_name:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_artifacts/{dl_model_path.name}"
            mlflow.register_model(model_uri=model_uri, name=register_name)
            print(f"mlflow_model_registered name={register_name}")


def main() -> None:
    dataset = generate_synthetic_dataset()
    stat_artifact = train_stat_model(dataset)
    save_artifact(stat_artifact, ARTIFACT_PATH)

    sequences, rul_targets, feature_mean, feature_std = build_sequences(dataset, window_size=16)
    dl_model, final_loss = train_dl_model(sequences, rul_targets, epochs=12)
    dl_metadata = {
        "input_dim": 3,
        "hidden_dim": 16,
        "latent_dim": 8,
        "window_size": 16,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "max_expected_wear_hours": 1000.0,
    }
    save_dl_artifacts(dl_model, dl_metadata, DL_MODEL_PATH, DL_METADATA_PATH)

    print(f"trained_model_artifact_saved path={ARTIFACT_PATH}")
    print(f"dl_model_artifact_saved path={DL_MODEL_PATH}")
    print(f"dl_metadata_saved path={DL_METADATA_PATH}")
    log_to_mlflow(ARTIFACT_PATH, stat_artifact, DL_MODEL_PATH, DL_METADATA_PATH, final_loss)


if __name__ == "__main__":
    main()
